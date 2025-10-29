import logging
from dataclasses import dataclass
import pickle

import numpy as np
import scipy
import matplotlib.pyplot as plt

from qbml.dynamics.hami import Hami

_logger = logging.getLogger(__name__)

@dataclass
class RedfieldSummary:
    rdms: np.array
    nmspds: list
    mspds: list
    Rmnij: np.array
    Rmij: np.array
    times: np.array
    freqs: np.array
    qfreq: float

def save_RedfieldSummary(summary, pth):
    with open(pth, 'wb') as f:
        pickle.dump(summary, f)

def open_RedfieldSummary(pth):
    with open(pth, 'rb') as f:
        x = pickle.load(f)
    return x


class SplitRedfield:
    """
    Compute Redfield dynamics as R = R_nm + R_m

    This is EXPLICITELY for the exploration of Markvian vs. non-Markovian effects in spectral densities.
    """

    def __init__(
        self,
        hamiltonian : Hami,
        beta : float,
        times : np.array,
        hbar : float = 1,
        nonmarkovian_spds : list = None,
        markovian_spds : list = None,
    ):
        """Initialize the Redfield object."""
        self.hbar = hbar
        self.hami = hamiltonian
        self.basis_L = self._initialize_site_basis_L()
        self.beta = beta
        self.trace_commutator = self._trace_commutator()
        self.nm_bcf = self._bath_corr_func(nonmarkovian_spds, times)
        self.m_bcf = self._bath_corr_func(markovian_spds, times)
        # self.bcf = self._bath_corr_func(times)
        self.times = times
        self.dt = times[1]

    def _initialize_site_basis(self):
        """Grab the site basis vectors in the Schrodinger picture."""
        dim = self.hami.sys_hami.shape[0]
        basis = np.identity(dim)
        self.n = dim
        return basis, dim

    def _initialize_site_basis_L(self):
        """Grab the site basis in the Liouville picture."""
        basis_vec, n = self._initialize_site_basis()
        basis_L = np.array(
            [np.outer(basis_vec[i], basis_vec[j]) for i in range(n) for j in range(n)]
        )
        return basis_L

    def _initialize_system_time_evo(self, time: np.array):
        """Precompute the value of the sb operator(s) for all times t."""
        operator_time = []
        for i in range(len(self.hami.spdn)):
            operator_time.append(
                np.array(
                    [
                        self.hami.ev_sys_operator(self.hami.sb_hami[i], t_index)
                        for t_index, t in enumerate(time)
                    ]
                )
            )
        self.sys_op_time = np.array(operator_time, dtype=complex)


    def _bath_corr_func(self, spds : list, time: np.array) -> np.array:
        """Calculate the value of the bath correlation function."""
        bcf = np.zeros((len(spds), len(time)), dtype=complex)
        for i, spdlist in enumerate(spds):
            bcf[i] += spdlist.construct_bcf(beta=self.beta, time=time)
        return bcf

    def _trace_commutator(self):
        """Calculate the commutator that is present in each term of the derivation."""
        commutator = []
        for j in range(len(self.hami.spdn)):
            sys_a = np.array(
                [self.hami.sb_hami[j] @ self.basis_L[i].T for i in range(self.n**2)]
            )
            a_sys = np.array(
                [self.basis_L[i].T @ self.hami.sb_hami[j] for i in range(self.n**2)]
            )
            commutator.append(sys_a - a_sys)
        return np.array(commutator)

    def _construct_sys_rft(self):
        """Construct the system contribution to the RDM dynamics."""
        L_sys = []
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for m in range(self.n):
                        t1 = 0
                        t2 = 0
                        if j == m:
                            t1 = self.hami.sys_hami[i, k]
                        if i == k:
                            t2 = np.conjugate(self.hami.sys_hami[j, m])
                        L_sys.append(t1 - t2)
        L_sys = np.array(L_sys, dtype=complex)
        self.sys_rft = L_sys.reshape((self.n**2, self.n**2))

    def _construct_rft_t(self, bcf: np.array, t: float, t_index: int):
        """Construct the Redfield tensor at a particular point in time array."""
        # Construct the integrand at time t and then can easily integrate over.
        # Work out the first term.
        rft_t = np.zeros((self.n**2, self.n**2), dtype=complex)
        for op in range(len(bcf)):
            trace_1 = np.array(
                [
                    np.trace(
                        self.trace_commutator[op, i]
                        @ self.sys_op_time[op, t_index]
                        @ self.basis_L[j]
                    )
                    for i in range(self.n**2)
                    for j in range(self.n**2)
                ]
            )
            trace_1 = trace_1.reshape((self.n**2, self.n**2))
            trace_1 = trace_1 * bcf[op, t_index]
            trace_2 = np.array(
                [
                    np.trace(
                        self.trace_commutator[op, i]
                        @ self.basis_L[j]
                        @ self.sys_op_time[op, t_index]
                    )
                    for i in range(self.n**2)
                    for j in range(self.n**2)
                ]
            )
            trace_2 = trace_2.reshape((self.n**2, self.n**2))
            trace_2 = trace_2 * np.conjugate(bcf[op, t_index])
            rft_t += trace_1 - trace_2
        return rft_t

    def _construct_rft(self, bcf : list):
        """Wrap for private function to construct RFT at each time point."""
        self._construct_sys_rft()
        self._initialize_system_time_evo(self.times)
        R_ij_integrand = np.zeros(
            (len(self.times), self.n**2, self.n**2), dtype=complex
        )
        R_ij = np.zeros((len(self.times), self.n**2, self.n**2), dtype=complex)
        for t_index, time in enumerate(self.times):
            # Now just need to integrate each time step over the correct indicies.
            R_ij_integrand[t_index] += self._construct_rft_t(bcf, time, t_index)
            # May need changed. Need results first.
            R_ij_t_array = np.array(
                [
                    np.trapz(
                        R_ij_integrand[: t_index + 1, i, j], self.times[: t_index + 1]
                    )
                    for i in range(self.n**2)
                    for j in range(self.n**2)
                ],
                dtype=complex,
            )
            R_ij_t = R_ij_t_array.reshape((self.n**2, self.n**2))
            R_ij[t_index] += R_ij_t
        return R_ij

    def propagate(self, initial_condition, approx : int = 1):
        """Run the dynamics son."""
        logging.basicConfig(level=logging.INFO)
        _logger.info('Starting creation of R_nm')
        self.R_nm_ij = self._construct_rft(self.nm_bcf)
        _logger.info('Finished creation of R_nm')
        _logger.info('Starting creation of R_m')
        self.R_m_ij = self._construct_rft(self.m_bcf)
        _logger.info('Finished creation of R_m')
        # mat contains all pairs ti - tj, i > j.
        rdm = np.zeros((5,len(self.times),4), dtype=complex)
        sys = scipy.linalg.expm(-1j * self.sys_rft * self.dt)
        for t_index, t in enumerate(self.times):
            if t_index == 0:
                rdm[0,0] += initial_condition
                rdm[1,0] += initial_condition
                rdm[2,0] += initial_condition
                rdm[3,0] += initial_condition
                rdm[4,0] += initial_condition
            else:
                mat_nm = (self.R_nm_ij[t_index] + self.R_nm_ij[t_index - 1]) * self.dt / 2
                mat_m = (self.R_m_ij[t_index] + self.R_m_ij[t_index - 1]) * self.dt / 2
                # Just non-Markovian
                # Normal way. No approximation.
                mat2 = scipy.linalg.expm(mat_nm)
                mmat2 = scipy.linalg.expm(mat_m)
                mmat2_inv = scipy.linalg.expm(-mat_m)
                mat2_approx0 = scipy.linalg.expm(mat_nm+mat_m)
                # Taking the last value of the Markovian Redfield tensor. Approximation 1.
                mat2_approx1 = scipy.linalg.expm(mat_nm + self.R_m_ij[-1]*self.dt)
                # exp{R} = exp{R_m+R_mn}
                rho_t_a0 = sys @ mat2_approx0 @ rdm[0,t_index -1]
                # exp{R} = exp{R_mn+R_m(∞)}
                rho_t_a1 = sys @ mat2_approx1 @ rdm[1,t_index -1]
                # exp{R} = exp{R_m}exp{H_s}exp{R_mn}
                rho_t_a2 = mmat2 @ sys @ mat2 @ rdm[3,t_index -1]
                # exp{R} = exp{H_s}exp{R_mn}
                rho_t = sys @ mat2 @ rdm[2,t_index -1]
                # exp{R} = exp{-R_m}exp{R_m}exp{H_s}exp{R_mn}
                removeit = mmat2 @ sys @ mat2 @ rdm[3,t_index -1]
                removeit = mmat2_inv @ removeit
                pops = rho_t_a0[0]+rho_t_a0[3]
                assert 0.99 < pops < 1.01, "Population is not conserved"
                rdm[0,t_index] += rho_t_a0
                rdm[1,t_index] += rho_t_a1
                rdm[2,t_index] += rho_t
                rdm[3,t_index] += rho_t_a2
                rdm[4,t_index] += removeit
                # rdm.append(rho_t)
        return rdm


def _coth(w):
    return 1 / np.tanh(w)
