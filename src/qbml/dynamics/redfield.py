import numpy as np
import scipy
import matplotlib.pyplot as plt

from qbml.dynamics.hami import Hami


class Redfield:
    """
    Runs Redfield dynamics for the provided system.

    This class object facilitates the propagation of redfield dynamics.
    To effectively run dynamics, you must have the following:
    - A Hamiltonian (Hami).
    - An inverse temperature.
    - The times over which to calculate the solution.
    - The frequencies over which to integrate the bath correlation function.
    """

    def __init__(
        self,
        hamiltonian: Hami,
        beta: float,
        times: np.array,
        hbar: float = 1,
    ):
        """Initialize the Redfield object."""
        self.hbar = hbar
        self.hami = hamiltonian
        self.basis_L = self._initialize_site_basis_L()
        self.beta = beta
        self.trace_commutator = self._trace_commutator()
        self.bcf = self._bath_corr_func(times)
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


    def _bath_corr_func(self, time: np.array) -> np.array:
        """Calculate the value of the bath correlation function."""
        bcf = np.zeros((len(self.hami.spdn), len(time)), dtype=complex)
        for i, spdlist in enumerate(self.hami.spdn):
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
        # L_sys = np.zeros((self.n**2, self.n**2))
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

    def _construct_rft_t(self, t: float, t_index: int):
        """Construct the Redfield tensor at a particular point in time array."""
        # Construct the integrand at time t and then can easily integrate over.
        # Work out the first term.
        rft_t = np.zeros((self.n**2, self.n**2), dtype=complex)
        for op in range(len(self.hami.spdn)):
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
            trace_1 = trace_1 * self.bcf[op, t_index]
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
            trace_2 = trace_2 * np.conjugate(self.bcf[op, t_index])
            rft_t += trace_1 - trace_2
        return rft_t

    def _construct_rft(self):
        """Wrap for private function to construct RFT at each time point."""
        self._construct_sys_rft()
        self._initialize_system_time_evo(self.times)
        R_ij_integrand = np.zeros(
            (len(self.times), self.n**2, self.n**2), dtype=complex
        )
        R_ij = np.zeros((len(self.times), self.n**2, self.n**2), dtype=complex)
        for t_index, time in enumerate(self.times):
            # Now just need to integrate each time step over the correct indicies.
            R_ij_integrand[t_index] += self._construct_rft_t(time, t_index)
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
        self.R_ij = R_ij

    def propagate(self, initial_condition):
        """Run the dynamics son."""
        self._construct_rft()
        # mat contains all pairs ti - tj, i > j.
        rdm = []
        sys = scipy.linalg.expm(-1j * self.sys_rft * self.dt)
        for t_index, t in enumerate(self.times):
            if t_index == 0:
                rdm.append(initial_condition)
            else:
                mat2 = scipy.linalg.expm(
                    (self.R_ij[t_index] + self.R_ij[t_index - 1]) * self.dt / 2
                )
                rho_t = sys @ mat2 @ rdm[t_index -1]
                pops = rho_t[0]+rho_t[3]
                assert 0.99 < pops < 1.01, "Population is not conserved"
                rdm.append(rho_t)
        return rdm


def _coth(w):
    return 1 / np.tanh(w)
