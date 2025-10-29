"""
Generate a dummy Redfield tensor given system parameters and guess λ's.

Plug into removemarkovianity.py:propagate()
"""

import logging
from dataclasses import dataclass
import numpy as np
import scipy

from qbml.dynamics.redfield import Redfield
from qbml.dynamics.hami import Hami
from qbml.dynamics.spectraldensity import Lorentzians
import qbml.dynamics.tools as qbmltools


_logger = logging.getLogger(__name__)


@dataclass
class RedfieldTensorGuess:
    """Class to wrap guess Redfield Tensor and λ{x,z} parameters."""
    λ_x : float
    λ_z : float
    Rft : np.array
    invMark : np.array
    spds : list
    rhotated : np.array = None
    tomos : np.array = None

    def apply_rotations(self, set_of_rhos):
        rho_tilda = np.zeros_like(set_of_rhos, dtype=complex)
        for t_idx, ρ in enumerate(set_of_rhos):
            rho_tilda[t_idx] = self.invMark[t_idx] @ ρ
        self.rhotated = rho_tilda
        self.compute_tomo()

    def compute_tomo(self):
        self.tomos = qbmltools.compute_full_tomography_from_rdm(self.rhotated)


def propagate(
    Rmij_ts : np.array,
    dt : float) -> np.array:
    """
    Compute exp{-R_m(t)} and invert each snapshot of the density matrix.
    """
    rotations = [np.eye(4, dtype=complex)]
    for t_index, R_t in enumerate(Rmij_ts):
        if t_index != 0:
            mat_m = (Rmij_ts[t_index] + Rmij_ts[t_index - 1]) * dt / 2
            mmat2 = scipy.linalg.expm(-mat_m)
            rotations.append(mmat2 @ rotations[t_index - 1])
    return np.array(rotations)


def generate_guess_rft(
        sys_hami : np.array,
        sb_hami : np.array,
        beta : float,
        times : np.array,
        guess_reorgs : list,
        qubit_frequency : float,
        N_BATHS : int,
        hbar : float = 1,
):
    """
    Generate a set of guess Markovian Redfield tensors given reorganization energies.
    """
    SPECDEN = [Lorentzians(
        centers=[100.], # in units of qfreq
        heights=[1.], # doesn't matter, gets rescaled
        widths=[1.5], # May matter
        beta=beta, # Needs to be the dimensionless one
        tgt_reorg=guess_reorgs[i],
    ) for i in range(N_BATHS)]

    HAMI = Hami(sys_hami, 0, sb_hami, SPECDEN, times[1])
    r = Redfield(HAMI,
                 beta,
                 times,
                 hbar=hbar,)
    r._construct_rft()
    return propagate(r.R_ij, times[1]), SPECDEN, r.R_ij

