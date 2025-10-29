import random
import numpy as np
import matplotlib.pyplot as plt

from qbml.dynamics.hami import Hami
from qbml.redfield import Redfield
import qbml.dynamics.spectraldensity as SPD
from qbml.dynamics.tools import rand_given_range


def simulation(
        spd_type: str,
        rand_spd: bool,
        spd_params: dict,
        β: float, # has dimensions
        BETA: float, # no dimensions
        HBAR: float,
        qubit_frequency: float,
        TIMES: np.array,
        N_BATHS: int,
        SYS_HAMI: np.array,
        SB_HAMI: np.array,
        ρ_0: np.array,

) -> np.array:
    # Generate random spectral densities.
    spd_class = getattr(SPD, spd_type)
    if rand_spd:
        SPECDEN = [spd_class.rand(spd_params, HBAR, qubit_frequency, β) for _ in range(N_BATHS)]
        norm, scale = spd_params.scale
        # if norm:
        #     SPECDEN = [SPD.NormalizedSpecDen(spd, random.uniform(scale[0], scale[1])/qubit_frequency) for spd in SPECDEN]

    # Run dynamics.
    HAMI = Hami(SYS_HAMI, 0, SB_HAMI, SPECDEN, TIMES[1])
    r = Redfield(HAMI, BETA, TIMES, hbar=HBAR)

    rdm = np.array(r.propagate(ρ_0))
    sigma_x = np.expand_dims(rdm[:, 1] + rdm[:, 2], axis=-1)
    sigma_y = np.expand_dims(1j * rdm[:, 1] - 1j * rdm[:, 2], axis=-1)
    sigma_z = np.expand_dims(rdm[:, 0] - rdm[:, 3], axis=-1)
    spins_t = np.concatenate((sigma_x, sigma_y, sigma_z), axis=-1)
    return spins_t, SPECDEN, r.R_ij
