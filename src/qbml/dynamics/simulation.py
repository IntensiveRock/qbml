import random
import numpy as np

from qbml.dynamics.hami import Hami
from qbml.dynamics.redfield import Redfield
import qbml.dynamics.spectraldensity as SPD


def simulation(
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
    coupling_assignment_keys = [key for key in spd_params.keys()]
    SPECDEN = []
    for coupling in coupling_assignment_keys:
        coupling_params = spd_params[coupling]
        types = [key for key in coupling_params.types.keys()]
        ntypes = len(types)
        if len(types) == 1:
            spd_class = getattr(SPD, types[0])
            SPECDEN.append(spd_class.rand(coupling_params.types[types[0]], qubit_frequency, BETA))
        else:
            # Random split the number of peaks
            n_peaks_per_type = np.zeros((ntypes), dtype=int)
            spdlist = []
            for i in range(ntypes-1):
                if np.sum(n_peaks_per_type) == coupling_params.n_peaks:
                    break
                else:
                    n_peaks_per_type[i] += random.randint(0,coupling_params.n_peaks-np.sum(n_peaks_per_type))
            n_peaks_per_type[-1] = coupling_params.n_peaks - np.sum(n_peaks_per_type)
            for i, spd_type in enumerate(types):
                if int(n_peaks_per_type[i]) == 0:
                    break
                spd_class = getattr(SPD, spd_type)
                coupling_params.types[spd_type].n_peaks = int(n_peaks_per_type[i])
                spdlist.append(spd_class.rand(coupling_params.types[spd_type], qubit_frequency, BETA))
            SPECDEN.append(SPD.CompositeSpecDen(spdlist, random.uniform(coupling_params.total_reorg[0], coupling_params.total_reorg[1])))

    # Run dynamics.
    HAMI = Hami(SYS_HAMI, 0, SB_HAMI, SPECDEN, TIMES[1])
    r = Redfield(HAMI, BETA, TIMES, hbar=HBAR)

    rdm = np.array(r.propagate(ρ_0))
    sigma_x = np.expand_dims(rdm[:, 1] + rdm[:, 2], axis=-1)
    sigma_y = np.expand_dims(1j * rdm[:, 1] - 1j * rdm[:, 2], axis=-1)
    sigma_z = np.expand_dims(rdm[:, 0] - rdm[:, 3], axis=-1)
    spins_t = np.concatenate((sigma_x, sigma_y, sigma_z), axis=-1)
    return spins_t, SPECDEN, r.R_ij
