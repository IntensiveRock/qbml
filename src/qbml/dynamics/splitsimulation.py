import random
import logging
import numpy as np
import matplotlib.pyplot as plt

from qbml.dynamics.hami import Hami
from qbml.dynamics.splitredfield import SplitRedfield
import qbml.dynamics.spectraldensity as SPD

_logger = logging.getLogger(__name__)

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
    logging.basicConfig(level=logging.INFO)
    spd_class = getattr(SPD, spd_type)
    which_bath = {0 : "x", 1 : "z"}
    if rand_spd:
        nmSPECDEN = [spd_class.rand(spd_params.nmarkovian[which_bath[i]], HBAR, qubit_frequency, BETA) for i in range(N_BATHS)]
        mSPECDEN = [spd_class.rand(spd_params.markovian[which_bath[i]], HBAR, qubit_frequency, BETA) for i in range(N_BATHS)]

    # Run dynamics.
    HAMI = Hami(SYS_HAMI, 0, SB_HAMI, nmSPECDEN, TIMES[1])
    _logger.info('Constructing SplitRedfield object (making BCFs)')
    r = SplitRedfield(HAMI,
                      BETA,
                      TIMES,
                      hbar=HBAR,
                      nonmarkovian_spds=nmSPECDEN,
                      markovian_spds=mSPECDEN)
    _logger.info('Finished BCFs')
    rdm = np.array(r.propagate(ρ_0))
    return rdm, nmSPECDEN, mSPECDEN, [r.R_nm_ij, r.R_m_ij]
