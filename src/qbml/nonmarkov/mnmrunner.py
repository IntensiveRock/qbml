import os
import random
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from qbml.dynamics.splitsimulation import simulation
from qbml.dynamics.splitredfield import RedfieldSummary


def mnmsimrunner(cfg_pth : Path):

    # Ensure the user follows the proper procedure.
    cfg = OmegaConf.load(cfg_pth)

    # Unit dictionaries.
    HBAR = {
    "cm-1 fs": 5308.8,
    "cm-1 ps": 5.3088,
    "cm-1 ns": 5.3088e-3,
    "cm-1 us": 5.3088e-6,
    "MHz us" : 0.15915,
    "dimensionless": 1.0,
    }
    k = {"cm-1 K-1": 0.695,
         "MHz K-1" : 2.0835e4,
         "dimensionless" : 1.}
    c = {
        "cm GHz": 30,
        "cm MHz": 3e4,
        "cm THz": 3e7,
        "dimensionless": 1.,
    }
    random.seed(cfg.simulation_parameters.seed)

    # Define the units and constants.
    HBAR = HBAR[cfg.units.hbar]
    SoL = c[cfg.units.speed_of_light]

    # Model parameters.
    ε_in_frequency = cfg.system.epsilon_frequency
    Δ_in_frequency = cfg.system.delta_frequency
    ε = ε_in_frequency / SoL  # Convert frequency to wavenumber
    Δ = Δ_in_frequency / SoL  # Convert frequency to wavenumber
    qubit_frequency = (Δ ** 2 + ε ** 2) ** 0.5
    SYS_HAMI = np.array([[ε,  Δ],
                         [Δ, -ε]])
    SB_HAMI = np.array(cfg.coupling.sb_operators)
    N_BATHS = len(SB_HAMI)

    # Simulation specific parameters.
    T = cfg.simulation_parameters.temperature
    kT = k[cfg.units.k] * T
    β = 1 / kT  # * k
    # print("The value of β is: ", β)
    ρ_0 = cfg.simulation_parameters.initial_condition
    t_min = cfg.simulation_parameters.t_min
    t_max = cfg.simulation_parameters.t_max
    δt = cfg.simulation_parameters.dt
    times = np.arange(t_min, t_max, δt)

    # Nondimensionalize by qubit frequency and ħ.
    SYS_HAMI = SYS_HAMI / qubit_frequency
    BETA = β * qubit_frequency
    TIMES = times * qubit_frequency / HBAR
    FREQS = np.arange(cfg.simulation_parameters.ω_min,
                      cfg.simulation_parameters.ω_max,
                      cfg.simulation_parameters.dω)

    # Initialize the data storage for simulations.
    tomography = np.zeros((cfg.simulation_parameters.num_sims,
                           len(TIMES),
                           3))
    spectral_densities = np.zeros((cfg.simulation_parameters.num_sims,
                                   len(FREQS),
                                   2))
    spd_params = []
    # Run the simulations.
    rdms, nmspds, mspds, R_ijs = simulation(
        cfg.specden.type,
        cfg.specden.random,
        cfg.specden.params,
        β,
        BETA,
        HBAR,
        qubit_frequency,
        TIMES,
        N_BATHS,
        SYS_HAMI,
        SB_HAMI,
        ρ_0,
    )
    Rmnij = R_ijs[0]
    Rmij = R_ijs[1]
    summary = RedfieldSummary(
                rdms=rdms,
                nmspds=nmspds,
                mspds=mspds,
                Rmnij=Rmnij,
                Rmij=Rmij,
                times=TIMES,
                freqs=FREQS/qubit_frequency,
                qfreq=qubit_frequency
            )
    return summary
