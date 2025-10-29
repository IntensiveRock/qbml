import os
import pickle
import random
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

import numpy as np

from qbml.dynamics.simulation import simulation


@hydra.main(version_base=None, config_path=f'{os.getcwd()}/configs/setgen/')
def main(cfg: DictConfig):

    # Ensure the user follows the proper procedure.
    assert cfg.title != "pleasechange", "Please change the title by passing: qmlviewer -cn <configname> title=<title_name> and that the config is in the PWD/configs/setgen/ directory!"

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
    ε = ε_in_frequency #/ SoL  # Convert frequency to wavenumber
    Δ = Δ_in_frequency #/ SoL  # Convert frequency to wavenumber
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
    for sim in range(cfg.simulation_parameters.num_sims):
        tomo, spds, R_ij = simulation(
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
        # for i, spd in enumerate(spds):
        #     spectral_densities[sim, :, i] += spd(FREQS)
        tomography[sim] += np.real(tomo)
        pred_fig = plt.figure(layout="constrained")

        gs = pred_fig.add_gridspec(nrows=2, ncols=2)
        dyn_ax = pred_fig.add_subplot(gs[0, :])
        j_x_ax = pred_fig.add_subplot(gs[1, 0])
        j_z_ax = pred_fig.add_subplot(gs[1, 1])
        lw = 0.5
        dyn_ax.plot(TIMES, tomo.real)
        dyn_ax.set_xlabel(r"Time [$\omega_q^{-1}$]")
        j_x_ax.plot(FREQS/qubit_frequency, spds[0](FREQS/qubit_frequency), lw=lw)
        j_z_ax.plot(FREQS/qubit_frequency, spds[1](FREQS/qubit_frequency), lw=lw)
        j_x_ax.set_xlabel(r"Frequency [$\omega_q$]")
        j_z_ax.set_xlabel(r"Frequency [$\omega_q$]")
        j_x_ax.set_ylabel(r"$J_x$")
        j_z_ax.set_ylabel(r"$J_z$")
        j_x_ax.set_ylim(bottom=0)
        j_z_ax.set_ylim(bottom=0)
        plt.show()
        fig = plt.figure(layout='constrained')
        figgs = fig.add_gridspec(nrows=4, ncols=4)
        for i in range(4):
            for j in range(4):
                tmp_axs = fig.add_subplot(figgs[i,j])
                tmp_axs.plot(TIMES, R_ij[:,i,j].real, color='black', lw=1)
                tmp_axs.plot(TIMES, R_ij[:,i,j].imag, color='red', lw=1)
        plt.show()
        spd_params.append([vars(spd) for i in spds])


if __name__ == "__main__":
    main()
