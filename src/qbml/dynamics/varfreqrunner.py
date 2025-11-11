import os
import random
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np
from tqdm import tqdm

from qbml.dynamics.simulation import simulation
from qbml.ml.variablefreqdataset import VariableFreqDataSet
from qbml.ml.spddb import save_spddb


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # Get pathing set for save.
    set_path = Path(cfg.prj_dir) / 'data' / cfg.title
    os.mkdir(set_path)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

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
    random.seed(cfg.simulation_parameters.seed)

    # Define the units and constants.
    HBAR = HBAR["dimensionless"]

    # Generate a list of Parameters for each.
    ɛ_list = [random.uniform(cfg.system.ɛ_min, cfg.system.ɛ_max) for _ in range(cfg.simulation_parameters.num_sims)]
    Δ_list = [random.uniform(cfg.system.Δ_min, cfg.system.Δ_max) for _ in range(cfg.simulation_parameters.num_sims)]

    # Hami
    SB_HAMI = np.array(cfg.coupling.sb_operators)
    N_BATHS = len(SB_HAMI)

    # Simulation specific parameters.
    T = cfg.simulation_parameters.temperature
    kT = k["dimensionless"] * T
    β = 1 / kT  # * k
    # print("The value of β is: ", β)
    ρ_0 = cfg.simulation_parameters.initial_condition
    t_min = cfg.simulation_parameters.t_min
    t_max = cfg.simulation_parameters.t_max
    δt = cfg.simulation_parameters.dt
    times = np.arange(t_min, t_max, δt)

    TIMES = times
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
    for i, sim in tqdm(enumerate(range(cfg.simulation_parameters.num_sims))):
        ɛ = ɛ_list[i]
        Δ = Δ_list[i]
        qubit_frequency = (Δ ** 2 + ɛ ** 2) ** 0.5
        SYS_HAMI = np.array([[ɛ,  Δ],
                             [Δ, -ɛ]])
        SYS_HAMI = SYS_HAMI / qubit_frequency
        BETA = β * qubit_frequency
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
        for i, spd in enumerate(spds):
            spectral_densities[sim, :, i] += spd(FREQS)
        tomography[sim] += np.real(tomo)
        spd_params.append(spds)

    # Create the Tomographydataset.
    dataset = VariableFreqDataSet(tomography,
                                  spectral_densities,
                                  times,
                                  FREQS,
                                  [ɛ_and_Δ for ɛ_and_Δ in zip(ɛ_list, Δ_list)],
                                  torch.from_numpy)
    torch.save(dataset, set_path / f'{cfg.title}.ds')
    save_spddb(spd_params, cfg.title, set_path)
    shutil.move(output_dir / '.hydra', set_path / 'hydra')
    if cfg.to_pics:
        dataset.to_pics()
        os.system("feh tmp")
        if not cfg.save_pics:
            os.system("rm -r tmp")

if __name__ == "__main__":
    main()
