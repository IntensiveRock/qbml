import random
from prettytable import PrettyTable
import numpy as np


def rand_given_range(minimum: float,
                     maximum: float,
                     ):
    """Returns a function that selects values in the given range."""
    return lambda : random.uniform(minimum, maximum)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def compute_full_tomography_from_rdm(rdm : np.array):
    """
    Compute the full tomography of a given reduced density matrix.
    """
    sigma_x = np.expand_dims(rdm[:, 1] + rdm[:, 2], axis=-1)
    sigma_y = np.expand_dims(1j * rdm[:, 1] - 1j * rdm[:, 2], axis=-1)
    sigma_z = np.expand_dims(rdm[:, 0] - rdm[:, 3], axis=-1)
    spins_t = np.concatenate((sigma_x, sigma_y, sigma_z), axis=-1)
    return spins_t
