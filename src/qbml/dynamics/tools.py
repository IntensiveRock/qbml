import random
from dataclasses import dataclass
from prettytable import PrettyTable
import numpy as np
import h5py


@dataclass
class RedfieldSummary:
    rdms: np.array
    bcf_m: np.array
    bcf_nm: np.array
    nm_spds: list
    m_spds: list
    Rnm: np.array
    Rm: np.array
    times: np.array
    freqs: np.array
    qfreq: float

def redfield_to_hdf5(summary : RedfieldSummary, fname : str):
    """
    Save a trajectory as a .hdf5 file.
    """
    redsum_dict = vars(summary)
    print(redsum_dict)
    with h5py.File(f"{fname}.hdf5", "w") as f:
        for key in redsum_dict.keys():
            if key == "nm_spds" or key == "m_spds":
                # for i, letter in enumerate(["x","z"]):
                #     f[key+letter] = redsum_dict[key][i].export_spd()
                pass
            else:
                f[key] = redsum_dict[key]
            print(f"Successfully added: {key}")


def resize_redfield_tensor(rft : np.array):
    """
    Resize the Redfield Tensor from (times,n**2,n**2) to (times,(n**2)*(n**2)*2) for use in training on R(t).
    """
    ntimes, nrows, ncols = rft.shape
    new_rft = np.zeros((ntimes, nrows*ncols*2))
    for n, r_t in enumerate(rft):
        flattened_time = np.array([[elem.real, elem.imag] for row in r_t for elem in row])
        if n == 100:
            print(np.concatenate(flattened_time))
        new_rft[n] += np.concatenate(flattened_time)
    return new_rft
    

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
