import logging
import os
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import Tensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


_logger = logging.getLogger(__name__)


class VariableFreqDataSet(Dataset):
    """
    Generate simulation datasets with varying qubit frequencies.
    """

    def __init__(
            self,
            tomography: np.array = None,
            spec_dens: np.array = None,
            set_time_axis: np.array = None,
            set_freq_axis: np.array = None,
            ɛ_and_Δ: np.array = None,
            transform=None,
    ):
        """
        Initialize the VariableFreq dataset.
        """
        self.tomography = tomography
        self.spec_dens = spec_dens
        self.time_axis = set_time_axis
        self.freq_axis = set_freq_axis
        self.ɛ_and_Δ = ɛ_and_Δ
        self.transform = transform
        

    def __len__(self):
        """Redefine len to return the number of simulations in the set."""
        return len(self.tomography)

    def __getitem__(self, idx: int = None) -> Tensor:
        """
        Define getitem to agree with the Pytorch API.

        :param idx: index of the tomography/spec_den pair to retrieve.
        :type idx: int
        :returns: The tomography and spec_dens at the provided index.
        :rtype: torch.Tensor
        
        """
        if self.transform:
            tomography_item = self.transform(self.tomography[idx])
            tomography_item = tomography_item.float()
            spec_dens_item = self.transform(self.spec_dens[idx])
            spec_dens_item = spec_dens_item.float()
        else:
            tomography_item = self.tomography[idx]
            tomography_item = tomography_item.float()
            spec_dens_item = self.spec_dens[idx]
            spec_dens_item = spec_dens_item.float()
        return tomography_item, spec_dens_item

    def to_pics(self):
        """
        Generate pictures of the dataset.

        This should be used for testing purposes only!
        Will consume much storage space.
        """
        _logger.info("Preparing images...")
        working_dir = os.getcwd()
        tmp_dir_pth = Path(working_dir) / Path("tmp")
        os.mkdir(tmp_dir_pth)
        for n, (tomos, spds) in enumerate(self):
            tmp_fig = plt.figure(layout="constrained", figsize=[8,6], dpi=100)
            tmp_gs = tmp_fig.add_gridspec(nrows=2, ncols=2)
            tmp_dyn_ax = tmp_fig.add_subplot(tmp_gs[0,:])
            tmp_jx_ax = tmp_fig.add_subplot(tmp_gs[1,0])
            tmp_jz_ax = tmp_fig.add_subplot(tmp_gs[1,1])
            for measurement in range(3):
                tmp_dyn_ax.plot(self.time_axis, tomos[:,measurement])
            tmp_jx_ax.plot(self.freq_axis, spds[:,0])
            tmp_jz_ax.plot(self.freq_axis, spds[:,1])
            tmp_dyn_ax.set_xlabel("Time [$\omega_q^{-1}$]")
            tmp_dyn_ax.set_ylim([-1.1, 1.1])
            tmp_fig.savefig(tmp_dir_pth / f"{n}.png")
            plt.close(tmp_fig)



def combine_tomos(
        set_1 : VariableFreqDataSet,
        set_2 : VariableFreqDataSet,
):
    assert set_1.time_axis.all() == set_2.time_axis.all(), "Sets have different time axes."
    assert set_1.freq_axis.all() == set_2.freq_axis.all(), "Sets have different frequency axes."
    dyns_1, spds_1 = set_1[:]
    dyns_2, spds_2 = set_2[:]
    new_dyns = torch.concatenate((dyns_1, dyns_2))
    new_spds = torch.concatenate((spds_1, spds_2))
    new_qfreqs = torch.concatenate((set_1.ɛ_and_Δ, set_2.ɛ_and_Δ))
    new_set = VariableFreqDataSet(
        tomography=new_dyns,
        spec_dens=new_spds,
        set_time_axis=set_1.time_axis,
        set_freq_axis=set_1.freq_axis,
        ɛ_and_Δ=new_qfreqs,
    )
    return new_set
    

def add_error_to_set(
        tomography_set : VariableFreqDataSet,
        error_percent : float,
) -> VariableFreqDataSet:
    """
    Add 'measurement error' to dataset. A random number is sample from a Gaussian with σ=error_percent.

    :param tomography_set: The dataset to add error to.
    :type tomography_set: TomographyDataSet
    :param error_percent: The standard deviation, σ, of the Gaussian sampled. Ex. 5% = 0.05
    :type error_percent: float
    :returns: A new dataset that has error added to the provided dataset's tomography.
    :rtype: TomographyDataSet

    """
    time_axis = tomography_set.time_axis
    freq_axis = tomography_set.freq_axis
    tomography, spec_dens = tomography_set[:]
    error_tensor = torch.randn(tomography.shape)
    error_tensor[:, :, :2] = error_percent * error_tensor[:, :, :2]
    error_tensor[:, :, 2] = error_percent * torch.max(tomography[:, :, 2]) * error_tensor[:, :, 2]
    return VariableFreqDataSet(tomography + error_tensor, spec_dens, time_axis, freq_axis, tomography_set.ɛ_and_Δ)


def sparsify_set(
        tomography_set : VariableFreqDataSet,
        every_nth : int,
) -> VariableFreqDataSet:
    """
    Make the provided set sparse by grabbing every_nth point from the tomography for each simulation.

    :param tomography_set: The dataset to sparsify.
    :type tomography_set: TomographyDataSet
    :param every_nth: Grab every nth point from each simulation and the corresponding time_axis.
    :type every_nth: int
    :returns: A tomography dataset with the desired sparsity.
    :rtype: TomographyDataSet

    """
    time_axis = tomography_set.time_axis[::every_nth]
    freq_axis = tomography_set.freq_axis
    tomography, spec_dens = tomography_set[:]
    tomography = tomography[:, ::every_nth]
    return VariableFreqDataSet(tomography, spec_dens, time_axis, freq_axis, tomography_set.ɛ_and_Δ)


# def increase_resolution(
#         tomography_set
# ) -> TomographyDataSet:

def scale_set_specdens(
        tomography_set : VariableFreqDataSet,
        scale_factor : float,
) -> VariableFreqDataSet:
    """
    Scale the spectral density values at each frequency by the provided scale_factor.

    Effectively a y-axis unit change. This can make the ML process easier by scaling the values into a float32 regime.
    """
    time_axis = tomography_set.time_axis
    freq_axis = tomography_set.freq_axis
    tomography, spec_dens = tomography_set[:]
    spec_dens *= scale_factor
    return VariableFreqDataSet(tomography, spec_dens, time_axis, freq_axis, tomography_set.ɛ_and_Δ)


def _assert_seq_len(
    self,
    spins: np.array,
    model_seq_len: int,
    p_time_axis: np.array,
    m_time_axis: np.array,
):
    provided_seq_len = spins.shape[-2]
    if model_seq_len is None:
        return spins
    elif provided_seq_len < model_seq_len:
        # Spline those dudes.
        splined_spins = np.zeros((spins.shape[0], model_seq_len, spins.shape[-1]))
        for index, sim in enumerate(spins):
            sigma_x = CubicSpline(p_time_axis, sim[:, 0])
            splined_spins[index, :, 0] = sigma_x(m_time_axis)
            sigma_y = CubicSpline(p_time_axis, sim[:, 1])
            splined_spins[index, :, 1] = sigma_y(m_time_axis)
            sigma_z = CubicSpline(p_time_axis, sim[:, 2])
            splined_spins[index, :, 2] = sigma_z(m_time_axis)

        return splined_spins
    elif provided_seq_len == model_seq_len:
        return spins
    else:
        _logger.critical(
            "The provided sequence is larger than the model sequence length!"
        )
