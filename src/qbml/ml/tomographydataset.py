import logging
import random
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

#from qubitml.tools import noisereduction as nr

_logger = logging.getLogger(__name__)


class TomographyDataSet(Dataset):
    """
    Generate qubit full tomography/spectral density datasets.
    Ensures that the dynamics correspond to the correct spectral densities.

    ...

    Attributes:
    ----------
    tomography : np.array
        Set of full tomography simulations that belong to the dataset.
    spec_dens : np.array
        Set of spectral densities that belong to the dataset.
    set_time_axis : np.array
        The frequency axis for the spectral densities.
    set_freq_axis : np.array
        The frequency axis for the spectral densities.
    transform : transform numpy arrays to different object. Ex. torch.from_numpy()
    """

    def __init__(
            self,
            tomography: np.array = None,
            spec_dens: np.array = None,
            set_time_axis: np.array = None,
            set_freq_axis: np.array = None,
            transform=None,
    ):
        """
        Initialize the Tomography dataset.

        :param tomography: Set of full tomography simulations that belong to the dataset.
        :type tomography: np.array
        :param spec_dens: Set of spectral densities that belong to the dataset.
        :type spec_dens: np.array
        :param set_time_axis: The time axis for the tomography.
        :type set_time_axis: np.array
        :param set_freq_axis: The frequency axis for the spectral densities.
        :type set_freq_axis: np.array
        :param transform: Change from numpy array to torch tensor.
        :returns: A customized pytorch dataset class that contains tomography and spec_dens.
        :rtype: torch.utils.data.Dataset
        """
        self.tomography = tomography
        self.spec_dens = spec_dens
        self.time_axis = set_time_axis
        self.freq_axis = set_freq_axis
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


def combine_tomos(
        set_1 : TomographyDataSet,
        set_2 : TomographyDataSet,
):
    assert set_1.time_axis.all() == set_2.time_axis.all(), "Sets have different time axes."
    assert set_1.freq_axis.all() == set_2.freq_axis.all(), "Sets have different frequency axes."
    dyns_1, spds_1 = set_1[:]
    dyns_2, spds_2 = set_2[:]
    new_dyns = torch.concatenate((dyns_1, dyns_2))
    new_spds = torch.concatenate((spds_1, spds_2))
    new_set = TomographyDataSet(tomography=new_dyns,
                                spec_dens=new_spds,
                                set_time_axis=set_1.time_axis,
                                set_freq_axis=set_1.freq_axis,
                                )
    return new_set


def construct_qubitml_dataloader(
        tomography_set : TomographyDataSet,
        mdl_input_seq_len: int,
        mdl_target_seq_len : int,
        shuffle : bool = False,
        batch_size: int = None,
        split: list = None,
) -> DataLoader:
    """
    Construct a qubitml dataset for neural network training or prediction.

    :param tomography_set: The set of simulations and spec_dens in the set.
    :type tomography_set: TomographyDataSet
    :param mdl_input_seq_len: The model input sequence length. Likely the length of the sim.
    :type mdl_input_seq_len: int
    :param mdl_target_seq_len: The model target sequence length. Likely the length of a spec_den.
    :type mdl_target_seq_len: int
    :param shuffle: Whether of not to shuffle the dataset. Suggestion: True for training, false for predictions.
    :type shuffle: bool, default=False
    :param split: How to split the dataset, ex. [0.9,0.1]. Useful for generating training and validation sets.
    :type split: list, optional
    :returns: A dataloader containing a qubitml dataset for training or predictions.
    :rtype: DataLoader
    """
    if mdl_input_seq_len != len(tomography_set.time_axis):
        _logger.critical("The model and dataset input sizes are unequal.")
    if mdl_target_seq_len != len(tomography_set.freq_axis):
        _logger.critical("The model and dataset target sizes are unequal.")
    if split is not None:
        set_0, set_1 = random_split(tomography_set, split)
        loader_0 = DataLoader(set_0, batch_size=batch_size, shuffle=shuffle)
        loader_1 = DataLoader(set_1, batch_size=batch_size, shuffle=shuffle)
        return loader_0, loader_1
    else:
        loader = DataLoader(tomography_set, batch_size=batch_size, shuffle=shuffle)
        return loader
    

def add_error_to_set(
        tomography_set : TomographyDataSet,
        error_percent : float,
) -> TomographyDataSet:
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
    return TomographyDataSet(tomography + error_tensor, spec_dens, time_axis, freq_axis)


def sparsify_set(
        tomography_set : TomographyDataSet,
        every_nth : int,
) -> TomographyDataSet:
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
    return TomographyDataSet(tomography, spec_dens, time_axis, freq_axis)


# def increase_resolution(
#         tomography_set
# ) -> TomographyDataSet:

def scale_set_specdens(
        tomography_set : TomographyDataSet,
        scale_factor : float,
) -> TomographyDataSet:
    """
    Scale the spectral density values at each frequency by the provided scale_factor.

    Effectively a y-axis unit change. This can make the ML process easier by scaling the values into a float32 regime.
    """
    time_axis = tomography_set.time_axis
    freq_axis = tomography_set.freq_axis
    tomography, spec_dens = tomography_set[:]
    spec_dens *= scale_factor
    return TomographyDataSet(tomography, spec_dens, time_axis, freq_axis)


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

class FitTomo(nn.Module):

    def __init__(
            self,
            initial_frq : float = 1.,
    ):
        super().__init__()
        self.w_q = nn.Parameter(
            torch.Tensor([initial_frq]),
            requires_grad=True
        )
        self.exp_w = nn.Parameter(
            torch.rand(4, dtype=torch.float) * 0.1,
            requires_grad=True
        )
        self.linear_w = nn.Parameter(
            torch.rand(2, dtype=torch.float) * 0.01,
            requires_grad=True
        )
        self.quad_w = nn.Parameter(
            torch.rand(1, dtype=torch.float) * 0.01,
            requires_grad=True
        )
        self.scale = nn.Parameter(
            torch.rand(1, dtype=torch.float),
            requires_grad=True
        )

    def forward(
            self,
            x_tensor : torch.Tensor,
    ):
        sigma_x = torch.exp(-self.exp_w[0] * x_tensor) * torch.cos(self.w_q * x_tensor) - self.linear_w[0] * x_tensor + self.quad_w[0] * x_tensor
        sigma_y = torch.exp(-self.exp_w[1] * x_tensor) * torch.sin(self.w_q * x_tensor)
        sigma_z = (torch.exp(-self.exp_w[2] * x_tensor) * torch.cos(self.w_q * x_tensor + torch.pi) + torch.exp(-self.exp_w[3] * x_tensor)  - self.linear_w[1] * x_tensor) * self.scale
        sigma_x = sigma_x.unsqueeze(1)
        sigma_y = sigma_y.unsqueeze(1)
        sigma_z = sigma_z.unsqueeze(1)
        return torch.cat((sigma_x, sigma_y, sigma_z), 1)
    


def fit(fitter, dataloader, loss_fn, optimizer, time_axis, idx : int):
    """Define the training loop for the neural network."""
    fitter.train()
    training_loss = 0
    fitted_tomos = torch.zeros(len(dataloader), len(time_axis), 3)
    for batch, (src, tgt) in tqdm(enumerate(dataloader)):
        src = src[0]
        tmp_fit = torch.zeros(len(time_axis), 3)
        for epoch in range(500):
            logits = fitter(time_axis.float())
            tmp_fit = logits.detach()
            loss = loss_fn(logits, src)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            for p in fitter.parameters():
                p.data.clamp_(0)
        fitted_tomos[batch] += tmp_fit
    torch.save(fitted_tomos, f"{idx}fit.ds")
