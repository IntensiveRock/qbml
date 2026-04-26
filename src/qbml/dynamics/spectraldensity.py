import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import random

import qbml.dynamics.bcf as BCF


class SpecDen:
    """Spectral density object."""

    def __init__(self, function, omega_infinity):
        """Initialize the spectral density. The function should calculate J(w) at w."""
        self.function = function
        self.omega_infinity = omega_infinity
        self.dlreorg_ratio = 1.

    def __call__(self, freq) -> np.array:
        """
        Callable spectral density calculates value at frequency.

        Wraps the function that represents the spectral density.
        """
        return self.function(freq)

    def plot(self, freqs: list) -> plt.figure:
        """Plot spectral densities based on defined API."""
        plt.plot(freqs, self(freqs))
        plt.show()

    def export_spd(self):
        """Export the spectral density parameters in a dictionary."""
        return vars(self)

    def construct_bcf(self, beta : float, time : np.array):
        bcf = BCF.RedfieldBCF(jw=self, beta=beta, times=time)
        return bcf

    def _calc_reorg_scaling_constant(self, tgt_reorg):
        """
        Compute the constant that scales spectral density to specified reorganization energy.
        """
        self.scaling_constant = 1
        omegas = np.linspace(0, self.omega_infinity, 100000)
        omegas = omegas[1:]
        j_w = self(omegas)
        reorg = (1 / np.pi) * np.trapezoid(j_w / omegas, omegas)
        return tgt_reorg / reorg


class DebyeSpecDen(SpecDen):
    """Debye Spectral Density."""

    def __init__(
        self,
        wc: float,
        lam: float,
    ):
        """Initialize the debye spectral density."""
        self.function = lambda w: 2 * lam * wc * w / (w**2 + wc**2)
        self.reorg_ratio = 1
        self.dlreorg_ratio = 1
        self.omega_infinity = 20 * wc


class Gaussians(SpecDen):
    """
    N-peak Gaussian shaped spectral density.

    Place a normalized Gaussian at center and scale the height by a scalar.
    """
    def __init__(
            self,
            centers : list[float],
            heights : list[float],
            fwhms : list[float],
            beta : float,
            tgt_reorg : float = 1,
    ):
        self.centers = centers
        self.heights = heights
        self.fwhms = fwhms
        self.beta = beta
        self.omega_infinity = 4 * fwhms[np.argmax(centers)]
        self.scaling_constant = self._calc_reorg_scaling_constant(tgt_reorg)

    def __call__(
            self,
            freq : np.array,
    ) -> np.array:
        j_w = np.zeros_like(freq)
        log2 = np.log(2)
        for i in range(len(self.heights)):
            j_w += self.heights[i] * np.exp(-4 * log2 * (freq - self.centers[i]) / self.fwhms[i]**2)
        tanh_prefactor = np.tanh(freq * self.beta / 2)
        return self.scaling_constant * tanh_prefactor * j_w

    @classmethod
    def rand(cls, spd_params : dict, qfreq : float, beta : float):
        """
        spd_params = {centers : [low, high],
                      fwhms  : [low, high],
                      scale   : [T/F, [low, high]],
                      n_peaks : int}
        """
        n_peaks = spd_params.n_peaks
        centers = [random.uniform(spd_params["centers"][0],spd_params["centers"][1]) for _ in range(n_peaks)]
        heights = [random.uniform(spd_params["heights"][0],spd_params["heights"][1]) for _ in range(n_peaks)]
        fwhms = [random.uniform(spd_params["fwhms"][0],spd_params["fwhms"][1]) for _ in range(n_peaks)]
        norm, scale = spd_params.scale
        reorganization_energy = random.uniform(scale[0], scale[1])
        return cls(centers=centers, heights=heights, fwhms=fwhms, beta=beta, tgt_reorg=reorganization_energy)


class Lorentzians(SpecDen):
    """N-peak lorentzian spectral density."""

    def __init__(
            self,
            centers: np.array,
            heights: np.array,
            widths: np.array,
            beta: float,
            tgt_reorg : float = 1,
    ):
        self.centers = centers
        self.heights = heights
        self.widths = widths
        self.beta = beta
        self.omega_infinity = (1.5 * max(self.centers)) + (30 * max(self.widths))
        self.scaling_constant = self._calc_reorg_scaling_constant(tgt_reorg)

    def __call__(
            self,
            freq
    ) -> np.array:
        j_w = np.zeros_like(freq)
        for i in range(len(self.heights)):
            term1 = self.heights[i] / (
                1 + ((np.sign(freq) * freq - self.centers[i]) / self.widths[i]) ** 2
            )
            j_w += term1
        tanh_prefactor = np.tanh(freq * self.beta / 2)
        return self.scaling_constant * tanh_prefactor * j_w

    @classmethod
    def rand(cls, spd_params : dict, qfreq : float, beta : float):
        """
        spd_params = {centers : [low, high],
                      heights : [low, high],
                      widths  : [low, high],
                      scale   : [T/F, [low, high],
                      n_peaks : int}
        """
        n_peaks = spd_params.n_peaks
        centers = [random.uniform(spd_params["centers"][0],spd_params["centers"][1]) for _ in range(n_peaks)]
        heights = [random.uniform(spd_params["heights"][0],spd_params["heights"][1]) for _ in range(n_peaks)]
        widths = [random.uniform(spd_params["widths"][0],spd_params["widths"][1]) for _ in range(n_peaks)]
        norm, scale = spd_params.scale
        reorganization_energy = random.uniform(scale[0], scale[1])
        return cls(centers, heights, widths, beta, reorganization_energy)


class TPSpecDen(SpecDen):
    """Class to handle three peak lorentzian spectral density."""

    def __init__(
        self,
        lams: list = None,
        wcs: list = None,
        dds: list = None,
        rand_lamb: float = None,
        beta: float = None,
        dimless: bool = False,
        hbar: float = 1,
        delta: float = None,
    ):
        """Initialize the three peak spectral density."""
        # Non-dimensionalizing parameters.
        self.hbar = hbar
        self.delta = delta
        # Dimensionful parameters.
        self.dim_full = {
            'beta' : beta * hbar,
            'lams' : lams,
            'wcs'  : wcs,
            'dds'  : dds,
            'rand_lamb' : rand_lamb
        }
        self.unnorm_reorg, self.omegas = self._prenorm_reorg(restore_dims=True,
                                                             setup=True
                                                             )
        self.reorg_ratio = self.dim_full['rand_lamb'] / self.unnorm_reorg
        # Dimless parameters.
        self.dimless = {
            'beta' : beta * delta,
            'lams' : lams / delta,
            'wcs'  : wcs * hbar / delta,
            'dds'  : dds * hbar / delta,
            'rand_lamb' : rand_lamb / delta
        }
        self.dlunnorm_reorg, self.dlomegas = self._prenorm_reorg(setup=True)
        self.dlreorg_ratio = self.dimless['rand_lamb'] / self.dlunnorm_reorg
        self.omega_infinity = np.max(self.dimless['dds']) * 40

    @classmethod
    def rand(cls, spd_params : dict):
        """
        Generate random TPSpecdens.
        spd_params = {lams : [low, high],
                      wcs  : [low, high],
                      dds : [low, high],
                      hi_height  : [low, high],
                      low_width  : [low, high],
                      hi_width   : [low, high]}
        """
        # return TPSpecDen(
        #     lams=np.array([rand_height() for _ in range(n_peaks[i])]),
        #     wcs=np.array([rand_bath_speed() for _ in range(n_peaks[i])]),
        #     dds=np.array([rand_centers() for _ in range(n_peaks[i])]),
        #     rand_lamb=rand_reorg(),
        #     hbar=HBAR,
        #     beta=β,
        #     delta=qubit_frequency)
        return 0


    def _prenorm_reorg(self, restore_dims: bool = False, setup: bool = False):
        """
        Normalize randomized spectral densities to randomized reorganization energy.
        """
        if restore_dims:
            params = self.dim_full
        else:
            params = self.dimless
        omegas = np.linspace(0, np.max(params['dds'])*40, 10000)
        omegas = omegas[1:]
        j_w = self(omegas, restore_dims=restore_dims, setup=setup)
        reorg = (1 / np.pi) * np.trapezoid(j_w / omegas, omegas)
        return reorg, omegas

    def __call__(self, freq, restore_dims: bool = False, setup: bool = False):
        """Make the spectral density callable in the form."""
        if setup:
            ratio = 1
            if restore_dims:
                params = self.dim_full
            else:
                params = self.dimless
        else:
            if restore_dims and not setup:
                params = self.dim_full
                ratio = self.reorg_ratio
            elif not restore_dims and not setup:
                params = self.dimless

        j_w = np.zeros_like(freq)
        for i in range(len(params['lams'])):
            term1 = params.get('lams')[i] / (
                1 + ((np.sign(freq) * freq - params.get('dds')[i]) / params.get('wcs')[i]) ** 2
            )
            j_w += term1
        tanh_prefactor = np.tanh(freq * params.get('beta') / 2)
        return tanh_prefactor * j_w


    def plot(self, freqs: list, restore_dims: bool = False) -> plt.figure:
        """Plot spectral densities based on defined API."""
        if restore_dims:
            plt.plot(freqs, self(freqs, restore_dims=restore_dims)*self.reorg_ratio)
        else:
            plt.plot(freqs, self(freqs, restore_dims=restore_dims)*self.dlreorg_ratio)
        plt.show()


class SplineSpecDen(SpecDen):
    """Spline spectral density predictions to rerun dynamics."""

    def __init__(self, freqs: list, j_w: list):
        """Initialize the Spline spectral density."""
        self.function = CubicSpline(freqs, j_w)
        self.dlreorg_ratio = 1
        self.omega_infinity = freqs[200]



class CompositeSpecDen(SpecDen):
    """Combine multiple spectral density types together."""

    def __init__(
            self,
            spd_list : list[SpecDen],
            tgt_reorg : float,
    ):
        self.spds = spd_list
        self.omega_infinity = np.max([spd.omega_infinity for spd in spd_list])
        scaling_constant = self._calc_reorg_scaling_constant(tgt_reorg)
        self._reset_spd_scaling_constants(scaling_constant)
        self.scaling_constant = 1

    def __call__(
            self,
            freq
    ):
        """Compute the value of J(ω) for the composite spectral density."""
        j_w = np.zeros_like(freq)
        for spd in self.spds:
            j_w += spd(freq)
        return self.scaling_constant * j_w

    def _reset_spd_scaling_constants(self, scaling_constant):
        """Recalculated the individual spd scaling constants."""
        for spd in self.spds:
            spd.scaling_constant *= scaling_constant

    def export_spd(self,):
        """Custom for exporting multiple spds together of database."""
        spd_dict = vars(self)
        # create a vars() dict for each spectral density in the list and use that to override initial_dict[spds]
        # then need a way to get them back. That should be easy enough.
        spd_dict['spds'] = {type(spd).__name__ : vars(spd) for spd in self.spds}
        return spd_dict
        
        

def _coth(w):
    return 1 / np.tanh(w)
