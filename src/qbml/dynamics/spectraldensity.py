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
        reorg = (1 / np.pi) * np.trapz(j_w / omegas, omegas)
        return tgt_reorg / reorg, omegas


class NormalizedSpecDen(SpecDen):
    """
    Normalized spectral density object.

    Wraps any spectral density type. Ensures desired reorganization energy.
    """
    def __init__(self, spd : SpecDen, reorganization_energy : float):
        self.spd = spd
        self.reorganization_energy = reorganization_energy
        self.scaling_constant, self.omegas = self._calc_reorg_scaling_constant(spd, reorganization_energy)
        self.omega_infinity = spd.omega_infinity

    def __call__(self, omegas):
        """
        Redefine call function to compute the normalized spectral density.
        """
        return self.scaling_constant * self.spd(omegas)

    def _calc_reorg_scaling_constant(self, spd, tgt_reorg):
        """
        Compute the constant that scales spectral density to specified reorganization energy.
        """
        omegas = np.linspace(0, spd.omega_infinity, 10000)
        omegas = omegas[1:]
        j_w = spd(omegas)
        reorg = (1 / np.pi) * np.trapz(j_w / omegas, omegas)
        return tgt_reorg / reorg, omegas


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
        self.omega_infinity = max(self.centers) + max(self.centers)
        self.scaling_constant, self.omegas = self._calc_reorg_scaling_constant(tgt_reorg)

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
    def rand(cls, spd_params : dict, hbar : float, qfreq : float, beta : float):
        """
        spd_params = {centers : [low, high],
                      heights : [low, high],
                      widths  : [low, high],
                      scale   : [T/F, [low, high],
                      n_peaks : int}
        """
        n_peaks = spd_params.n_peaks
        centers = [random.uniform(spd_params["centers"][0],spd_params["centers"][1])*hbar/qfreq for _ in range(n_peaks)]
        heights = [random.uniform(spd_params["heights"][0],spd_params["heights"][1]) for _ in range(n_peaks)]
        widths = [random.uniform(spd_params["widths"][0],spd_params["widths"][1])*hbar/qfreq for _ in range(n_peaks)]
        norm, scale = spd_params.scale
        reorganization_energy = random.uniform(scale[0], scale[1])/qfreq
        return cls(centers, heights, widths, beta, reorganization_energy)


class NonMarkovLorentz(SpecDen):
    """
    Spectral density with low and very high frequency Lorentzian peaks.

    One low frequency peak contributes to Non-Markovian evolution.
    One high frequency peak drives Markovian evolution.
    """
    def __init__(self, low_center, hi_center, heights, widths, beta, tgt_reorg : float = 1):
        self.centers = [low_center, hi_center]
        self.heights = heights
        self.widths = widths
        self.beta = beta
        self.omega_infinity = max(self.centers) + 0.2*hi_center
        self.scaling_constant, self.omegas = self._calc_reorg_scaling_constant(tgt_reorg)

    @classmethod
    def rand(cls, spd_params : dict, hbar : float, qfreq : float, beta : float):
        """
        spd_params = {low_center : [low, high],
                      hi_center  : [low, high],
                      low_height : [low, high],
                      hi_height  : [low, high],
                      low_width  : [low, high],
                      hi_width   : [low, high]}
        """
        # Add normalization to specified
        low_center = random.uniform(spd_params["low_center"][0],spd_params["low_center"][1])
        hi_center = random.uniform(spd_params["hi_center"][0],spd_params["hi_center"][1])
        heights = [random.uniform(spd_params["low_height"][0],spd_params["low_height"][1]),
                   random.uniform(spd_params["hi_height"][0],spd_params["hi_height"][1])]
        widths = [random.uniform(spd_params["low_width"][0],spd_params["low_width"][1])*hbar/qfreq,
                   random.uniform(spd_params["hi_width"][0],spd_params["hi_width"][1])*hbar/qfreq]
        norm, scale = spd_params.scale
        reorganization_energy = random.uniform(scale[0], scale[1])/qfreq
        print(reorganization_energy)
        return cls(low_center*hbar/qfreq, hi_center*hbar/qfreq, heights, widths, beta*qfreq, reorganization_energy)

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

    def computelow(self, freq):
        j_w = np.zeros_like(freq)
        term1 = self.heights[0] / (
            1 + ((np.sign(freq) * freq - self.centers[0]) / self.widths[0]) ** 2
        )
        j_w += term1
        tanh_prefactor = np.tanh(freq * self.beta / 2)
        return self.scaling_constant * tanh_prefactor * j_w

    def computehi(self, freq):
        j_w = np.zeros_like(freq)
        term1 = self.heights[1] / (
            1 + ((np.sign(freq) * freq - self.centers[1]) / self.widths[1]) ** 2
        )
        j_w += term1
        tanh_prefactor = np.tanh(freq * self.beta / 2)
        return self.scaling_constant * tanh_prefactor * j_w

    def construct_bcf(self, beta : float, time : np.array):
        bcf = np.zeros((len(time)), dtype=complex)
        center = self.centers[1]
        lower_limit = center - (self.omega_infinity - center)
        low_freqs = np.linspace(0, self.omega_infinity, 10000)
        low_freqs = low_freqs[1:]
        hi_freqs = np.linspace(lower_limit, self.omega_infinity, 100000)
        hi_freqs = hi_freqs[1:]
        calc_j_w_low = self.computelow(low_freqs)
        calc_j_w_hi = self.computehi(hi_freqs)
        for t_index, t in enumerate(time):
            # Compute the low part
            bcf_integrand_real = calc_j_w_low*_coth(low_freqs*self.beta/2)*np.cos(low_freqs*t)
            bcf_integrand_imag = calc_j_w_low*np.sin(low_freqs*t)
            bcf_real = np.trapz(bcf_integrand_real, low_freqs)
            bcf_imag = np.trapz(bcf_integrand_imag, low_freqs)
            bcf_t = (1 / np.pi) * (bcf_real - 1j * bcf_imag)
            bcf[t_index] += bcf_t
            # Compute the hi part
            bcf_hi_integrand_real = calc_j_w_hi*np.cos(hi_freqs*t)
            bcf_hi_integrand_imag = calc_j_w_hi*np.sin(hi_freqs*t)

            bcf_hi_real = np.trapz(bcf_hi_integrand_real, hi_freqs)
            bcf_hi_imag = np.trapz(bcf_hi_integrand_imag, hi_freqs)
            bcf_hi_t = (1 / np.pi) * (bcf_hi_real - 1j * bcf_hi_imag)
            bcf[t_index] += bcf_hi_t
            # if t_index % 500 == 0:
            #     pred_fig = plt.figure(layout="constrained")
            #     gs = pred_fig.add_gridspec(nrows=2, ncols=1)
            #     integrand_ax = pred_fig.add_subplot(gs[0])
            #     integrated_ax = pred_fig.add_subplot(gs[1])
            #     integrand_ax.plot(hi_freqs, bcf_hi_integrand_real, label='real')
            #     integrand_ax.plot(hi_freqs, calc_j_w_hi, label=t)
            #     integrand_ax.plot(hi_freqs, bcf_hi_integrand_imag, label='imag')
            #     integrated_ax.plot(bcf[:t_index+1].real, label='Real BCF')
            #     integrated_ax.plot(bcf[:t_index+1].imag, label='Imag BCF')
            #     integrand_ax.legend()
            #     integrated_ax.legend()
            #     plt.show()
        return bcf



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
        reorg = (1 / np.pi) * np.trapz(j_w / omegas, omegas)
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


def _coth(w):
    return 1 / np.tanh(w)
