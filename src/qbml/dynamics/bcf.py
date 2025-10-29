import numpy as np

# from qubitml.dynamics.spectraldensity import SpecDen


class BathCorrelationFunction():
    """
    Bath correlation function object. Define how one computes the bath correlation function.

    Standard is RedfieldBCF.
    Custom: Far frequency gaussian.
    """

    def __init__(self, jw, beta : float, times : np.array):
        self.jw = jw
        self.beta = beta
        self.bcf = self._calc_bcf(times)

    def __mul__(self, b) -> np.array:
        return self.bcf * b

    def __add__(self, b) -> np.array:
        return self.bcf + b

    def __len__(self,):
        return len(self.bcf)

    def __getitem__(self, idx : int):
        return self.bcf[idx]

    def _calc_bcf(self, time : np.array) -> np.array:
        ...


class RedfieldBCF(BathCorrelationFunction):
    """
    Redfield bath correlation function.

    Computed as written in the paper.
    """
    def __init__(self, jw, beta : float, times : np.array):
        super().__init__(jw, beta, times)

    def _calc_bcf(self, time : np.array) -> np.array:
        bcf = np.zeros((len(time)), dtype=complex)
        j_w = self.jw
        freqs = np.linspace(0., j_w.omega_infinity, 10000)
        freqs = freqs[1:]
        calc_j_w = j_w(freqs)
        for t_index, t in enumerate(time):
            bcf_integrand_real = calc_j_w*_coth(freqs*self.beta/2)*np.cos(freqs*t)
            bcf_integrand_imag = calc_j_w*np.sin(freqs*t)
            bcf_real = np.trapz(bcf_integrand_real, freqs)
            bcf_imag = np.trapz(bcf_integrand_imag, freqs)
            bcf_t = (1 / np.pi) * (bcf_real - 1j * bcf_imag)
            bcf[t_index] += bcf_t
        return bcf


class FarFrequencyBCF(BathCorrelationFunction):
    """
    Assumes that the spectral density peak is Gaussian and located very far from zero.

    Here, coth(βω/2) -> 1. BCF becomes fourier transform of Gaussian.
    """
    def __init__(self, jw, beta : float, times : np.array):
        super().__init__(jw, beta, times)

    def _calc_bcf(self, time : np.array):
        bcf = np.zeros((len(time)), dtype=complex)
        j_w = self.jw
        center = j_w.centers[0]
        lower_limit = center - (j_w.omega_infinity - center)
        freqs = np.linspace(lower_limit, j_w.omega_infinity, 10000)
        freqs = freqs[1:]
        calc_j_w = j_w(freqs)
        for t_index, t in enumerate(time):
            bcf_integrand_real = calc_j_w*np.cos(freqs*t)
            bcf_integrand_imag = calc_j_w*np.sin(freqs*t)
            bcf_real = np.trapz(bcf_integrand_real, freqs)
            bcf_imag = np.trapz(bcf_integrand_imag, freqs)
            bcf_t = (1 / np.pi) * (bcf_real - 1j * bcf_imag)
            bcf[t_index] += bcf_t
        return bcf


class CompoundBCF(BathCorrelationFunction):
    """
    Compute BCFs in different ways.

    Suppose that the spectral density has distinct frequency regimes.
    J(ω) = J_low + J_hi
    C(t) = C_low(t) + C_hi(t)

    Compute the BCFs in each region differently and add them up.
    """

    def __init__(self, bcfs : list, beta : float, times : float):
        self.jws = [bathcorr.jw for bathcorr in bcfs]
        self.beta = beta
        self.bcf = self._sum_bcfs(bcfs)

    def _sum_bcfs(self, bcfs : list):
        """
        Sum all of the bath correlation functions provided to CompoundBCF.
        """
        tmp_bcf = np.zeros(len(bcfs[0]))
        for bcf in bcfs:
            tmp_bcf += bcf
        return tmp_bcf


def _coth(w):
    return 1 / np.tanh(w)
