import numpy as np

from qbml.dynamics.spectraldensity import SpecDen


class Hami:
    """
    Define the problem Hamiltonian.

    Provide the hamiltonians that are relevant to the problem.
    This could include any of the hamiltonians included below.
    If you do not provide any, nothing will happen.

    Note: Spectral Density should be passed as a list of objects.
    """

    def __init__(
        self,
        system_hami: np.array = None,
        bath_hami: np.array = None,
        sb_hami: np.array = None,
        spectral_density: SpecDen = None,
        dt: float = None,
    ):
        """Initialize the Hamiltonian."""
        self.sys_hami = system_hami
        self.bath_hami = bath_hami
        self.sb_hami = sb_hami
        self.spdn = spectral_density
        # Maybe in the future add some conditional language to determine the sign of dt.
        self.u_dt = self._system_evolution_operator(-dt)

    def _system_evolution_operator(self, t: float):
        """Determine the value of exp(-iHst) at time t. Private class method."""
        e_val, e_vect = np.linalg.eig(self.sys_hami)
        d_s = np.diagflat(np.exp(-1j * e_val * t))
        u_t = e_vect @ d_s @ np.matrix(e_vect).H
        return u_t

    def ev_sys_operator(self, op: np.array, t_index: int):
        """Return system operator at time, t, with respect to system hamiltonian."""
        u_t = np.linalg.matrix_power(self.u_dt, t_index)
        u_t_dagger = u_t.H
        ev_op = u_t_dagger @ op @ u_t
        return ev_op
