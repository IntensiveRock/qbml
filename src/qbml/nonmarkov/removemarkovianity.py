from pathlib import Path
import logging

import numpy as np
import scipy
import matplotlib.pyplot as plt

from qbml.nonmarkov.mnmrunner import mnmsimrunner
import qbml.dynamics.tools as qbmltools

_logger = logging.getLogger(__name__)


"""
GOAL:
Test to determine:
 - if the Markovian contribution to the density matrix can be removed
 - and recover original dynamics.

STEPS:
1. Build spectral density with Markovian and non-Markovian contributions.
2. Evolve density matrix for entire spectral density.
3. Evolve density matrix for non-Markovian spectral density ONLY.
4. Rotate result of step 2 by inverse of R_m(∞).
5. Compare results of 3 and 4.
"""

config_path = Path("/home/dewdrop/pypkgs/qubitml/src/qubitml/nonmarkov/mnm.yaml")

def propagate(
        set_of_rhots : np.array,
        Rmij_ts : np.array,
        dt : float) -> np.array:
    """
    Compute exp{-R_m(t)} and invert each snapshot of the density matrix.
    """
    n_steps = len(Rmij_ts)
    rho_tilda = np.zeros_like(set_of_rhots, dtype=complex)
    # rotations = np.array([np.eye(4) for _ in range(n_steps)], dtype=complex)
    rotations = [np.eye(4, dtype=complex)]
    for t_index, R_t in enumerate(Rmij_ts):
        if t_index != 0:
            mat_m = (Rmij_ts[t_index] + Rmij_ts[t_index - 1]) * dt / 2
            mmat2 = scipy.linalg.expm(-mat_m)
            rotations.append(mmat2 @ rotations[t_index - 1])
    for t_idx, ρ in enumerate(set_of_rhots):
        rho_tilda[t_idx] = rotations[t_idx] @ ρ
    return rho_tilda


mnm_summary = mnmsimrunner(config_path)
dt = mnm_summary.times[1]
times = mnm_summary.times

# TEST:: take actual R_{m,ij}, invert it and apply to density matrix.

Rmij = mnm_summary.Rmij
# _logger.info("Inverted Markovian Redfield Tensor")
# propagate(mnm_summary.rdms[0], Rmij, dt)

# Already computing only the non-Markovian part as part of the Summary.rdms.
# Inversion time!!
# Index 2 is the purely non-Markovian part.
# Index 3 is the last approximation.
# Index 0 is no approximation.
tomo_0 = qbmltools.compute_full_tomography_from_rdm(mnm_summary.rdms[0])
tomo_3 = qbmltools.compute_full_tomography_from_rdm(mnm_summary.rdms[3])
tomo_removeit = qbmltools.compute_full_tomography_from_rdm(mnm_summary.rdms[4])

param_loop = True
while param_loop:

    # x_params, z_params = ask_for_params()

    # rotated_0 = propagate(mnm_summary.rdms[0], Rmij, dt)

    rotated_3 = propagate(mnm_summary.rdms[3], Rmij, dt)

    tomo_nm = qbmltools.compute_full_tomography_from_rdm(mnm_summary.rdms[2])
    rot_tomo_3 = qbmltools.compute_full_tomography_from_rdm(rotated_3)

    pred_fig = plt.figure(layout="constrained", dpi=200, figsize=[8,8])

    gs = pred_fig.add_gridspec(nrows=3, ncols=1)
    x_ax = pred_fig.add_subplot(gs[0, :])
    y_ax = pred_fig.add_subplot(gs[1, :])
    z_ax = pred_fig.add_subplot(gs[2, :])

    difference = ((tomo_nm.real-rot_tomo_3.real)**2)**0.5
    baseline = np.zeros_like(times)

    x_ax.plot(times, tomo_0[:,0].real, color='pink', label=r"$\exp\{R_m+R_{nm}\}$", alpha=0.5, ls='-.')
    x_ax.plot(times, tomo_3[:,0].real, color='blue', label=r"$\exp\{R_m\}\exp\{R_{nm}\}$", alpha=0.5, ls='--')
    # x_ax.plot(tomo_removeit[:,0], color='green', label="removeit", alpha=0.5)
    x_ax.plot(times, tomo_nm[:,0].real, color='teal', label=r"$\exp\{R_{nm}\}$", alpha=0.5)
    # x_ax.plot(rot_tomo_0[:,0].real, color='salmon', ls='-.', label="NM+M no approx")
    x_ax.plot(times, rot_tomo_3[:,0].real, color='#3e4e50', ls=':', label=r"$\tilde{\rho}(t)$")
    x_ax.plot(times, baseline, color='black')
    x_ax.plot(times, difference[:,0], color='red', label="Error (teal-dots)", ls='--')
    x_ax.legend(frameon=False,
                loc="upper right",
                ncols=3,
                fontsize='x-small',
                )

    y_ax.plot(times, tomo_0[:,1].real, color='pink', label="Full", alpha=0.5, ls='-.')
    y_ax.plot(times, tomo_3[:,1].real, color='blue', label="Full", alpha=0.5, ls='--')
    # y_ax.plot(tomo_removeit[:,1], color='green', label="removeit", alpha=0.5)
    y_ax.plot(times, tomo_nm[:,1].real, color='teal')
    # y_ax.plot(rot_tomo_0[:,1].real, color='salmon', ls='-.')
    y_ax.plot(times, rot_tomo_3[:,1].real, color='#3e4e50', ls=':')
    y_ax.plot(times, baseline, color='black')
    y_ax.plot(times, difference[:,1], color='red', label="Error", ls='--')

    z_ax.plot(times, tomo_0[:,2].real, color='pink', label="Full", alpha=0.5, ls='-.')
    z_ax.plot(times, tomo_3[:,2].real, color='blue', label="Full", alpha=0.5, ls='--')
    # z_ax.plot(tomo_removeit[:,2], color='green', label="removeit", alpha=0.5)
    z_ax.plot(times, tomo_nm[:,2].real, color='teal')
    # z_ax.plot(rot_tomo_0[:,2].real, color='salmon', ls='-.')
    z_ax.plot(times, rot_tomo_3[:,2].real, color='#3e4e50', ls=':')
    z_ax.plot(times, baseline, color='black')
    z_ax.plot(times, difference[:,2], color='red', label="Error", ls='--')

    x_ax.set_ylim([-1.1,1.2])
    y_ax.set_ylim([-1.1,1.1])
    z_ax.set_ylim([-1.1,1.1])

    # x_ax.set_xlim([0,times[-1]])
    # y_ax.set_xlim([0,times[-1]])
    # z_ax.set_xlim([0,times[-1]])

    x_ax.set_title(r"$\sigma_x(t)$")
    y_ax.set_title(r"$\sigma_y(t)$")
    z_ax.set_title(r"$\sigma_z(t)$")

    z_ax.set_xlabel(r"$t\quad [\omega_q^{-1}]$")

    # og_ax.plot(tomo_nm.real, color='black')
    # og_ax.plot(tomo_0.real, ls='-.', color='red')
    # og_ax.plot(tomo_3.real, ls=':', color='green')
    plt.show()
    more_params = input("Try new params? (y/n) ")
    if more_params == "n":
        param_loop = False
