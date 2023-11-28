from matplotlib.pyplot import gca, Axes, subplots, Figure
from .QV import BeckersQVpy
from .constants import e
from dataclasses import dataclass
from scipy.interpolate import interp1d

def plot_bands(mdl: BeckersQVpy, ax: Axes = None):
    ax = ax or gca()
    ax.axhline(mdl.E_v / e, color='k', label='E_v')
    ax.axhline(mdl.E_c / e, color='k', label='E_c')
    ax.axhline(mdl.E_f / e, color='r', label='E_f')
    if mdl.N_A != 0:
        ax.axhline(mdl.E_A / e, linestyle='--', label='E_A')
    if mdl.N_D != 0:
        ax.axhline(mdl.E_D / e, linestyle='--', label='E_D')
    ax.axhline(mdl.E_i / e, color='b', label='E_i')

    ax.set_ylabel('energy [ev]')
    ax.legend()


@dataclass
class MoscapPlot:
    fig: Figure
    ax_bands: Axes
    ax_carriers: Axes
    tox: float
    vgb: float = None


def plot_moscap(mdl: BeckersQVpy, v_gb=None, v_ch=0, d=None, Eg_siox=8.9, si_phi_m=0.0):
    from .constants import eps_siox
    tox = eps_siox / mdl.cox

    if v_gb is None:
        v_gb = mdl.v_th

    y, psi = mdl.y_psi(v_gb, v_ch=v_ch)
    if d is None:
        d = y.max() * 1e6

    psi_int = interp1d(y, psi, fill_value="extrapolate")
    psi_d = psi_int(d / 1e6)

    psi_s = psi[0]

    del_psi_gb = -mdl.Es(psi_s, v_ch=v_ch) * tox * mdl.eps_si / eps_siox

    gt_l = -tox * 2e6
    gt_r = -tox * 1e6

    fig, [ax1, ax2] = subplots(2, sharex=True)
    ax1.plot(y * 1e6, mdl.psi_c - psi, 'r')
    ax1.plot(y * 1e6, mdl.psi_v - psi, 'b')
    ax1.plot(y * 1e6, mdl.psi_a - psi, 'b--')
    ax1.plot(y * 1e6, 0 - psi, 'k--', alpha=0.5)
    ax1.plot([0, y[-1] * 1e6], [0, 0], c='k', ls='-')
    ax1.plot([0, 0, gt_r],
             [mdl.psi_c - psi_s, Eg_siox / 2 - psi_s, Eg_siox / 2 - psi_s + del_psi_gb], 'r')
    ax1.plot([0, 0, gt_r],
             [mdl.psi_v - psi_s, -Eg_siox / 2 - psi_s, -Eg_siox / 2 - psi_s + del_psi_gb], 'b')

    if si_phi_m is None:  # plot gate fermi energy
        ax1.plot([gt_r, gt_r],
                 [Eg_siox / 2 - psi_s + del_psi_gb, del_psi_gb], 'r')
        ax1.plot([gt_r, -tox * 1e6],
                 [-Eg_siox / 2 - psi_s + del_psi_gb, del_psi_gb], 'b')
        ax1.plot([gt_r, gt_l], [-v_gb, -v_gb], 'k-')
    else:  # plot gate band diagram
        psi_c = -v_gb + si_phi_m
        psi_v = -v_gb - mdl.E_g/e + si_phi_m
        ax1.plot([gt_r, gt_r, gt_l], [Eg_siox / 2 - psi_s + del_psi_gb, psi_c, psi_c], 'r-')
        ax1.plot([gt_r, gt_r, gt_l], [-Eg_siox / 2 - psi_s + del_psi_gb, psi_v, psi_v], 'b-')

        ax1.plot([gt_r, gt_l], [-v_gb, -v_gb], 'k--')

    ax1.annotate(' $E_c$', (d, mdl.psi_c - psi[-1]), c='r')
    ax1.annotate(' $E_i$', (d, 0 - psi[-1]), c='k', alpha=0.5)
    ax1.annotate(' $E_v$', (d, mdl.psi_v - psi[-1]), c='b')

    ax2.semilogy(y * 1e6, mdl.n_psi(psi), 'r')
    ax2.semilogy(y * 1e6, mdl.p_psi(psi), 'b')
    ax2.semilogy(y * 1e6, mdl.N_Am_psi(psi), 'r--')
    ax2.axhline(mdl.n_i, c='k', ls='--', alpha=0.5)
    ax2.axvline(0, c='k')
    ax2.annotate(' $n_i$', (d, mdl.n_i), c='k', alpha=0.5)
    # ax2.annotate(' $N_A^-$', (d, mdl.N_Am_psi(mdl.psi_b)), c='b')
    # ax2.axhline(mdl.N_A, c='r', ls='--')
    ax2.annotate(' $N_A^-$', (d, mdl.N_Am_psi(psi_d)), c='r')
    ax2.annotate(' $p$', (d, mdl.p_psi(psi_d)), c='b')
    # ax2.annotate(' $N_A$', (d, mdl.N_A), c='r')
    ax2.set_ylim(1e8, 1e24)
    ax2.set_xlim(-2 * tox * 1e6, d)
    ax2.set_xlabel('$y$ ($\mu$m)')
    ax1.set_ylabel('$\phi$ (eV)')
    ax2.set_ylabel('$n,p$ (1/m$^3$)')
    ax1.grid(), ax2.grid()
    return MoscapPlot(fig=fig, ax_bands=ax1, ax_carriers=ax2, tox=tox, vgb=v_gb)
