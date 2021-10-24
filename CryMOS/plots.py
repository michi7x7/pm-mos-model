from matplotlib.pyplot import gca, Axes, subplots
from .QV import BeckersQVpy
from .constants import e


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


def plot_moscap(mdl: BeckersQVpy, v_gb=None, v_ch=0, d=0.125, Eg_siox=8.9):
    from .constants import eps_siox
    tox = eps_siox / mdl.cox

    v_gb = v_gb or mdl.v_th
    y, psi = mdl.y_psi(v_gb, v_ch=v_ch)
    psi_s = psi[0]

    del_psi_gb = -mdl.Es(psi_s, v_ch=v_ch) * tox * mdl.eps_si / eps_siox

    fig, [ax1, ax2] = subplots(2, sharex=True)
    ax1.plot(y * 1e6, mdl.psi_c - psi, 'r')
    ax1.plot(y * 1e6, mdl.psi_v - psi, 'b')
    ax1.plot(y * 1e6, mdl.psi_a - psi, 'b--')
    ax1.plot(y * 1e6, 0 - psi, 'k--', alpha=0.5)
    ax1.plot([0, y[-1] * 1e6], [0, 0], c='k', ls='-')
    ax1.plot([-tox * 1e6, -2 * tox * 1e6], [-v_gb, -v_gb], c='k', ls='-')
    ax1.plot([0, 0, -tox * 1e6, -tox * 1e6],
             [mdl.psi_c - psi_s, Eg_siox / 2 - psi_s, Eg_siox / 2 - psi_s + del_psi_gb, del_psi_gb], 'r')
    ax1.plot([0, 0, -tox * 1e6, -tox * 1e6],
             [mdl.psi_v - psi_s, -Eg_siox / 2 - psi_s, -Eg_siox / 2 - psi_s + del_psi_gb, del_psi_gb], 'b')

    ax1.annotate(' $E_c$', (d, mdl.psi_c - psi[-1]), c='r')
    ax1.annotate(' $E_i$', (d, 0 - psi[-1]), c='k', alpha=0.5)
    ax1.annotate(' $E_v$', (d, mdl.psi_v - psi[-1]), c='b')

    ax2.semilogy(y * 1e6, mdl.n_psi(psi), 'r')
    ax2.semilogy(y * 1e6, mdl.p_psi(psi), 'b')
    ax2.axhline(mdl.N_A, c='r', ls='--')
    ax2.axhline(mdl.n_i, c='k', ls='--', alpha=0.5)
    ax2.axvline(0, c='k')
    ax2.annotate(' $N_A$', (d, mdl.N_A), c='r')
    ax2.annotate(' $n_i$', (d, mdl.n_i), c='k', alpha=0.5)
    ax2.set_ylim(1e8, 1e24)
    ax2.set_xlim(-2 * tox * 1e6, d)
    ax2.set_xlabel('$y$ ($\mu$m)')
    ax1.set_ylabel('$\phi$ (eV)')
    ax2.set_ylabel('$n,p$ (1/m$^3$)')
    ax1.grid(), ax2.grid()
    fig.show()
    return fig
