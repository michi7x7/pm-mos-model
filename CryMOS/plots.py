from matplotlib.pyplot import gca, Axes
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
