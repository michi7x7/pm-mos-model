from scipy.constants import (
    e, pi, k, h, m_e, hbar, epsilon_0,
    value as sc_value)

from .utils import SiMemo

import numpy as np
from math import sqrt

kb_eV = sc_value('Boltzmann constant in eV/K')

eps_si = 11.68 * epsilon_0
eps_siox = 3.9 * epsilon_0

@SiMemo
def E_g_Si_Caiafa(T):
    """ taken from Cryogenic Study and Modeling of IGBTs, Caiafa et al"""
    T = np.array(T, dtype=np.longdouble)

    ret = np.empty_like(T)
    ic = T > 170.
    if not np.all(ic):
        ret[~ic] = 1.17 + 1.059e-6 * T[~ic] - 6.05e-7 * T[~ic]**2
    if np.any(ic):
        ret[ic] = 1.1785 - 9.025e-5 * T[ic] - 3.05e-7 * T[ic]**2

    return ret * e

@SiMemo
def E_g_Si_Varshni(T):
    """ model ed after Varshni semiempirical E_G model

    values taken from Temperature dependence of the indirect energy gap in crystalline silicon; Alex et al. """
    Eg_0 = 1.1692  # [eV]
    alpha = 4.9e-4  # [eV/K]
    beta = 655  # [K]
    ret = Eg_0 - (alpha*T**2)/(T + beta)
    return ret * e  # [J]


E_g_Si = E_g_Si_Varshni

@SiMemo
def n_i_Si(T):
    """ theoretical formula for intrinsic carrier concentration, see e.g Hadley """
    return np.sqrt(N_c_Si(T) * N_v_Si(T)) * np.exp(-E_g_Si(T) / (2. * k * T), dtype=np.longdouble)


@SiMemo
def n_i_Si_CAIAFA(T):
    """ intrinsic carrier concentration fit according to CRYOGENIC STUDY AND MODELING OF IGBTS, CAIAFA et al """
    Z = 1.347 * np.cosh(E_g_Si(T) / (k * T)) + 1.251 * np.sinh(E_g_Si(T) / (k * T))
    - (1.095 * np.cosh(E_g_Si(T) / (2. * k * T)) + 0.742 * np.sinh(E_g_Si(T) / (2. * k * T))) * np.cosh(
        0.75 * np.log(m_p_Si(T) / m_n_Si(T)))
    + 0.1624 + 0.5 * np.cosh(1.5 * np.log(m_p_Si(T) / m_n_Si(T)))
    ret = (4.221e15) * T ** 1.5 * (m_n_Si(T) * m_p_Si(T) / m_e ** 2) ** 0.75 * np.exp(-E_g_Si(T) / (k * T)) * Z ** 0.5
    return ret * 1e6


@SiMemo
def E_i_Si(T):
    """this corresponds to the fermi energy for intrinsic Si"""
    return E_g_Si(T) / 2 + k * T / 2 * np.log(N_v_Si(T) / N_c_Si(T))


@SiMemo
def N_c_Si(T):
    """ effective density of states in the conduction band"""
    factor = 2. * pi * m_n_Si(T) * k * T / (h ** 2.)
    return 2. * np.power(factor, 1.5)


@SiMemo
def N_v_Si(T):
    """effective density of states in the valence band """
    # print(T)
    T = np.array(T)
    factor = 2*pi * m_p_Si(T) * k * T / (h ** 2.)
    return 2. * np.power(factor, 1.5)


@SiMemo
def D_c_Si(T):
    return np.power(2*m_n_Si(T), 1.5)/(2 * pi**2 * hbar**3)


@SiMemo
def D_v_Si(T):
    return np.power(2*m_p_Si(T), 1.5)/(2 * pi**2 * hbar**3)


@SiMemo
def m_n_Si(T):
    """ effective hole mass for Si, taken from CAIAFA et al. """
    a = -1.084e-9
    b = +7.580e-7
    c = +2.862e-4
    d = +1.057
    return (a * T ** 3 + b * T ** 2 + c * T + d) * m_e


@SiMemo
def m_p_Si(T):
    """ effective hole mass for Si, taken from taken from CAIAFA et al. """
    a = +1.872e-11
    b = -1.969e-8
    c = +5.857e-6
    d = +2.712e-4
    e = +0.584
    return (a * T ** 4 + b * T ** 3 + c * T ** 2 + d * T + e) * m_e


def v_t(T):
    """ thermal voltage"""
    return kb_eV * T


def exp_vt(A, T):
    """calculates the very often appearing term exp(A/vt)"""
    return np.exp(A / kb_eV / T)


def exp_kb(A, T):
    """ returns the very often appearing term exp(A/kb*T)"""
    return np.exp(A / (k * T))


def fermi_dirac_integral_slow(E, T):
    from scipy import integrate
    """ Sze. eq. (19) """
    E = np.atleast_1d(E)
    T = np.atleast_1d(T)
    sols = np.zeros([len(E), len(T)])
    for i, Ei in enumerate(E):
        for j, Tj in enumerate(T):
            eta_f = Ei / (k * Tj)
            # print(eta_f)
            func = lambda eta: np.sqrt(eta) / (1 + np.exp(eta - eta_f))
            sol = integrate.quad(func, 0, np.inf)
            sols[i, j] = sol[0]
    return np.squeeze(sols)


def fermi_dirac_integral(E, T):
    from fdint import fdk
    return fdk(k=0.5, phi=E / (k * T))


def fermi_dirac_factor(E1, E2, g, T):
    return 1 / (1 + g * np.exp((E1 - E2) / (k * T)))


def fermi_dirac_p(T, E):
    """ Sze eq (23)"""
    return N_v_Si(T) * 2 / np.sqrt(pi) * fermi_dirac_integral(E, T)


def fermi_dirac_n(T, E):
    """ Sze eq (17)"""
    return N_c_Si(T) * 2 / np.sqrt(pi) * fermi_dirac_integral(E, T)


def g_fun(u_, lam):
    return 1./lam * np.log(1. + lam*(np.exp(u_) - 1))
