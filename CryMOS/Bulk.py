""" Bulk model """
from .constants import *
from .base import writeable_property

from math import sqrt
import numpy as np
from warnings import warn

__all__ = ['BulkModel', 'BulkModelFD', 'BulkModelTails']


class BulkModel:
    """ Basic Bulk model for Silicon

    The terminology and derivations are similar to "Cryogenic MOS transistor model" (Beckers.2018)

    This is a strongly approximated model for a Si-Bulk, that uses Boltzman-statistics everywhere and a DOS
    in the basic DOS(E) = H(E > E_c)*D_c*sqrt(E - E_c) form.

    The model is valid for temperatures down to ~10 K due to limitations in double float datatype
    (n_i becomes very small) """

    params = ('temp', 'N_A', 'N_D', 'g_A', 'g_D', '_E_D', '_E_A', '_D_c', '_D_v', '_E_g', '_chi')
    memos = ['_psi_b_memo', '_Ei_memo', '_Ef_memo', '_bulk_p_memo', '_bulk_n_memo']

    def __init__(self, **kwargs):
        self._psi_b_memo = None
        self._Ef_memo = None
        self._bulk_p_memo = None
        self._bulk_n_memo = None
        self._Ei_memo = None

        self.temp = 300.

        self.N_A = 1e23  # bulk doping in 1/m^3
        self.N_D = 0.

        self._E_A = None
        self._E_D = None
        self._D_c = None
        self._D_v = None
        self._E_g = None
        self._chi = None

        self.ionization = 'incomplete'
        self.update_params(**kwargs)

    def __setattr__(self, key, value):
        # TODO: there has to be a better way
        if key in BulkModel.params:
            for m in BulkModel.memos:
                super().__setattr__(m, None)

        return super().__setattr__(key, value)

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise TypeError(f"Invalid propert {k}!")
            setattr(self, k, v)

    @writeable_property
    def E_D(self):
        """ donor energy """
        # phosporous doping in si: 45 meV taken from Sze
        return self.E_c - 0.045 * e

    @writeable_property
    def E_A(self):
        """ acceptor energy """
        # boron doping in si: 45meV taken from Sze
        return self.E_v + 0.044 * e

    @writeable_property
    def D_c(self):
        return D_c_Si(self.temp)

    @writeable_property
    def D_v(self):
        return D_v_Si(self.temp)

    @writeable_property
    def chi(self):
        """ semiconductor electron affinity """
        chi0 = 4.05*e
        return E_g_Si(300)/2 - self.E_g + self.E_i + chi0

    @property
    def N_c(self):
        return self.D_c * sqrt(pi) / 2 * (k*self.temp) ** 1.5

    @N_c.setter
    def N_c(self, N_c):
        self._D_c = 2 / sqrt(pi) * N_c / (k*self.temp)**1.5

    @property
    def N_v(self):
        return self.D_v * sqrt(pi) / 2 * (k*self.temp) ** 1.5

    @N_v.setter
    def N_v(self, N_v):
        self._D_c = 2 / sqrt(pi) * N_v / (k*self.temp)**1.5

    @writeable_property
    def E_g(self):
        return E_g_Si(self.temp)

    @property
    def ionization(self):
        return None

    @ionization.setter
    def ionization(self, ionization):
        if ionization == 'incomplete':
            self.g_A = 4.
            self.g_D = 2.
        elif ionization == 'complete':
            self.g_A = 0.
            self.g_D = 0.
        else:
            raise RuntimeError("invalid ionization setting")

    # -- use this for E_v = 0 reference --
    @property
    def E_v(self):
        return 0.
    @property
    def E_c(self):
        return self.E_v + self.E_g
    @property
    def E_i(self):
        return self.E_g / 2 + k * self.temp / 2 * np.log(self.N_v / self.N_c)

    @property
    def E_vac(self):
        return self.E_c + self.chi

    @property
    def phi_s(self):
        return (self.E_vac - self.E_f) / e

    E_i_boltzmann = E_i

    @property
    def E_f(self):
        if self._Ef_memo is None:
            self._Ef_memo = self._fermi_energy_bulk()
        return self._Ef_memo

    def _fermi_energy_bulk(self):
        """calculates the doped Si Fermi energy

        TODO: implement N_D?
        """
        assert self.N_A > self.N_D, "NMOS only"

        fac = 1 + np.sqrt(1 + 4 * self.g_A * self.N_A / self.N_v * np.exp((self.E_A - self.E_v) / (k * self.temp)))
        return k * self.temp * np.log(self.N_v / self.N_A) + k * self.temp * np.log(fac / 2)

    @property
    def E_f_boltzmann(self):
        return BulkModel._fermi_energy_bulk(self)

    def n(self, E_f):
        """ electron density for given fermi-energy """
        return self.N_c * np.exp((E_f - self.E_c) / (k * self.temp))

    def p(self, E_f):
        """ hole density for given fermi-energy """
        return self.N_v * np.exp((self.E_v - E_f) / (k * self.temp))

    def n_psi(self, psi):
        """ electron density for given psi = (E_f - E_i)/e """
        return self.n(psi*e + self.E_i)

    def p_psi(self, psi):
        """ hole density for given psi = (E_f - E_i)/e """
        return self.p(psi*e + self.E_i)

    @property
    def n_i(self):
        """ intrinsic electron/hole density """
        return np.sqrt(self.N_c * self.N_v) * np.exp(-self.E_g / (2. * k * self.temp), dtype=np.longdouble)

    def N_Am(self, E_f):
        """ ionized acceptor density for given fermi-energy """
        return self.N_A * fermi_dirac_factor(self.E_A - E_f, 0, self.g_A, self.temp)

    def N_Dp(self, E_f):
        """ ionized donor density for given fermi-energy """
        return self.N_D * fermi_dirac_factor(E_f - self.E_D, 0, self.g_D, self.temp)

    def N_Am_psi(self, psi):
        """ ionized acceptor density for given fermi-energy """
        return self.N_Am(psi * e + self.E_i)

    def N_Dp_psi(self, psi):
        """ ionized donor density for given fermi-energy """
        return self.N_Dp(psi * e + self.E_i)

    @property
    def kBT(self):
        return k * self.temp

    @property
    def phi_t(self):
        """ thermal voltage """
        return kb_eV * self.temp

    def exp_phi_t(self, a):
        """ very often appearing term"""
        return np.exp(a / (kb_eV * self.temp))

    @property
    def bulk_p(self):
        """ Sze eq (23)"""
        if self._bulk_p_memo is None:
            self._bulk_p_memo = self.p(self.E_f)
        return self._bulk_p_memo

    @property
    def bulk_n(self):
        """ Sze eq (17)"""
        if self._bulk_n_memo is None:
            self._bulk_n_memo = self.n(self.E_f)
        return self._bulk_n_memo

    @property
    def psi_a(self):
        """ defined below eq. (2)"""
        return (self.E_A - self.E_i) / e  # acceptor potential

    @property
    def psi_b(self):
        # Beckers, eq. (10)
        # f1 = self.n_i/self.N_A
        # f2 = 1. + np.sqrt(1. + 4. / f1 * self.g_A * self.exp_phi_t(self.psi_a))
        # return self.phi_t * (np.log(f1) + np.log(f2/2.))

        if self._psi_b_memo is None:
            self._psi_b_memo = (self.E_f - self.E_i) / e
        return self._psi_b_memo

    @property
    def psi_th(self):
        """ potential where n = N_A  (NMOS only) """
        assert self.N_A > self.N_D, "NMOS only"

        # complete ionization at the surface is assured
        E_f_ionized = k * self.temp * np.log(self.N_v / self.N_A)
        return -(E_f_ionized - self.E_i) / e

    def DOS(self, E):
        return np.heaviside(E - self.E_c, 0) * self.D_c * np.sqrt(E - self.E_c)

    def plot_band_diagram(self, ax=None):
        warn("not tested for a long time", DeprecationWarning)
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.axhline(self.E_v/e, color='k', label='E_v')
        ax.axhline(self.E_c/e, color='k', label='E_c')
        ax.axhline(self.E_f/e, color='r', label='E_f')
        if self.N_A != 0:
            ax.axhline(self.E_A/e, linestyle='--', label='E_A')
        if self.N_D != 0:
            ax.axhline(self.E_D/e, linestyle='--', label='E_D')
        ax.axhline(self.E_i/e, color='b', label='E_i')

        ax.set_ylabel('energy [ev]')
        ax.legend()
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

    def plot_fermi_calc(self, ax=None):
        """ plots the curves whichs intersection are solved implicitly to get E_f """
        import matplotlib.pyplot as plt
        warn("not tested for a long time", DeprecationWarning)

        ax = ax or plt.gca()
        E_f = np.linspace(self.E_v, self.E_c, 1000)
        p = np.squeeze(fermi_dirac_p(self.temp, self.E_v-E_f))  # TODO: uses wrong N_v
        N_D_plus = np.squeeze(self.N_D * fermi_dirac_factor(E_f - self.E_D, 0., self.g_D, self.temp))
        n = np.squeeze(fermi_dirac_n(self.temp, self.E_c - E_f))  # TODO: uses wrong N_c
        N_A_minus = np.squeeze(self.N_A * fermi_dirac_factor(self.E_A - E_f, 0., self.g_A, self.temp))

        pos_charge = p + N_D_plus
        neg_charge = n + N_A_minus

        ax.plot(E_f / e, pos_charge, color='r', label='positive charges')
        ax.plot(E_f / e, neg_charge, color='b', label='negative charges')
        ax.plot(E_f / e, p, linestyle='--', label='p')
        ax.plot(E_f / e, N_D_plus, linestyle='--', label='N_D+')
        ax.plot(E_f / e, n, linestyle='--', label='n')
        ax.plot(E_f / e, N_A_minus, linestyle='--', label='N_A-')
        ax.axvline(self.E_f / e, label='E_f = %.1E eV' % (self.E_f / e), color='k', linestyle='--')
        ax.legend()
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel('carrier concentration [1/m^3]')
        #ax.title('N_A = %.1E , N_D=%.1E' % (self.N_A, self.N_D))


class BulkModelFD(BulkModel):
    """ Bulk model for Silicon that uses FD-statics instead of Boltzmann-approximations
    
    E_i and E_f are implicitly calculated by rooting the charge-balance equations"""

    memos = BulkModel.memos + ['_psi_th_memo']

    def __init__(self, **kwargs):
        super().__init__()

        self._psi_th_memo = None
        self.update_params(**kwargs)

    def _Ei_calc(self):
        # calculate intrinsic carrier concentration from charge balance
        from scipy.optimize import root_scalar

        res = root_scalar(lambda E_i: self.p(E_i) - self.n(E_i),
                          bracket=[self.E_v + 3*self.kBT, self.E_c - 3*self.kBT], xtol=1e-60)
        assert res.converged, "root failed!"
        return res.root

    @property
    def E_v(self):
        return super().E_v

    @property
    def E_c(self):
        return super().E_c

    @property
    def E_i(self):
        if self._Ei_memo is None:
            self._Ei_memo = self._Ei_calc()
        return self._Ei_memo

    def _fermi_energy_bulk(self):
        """calculates the doped Si Fermi energy, solving the implicit equation for E_f """
        from scipy.optimize import root_scalar

        def root_fun(E_f):
            p_term = self.p(E_f) - self.N_Am(E_f)
            n_term = self.n(E_f) - self.N_Dp(E_f)
            return p_term - n_term
        try:  # n_approx2 and p_approx2 are only stable to a few kBT in band
            res = root_scalar(root_fun, bracket=[self.E_v + 3*self.kBT, self.E_c - 3*self.kBT], xtol=1e-60)
        except ValueError:  # degenerate doping?
            res = root_scalar(root_fun, bracket=[self.E_v - 3*self.kBT, self.E_c + 3*self.kBT], xtol=1e-60)

        assert res.converged, "root failed!"
        return res.root

    def n(self, E_f):
        """ electron density for given fermi-energy """
        from fdint import fd1h
        return self.N_c * 2 / sqrt(pi) * fd1h((E_f - self.E_c)/(k*self.temp))

    def p(self, E_f):
        """ hole density for given fermi-energy """
        from fdint import fd1h
        return self.N_v * 2 / sqrt(pi) * fd1h((self.E_v - E_f)/(k * self.temp))

    @property
    def n_i(self):
        """ intrinsic electron/hole density """
        return np.sqrt(self.bulk_n*self.bulk_p)

    @property
    def psi_th(self):
        if self._psi_th_memo is None:
            from scipy.optimize import root_scalar

            res = root_scalar(lambda psi: self.n_psi(psi) - self.N_A,
                              bracket=[0, 1])
            assert res.converged, f"root failed! {res.flag}"
            self._psi_th_memo = res.root

        return self._psi_th_memo


class BulkModelTails(BulkModelFD):
    """ Bulk model for Silicon that implements a band-tail DOS

    The equations for the DOS are exponential below E_c and a sqrt above E_c (see `DOS`)
    """
    params = BulkModelFD.params + ('W_TA', 'W_TD')
    memos = BulkModelFD.memos + ['_C1_n_memo', '_C2_n_memo', '_C1_p_memo', '_C2_p_memo']

    def __init__(self, **kwargs):
        super().__init__()

        self._C1_n_memo = None
        self._C2_n_memo = None
        self._C1_p_memo = None
        self._C2_p_memo = None

        self.W_TA = 4e-3 * e  # 4meV
        self.W_TD = 4e-3 * e  # 4meV

        self.w_fac = 5.  # development point (W_T/2)

        self.update_params(**kwargs)

    def DOS(self, E):
        E_cme = self.E_cme
        N_t = self.D_c * np.sqrt(E_cme - self.E_c)

        return np.piecewise(E, [E < E_cme, E >= E_cme], [
            lambda E: N_t * np.exp((E-E_cme) / self.W_TA),
            lambda E: self.D_c * np.sqrt(E - self.E_c)
        ])

    @property
    def E_cme(self):
        return self.E_c + self.W_TA / 2

    @property
    def E_vme(self):
        return self.E_v - self.W_TD / 2

    def n_integral(self, E_f):
        """ electron density for given fermi-energy """
        from scipy.integrate import quad

        def fdn_fun(E, E_f):
            return self.DOS(E) / (1. + np.exp((E - E_f) / (k * self.temp)))

        return np.vectorize(
            lambda E_fx: quad(fdn_fun, self.E_v + self.E_g/2, self.E_c+self.phi_t*e*10,
                              args=E_fx, points=[self.E_cme, self.E_c])[0] #, epsrel=1e-60
            )(E_f)

    def n_approx(self, E_f):
        from scipy.special import erfc, hyp2f1
        kT = k * self.temp
        Wt = self.W_TA
        Ec = self.E_c
        Ecme = self.E_cme
        Nt = self.D_c * np.sqrt(Ecme - Ec)

        i1 = 2*(Ecme - Ec) * Nt * \
            hyp2f1(1, kT / Wt, 1 + kT/Wt, -np.exp((Ecme - E_f)/kT))

        sqrt_c = np.sqrt((Ecme - Ec) / kT)
        i2 = self.N_c * np.exp((E_f - Ecme) / kT) * (
            2/sqrt(pi) * sqrt_c + np.exp((Ecme - Ec) / kT) * erfc(sqrt_c)
        )
        return i1 + i2

    @property
    def C1_n(self):
        if self._C1_n_memo is None:
            from scipy.special import hyp2f1
            from math import exp
            kT = k * self.temp
            Wt = self.W_TA
            Nt = self.D_c * sqrt(Wt/2)
            a = kT / Wt
            assert abs(a - 1) > 1e-8, "model is singular for W_TA = kB*T!"

            # x = w*Wt / kT
            w = self.w_fac
            c1 = exp(w) * hyp2f1(1, a, 1+a, -exp(w/a)) + \
                -a/(a - 1) * exp(w - w/a)

            self._C1_n_memo = Nt * Wt * c1
        return self._C1_n_memo

    @property
    def C2_n(self):
        if self._C2_n_memo is None:
            from math import exp, sqrt, pi, erfc
            kT = k * self.temp
            Wt = self.W_TA
            Nt = self.D_c * sqrt(Wt/2)
            w = Wt / (2*kT)

            self._C2_n_memo = Nt * Wt * kT / (kT - Wt) + \
                self.N_c * (2 / pi * sqrt(w) + exp(w) * erfc(sqrt(w)))

        return self._C2_n_memo

    def n_approx2(self, E_f):
        return self.C1_n * np.exp((E_f - self.E_cme) / self.W_TA) + \
               self.C2_n * np.exp((E_f - self.E_cme) / (k * self.temp))

    n = n_approx2

    def p_approx(self, E_f):
        from scipy.special import erfc, hyp2f1
        kT = k * self.temp
        Wt = self.W_TD
        Ev = self.E_v
        Evme = self.E_vme
        Nt = self.D_v * np.sqrt(Ev - Evme)

        i1 = 2*(Ev - Evme) * Nt * \
            hyp2f1(1, kT / Wt, 1 + kT/Wt, -np.exp((E_f - Evme)/kT))

        sqrt_v = np.sqrt((Ev - Evme) / kT)
        i2 = self.N_v * np.exp((Evme - E_f) / kT) * (
            2/sqrt(pi) * sqrt_v + np.exp((Ev - Evme) / kT) * erfc(sqrt_v)
        )
        return i1 + i2

    @property
    def C1_p(self):
        if self._C1_p_memo is None:
            from scipy.special import hyp2f1
            from math import exp
            kT = k * self.temp
            Wt = self.W_TD
            Nt = self.D_v * sqrt(Wt/2)
            a = kT / Wt
            assert abs(a - 1) > 1e-8, "model is singular for W_TA = kB*T!"

            # x = w*Wt / kT
            w = self.w_fac
            c1 = exp(w) * hyp2f1(1, a, 1+a, -exp(w/a)) + \
                -a/(a - 1) * exp(w - w/a)

            self._C1_p_memo = Nt * Wt * c1
        return self._C1_p_memo

    @property
    def C2_p(self):
        if self._C2_p_memo is None:
            from math import exp, sqrt, pi, erfc
            kT = k * self.temp
            Wt = self.W_TD
            Nt = self.D_v * sqrt(Wt/2)
            w = Wt / (2*kT)

            self._C2_p_memo = Nt * Wt * kT / (kT - Wt) + \
                self.N_v * (2 / pi * sqrt(w) + exp(w) * erfc(sqrt(w)))

        return self._C2_p_memo

    def p_approx2(self, E_f):
        return self.C1_p * np.exp((self.E_vme - E_f) / self.W_TD) + \
               self.C2_p * np.exp((self.E_vme - E_f) / (k * self.temp))

    p = p_approx2
