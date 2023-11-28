import numpy as np

from .constants import *
from .Bulk import BulkModel, BulkModelFD, BulkModelTails
from .base import MosModelBase, writeable_property

from math import sqrt
from scipy.integrate import quad

__all__ = ['DefaultQV', 'BeckersQVpy',
           'DiracQVpy', 'TailsQVpy',
           'GildenblatQVpy', 'DefaultQV']


class BeckersQVpy(MosModelBase, BulkModel):
    """ modelled after CRYOGENIC MOS TRANSISTOR MODEL """

    new_params = ('cox', 'N_t', 'psi_t', 'g_t', 'Q_0', '_phi_m')
    params = MosModelBase.params + BulkModel.params + new_params
    pandas_default = ('temp',)  # TODO

    def __init__(self, **kwargs):
        self.eps_si = eps_si  # DO NOT CHANGE! many parts of the model refer to the global eps_si
        self.cox = 0.005755

        self._phi_m = None

        self.N_t = None
        self.psi_t = []
        self.g_t = 4.
        self.Q_0 = 0.  # fixed oxide charge

        BulkModel.__init__(self)
        MosModelBase.__init__(self, **kwargs)

        self.update_params(**kwargs)

    @writeable_property
    def phi_m(self):
        """ gate work function / electron affinity. Default: degenerately doped silicon E_f = E_c """
        return self.chi/e

    @property
    def phi_ms(self):
        """ work function difference between gate/bulk ~ flatband voltage

        for a poly-gate, both add chi, thus chi cancels """
        return self.phi_m - self.phi_s

    @phi_ms.setter
    def phi_ms(self, phi_ms):
        """ just another way to set phi_m, does not keep phi_ms constant """
        self._phi_m = phi_ms + self.phi_s

    def fs_ea(self, psi_s, V_ch):
        """ eq (8)"""
        return 1. / (1. + self.g_A * self.exp_phi_t(self.psi_a - psi_s + V_ch))

    def fb_ea(self):
        """ eq (9) """

        assert self.N_A > self.N_D, "NMOS only"
        return 1. / (1. + self.g_A * self.exp_phi_t(self.psi_a - self.psi_b))

    @property
    def gamma(self):
        return sqrt(2 * e * self.N_A * eps_si) / self.cox

    def Es_square(self, psi_s, v_ch, psi_b=None, fb_ea=None):
        """ eq (7) """

        # these are kinda hard to calculate, precalculate and use just once
        phi_t = self.phi_t
        if psi_b is None:
            psi_b = self.psi_b
        if fb_ea is None:
            fb_ea = self.fb_ea()

        # exp_phi_t = self.exp_phi_t
        exp_phi_t = lambda a: np.exp(a / phi_t)

        fs_ea = self.fs_ea(psi_s, v_ch)

        fac1 = 2. * e / eps_si
        fac2 = exp_phi_t(psi_s - v_ch) + exp_phi_t(-psi_s) - exp_phi_t(psi_b - v_ch) - exp_phi_t(-psi_b)
        fac3 = psi_s - psi_b - phi_t * np.log(fs_ea / fb_ea)
        return fac1 * (self.n_i * phi_t * fac2 + self.N_A * fac3)

    def Es(self, psi_s, v_ch, psi_b=None, **kwargs):
        """ sqrt of eq (7)"""
        psi_b = psi_b or self.psi_b
        return np.sign(psi_s-psi_b) * np.sqrt(self.Es_square(psi_s, v_ch, psi_b=psi_b, **kwargs))

    def v_fb(self, psi_s, v_ch):
        return self.phi_ms + (self.Q_0 - self.Q_it(psi_s, v_ch)) / self.cox

    @property
    def v_th0(self):
        """ approximated threshold voltage """
        phi0 = self.psi_th - self.psi_b  # + 5.*self.phi_t
        dphi = self.phi_t * np.log(self.fb_ea())  # E_f > E_i, fs_ea == 1

        return self.v_fb(self.psi_th, 0.0) + phi0 + \
            self.gamma * sqrt(phi0 + dphi)

    @property
    def v_th(self):
        """ threshold voltage from full v_gb expression (psi_s = psi_th) """
        return self.v_gb(self.psi_th, 0.0)

    def v_th_d(self, dpsi=None):
        """ threshold voltage with a shift in psi """
        dpsi = dpsi or np.sign(self.psi_th)*4*self.phi_t
        return self.v_gb(self.psi_th + dpsi, 0.0)

    @property
    def v_th1(self):
        phi_f0 = self.E_g/(2*e) + self.phi_t * np.log(self.N_A/np.sqrt((self.N_c * self.N_v)))

        # this includes incomplete ionization if the instance has ionization = incomplete
        phi_f1 = -self.psi_b
        return phi_f0 + self.phi_m - self.chi/e - (self.E_c-self.E_i)/e + self.gamma * np.sqrt(phi_f0 + phi_f1)

    def v_gb(self, psi_s, v_ch):
        return self.v_fb(psi_s, v_ch) + eps_si * self.Es(psi_s, v_ch) / self.cox + psi_s - self.psi_b

    def psi_s(self, v_ch, v_gb):
        """solves the implicit equation (pot_loop) to get the surface potential as a function of v_ch and v_gb"""
        from scipy.optimize import root_scalar
        v_gb = np.atleast_1d(v_gb)
        psi_s = 0. * v_gb
        bracket = [-2., 2.]
        # bracket = [(self.E_v-self.E_i)/e-v_ch, (self.E_c-self.E_i)/e-v_ch]

        psi_b = self.psi_b
        fb_ea = self.fb_ea()

        Es = self.Es
        v_fb = self.v_fb

        # surface boundary condition:
        # going around the loop, all appearing voltage must cancel each other out, statet a bit before eq. (13)
        def pot_loop(psi_s, v_ch, v_gb):
            return v_fb(psi_s, v_ch) + \
                   eps_si * Es(psi_s, v_ch, psi_b=psi_b, fb_ea=fb_ea) / self.cox + \
                   psi_s - self.psi_b - v_gb

        for i, v in enumerate(v_gb):
            res = root_scalar(pot_loop, args=(v_ch, v), bracket=bracket, xtol=1e-30)
            if not res.converged:
                psi_s[i] = np.nan
                raise RuntimeError("root did not converge!")
            else:
                psi_s[i] = res.root
        return psi_s

    def Q_m_1(self, psi_s, v_ch):
        """ Q_m exploiting the charge neutrality, here mobile = holes+electrons """
        return self.Q_sc(psi_s, v_ch) - self.Q_f(psi_s, v_ch)

    def Q_m(self, psi_s, v_ch):
        """ Q_m only including electron terms """
        log = np.log(self.fs_ea(psi_s, v_ch) / self.fb_ea())
        sqrt1 = - np.sqrt(2. * e * self.n_i * self.phi_t * eps_si * (
                    self.exp_phi_t(psi_s - v_ch) - self.exp_phi_t(self.psi_b - v_ch)) + 2. * e * self.N_A * eps_si * (
                                      psi_s - self.psi_b - self.phi_t * log))

        sqrt2 = np.sqrt(2. * e * self.N_A * eps_si * (psi_s - self.psi_b - self.phi_t * log))
        return sqrt1 + sqrt2

    def fs_Et(self, g_t, psi_t, psi_s, v_ch):
        return 1. / (1. + g_t * self.exp_phi_t((+psi_t - psi_s + v_ch)))

    def Q_sc(self, psi_s, v_ch):
        """ total semiconductor charge per unit area, text after eq (10)"""
        return -eps_si * self.Es(psi_s, v_ch)

    def Q_f(self, psi_s, v_ch):
        """ fixed charge density per unit area, eq (11)"""
        log = np.log(self.fs_ea(psi_s, v_ch) / self.fb_ea())
        return -np.sqrt(
            2. * e * self.N_A * eps_si * (psi_s - self.psi_b) - 2. * e * self.N_A * self.phi_t * eps_si * log)

    def Q_it(self, psi_s, v_ch):
        """ interface charge per unit area, eq (13) and eq (14) and text above"""
        ret = 0. * psi_s
        if self.N_t is not None and self.psi_t is not None:
            for psi_t, N_t in zip(np.atleast_1d(self.psi_t), np.atleast_1d(self.N_t)):
                # catch the case very complete ionization is assumed in order to avoid computational errors in fs_Et in this case
                if self.g_t != 0.:
                    ret = ret + (-e * N_t * self.fs_Et(self.g_t, psi_t, psi_s, v_ch))
                else:
                    ret = ret + (-e) * N_t
        return ret

    def set_arnout_traps(self, psi_t_c=0.58, N_t=1.2e15, fac=None):
        """ sets the interface traps similar to what Arnout did in his paper"""
        fac = fac or np.linspace(-2, 2, 5)
        self.psi_t = psi_t_c + self.phi_t * fac
        self.N_t = np.full_like(self.psi_t, N_t)

    def y_psi(self, v_gb, v_ch=0, linlog=0.5, logend=1e-3) -> (np.ndarray, np.ndarray):
        """ calculate the band structure in vertical direction

        returns: y, psi
        """

        from math import log10, fabs

        psi_s = self.psi_s(v_ch, v_gb)
        psi_b = self.psi_b

        integr = lambda psi: 1/self.Es(psi, v_ch, psi_b=psi_b)

        if np.isclose(psi_s, psi_b):
            return [0, 1e-6], [psi_s, psi_b]

        del_psi = psi_s - psi_b

        # linear close to the interface, log further away
        # as per suggestion in https://h-gens.github.io/automated-drawing-of-the-mos-band-diagram.html
        psis = psi_b + del_psi*np.hstack((
            np.linspace(1, linlog, 51),
            np.logspace(log10(linlog), log10(logend), 101)[1:]
        ))

        @np.vectorize
        def get_y(psi):
            return quad(integr, psi, psi_s)[0]

        y = get_y(psis)
        return y, psis


class DiracQVpy(BulkModelFD, BeckersQVpy):
    """ QV model that uses FD-Integrals for E**2

        TODO: check whether psi_b and psi_s fit together in flatband condition!!!
    """

    def Es_square(self, psi_s, v_ch, psi_b=None, fb_ea=None, E_i=None):
        # calculate Es_square via the fermi dirac integrals
        psi_b = psi_b or self.psi_b
        fac = 2. * e / eps_si

        def int_fun(psi):
            return self.n_psi(psi - v_ch) - self.p_psi(psi) + self.N_Am_psi(psi - v_ch) - self.N_Dp_psi(psi)

        intfun = lambda psi: fac * quad(int_fun, psi_b, psi)[0]
        return np.vectorize(intfun)(psi_s)

    def Q_f(self, psi_s, v_ch, psi_b=None):
        psi_b = psi_b or self.psi_b
        fac = 2. * e / eps_si

        def int_fun(psi):
            return fac * quad(
                lambda psi: self.N_Am_psi(psi) - self.N_Dp_psi(psi),
                psi_b - v_ch, psi - v_ch)[0]

        Es2 = np.vectorize(int_fun)(psi_s)

        # TODO: this is hideous... and probably wrong, is there no better way?
        return -eps_si * np.sign(psi_s - psi_b) * np.sqrt(np.abs(Es2))

    def Q_m(self, psi_s, v_ch, psi_b=None):
        fac = 2. * e / eps_si
        psi_b = psi_b or self.psi_b

        def int_fun_Qsc(psi):
            return self.n_psi(psi - v_ch) + self.N_Am_psi(psi - v_ch)

        Es_electrons = np.vectorize(
                lambda psi: fac * quad(int_fun_Qsc, psi_b, psi)[0]
            )(psi_s)

        return -eps_si * np.sqrt(Es_electrons) - self.Q_f(psi_s, v_ch, psi_b=psi_b)


class TailsQVpy(BulkModelTails, DiracQVpy):
    """ QV-model that includes bandtail-states """
    pass


class GildenblatQVpy(BeckersQVpy):
    """ QV model that uses the H(u) description for Es_square

    The relevant paper is "Surface potential equation for bulk MOSFET" (Gildenblat 2009)
    """

    @property
    def lam_bulk(self):
        return self.fb_ea()

    def Es_square(self, psi_s, v_ch, psi_b=None, fb_ea=None):
        from math import log, exp  # this is substantially faster than np
        from warnings import warn
        warn("bulk_n, bulk_p and psi_b do not fit togeter: ERROR between psi_s and psi_b!")

        psi_b = psi_b or self.psi_b
        phi_s = psi_s - psi_b
        phi_t = self.phi_t

        lam = self.lam_bulk

        n_b = self.bulk_n
        p_b = self.bulk_p

        k_0 = exp(-v_ch / phi_t)

        u = np.array(phi_s / phi_t, dtype=np.longdouble)
        g_fun = 1. / lam * np.log(1. + lam * (np.exp(u) - 1))
        # g_fun = 1. / lam * np.logaddexp(log(1. - lam), log(lam)+u)  # only a single call to numpy: faster
        h2 = np.exp(-u) - 1 + g_fun + n_b / p_b * k_0 * (np.exp(u) - 1. - g_fun)

        return 2 * e * p_b * phi_t / eps_si * h2


class Dirac2DQV(BeckersQVpy):
    """ QV model that uses FD-Integrals for E**2 """
    def Es_square(self, psi_s, v_ch, psi_b=None, fb_ea=None, E_i=None):
        # calculate es_square via the fermi dirac integrals
        phi_t = self.phi_t
        E_i = E_i or self.E_i
        psi_a = self.psi_a
        psi_b = psi_b or self.psi_b
        exp_phi_t = lambda a: np.exp(a / phi_t)
        fac = 2. * e / eps_si

        def fermi_dirac_integral(E, T):
            from fdint import fdk
            return fdk(k=0.5, phi=E / (k * T))

        def int_fun(psi):
            n_fd = self.N_c * 2 / np.sqrt(pi) * fermi_dirac_integral(e * (psi - v_ch) + E_i - self.E_c, self.temp)
            p_fd = self.N_v * 2 / np.sqrt(pi) * fermi_dirac_integral(self.E_v - e * psi - E_i, self.temp)
            na_min = self.N_A / (1. + self.g_A * exp_phi_t(psi_a - psi + v_ch))
            return n_fd - p_fd + na_min

        intfun = lambda psi: fac * quad(int_fun, psi_b, psi)[0]
        return np.vectorize(intfun)(psi_s)

    Q_m = BeckersQVpy.Q_m_1


# default implementation
DefaultQV = BeckersQVpy
