from .QV import *
import numpy as np
from scipy.constants import e

__all__ = ['BeckersIVpy', 'TailsIVpy', 'DefaultIV']


class BeckersIVpy(BeckersQVpy):
    """ modelled after CRYOGENIC MOS TRANSISTOR MODEL """

    new_params = ('u0', 'theta', 'phi_ms')
    params = BeckersQVpy.params + new_params
    pandas_default = ('temp',)  # TODO

    def __init__(self, **kwargs):
        self.u0 = 0.04  # zero field mobility
        self.mobmod = 0

        # mobmod==0 -mobility model as stated in paper
        self.theta = 0.  # mobility reduction factor

        # mobmod ==1 - effective field mobility model
        self.ualpha = 0.045
        self.ub = 4e10
        self.ue = 1.5
        self.eta_e = 0.5
        # mobmod == 2 - mobility model with interface traps
        self.alpha = 1e-16

        super().__init__(**kwargs)

    def ueff(self, v_gb, v_ch):
        if self.mobmod == 0:
            ret = self.u0 / (1 + self.theta * v_gb)
        elif self.mobmod == 1:
            qb = self.Q_f(self.psi_s(v_ch, v_gb), v_ch)
            qi = self.Q_m(self.psi_s(v_ch, v_gb), v_ch)
            u_c = self.u0 / np.abs((self.ualpha * qb) / (qi + self.ualpha * qb))
            e_eff = (qb + self.eta_e * qi) / self.eps_si  # = self.e_eff(v_g, v_ds)
            u_sr = self.ub * e_eff ** (-self.ue)
            ret = 1 / (1 / u_c + 1 / u_sr)
        elif self.mobmod == 2:
            qt = self.Q_it(self.psi_s(v_ch, v_gb), v_ch)
            ret = self.u0 / (1 + self.alpha * np.abs(qt) / e)
        else:
            raise NotImplementedError()
        return ret

    def i_ds_integral_single(self, v_ds, v_gb):
        import scipy.integrate as integrate

        def intfun(v_ch):
            mob = self.ueff(v_gb, v_ch)
            return -mob * self.Q_m(self.psi_s(v_ch, v_gb), v_ch)

        res = integrate.quad(intfun, 0., v_ds)

        return self.w / self.l * res[0]

    @property
    def i_ds_integral(self):
        """ calculating the current via the integral Q_m(V_ch) dV_ch, stated in text before eq (15)"""
        return np.vectorize(self.i_ds_integral_single)

    def i_ds_sat_single(self, v_ds, v_gb):
        Q_mD = self.Q_m(self.psi_s(v_ds, v_gb), v_ds)
        Q_mS = self.Q_m(self.psi_s(0., v_gb), 0.)

        raise NotImplementedError()
        return self.w / self.l * self.ueff(v_gb, v_ds) * ()

    i_ds = i_ds_integral

    def i_ds_lin_single(self, v_ds, v_gb):
        """calculating the current in the linear regiome """
        mob = self.u0 / (1. + self.theta * v_gb)  # mobility reduction as described in paper
        return -mob * self.w / self.l * self.Q_m(self.psi_s(0.005, v_gb), 0.005) * v_ds

    @property
    def i_ds_lin(self):
        return np.vectorize(self.i_ds_lin_single)

    def fit_iv(self, v_gs_meas, i_ds_meas, v_ds=0.02, xlims=None, ylims=None, log=False, **kwargs):
        xlims = xlims or [-np.inf, np.inf]
        ylims = ylims or [-np.inf, np.inf]

        ind = (xlims[0] < v_gs_meas) & (v_gs_meas < xlims[1]) & (ylims[0] < i_ds_meas) & (i_ds_meas < ylims[1])
        v_gs_meas = v_gs_meas[ind]
        i_ds_meas = i_ds_meas[ind]

        if log:
            def i_ds_fit(mdlx, v_g):
                return np.log(mdlx.i_ds(v_ds, v_g))
            self.fit_arb(i_ds_fit, v_gs_meas, np.log(i_ds_meas), **kwargs)
        else:
            def i_ds_fit(mdlx, v_g):
                return mdlx.i_ds(v_ds, v_g)
            self.fit_arb(i_ds_fit, v_gs_meas, i_ds_meas, **kwargs)

    def fit_iv_lin(self, v_gs_meas, i_ds_meas, v_ds=0.02,sigma=None, **kwargs):
        def i_ds_fit(mdlx, v_g):
            return mdlx.i_ds_lin(v_ds, v_g)
        self.fit_arb(i_ds_fit, v_gs_meas, i_ds_meas, sigma=sigma, **kwargs)


class TailsIVpy(TailsQVpy, BeckersIVpy):
    new_params = ('Tr0', 'Wt', 'N_SD')
    params = TailsQVpy.params + BeckersIVpy.new_params + new_params

    def __init__(self, **kwargs):
        # raise NotImplementedError("This model does not work")
        super().__init__()

        self.Tr0 = 0.01  # bandtail-tunneling at E_c
        self.Wt = 3e-3  # bandtail-width in eV
        self.N_SD = 1e24  # degenerate source/drain doping

        self.update_params(**kwargs)

    def i_ds_bt(self, v_ds, psi_s):
        """ bandtail-tunnelling current

            psi_s is assumed to be independent of psi_s in subthreshold

            TODO: W/L dependence?
        """

        from .constants import h, e, k, kb_eV

        Eg = self.E_g

        kBT = k * self.temp
        kBT_eV = kb_eV * self.temp
        eta = 1 - self.Wt / kBT

        nind = (self.n_i / self.N_SD) ** (kBT_eV / self.Wt - 1)

        psi = psi_s

        return -e/h * self.Tr0 / eta * np.exp(-Eg / (2*kBT)) * (
            np.exp(psi/kBT_eV) * (1 - np.exp(-v_ds/kBT_eV))
            - np.exp(psi/self.Wt) * nind * (1 - np.exp(-v_ds/self.Wt))
        )


DefaultIV = BeckersIVpy
