""" C++ implementations of QV-models """

from .QV import BeckersQVpy, DiracQVpy, GildenblatQVpy
import numpy as np

__all__ = ['BeckersQVcpp', 'DiracQVcpp', 'GildenblatQVcpp']


class BeckersQVcpp(BeckersQVpy):
    from .cpp import available as cpp_avail

    def _get_solver(self):
        if not self.cpp_avail:
            raise RuntimeError("No Solver found!")

        from .cpp import PotLoop
        solver = PotLoop()
        for k in "phi_t,g_A,psi_a,psi_b,n_i,N_A,eps_si,phi_ms,Q_0,g_t,cox".split(","):
            setattr(solver, k, getattr(self, k))

        if self.N_t is not None and self.psi_t is not None:
            solver.N_t = np.atleast_1d(self.N_t)
            solver.psi_t = np.atleast_1d(self.psi_t)
        else:
            solver.N_t = []
            solver.psi_t = []

        return solver

    psi_s_py = BeckersQVpy.psi_s

    if cpp_avail:
        def psi_s(self, v_ch, v_gb):
            return self._get_solver().psi_s(v_ch, v_gb)

        def Es(self, psi_s, v_ch):
            return self._get_solver().Es(psi_s, v_ch)


class DiracQVcpp(DiracQVpy):
    """ QV model that uses FD-Integrals for E**2 in C++ """
    from .cpp import available as cpp_avail

    def _get_solver(self):
        if not self.cpp_avail:
            raise RuntimeError("No Solver found!")

        from .cpp import PotLoopFD
        solver = PotLoopFD()
        for k in "phi_t,g_A,psi_a,psi_b,n_i,N_A,eps_si,phi_ms,Q_0,g_t,cox,E_i,E_v,E_c,N_c,N_v".split(","):
            setattr(solver, k, getattr(self, k))

        if self.N_t is not None and self.psi_t is not None:
            solver.N_t = np.atleast_1d(self.N_t)
            solver.psi_t = np.atleast_1d(self.psi_t)
        else:
            solver.N_t = []
            solver.psi_t = []

        return solver

    psi_s_py = DiracQVpy.psi_s

    if cpp_avail:
        def psi_s(self, v_ch, v_gb):
            return self._get_solver().psi_s(v_ch, v_gb)

        def Es(self, psi_s, v_ch):
            return self._get_solver().Es(psi_s, v_ch)


class GildenblatQVcpp(GildenblatQVpy):
    """ QV model that uses Gildenblat in C++ """
    from .cpp import available as cpp_avail

    def _get_solver(self):
        if not self.cpp_avail:
            raise RuntimeError("No Solver found!")

        from .cpp import PotLoopGildenblat
        solver = PotLoopGildenblat()
        for k in "phi_t,g_A,psi_a,psi_b,n_i,N_A,eps_si,phi_ms,Q_0,g_t,cox,lam_bulk,bulk_n,bulk_p".split(","):
            setattr(solver, k, getattr(self, k))

        if self.N_t is not None and self.psi_t is not None:
            solver.N_t = np.atleast_1d(self.N_t)
            solver.psi_t = np.atleast_1d(self.psi_t)
        else:
            solver.N_t = []
            solver.psi_t = []

        return solver

    psi_s_py = GildenblatQVpy.psi_s

    if cpp_avail:
        def psi_s(self, v_ch, v_gb):
            return self._get_solver().psi_s(v_ch, v_gb)

        def Es(self, psi_s, v_ch):
            return self._get_solver().Es(psi_s, v_ch)
