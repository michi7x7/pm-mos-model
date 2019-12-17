#include <functional> //bind
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

using Float=double;

// include after all global headers
#include "fdint.hpp"
#include "PotLoop.hpp"

PYBIND11_MODULE(mod, m) {

#define DEF_PL_MEMBER(a)      .def_readwrite(#a, &PotLoop::a)
#define DEF_PL_MEMBER2(a,...) .def_readwrite(#a, &PotLoop::a, __VA_ARGS__)
    py::class_<PotLoop>(m, "PotLoop",
        "Potential Loop Solver")
        .def(py::init<>())
        DEF_PL_MEMBER(phi_t)
        DEF_PL_MEMBER(g_A)
        DEF_PL_MEMBER(psi_a)
        DEF_PL_MEMBER(psi_b)
        DEF_PL_MEMBER(n_i)
        DEF_PL_MEMBER(N_A)
        DEF_PL_MEMBER(eps_si)
        DEF_PL_MEMBER(phi_ms)
        DEF_PL_MEMBER(Q_0)
        DEF_PL_MEMBER(g_t)
        DEF_PL_MEMBER(cox)
        DEF_PL_MEMBER2(psi_t, "trap energies")
        DEF_PL_MEMBER2(N_t, "trap count for each energy")
        DEF_PL_MEMBER2(tol_bits, "result for phi_s will be exact to 2^(1-tol_bits)")
        .def("Es", py::vectorize(&PotLoop::Es))
        .def("psi_s", py::vectorize(&psi_s<PotLoop>), "v_ch"_a, "v_gb"_a, "solve psi_s for v_gb and v_ch")
        .def("psi_s2", &psi_s2<PotLoop>, "v_ch"_a, "v_gb"_a,
            "debug method, returns (root.first, root.second, n_iter)");

#undef DEF_PL_MEMBER
#undef DEF_PL_MEMBER2
#define DEF_PL_MEMBER(a)      .def_readwrite(#a, &PotLoopFD::a)
#define DEF_PL_MEMBER2(a,...) .def_readwrite(#a, &PotLoopFD::a, __VA_ARGS__)
    py::class_<PotLoopFD>(m, "PotLoopFD",
        "Potential Loop Solver")
        .def(py::init<>())
        DEF_PL_MEMBER(phi_t)
        DEF_PL_MEMBER(g_A)
        DEF_PL_MEMBER(psi_a)
        DEF_PL_MEMBER(psi_b)
        DEF_PL_MEMBER(n_i)
        DEF_PL_MEMBER(N_A)
        DEF_PL_MEMBER(eps_si)
        DEF_PL_MEMBER(phi_ms)
        DEF_PL_MEMBER(Q_0)
        DEF_PL_MEMBER(g_t)
        DEF_PL_MEMBER(cox)
        DEF_PL_MEMBER2(psi_t, "trap energies")
        DEF_PL_MEMBER2(N_t, "trap count for each energy")
        DEF_PL_MEMBER2(tol_bits, "result for phi_s will be exact to 2^(1-tol_bits)")
        // -- THESE ARE NEW --
        DEF_PL_MEMBER(E_i)
        DEF_PL_MEMBER(E_v)
        DEF_PL_MEMBER(E_c)
        DEF_PL_MEMBER(N_c)
        DEF_PL_MEMBER(N_v)
        // -- --
        .def("Es", py::vectorize(&PotLoopFD::Es))
        .def("psi_s", py::vectorize(&psi_s<PotLoopFD>), "v_ch"_a, "v_gb"_a, "solve psi_s for v_gb and v_ch")
        .def("psi_s2", &psi_s2<PotLoopFD>, "v_ch"_a, "v_gb"_a,
            "debug method, returns (root.first, root.second, n_iter)");

#undef DEF_PL_MEMBER
#undef DEF_PL_MEMBER2
#define DEF_PL_MEMBER(a)      .def_readwrite(#a, &PotLoopGildenblat::a)
#define DEF_PL_MEMBER2(a,...) .def_readwrite(#a, &PotLoopGildenblat::a, __VA_ARGS__)
    py::class_<PotLoopGildenblat>(m, "PotLoopGildenblat",
        "Potential Loop Solver (Gildenblat)")
        .def(py::init<>())
        DEF_PL_MEMBER(phi_t)
        DEF_PL_MEMBER(g_A)
        DEF_PL_MEMBER(psi_a)
        DEF_PL_MEMBER(psi_b)
        DEF_PL_MEMBER(n_i)
        DEF_PL_MEMBER(N_A)
        DEF_PL_MEMBER(eps_si)
        DEF_PL_MEMBER(phi_ms)
        DEF_PL_MEMBER(Q_0)
        DEF_PL_MEMBER(g_t)
        DEF_PL_MEMBER(cox)
        DEF_PL_MEMBER2(psi_t, "trap energies")
        DEF_PL_MEMBER2(N_t, "trap count for each energy")
        DEF_PL_MEMBER2(tol_bits, "result for phi_s will be exact to 2^(1-tol_bits)")
        // -- THESE ARE NEW --
        DEF_PL_MEMBER(lam_bulk)
        DEF_PL_MEMBER(bulk_n)
        DEF_PL_MEMBER(bulk_p)
        // -- --
        .def("Es", py::vectorize(&PotLoopGildenblat::Es))
        .def("psi_s", py::vectorize(&psi_s<PotLoopGildenblat>), "v_ch"_a, "v_gb"_a, "solve psi_s for v_gb and v_ch")
        .def("psi_s2", &psi_s2<PotLoopGildenblat>, "v_ch"_a, "v_gb"_a,
            "debug method, returns (root.first, root.second, n_iter)");

    m.def("test_fdint", &test_fdint);
}