//NAMING: [type][m][N]h
// type:
//  - fd: Fermi-Dirac Integral
//  - dfd: first derivative
//  - ifd: Inverse Fermi-Dirac Integral
//  - gfd: Generalized Fermi-Dirac Integrals
//  - dgfd: first derivative
//  - scfd: Semiconductor Fermi-Dirac Integrals (nonparabolic, vnonparabolic, dnonparabolic, vdnonparabolic)
// m: MINUS
// N: the order is +/- N/2
// Example: fdm9h: k=-9/2

#include <iostream>

namespace py = pybind11;

using fd_fun_t = double(double); // all fd*, dfd* and ifd* methods have this signature
using gfd_fun_t = double(double, double);

template<class F=fd_fun_t>
struct fdint_method {
    static py::object get_fd_mod() {
        static py::object fdint = py::module::import("fdint._fdint");
        return fdint;
    }

    static F* get_fd_fun(std::string name) {
        auto fun_dict = get_fd_mod().attr("__pyx_capi__").attr("__getitem__");
        py::capsule funptr = py::cast<py::capsule>(fun_dict(name));
        void* fn = static_cast<void*>(funptr);
        return reinterpret_cast<F*>(fn); //undefined behaviour in C++
    }

    using result_type = typename std::function<F>::result_type;

    fdint_method(std::string name)
    : callable(get_fd_fun(name))
    {}

    template<class... T>
    result_type operator()(T... args) const {
        return callable(args...);
    }

    void* addr() const {
        return reinterpret_cast<void*>(callable);
    }

    F * const callable;
};

void test_fdint() {
    fdint_method<> fdm9h("fdm9h");
    std::cout << "found function at " << std::hex << fdm9h.addr() << "\n";
    std::cout << "fdm9h(0.789) = " << fdm9h(0.789) << "\n";
}
