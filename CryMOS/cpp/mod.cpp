#include <functional> //bind
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

using Float=double;

int test() {
    return 4;
}

PYBIND11_MODULE(mod, m) {
    m.def("test", &test);
}