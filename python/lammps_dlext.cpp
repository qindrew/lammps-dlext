// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "FixDLExt.h"
#include "PyDLExt.h"

#include "pybind11/functional.h"
#include "pybind11/stl.h"

using namespace LAMMPS_NS;
using namespace LAMMPS_NS::dlext;

namespace py = pybind11;

LAMMPS_NS::LAMMPS* to_lammps_ptr(py::object lmp)
{
    auto pyptr = lmp.attr("lmp");
    auto pyaddr = pyptr.attr("value");
    auto lmp_addr = pyaddr.cast<uintptr_t>();
    return reinterpret_cast<LAMMPS*>(lmp_addr);
}

void export_LAMMPSView(py::module& m)
{
    py::class_<LAMMPSView, SPtr<LAMMPSView>>(m, "LAMMPSView")
        .def(py::init([](py::object lmp) {
            return std::make_shared<LAMMPSView>(to_lammps_ptr(lmp));
        }))
        .def("has_kokkos_cuda_enabled", &LAMMPSView::has_kokkos_cuda_enabled)
        .def("local_particle_number", &LAMMPSView::local_particle_number)
        .def("global_particle_number", &LAMMPSView::global_particle_number)
        .def("synchronize", &LAMMPSView::synchronize, py::arg("space") = kOnDevice)
        ;
}

void export_FixDLExt(py::module& m)
{
    py::class_<FixDLExt>(m, "FixDLExt")
        .def(py::init([](py::object lmp, std::vector<std::string> args) {
            auto lmp_ptr = to_lammps_ptr(lmp);
            std::vector<char*> cargs;
            cargs.reserve(args.size());
            for (auto& arg : args)
                cargs.push_back(const_cast<char*>(arg.c_str()));
            int narg = cargs.size();
            return cxx11::make_unique<FixDLExt>(lmp_ptr, narg, cargs.data());
        }))
        .def("set_callback", &FixDLExt::set_callback)
        .def_property_readonly("view", &FixDLExt::get_view)
        ;
}

PYBIND11_MODULE(_api, m)
{
    // We want to display the members of the module as `lammps.dlext.x`
    // instead of `lammps.dlext._api.x`.
    auto module_name = m.attr("__name__");
    m.attr("__name__") = "lammps.dlext";

    // Enums
    py::enum_<ExecutionSpace>(m, "ExecutionSpace")
        .value("kOnDevice", kOnDevice)
        .value("kOnHost", kOnHost)
        .export_values();

    // Classes
    export_LAMMPSView(m);
    export_FixDLExt(m);

    // Methods
    m.def("positions", enpycapsulate<&positions>);
    m.def("velocities", enpycapsulate<&velocities>);
    m.def("masses", enpycapsulate<&masses>);
    m.def("forces", enpycapsulate<&forces>);
    m.def("images", enpycapsulate<&images>);
    m.def("tags", enpycapsulate<&tags>);
    m.def("types", enpycapsulate<&types>);

    // Other attributes
    m.attr("kImgMask") = IMGMASK;
    m.attr("kImgMax") = IMGMAX;
    m.attr("kImgBits") = IMGBITS;
    m.attr("kImg2Bits") = IMG2BITS;

    // Set back the module_name to its original value
    m.attr("__name__") = module_name;
}
