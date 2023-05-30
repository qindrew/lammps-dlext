// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "FixDLExt.h"
#include "PyDLExt.h"

#include "pybind11/functional.h"

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
    py::class_<LAMMPSView>(m, "LAMMPSView")
        .def(py::init([](py::object lmp) {
            return cxx11::make_unique<LAMMPSView>(to_lammps_ptr(lmp));
        }))
        .def("device_type", &LAMMPSView::device_type)
        .def("has_kokkos_cuda_enabled", &LAMMPSView::has_kokkos_cuda_enabled)
        .def("local_particle_number", &LAMMPSView::local_particle_number)
        .def("global_particle_number", &LAMMPSView::global_particle_number)
        .def("synchronize", &LAMMPSView::synchronize);
}

void export_FixDLExt(py::module& m)
{
    py::class_<FixDLExt>(m, "FixDLExt")
        .def(py::init([](py::object lmp, std::vector<std::string> args) {
            auto lmp_ptr = to_lammps_ptr(lmp);
            std::vector<char*> cstrs;
            cstrs.reserve(args.size());
            for (auto& s : args)
                cstrs.push_back(&s[0]);
            int narg = cstrs.size();
            return cxx11::make_unique<FixDLExt>(lmp_ptr, narg, cstrs.data());
        }))
        .def("set_callback", &FixDLExt::set_callback);
}

PYBIND11_MODULE(dlpack_extension, m)
{
    // Enums
    py::enum_<ExecutionSpace>(m, "ExecutionSpace")
        .value("OnDevice", kOnDevice)
        .value("OnHost", kOnHost)
        ;

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
}
