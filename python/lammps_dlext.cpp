// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "LAMMPSView.h"
#include "PyDLExt.h"

#include <pybind11/stl.h>

using namespace LAMMPS_NS;

namespace cxx11 = cxx11utils;
namespace dl = LAMMPS_NS::dlext;
namespace py = pybind11;

LAMMPS* to_lammps_ptr(py::object lmp)
{
    auto pyptr = lmp.attr("lmp");
    auto pyaddr = pyptr.attr("value");
    auto lmp_addr = pyaddr.cast<uintptr_t>();
    return reinterpret_cast<LAMMPS*>(lmp_addr);
}

void export_LAMMPSView(py::module& m)
{
    py::class_<dl::LAMMPSView>(m, "LAMMPSView")
        .def(py::init(
            [] (py::object lmp) { return cxx11::make_unique<dl::LAMMPSView>(to_lammps_ptr(lmp)); }
        ))
        .def("device_type", &dl::LAMMPSView::device_type)
        .def("has_kokkos_cuda_enabled", &dl::LAMMPSView::has_kokkos_cuda_enabled)
        .def("local_particle_number", &dl::LAMMPSView::local_particle_number)
        .def("global_particle_number", &dl::LAMMPSView::global_particle_number)
        .def("synchronize", &dl::LAMMPSView::synchronize)
        ;
}

void export_PySampler(py::module& m)
{
    using PyFunction = py::function;
    using PySampler = dl::Sampler<PyFunction, LMPDeviceType>;
    using PySamplerSPtr = std::shared_ptr<PySampler>;

    py::class_<PySampler>(m, "DLExtSampler")
        .def(py::init(
            [ ] (
                py::object lmp,
                std::vector<std::string> args,
                dl::AccessMode mode
            ) {
                auto lmp_ptr = to_lammps_ptr(lmp);
                std::vector<char *> cstrs;
                cstrs.reserve(args.size());
                for (auto &s : args) cstrs.push_back(&s[0]);
                int narg = cstrs.size();
                return (new dl::Sampler<PyFunction, LMPDeviceType>(lmp_ptr, narg, cstrs.data(), mode));
        }))
        .def("set_callback",   &PySampler::set_callback)
        .def("forward_data",   &PySampler::forward_data<PyFunction>)
        .def("get_positions",  &PySampler::get_positions)
        .def("get_velocities", &PySampler::get_velocities)
        .def("get_net_forces", &PySampler::get_net_forces)
        .def("get_type",       &PySampler::get_type)
        .def("get_tag",        &PySampler::get_tag)
        ;
}

PYBIND11_MODULE(dlpack_extension, m)
{
    // Enums
    py::enum_<ExecutionSpace>(m, "ExecutionSpace")
        .value("OnDevice", dl::kOnDevice)
        .value("OnHost", dl::kOnHost)
        ;

    py::enum_<dl::AccessMode>(m, "AccessMode")
        .value("Read", dl::kRead)
        .value("ReadWrite", dl::kReadWrite)
        .value("Overwrite", dl::kOverwrite)
        ;

    // Classes
    export_LAMMPSView(m);
    export_PySampler(m);
}
