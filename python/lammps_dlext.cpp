// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "PyDLExt.h"
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace LAMMPS_NS;
namespace LAMMPS_dlext = LAMMPS_NS::dlext;

void export_PySampler(py::module m)
{
    using PyFunction = py::function;
    using PySampler = LAMMPS_NS::dlext::Sampler<PyFunction, LMPDeviceType>;
    using PySamplerSPtr = std::shared_ptr<PySampler>;

    py::class_<PySampler>(m, "DLExtSampler")
        .def(py::init(
            [ ] (
                py::object lmp,
                std::vector<std::string> args,
                LAMMPS_dlext::AccessLocation location, LAMMPS_dlext::AccessMode mode
            ) {
                auto pyptr = lmp.attr("lmp");
                auto pyaddr = pyptr.attr("value");
                auto lmp_addr = pyaddr.cast<uintptr_t>();
                auto lmp_ptr = reinterpret_cast<LAMMPS_NS::LAMMPS*>(lmp_addr);
                std::vector<char *> cstrs;
                cstrs.reserve(args.size());
                for (auto &s : args) cstrs.push_back(&s[0]);
                int narg = cstrs.size();
                return (new LAMMPS_dlext::Sampler<PyFunction, LMPDeviceType>(lmp_ptr, narg, cstrs.data(), location, mode));
        }))
        .def("set_callback", &PySampler::set_callback)
        .def("forward_data", &PySampler::forward_data<PyFunction>)

        .def("get_positions",  &PySampler::get_positions<LAMMPS_dlext::kOnDevice>)
        .def("get_velocities", &PySampler::get_velocities<LAMMPS_dlext::kOnDevice>)
        .def("get_net_forces", &PySampler::get_net_forces<LAMMPS_dlext::kOnDevice>)
        .def("get_type",       &PySampler::get_type<LAMMPS_dlext::kOnDevice>)
        .def("get_tag",        &PySampler::get_tag<LAMMPS_dlext::kOnDevice>)
        #ifdef KOKKOS_ENABLE_CUDA
        .def("get_positions",  &PySampler::get_positions<LAMMPS_dlext::kOnHost>)
        .def("get_velocities", &PySampler::get_velocities<LAMMPS_dlext::kOnHost>)
        .def("get_net_forces", &PySampler::get_net_forces<LAMMPS_dlext::kOnHost>)
        .def("get_type",       &PySampler::get_type<LAMMPS_dlext::kOnHost>)
        .def("get_tag",        &PySampler::get_tag<LAMMPS_dlext::kOnHost>)
        #endif
        ;    
}

PYBIND11_MODULE(dlpack_extension, m)
{
    // Enums
    py::enum_<LAMMPS_dlext::AccessLocation>(m, "AccessLocation")
        .value("OnDevice", LAMMPS_dlext::kOnDevice)
        #ifdef KOKKOS_ENABLE_CUDA
        .value("OnHost", LAMMPS_NS::dlext::kOnHost)
        #endif
        ;

    py::enum_<LAMMPS_dlext::AccessMode>(m, "AccessMode")
        .value("Read", LAMMPS_dlext::kRead)
        .value("ReadWrite", LAMMPS_dlext::kReadWrite)
        .value("Overwrite", LAMMPS_dlext::kOverwrite);

    // Classes
    export_PySampler(m);
}
