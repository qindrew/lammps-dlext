// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "PyDLExt.h"
#include "Sampler.h"
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
        .def(py::init([](LAMMPS* lmp, std::vector<std::string> args, PyFunction function, LAMMPS_dlext::AccessLocation location, LAMMPS_dlext::AccessMode mode) {
          std::vector<char *> cstrs;
          cstrs.reserve(args.size());
          for (auto &s : args) cstrs.push_back(&s[0]);
          int narg = cstrs.size();
          return (new LAMMPS_dlext::Sampler<PyFunction, LMPDeviceType>(lmp, narg, cstrs.data(), function, location, mode));
        }))
        .def("forward_data", &PySampler::forward_data<PyFunction>)
        .def("get_positions", &PySampler::get_positions)
        .def("get_velocities", &PySampler::get_velocities)
        .def("get_net_forces", &PySampler::get_net_forces)
        .def("get_type", &PySampler::get_type)
        .def("get_tag", &PySampler::get_tag);
    
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

    // Methods
    // NOTE: PyEncapsulator is defined in PyDLext.h, with a member function wrap_property()
    //       Positions, Types, Velocities, NetForces, Tags and Images are structs defined in Sampler.h
    // TODO: PyEncapsulator should be aware of DLExtSampler
/*
    m.def("get_positions",  &PyEncapsulator<Positions>::wrap_property);    
    m.def("types",      &PyEncapsulator<Types>::wrap_property);
    m.def("velocities", &PyEncapsulator<Velocities>::wrap_property);
    m.def("net_forces", &PyEncapsulator<NetForces>::wrap_property);
    m.def("tags",       &PyEncapsulator<Tags>::wrap_property);
    m.def("images",     &PyEncapsulator<Images>::wrap_property);
*/
/*
    m.def("masses",      &PyEncapsulator<Masses>::wrap_property);
    m.def("rtags",       &PyEncapsulator<RTags>::wrap_property);
    m.def("net_torques", &PyEncapsulator<NetTorques>::wrap_property);
*/

/*    
    m.def("orientations", &PyEncapsulator<Orientations>::wrap);
    m.def("angular_momenta", &PyEncapsulator<AngularMomenta>::wrap);
    m.def("moments_of_intertia", &PyEncapsulator<MomentsOfInertia>::wrap);
    m.def("charges", &PyEncapsulator<Charges>::wrap);
    m.def("diameters", &PyEncapsulator<Diameters>::wrap);  
    m.def("net_virial", &PyEncapsulator<NetVirial>::wrap);
*/
}
