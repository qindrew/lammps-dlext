// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "PyDLExt.h"
#include "Sampler.h"
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace LAMMPS_NS;
using namespace LAMMPS_NS::dlext;

void export_PySampler(py::module m)
{
    //using SamplerSPtr = std::shared_ptr<Sampler>;
 
    using PyFunction = py::function;
    using PySampler = Sampler<PyFunction, PyEncapsulator, LMPDeviceType>;
    using PySamplerSPtr = std::shared_ptr<PySampler>;

    py::class_<PySampler>(m, "DLExtSampler")
        .def(py::init([](LAMMPS* lmp, std::vector<std::string> args, PyFunction function, AccessLocation location, AccessMode mode) {
          std::vector<char *> cstrs;
          cstrs.reserve(args.size());
          for (auto &s : args) cstrs.push_back(const_cast<char *>(s.c_str()));
          return (new Sampler<PyFunction, PyEncapsulator, LMPDeviceType>(lmp, cstrs.size(), cstrs.data(), function, location, mode));
        }))
        //.def("update", &PySampler::update)
        .def("get_positions", &PySampler::get_positions);
        //.def("get_velocities", &PySampler::get_velocities)
        //.def("get_net_forces", &PySampler::get_net_forces)
        //.def("get_types", &PySampler::get_types)
        //.def("get_images", &PySampler::get_images);
        
}

PYBIND11_MODULE(dlpack_extension, m)
{
    // Enums
    py::enum_<AccessLocation>(m, "AccessLocation")
        .value("OnHost", kOnHost)
#ifdef KOKKOS_ENABLE_CUDA
        .value("OnDevice", kOnDevice)
#endif
        ;

    py::enum_<AccessMode>(m, "AccessMode")
        .value("Read", kRead)
        .value("ReadWrite", kReadWrite)
        .value("Overwrite", kOverwrite);

    // Classes
    export_PySampler(m);

    // Methods
    // NOTE: PyEncapsulator is defined in PyDLext.h, with a member function wrap_property()
    //       Positions, Types, Velocities, NetForces, Tags and Images are structs defined in Sampler.h
    // TODO: PyEncapsulator should be aware of DLExtSampler
/*
    m.def("positions",  &PyEncapsulator<Positions>::wrap_property);
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
