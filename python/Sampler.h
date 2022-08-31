// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef SAMPLER_H
#define SAMPLER_H

// Maintainer: ndtrung
// TODO: Switch from HOOMD API to LAMMPS API via the KOKKOS package

//#include "lammps/fix_dlext_kokkos.h"
#include "KOKKOS/kokkos_type.h"
#include "fix.h"
#include "pybind11/pybind11.h"
#include "dlpack/dlpack.h"

struct DLDataBridge {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLManagedTensor tensor;
};

using namespace LAMMPS_NS;

template<class DeviceType>
class Sampler : public Fix  // FixDLextKOKKOS
    {
    public:
        //! Constructor
      Sampler(LAMMPS_NS::LAMMPS* lmp, int narg, char** arg, pybind11::function python_update);

      virtual void setSimulator(LAMMPS_NS::LAMMPS* lmp) override;

        //! Take one timestep forward
      virtual void post_force(int) override;

      // run a custom python function on data from lammps
      // access_mode is ignored for forces. Forces are returned in readwrite mode always.
      //void run_on_data(pybind11::function py_exec, const access_location::Enum location, const access_mode::Enum mode);
      void run_on_data(pybind11::function py_exec);

    private:
      typename ArrayTypes<DeviceType>::t_x_array x;
      typename ArrayTypes<DeviceType>::t_v_array v;
      typename ArrayTypes<DeviceType>::t_f_array f;
      typename ArrayTypes<DeviceType>::t_imageint_1d image;

      template<typename TS, typename TV>
      DLDataBridge wrap(TS* const ptr, const bool, const int64_t size2 = 1, const uint64_t offset=0, uint64_t stride1_offset = 0);
      pybind11::function m_python_update;
      LAMMPS_NS::LAMMPS* m_lmp;
    };

void export_Sampler(pybind11::module& m);

#endif//SAMPLER_H
