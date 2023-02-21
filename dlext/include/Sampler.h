// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef DLEXT_SAMPLER_H_
#define DLEXT_SAMPLER_H_

#include "DLExt.h"
#include "KOKKOS/kokkos_type.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "fix.h"
#include "lammps.h"

namespace LAMMPS_NS
{
namespace dlext
{

namespace cxx11 = cxx11utils;

//{ // Aliases

//! Specifies where to acquire the data
struct access_location {
    //! The enum
    enum Enum {
        host,   //!< Ask to acquire the data on the host
#ifdef KOKKOS_ENABLE_CUDA
        device  //!< Ask to acquire the data on the device
#endif
    };
};
using AccessLocation = access_location::Enum;
const auto kOnHost = access_location::host;
#ifdef KOKKOS_ENABLE_CUDA
const auto kOnDevice = access_location::device;
#endif

//! Specify how the data is to be accessed
struct access_mode  {
    //! The enum
    enum Enum {
        read,       //!< Data will be accessed read only
        readwrite,  //!< Data will be accessed for read and write
        overwrite   //!< The data is to be completely overwritten during this acquire
    };
};
using AccessMode = access_mode::Enum;
const auto kRead = access_mode::read;
const auto kReadWrite = access_mode::readwrite;
const auto kOverwrite = access_mode::overwrite;

using TimeStep = int;

/*
  Sampler is essentially a LAMMPS fix that allows an external updater
  to advance atom positions based on the instantaneous values of the CVs
  NOTE: A closely related example is the existing FixExternal in LAMMPS
    (docs.lammps.org/fix_external.html)
  
*/
template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType>
class DEFAULT_VISIBILITY Sampler : public Fix {
public:
    //! Constructor
    Sampler(LAMMPS* lmp, int narg, char** arg,
        ExternalUpdater update_callback,
        AccessLocation location,
        AccessMode mode
    );

    int set_mask() override;

    //! Wraps the system positions, velocities, reverse tags, images and forces as
    //! DLPack tensors and passes them to the external function `callback`.
    //!
    //! The (non-typed) signature of `callback` is expected to be
    //!     callback(positions, velocities, rtags, images, forces, n)
    //! where `n` ìs an additional `TimeStep` parameter.
    //!
    //! The data for the particles information is requested at the given `location`
    //! and access `mode`. NOTE: Forces are always passed in readwrite mode.

    template <typename Callback>
    void forward_data(Callback callback, AccessLocation location, AccessMode mode, TimeStep n);

    auto get_positions() { return x; }
    auto get_velocities() { return v; }
    auto get_net_forces() { return f; }
    auto get_types() { return type; }
    auto get_images() { return image; }

private:
    ExternalUpdater _update_callback;
    AccessLocation _location;
    AccessMode _mode;

    // the ArrayTypes namespace and its structs (t_x_array, t_v_array and so on) are defined in kokkos_type.h
    typename ArrayTypes<DeviceType>::t_x_array x;
    typename ArrayTypes<DeviceType>::t_v_array v;
    typename ArrayTypes<DeviceType>::t_f_array f;
    typename ArrayTypes<DeviceType>::t_v_array omega;
    typename ArrayTypes<DeviceType>::t_v_array angmom;
    typename ArrayTypes<DeviceType>::t_f_array torque;
    typename ArrayTypes<DeviceType>::t_float_1d mass;
    typename ArrayTypes<DeviceType>::t_int_1d_randomread type;
    typename ArrayTypes<DeviceType>::t_int_1d mask;
    typename ArrayTypes<DeviceType>::t_tagint_1d tag;
    typename ArrayTypes<DeviceType>::t_imageint_1d image;
};

template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType>
inline DLDevice dldevice(const Sampler<ExternalUpdater, Wrapper, DeviceType>& sampler,
                         bool gpu_flag)
{
    int gpu_id = 0;
    auto device_id = gpu_id; // be careful here 
    return DLDevice { gpu_flag ? kDLCUDA : kDLCPU, device_id };
}

/*
  wrap is called by Sampler::forward_data()
  data : a generic pointer to an atom property (x, v, f)
  location
  mode
  num particles,
*/
template <typename T>
DLManagedTensorPtr wrap(auto data, const AccessLocation location, const AccessMode mode,
                        const int num_particles,
                        int64_t size2 = 1, uint64_t offset = 0, uint64_t stride1_offset = 0)
{
    assert((size2 >= 1));

    #ifdef KOKKOS_ENABLE_CUDA
    bool gpu_flag = (location == kOnDevice);
    #else
    bool gpu_flag = false;
    #endif

    //auto location = gpu_flag ? kOnDevice : kOnHost;
    auto handle = cxx11utils::make_unique<T>(data, location, mode);
    auto bridge = cxx11utils::make_unique<DLDataBridge<T>>(handle);

    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = delete_bridge<T>;

    auto& dltensor = bridge->tensor.dl_tensor;
    // cast handle->data to void* -- no need
    dltensor.data = opaque(bridge->handle->data);
    dltensor.device = dldevice(sampler, gpu_flag);
    // dtype()
    dltensor.dtype = dtype<T>();

    auto& shape = bridge->shape;
    // first be the number of particles
    shape.push_back(num_particles);
    if (size2 > 1)
       shape.push_back(size2);
    // from one particle datum to the next one
    auto& strides = bridge->strides;
    strides.push_back(stride1<T>() + stride1_offset);
    if (size2 > 1)
        strides.push_back(1);

    dltensor.ndim = shape.size(); // 1 or 2 dims
    dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
    dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
    // offset for the beginning pointer
    dltensor.byte_offset = offset;

    return &(bridge.release()->tensor);
}

// the following structs provide a way to return the pointer to the corresponding atom property
//   used in PyDLExt.h and lammps_dlext.cc

struct Positions final {
    static DLManagedTensorPtr from(const AccessLocation location, AccessMode mode)
    {
        
        return wrap(x, location, mode, 3);
    }
};

struct Types final {
    static DLManagedTensorPtr from(auto 
        const AccessLocation location, AccessMode mode
    )
    {
        return wrap(sampler.get_types(), location, mode, 1);
    }
};

struct Velocities final {
    static DLManagedTensorPtr from(const Sampler<ExternalUpdater, Wrapper, DeviceType>& sampler,
        const AccessLocation location, AccessMode mode
    )
    {
        return wrap(sampler.get_velocities(), location, mode, 3);
    }
};

struct NetForces final {
    static DLManagedTensorPtr from(const Sampler<ExternalUpdater, Wrapper, DeviceType>& sampler,
        const AccessLocation location, AccessMode mode
    )
    {
        return wrap(sampler.get_net_forces(), location, mode, 3);
    }
};


} // namespace dlext

} // namespace LAMMPS_NS

#endif  // DLEXT_SAMPLER_H_
