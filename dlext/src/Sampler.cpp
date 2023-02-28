// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "Sampler.h"

using namespace LAMMPS_NS;
using namespace LAMMPS_NS::dlext;
using namespace FixConst;

#define SamplerT Sampler<ExternalUpdater, Wrapper, DeviceType>

template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType>
SamplerT::Sampler(LAMMPS* lmp, int narg, char** arg,
    ExternalUpdater update, AccessLocation location, AccessMode mode) : FixExternal(lmp, narg, arg),
    _update_callback { update },
    _location { location },
    _mode { mode }
{ 
    kokkosable = 1;
    atomKK = (AtomKokkos *) atom;

    execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
    
    datamask_read   = X_MASK | V_MASK | F_MASK | TYPE_MASK | IMAGE_MASK | OMEGA_MASK | MASK_MASK | TORQUE_MASK | ANGMOM_MASK;
    datamask_modify = X_MASK | V_MASK | F_MASK | OMEGA_MASK | TORQUE_MASK | ANGMOM_MASK;
}

template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType>
int SamplerT::setmask()
{
    int mask = 0;
    mask |= POST_FORCE;
    mask |= MIN_POST_FORCE;
    return mask;
}

template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType>
template <typename Callback>
void SamplerT::forward_data(Callback callback, AccessLocation location, AccessMode mode, TimeStep n)
{
    atomKK->sync(execution_space,datamask_read);

    x = atomKK->k_x.template view<DeviceType>();
    v = atomKK->k_v.template view<DeviceType>();
    f = atomKK->k_f.template view<DeviceType>();
    type = atomKK->k_type.template view<DeviceType>();
    image = atomKK->k_image.template view<DeviceType>();
    tag = atomKK->k_tag.template view<DeviceType>();

    if (atomKK->omega_flag)
        omega  = atomKK->k_omega.template view<DeviceType>();

    if (atomKK->angmom_flag)
        angmom = atomKK->k_angmom.template view<DeviceType>();

    if (atomKK->torque_flag)
        torque = atomKK->k_torque.template view<DeviceType>();

    // wrap these KOKKOS arrays into DLManagedTensor to pass to callback

    int nlocal = atom->nlocal;

    int offset = 0;
    auto pos_capsule = wrap(x.data(), location, mode, nlocal, 3);
    auto vel_capsule = wrap(v.data(), location, mode, nlocal, 3);
    auto type_capsule = wrap(type.data(), location, mode, nlocal, 1);
    auto tag_capsule = wrap(tag.data(), location, mode, nlocal, 1);
    auto force_capsule = wrap(f.data(), location, mode, nlocal, 3);

    // callback will be responsible for advancing the simulation for n steps

//    callback(pos_capsule, vel_capsule, rtags_capsule, img_capsule, force_capsule, n);
    
}

template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType>
inline DLDevice SamplerT::dldevice(bool gpu_flag)
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
template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType>
template <typename T>
DLManagedTensorPtr SamplerT::wrap(const T* data, const AccessLocation location, const AccessMode mode,
                                  const int num_particles, int64_t size2 /*=1*/,
                                  uint64_t offset /*= 0*/, uint64_t stride1_offset /*= 0*/)
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
    dltensor.device = dldevice(gpu_flag);
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


