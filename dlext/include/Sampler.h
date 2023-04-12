// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef DLEXT_SAMPLER_H_
#define DLEXT_SAMPLER_H_

#include "DLExt.h"
#include "KOKKOS/atom_kokkos.h"
#include "KOKKOS/kokkos_type.h"
#include "atom_masks.h"
#include "fix_external.h"
#include "update.h"

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
        host,  //!< Ask to acquire the data on the host
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
struct access_mode {
    //! The enum
    enum Enum {
        read,  //!< Data will be accessed read only
        readwrite,  //!< Data will be accessed for read and write
        overwrite  //!< The data is to be completely overwritten during this acquire
    };
};

using AccessMode = access_mode::Enum;
const auto kRead = access_mode::read;
const auto kReadWrite = access_mode::readwrite;
const auto kOverwrite = access_mode::overwrite;

using TimeStep = bigint;  // bigint depends on how LAMMPS was built

using namespace FixConst;

/*
  Sampler is essentially a LAMMPS fix that allows an external updater
  to advance atom positions based on the instantaneous values of the CVs
  NOTE: A closely related example is the existing FixExternal in LAMMPS
    (docs.lammps.org/fix_external.html)
*/

constexpr unsigned int DLEXT_MASK = X_MASK | V_MASK | F_MASK | TYPE_MASK | IMAGE_MASK | OMEGA_MASK
    | MASK_MASK | TORQUE_MASK | ANGMOM_MASK;

template <typename ExternalUpdater, typename DeviceType>
class DEFAULT_VISIBILITY Sampler : public FixExternal {
public:
    //! Constructor
    Sampler(
        LAMMPS* lmp, int narg, char** arg, ExternalUpdater update_callback, AccessLocation location,
        AccessMode mode
    )
        : FixExternal(lmp, narg, arg)
        , _update_callback { update_callback }
        , _location { location }
        , _mode { mode }
    {
        kokkosable = 1;
        atomKK = dynamic_cast<AtomKokkos*>(atom);
        execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

        datamask_read = DLEXT_MASK;
        datamask_modify = DLEXT_MASK;  // to enable restores from PySAGES
    }

    int setmask() override { return POST_FORCE; }

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
    void forward_data(Callback callback, AccessLocation location, AccessMode mode, TimeStep n)
    {
        if (location == kOnHost) {
            _forward_data<LMPHostType>(callback, location, mode, n);
        } else {
            _forward_data<LMPDeviceType>(callback, location, mode, n);
        }
    }

    template <typename RequestedLocation, typename Callback>
    void _forward_data(Callback callback, AccessLocation location, AccessMode mode, TimeStep n)
    {
        // TODO:
        // 1. Instead of datamask_read we should choose based on the AccessMode
        // 2. Check if execution_space should match the DeviceType or Location here
        atomKK->sync(execution_space, datamask_read);

        auto x = (atomKK->k_x).view<RequestedLocation>();
        auto v = (atomKK->k_v).view<RequestedLocation>();
        auto f = (atomKK->k_f).view<RequestedLocation>();
        auto type = (atomKK->k_type).view<RequestedLocation>();
        auto image = (atomKK->k_image).view<RequestedLocation>();
        auto tag = (atomKK->k_tag).view<RequestedLocation>();
/*
        if (atomKK->omega_flag)
            omega = (atomKK->k_omega).view<RequestedLocation>();
        if (atomKK->angmom_flag)
            angmom = (atomKK->k_angmom).view<RequestedLocation>();
        if (atomKK->torque_flag)
            torque = (atomKK->k_torque).view<RequestedLocation>();
*/
        // wrap these KOKKOS arrays into DLManagedTensor to pass to callback

        int nlocal = atom->nlocal;

        auto pos_capsule = wrap<Scalar3>(x.data(), location, mode, nlocal, 3);
        auto vel_capsule = wrap<Scalar3>(v.data(), location, mode, nlocal, 3);
        auto type_capsule = wrap<int>(type.data(), location, mode, nlocal, 1);
        auto tag_capsule = wrap<tagint>(tag.data(), location, mode, nlocal, 1);
        auto force_capsule = wrap<Scalar3>(f.data(), location, mode, nlocal, 3);

        // callback might require the info of the simulation timestep `n`
        // callback(pos_capsule, vel_capsule, rtags_capsule, img_capsule, force_capsule, n);
    }

    //! This function allows the external callback `_update_callback` to be called after
    //! every integration timestep
    void post_force(int) override
    {
        forward_data(_update_callback, _location, _mode, update->ntimestep);
    }

    template <typename RequestedLocation>
    auto get_positions()
    {
        int nlocal = atom->nlocal;
        auto x = (atomKK->k_x).view<RequestedLocation>();
        return wrap<Scalar3>(x.data(), _location, _mode, nlocal, 3);
    }

    template <typename RequestedLocation>
    auto get_velocities()
    {
        int nlocal = atom->nlocal;
        auto v = (atomKK->k_v).view<RequestedLocation>();
        return wrap<Scalar3>(v.data(), _location, _mode, nlocal, 3);
    }

    template <typename RequestedLocation>
    auto get_net_forces()
    {
        int nlocal = atom->nlocal;
        auto f = (atomKK->k_f).view<RequestedLocation>();
        return wrap<Scalar3>(f.data(), _location, _mode, nlocal, 3);
    }

    template <typename RequestedLocation>
    auto get_type()
    {
        int nlocal = atom->nlocal;
        auto type = (atomKK->k_type).view<RequestedLocation>();
        return wrap<int>(type.data(), _location, _mode, nlocal, 1);
    }

    template <typename RequestedLocation>
    auto get_tag()
    {
        int nlocal = atom->nlocal;
        auto tag = (atomKK->k_tag).view<RequestedLocation>();
        return wrap<tagint>(tag.data(), _location, _mode, nlocal, 1);
    }

    DLDevice dldevice(bool gpu_flag)
    {
        int gpu_id = 0;
        auto device_id = gpu_id;  // be careful here
        return DLDevice { gpu_flag ? kDLCUDA : kDLCPU, device_id };
    }

    template <typename T>
    DLManagedTensorPtr wrap(
        void* data, const AccessLocation location, const AccessMode mode, const int num_particles,
        int64_t size2 = 1, uint64_t offset = 0, uint64_t stride1_offset = 0
    )
    {
        assert((size2 >= 1));

#ifdef KOKKOS_ENABLE_CUDA
        bool gpu_flag = (location == kOnDevice);
#else
        bool gpu_flag = false;
#endif
        /*
                //auto location = gpu_flag ? kOnDevice : kOnHost;
                //auto handle = cxx11utils::make_unique<T>(data, location, mode);
                //auto bridge = cxx11utils::make_unique<DLDataBridge<T>>(handle);

                //bridge->tensor.manager_ctx = bridge.get();
                bridge->tensor.deleter = delete_bridge<T>;

                auto& dltensor = bridge->tensor.dl_tensor;
                // cast handle->data to void* -- no need
                //dltensor.data = opaque(bridge->handle->data);
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

        */
        std::vector<int64_t> shape;
        std::vector<int64_t> strides;
        DLManagedTensor* tensor = new DLManagedTensor;
        auto& dltensor = tensor->dl_tensor;
        
        dltensor.device = dldevice(gpu_flag);
        // dtype()
        dltensor.dtype = dtype<T>();

        // first be the number of particles
        shape.push_back(num_particles);
        if (size2 > 1)
        shape.push_back(size2);
        // from one particle datum to the next one
        strides.push_back(stride1<T>() + stride1_offset);
        if (size2 > 1)
            strides.push_back(1);

        dltensor.ndim = shape.size(); // 1 or 2 dims
        dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
        dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
        // offset for the beginning pointer
        dltensor.byte_offset = offset;

        return tensor;
    }

private:
    ExternalUpdater _update_callback;
    AccessLocation _location;
    AccessMode _mode;
    int _nlocal;

    // the ArrayTypes namespace and its structs (t_x_array, t_v_array and so on)
    // are defined in kokkos_type.h
    // typename ArrayTypes<DeviceType>::t_x_array x;
    // typename ArrayTypes<DeviceType>::t_v_array v;
    // typename ArrayTypes<DeviceType>::t_f_array f;
    // typename ArrayTypes<DeviceType>::t_v_array omega;
    // typename ArrayTypes<DeviceType>::t_v_array angmom;
    // typename ArrayTypes<DeviceType>::t_f_array torque;
    // typename ArrayTypes<DeviceType>::t_float_1d mass;
    // typename ArrayTypes<DeviceType>::t_int_1d type;
    // typename ArrayTypes<DeviceType>::t_int_1d mask;
    // typename ArrayTypes<DeviceType>::t_tagint_1d tag;
    // typename ArrayTypes<DeviceType>::t_imageint_1d image;
};

/*
struct Positions final {
    static DLManagedTensorPtr from(LAMMPS* lmp, AccessLocation location, AccessMode mode)
    {

       atomKK->sync(execution_space,datamask_read);
        return wrap(x, location, mode, 3);
    }
};

struct Types final {
    static DLManagedTensorPtr from(AccessLocation location, AccessMode mode
    )
    {
        return wrap(sampler.get_types(), location, mode, 1);
    }
};

struct Velocities final {
    static DLManagedTensorPtr from(const Sampler<ExternalUpdater, Wrapper, DeviceType>& sampler,
        AccessLocation location, AccessMode mode
    )
    {
        return wrap(sampler.get_velocities(), location, mode, 3);
    }
};

struct NetForces final {
    static DLManagedTensorPtr from(const Sampler<ExternalUpdater, Wrapper, DeviceType>& sampler,
        AccessLocation location, AccessMode mode
    )
    {
        return wrap(sampler.get_net_forces(), location, mode, 3);
    }
};
*/

}  // namespace dlext

}  // namespace LAMMPS_NS

#endif  // DLEXT_SAMPLER_H_
