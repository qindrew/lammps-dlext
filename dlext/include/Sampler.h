// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef DLEXT_SAMPLER_H_
#define DLEXT_SAMPLER_H_

#include "DLExt.h"
#include "KOKKOS/kokkos_type.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "fix_external.h"

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

using namespace FixConst;

/*
  Sampler is essentially a LAMMPS fix that allows an external updater
  to advance atom positions based on the instantaneous values of the CVs
  NOTE: A closely related example is the existing FixExternal in LAMMPS
    (docs.lammps.org/fix_external.html)
  
*/

template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType>
class DEFAULT_VISIBILITY Sampler : public FixExternal {
public:
    //! Constructor
    Sampler(LAMMPS* lmp, int narg, char** arg,
            ExternalUpdater update_callback,
            AccessLocation location,
            AccessMode mode
    ) : FixExternal(lmp, narg, arg),
    _update_callback { update_callback },
    _location { location },
    _mode { mode }
    { 
        kokkosable = 1;
        atomKK = (AtomKokkos *) atom;

        execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
        
        datamask_read   = X_MASK | V_MASK | F_MASK | TYPE_MASK | IMAGE_MASK | OMEGA_MASK | MASK_MASK | TORQUE_MASK | ANGMOM_MASK;
        datamask_modify = X_MASK | V_MASK | F_MASK | OMEGA_MASK | TORQUE_MASK | ANGMOM_MASK;
    }

    int setmask() override
    {
        int mask = 0;
        mask |= POST_FORCE;
        mask |= MIN_POST_FORCE;
        return mask;
    }

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

    auto get_positions()
    {
        int nlocal = atom->nlocal;
        return wrap(x.data(), _location, _mode, nlocal, 3);
    }

    auto get_velocities()
    {
        int nlocal = atom->nlocal;
        return wrap(v.data(), _location, _mode, nlocal, 3);
    }

    auto get_net_forces()
    {
       int nlocal = atom->nlocal;
       return wrap(f.data(), _location, _mode, nlocal, 3);
    }

    auto get_type()
    {
        int nlocal = atom->nlocal;
        return wrap(type.data(), _location, _mode, nlocal, 1);
    }

    auto get_tag()
    {
        int nlocal = atom->nlocal;
        return wrap(tag.data(), _location, _mode, nlocal, 1);
    }

    DLDevice dldevice(bool gpu_flag);

    template <typename T>
    DLManagedTensorPtr wrap(const T* data, const AccessLocation location, const AccessMode mode,
                        const int num_particles, int64_t size2 = 1, uint64_t offset = 0, uint64_t stride1_offset = 0);

private:
    ExternalUpdater _update_callback;
    AccessLocation _location;
    AccessMode _mode;
    int _nlocal;

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

} // namespace dlext

} // namespace LAMMPS_NS

#endif  // DLEXT_SAMPLER_H_
