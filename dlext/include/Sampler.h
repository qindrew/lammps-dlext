// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef DLEXT_SAMPLER_H_
#define DLEXT_SAMPLER_H_

#include "SystemView.h"
#include "KOKKOS/kokkos_type.h"
#include "fix.h"

using namespace LAMMPS::NS;

namespace dlext
{

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
    Sampler(
        LAMMPS* lmp, int narg, char** arg,
        ExternalUpdater update_callback,
        AccessLocation location,
        AccessMode mode
    );

    //! There is no need for SystemDefinition (a HOOMD term) for SystemView (a PySAGES term)
    //! because all Fix classes have pointers to all other data structures Atom, Pair
    void setSystemDefinition(void* sysdef) override
    {
        return;
    }

    //! Override Fix::post_force(): invoked after force computation 
    void post_force(TimeStep timestep) override
    {
        forward_data(_update_callback, _location, _mode, timestep);
    }

    //! simply returns this
    void* system_view() const;

    //! Wraps the system positions, velocities, reverse tags, images and forces as
    //! DLPack tensors and passes them to the external function `callback`.
    //!
    //! The (non-typed) signature of `callback` is expected to be
    //!     callback(positions, velocities, rtags, images, forces, n)
    //! where `n` ìs an additional `TimeStep` parameter.
    //!
    //! The data for the particles information is requested at the given `location`
    //! and access `mode`. NOTE: Forces are always passed in readwrite mode.
    //! TODO: Move Wrapper from SystemView.h to here
    template <typename Callback>
    void forward_data(Callback callback, AccessLocation location, AccessMode mode, TimeStep n)
    {
        atomKK->sync(execution_space,datamask_read);

        x = atomKK->k_x.view<DeviceType>();
        v = atomKK->k_v.view<DeviceType>();
        f = atomKK->k_f.view<DeviceType>();
        type = atomKK->k_type.view<DeviceType>();
        tag = atomKK->k_tag.view<DeviceType>();

        if (atomKK->omega_flag)
            omega  = atomKK->k_omega.view<DeviceType>();

        if (atomKK->angmom_flag)
            angmom = atomKK->k_angmom.view<DeviceType>();

        if (atomKK->torque_flag)
            torque = atomKK->k_torque.view<DeviceType>();

        /*
            TODO: wrap these KOKKOS arrays into DLManagedTensor to pass to callback
            Wrapper and the structs (PositionTypes, VelocitiesMasses, etc.) are currently residing in SystemView.h
        */
        
        auto pos_capsule = Wrapper<PositionsTypes>::wrap(lmp, location, mode);
        auto vel_capsule = Wrapper<VelocitiesMasses>::wrap(lmp, location, mode);
        auto rtags_capsule = Wrapper<RTags>::wrap(lmp, location, mode);
        auto img_capsule = Wrapper<Images>::wrap(lmp, location, mode);
        auto force_capsule = Wrapper<NetForces>::wrap(lmp, location, kReadWrite);

        // callback will be responsible for advancing the simulation for n steps

        callback(pos_capsule, vel_capsule, rtags_capsule, img_capsule, force_capsule, n);
    }

private:
    //SystemView _sysview;
    ExternalUpdater _update_callback;
    AccessLocation _location;
    AccessMode _mode;

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
};

template <typename ExternalUpdater, template <typename> class Wrapper>
Sampler<ExternalUpdater, Wrapper>::Sampler(
    //SystemView sysview,
    ExternalUpdater update, AccessLocation location, AccessMode mode)
    : Fix(lmp, narg, arg)
    //, _sysview { sysview }
    , _update_callback { update }
    , _location { location }
    , _mode { mode }
{ 
    this->setSimulator(lmp);

    kokkosable = 1;
    atomKK = (AtomKokkos *) atom;

    execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
    
    datamask_read   = X_MASK | V_MASK | F_MASK | TYPE_MASK | OMEGA_MASK | MASK_MASK | TORQUE_MASK | ANGMOM_MASK;
    datamask_modify = X_MASK | V_MASK | F_MASK | OMEGA_MASK | TORQUE_MASK | ANGMOM_MASK;
}

template <typename ExternalUpdater, template <typename> class Wrapper>
void* Sampler<ExternalUpdater, Wrapper>::system_view() const
{
    return this->lmp;
}





}  // namespace dlext

#endif  // DLEXT_SAMPLER_H_
