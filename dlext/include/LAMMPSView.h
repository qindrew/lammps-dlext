// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef LAMMPSVIEW_H_
#define LAMMPSVIEW_H_

#include "atom_masks.h"
#include "cxx11utils.h"
#include "dlpack/dlpack.h"
#include "pointers.h"

namespace LAMMPS_NS
{
namespace dlext
{

// { // Aliases

const auto kOnHost = ExecutionSpace::Host;
const auto kOnDevice = ExecutionSpace::Device;

// } // Aliases

constexpr unsigned int DLEXT_MASK = (
    X_MASK | V_MASK | F_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | IMAGE_MASK
);

//! LAMMPSView is a wrapper around a LAMMPS* instance which provides convenience methods
//! to retrieve some of the system information.
class DEFAULT_VISIBILITY LAMMPSView : public Pointers {
public:
    LAMMPSView(LAMMPS* lmp);

    // Provide easy access to the atom pointers
    Atom* atom_ptr() const;
    AtomKokkos* atom_kokkos_ptr() const;

    //! Given an execution space, returns kDLCUDA if LAMMPS was built with KOKKOS and
    //! Cuda supoprt, and it's available at runtime. Otherwise, returns kDLCPU.
    DLDeviceType device_type(ExecutionSpace requested_space) const;

    //! The device id where this class instances are being executed
    int device_id() const;

    bool has_kokkos_cuda_enabled() const;

    // Convenience methods for retriving the number of particles
    int local_particle_number() const;
    bigint global_particle_number() const;

    //! If KOKKOS is available, synchronize on the particle data on the requested space
    void synchronize(ExecutionSpace requested_space = kOnDevice);

private:
    ExecutionSpace try_pick(ExecutionSpace requested_space) const;
};

}  // namespace dlext
}  // namespace LAMMPS_NS

#endif  // LAMMPSVIEW_H_
