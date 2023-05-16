// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "LAMMPSView.h"
#include "atom.h"
#include "atom_masks.h"
#include "KOKKOS/atom_kokkos.h"

namespace LAMMPS_NS
{
namespace dlext
{

LAMMPSView::LAMMPSView(LAMMPS_NS::LAMMPS* lmp)
    : Pointers(lmp)
{ }

Atom* LAMMPSView::atom_ptr() const { return lmp->atom; }
AtomKokkos* LAMMPSView::atom_kokkos_ptr() const { return lmp->atomKK; }

DLDeviceType LAMMPSView::device_type(ExecutionSpace requested_space) const
{
    return (try_pick(requested_space) == kOnDevice) ? kDLCUDA : kDLCPU;
}

int LAMMPSView::device_id() const { return 0; }  // TODO: infer this from the LAMMPS instance

bool LAMMPSView::has_kokkos_cuda_enabled() const
{
    bool has_cuda = strcmp(LMPDeviceType::name(), "Cuda") == 0;
    return has_cuda & (lmp->kokkos != nullptr);
}

int LAMMPSView::local_particle_number() const { return atom_ptr()->nlocal; }
bigint LAMMPSView::global_particle_number() const { return atom_ptr()->natoms; }

void LAMMPSView::synchronize(ExecutionSpace requested_space)
{
#ifdef LMP_KOKKOS_GPU
    if (lmp->kokkos) {
        atom_kokkos_ptr()->sync(try_pick(requested_space), ALL_MASK);
    }
#endif
}

ExecutionSpace LAMMPSView::try_pick(ExecutionSpace requested_space) const
{
    return has_kokkos_cuda_enabled() ? requested_space : kOnHost;
}

} // dlext
} // LAMMPS_NS