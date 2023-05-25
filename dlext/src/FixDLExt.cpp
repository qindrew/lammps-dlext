// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "FixDLExt.h"

#include "accelerator_kokkos.h"
#include "error.h"
#include "update.h"

namespace LAMMPS_NS
{
namespace dlext
{

FixDLExt::FixDLExt(LAMMPS* lmp, int narg, char** arg)
    : Fix(lmp, narg, arg)
{
    auto on_host = true;
    auto bad_args = (narg != 3 || narg != 5);
    if (narg == 5) {
        auto on_device = (strcmp(arg[4], "device") == 0);
        on_host = (strcmp(arg[4], "host") == 0);
        bad_args |= (strcmp(arg[3], "space") != 0);
        bad_args |= (!on_host || !on_device);
    }
    if (bad_args)
        error->all(FLERR, "Illegal fix external command");

    view = cxx11::make_unique<LAMMPSView>(lmp);
    kokkosable = view->has_kokkos_cuda_enabled();
    atomKK = dynamic_cast<AtomKokkos*>(atom);
    execution_space = (on_host || !kokkosable) ? kOnHost : kOnDevice;
    datamask_read = EMPTY_MASK;
    datamask_modify = EMPTY_MASK;
}

int FixDLExt::setmask() { return FixConst::POST_FORCE; }
void FixDLExt::post_force(int) { callback(update->ntimestep); }
void FixDLExt::set_callback(DLExtCallback& cb) { callback = cb; }

} // dlext
} // LAMMPS_NS
