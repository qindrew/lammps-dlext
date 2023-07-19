# SPDX-License-Identifier: MIT
# This file is part of `lammps-dlext`, see LICENSE.md

from ._api import (  # noqa: F401 # pylint: disable=E0401
    # Enums
    ExecutionSpace,
    kOnDevice,
    kOnHost,
    # Classes
    LAMMPSView,
    # Methods
    forces,
    has_kokkos_cuda_enabled,
    images,
    masses,
    positions,
    tags,
    tags_map,
    types,
    velocities,
    # Other attributes
    kImgMask,
    kImgMax,
    kImgBits,
    kImg2Bits,
    kImgBitSize,
)


class FixDLExt(_api.FixDLExt):
    def __init__(self, lammps, args = "dlext all dlext space device"):
        super().__init__(lammps, args)
