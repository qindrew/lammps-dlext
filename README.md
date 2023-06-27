# LAMMPS-dlext

Provides access to [LAMMPS](https://www.lammps.org) simulation data on CPU or GPU through
[DLPack](https://github.com/dmlc/dlpack).

## Installation

Make sure that you have LAMMPS installed and it was built as a shared library, and that
the LAMMPS Python module is also installed. If appropriate, activate the python
environment where the LAMMPS Python module can be found.

Clone this repository:
```bash
git clone https://github.com/SSAGESLabs/lammps-dlext.git
cd lammps-dlext
```

Configure and build the plugin:
```bash
BUILD_PATH=build
cmake -S . -B $BUILD_PATH -DCMAKE_PREFIX_PATH=/path/to/lammps/top/level/folder
cmake --build $BUILD_PATH --target install -j4
```

Alternatively to setting `CMAKE_PREFIX_PATH` you can directly set the following:

 - The path to `LAMMPS_ROOT` (where the file `LAMMPSConfig.cmake` can be found).
 - When compiling with Kokkos support, the path to `Kokkos_ROOT` (where the file
   `KokkosConfig.cmake` can be found). See also [the Kokkos
   documentation](https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Compiling.html#using-kokkos-installed-package).

If during compilation you encounter an error stating that `dlpack/dlpack.h` cannot be
found, try adding ` -DFETCH_DLPACK=ON` at the end of the first cmake command above.

For additional information on how to build and install LAMMPS and lammps-dlext in specific
platforms, we invite you to visit our [wiki
page](https://github.com/SSAGESLabs/lammps-dlext/wiki).
