# LAMMPS-dlext

Provide access to [LAMMPS](https://www.lammps.org) simulation data on CPU or GPU through [DLPack](https://github.com/dmlc/dlpack)

## Installation

Install upstream LAMMPS
* Build LAMMPS with KOKKOS and PYTHON packages
* Install LAMMPS into some location, e.g., under a python virtual env named ```pysages```: ```$HOME/miniconda3/envs/pysages/lib/python3.9/site-packages/lammps```
* Copy the folder ```$LMP_PATH/src/fmt``` into the installation folder ```include/lammps/fmt```
* Copy several key headers from ```$LMP_PATH/src/KOKKOS``` (e.g. kokkos_type.h, atom_kokkos.h, memory_kokkos.h, comm_kokkos.h) into ```include/lammps/KOKKOS```
* Copy KOKKOS-generated header files from ```build/lib/kokkos/KokkosCore_*.hpp``` and ```build/lib/kokkos/KokkosCore_config.h```  into ```include/lammps/KOKKOS```
* Copy all the KOKKOS source files from ```$LMP_PATH/lib/kokkos/core/src```  into ```include/lammps/KOKKOS/core/src```
* Copy all the KOKKOS source files from ```$LMP_PATH/lib/kokkos/containers/src``` and ```include/lammps/KOKKOS/containers/src```

Build LAMMPS dlext (this package):

* TODO: Add to the include path in CMakeLists.txt the KOKKOS header files under ```lib/kokkos/core/src```
* Set the same install path as the LAMMPS python module (see above):
```
  mkdir build && cd build

  cmake ../ -DCMAKE_INSTALL_PATH=$HOME/miniconda3/envs/pysages/lib/python3.9/site-packages/lammps
```
