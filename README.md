# LAMMPS-dlext

Provide access to [LAMMPS](https://www.lammps.org) simulation data on CPU or GPU through [DLPack](https://github.com/dmlc/dlpack)

## Installation

Install upstream LAMMPS
* Activate the `pysages3` virtual environment
```
  source activate pysages3
```
* Configure LAMMPS with KOKKOS and PYTHON packages, and install LAMMPS to the same location as the `pysages` virtual environment
```
   git clone https://github.com/lammps/lammps.git
   cd lammps
   ccmake -S cmake -B build -D PKG_KOKKOS=on -D Kokkos_ENABLE_CUDA=on -D Kokkos_ARCH_AMPERE80=on \
     -D PKG_MOLECULE=on -D PKG_KSPACE=on -D BUILD_SHARED_LIBS=on -D PKG_PYTHON=on -D FFT=KISS \
     -D CMAKE_INSTALL_PREFIX=`python3 -c "import sys; print(sys.prefix)""`
   cd build
   make -j6
   make install
```
where `CMAKE_INSTALL_PREFIX` points to the environment level, not down to `lib/python3.x/site-packages`.
The following steps are needed to compile `lammpds-dlext` because the header files are not copied into the LAMMPS installation folder:
* Copy the folder ```$LMP_PATH/src/fmt``` into the installation folder ```include/lammps/fmt```
* Copy several key headers from ```$LMP_PATH/src/KOKKOS``` (e.g. kokkos_type.h, atom_kokkos.h, memory_kokkos.h, comm_kokkos.h) into ```include/lammps/KOKKOS```
* Copy KOKKOS-generated header files from ```build/lib/kokkos/KokkosCore_*.hpp``` and ```build/lib/kokkos/KokkosCore_config.h```  into ```include/lammps/KOKKOS```
* Copy all the KOKKOS source files from ```$LMP_PATH/lib/kokkos/core/src```  into ```include/lammps/KOKKOS/core/src```
* Copy all the KOKKOS source files from ```$LMP_PATH/lib/kokkos/containers/src``` into ```include/lammps/KOKKOS/containers/src```

Build LAMMPS dlext (this package):

* Set the same install path as the LAMMPS python module (see above):
```
  mkdir build && cd build

  cmake ../ -DCMAKE_INSTALL_PATH="`python3 -c "import site; print(site.getsitepackages()[0])"`/lammps"
```
* TODO: Add to the include path in CMakeLists.txt the KOKKOS header files under ```lib/kokkos/core/src```

