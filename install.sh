module load python
source activate /project/depablo/acqin2/environments/MACE_lammps_pysages_rtx6000 #clean conda environment
module unload python

module load cuda/12.2 \
             cudnn/9.4.0 \
             openmpi/4.1.2+gcc-10.2.0 \
             cmake/3.19 \
             clang/13.0.0 \
             ffmpeg/5.1 \
             fftw3/3.3.9 \
             openblas/0.3.13

git clone https://github.com/lammps/lammps.git
cd lammps
git checkout stable_29Aug2024_update2

# edit the cmake/Modules/Packages/KOKKOS.cmake file

# start interactive session
BUILD_PATH=build
PYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python
INSTALL_PREFIX=$(${PYTHON_EXECUTABLE} -c "import sys; print(sys.prefix)")

cmake -S cmake -B $BUILD_PATH \
    -C cmake/presets/most.cmake \
    -C cmake/presets/nolib.cmake \
    -C cmake/presets/kokkos-cuda.cmake \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DBUILD_SHARED_LIBS=ON \
    -DFFT=KISS \
    -DKokkos_ARCH_PASCAL60=OFF \
    -DKokkos_ARCH_AMPERE80=ON \
    -DLAMMPS_EXCEPTIONS=ON \
    -DPKG_MPI=ON \
    -DPKG_OPENMP=ON \
    -DPython_EXECUTABLE=${PYTHON_EXECUTABLE} \
    -DWITH_JPEG=no \
    -DWITH_PNG=no \
    -DBUILD_TOOLS=no \
#exit interactive session

cmake --build $BUILD_PATH --target install -j8
cd $BUILD_PATH
make install-python


git clone --branch comment_line https://github.com/qindrew/lammps-dlext.git #edited cmake files
cd lammps-dlext

BUILD_PATH=build
cmake -S . -B $BUILD_PATH -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib64/cmake
cmake --build $BUILD_PATH --target install -j4
