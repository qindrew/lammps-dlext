#!/bin/bash

PYTHON_EXECUTABLE=$(which python3)
LAMMPS_ROOT=$(${PYTHON_EXECUTABLE} -c 'import site; print(site.getsitepackages()[0])')/lammps/

cmake -S . -B build -DLAMMPS_ROOT=${LAMMPS_ROOT}
cmake --build build --target install
