# CMake script for finding LAMMPS and setting up all needed compile options to create and link a plugin library
#
# Variables taken as input to this module:
# LAMMPS_ROOT :          location to look for LAMMPS, if it is not in the python path
#
# Variables defined by this module:
# FOUND_LAMMPS :         set to true if LAMMPS is found
# LAMMPS_LIBRARIES :     a list of all libraries needed to link to to access lammps (uncached)
# LAMMPS_INCLUDE_DIR :   a list of all include directories that need to be set to include LAMMPS
# LAMMPS_LIB :           a cached var locating the lammps library to link to
#
# NOTE: various BUILD_ and ENABLE_ flags translated from lammps_config.h so this plugin build can match the ABI of the installed lammps
#
# as a convenience (for the intended purpose of this find script), all include directories and definitions needed
# to compile with all the various libs (boost, python, winsoc, etc...) are set within this script

set(LAMMPS_ROOT "" CACHE FILEPATH "Directory containing a lammps installation (i.e. _lammps.so)")

# Let LAMMPS_ROOT take precedence, but if unset, try letting Python find a lammps package in its default paths.
if(LAMMPS_ROOT)
  set(lammps_installation_guess ${LAMMPS_ROOT})
else(LAMMPS_ROOT)
  find_package(PythonInterp)

  set(find_lammps_script "
from __future__ import print_function;
import sys, os; sys.stdout = open(os.devnull, 'w')
import lammps
print(os.path.dirname(lammps.__file__), file=sys.stderr, end='')")

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${find_lammps_script}"
                  ERROR_VARIABLE lammps_installation_guess)
  message(STATUS "Python output: " ${lammps_installation_guess})
endif(LAMMPS_ROOT)

message(STATUS "Looking for a LAMMPS installation at " ${lammps_installation_guess})
find_path(FOUND_LAMMPS_ROOT
        NAMES _lammps.so __init__.py
        HINTS ${lammps_installation_guess}
        )

if(FOUND_LAMMPS_ROOT)
  set(LAMMPS_ROOT ${FOUND_LAMMPS_ROOT} CACHE FILEPATH "Directory containing a lammps installation (i.e. _lammps.so)" FORCE)
  message(STATUS "Found lammps installation at " ${LAMMPS_ROOT})
else(FOUND_LAMMPS_ROOT)
  message(FATAL_ERROR "Could not find lammps installation, either set LAMMPS_ROOT or set PYTHON_EXECUTABLE to a python which can find lammps")
endif(FOUND_LAMMPS_ROOT)



# search for the lammps include directory
find_path(LAMMPS_INCLUDE_DIR
          NAMES lammps.h
          HINTS ${CMAKE_PREFIX_PATH}/include/lammps
          HINTS ${LAMMPS_ROOT}/../../../../include/lammps
          )

if (LAMMPS_INCLUDE_DIR)
    message(STATUS "Found LAMMPS include directory at: ${LAMMPS_INCLUDE_DIR}")
    mark_as_advanced(LAMMPS_INCLUDE_DIR)
else(LAMMPS_INCLUDE_DIR)
    message(STATUS "Cannot find LAMMPS include directory at ${CMAKE_PREFIX_PATH}/include/lammps")
endif (LAMMPS_INCLUDE_DIR)

set(LAMMPS_FOUND FALSE)
if (LAMMPS_INCLUDE_DIR AND LAMMPS_ROOT)
    set(LAMMPS_FOUND TRUE)
    mark_as_advanced(LAMMPS_ROOT)
endif (LAMMPS_INCLUDE_DIR AND LAMMPS_ROOT)

if (NOT LAMMPS_FOUND)
    message(SEND_ERROR "LAMMPS Not found. Please specify the location of your lammps installation in LAMMPS_ROOT")
endif (NOT LAMMPS_FOUND)

#############################################################
## Now that we've found lammps, lets do some setup
if (LAMMPS_FOUND)

include_directories(${LAMMPS_INCLUDE_DIR})

# run all of LAMMPS's generic lib setup scripts
set(CMAKE_MODULE_PATH ${LAMMPS_ROOT}
                      ${LAMMPS_ROOT}/../../../../lib64/cmake/LAMMPS
                      ${LAMMPS_ROOT}/../../../../lib64/cmake/Kokkos
                      ${CMAKE_MODULE_PATH}
                      )
#message(STATUS "current module path " ${CMAKE_MODULE_PATH})
# grab previously-set lammps configuration
include (lammps_cache)

# Handle user build options
#include (CMake_build_options)
#include (CMake_preprocessor_flags)
# setup the install directories
#include (CMake_install_options)

# Find the python executable and libraries
include (LAMMPSPythonSetup)
# Find CUDA and set it up
#include (LAMMPSCUDASetup)
# Set default CFlags
#include (LAMMPSCFlagsSetup)
# include some os specific options
#include (LAMMPSOSSpecificSetup)
# setup common libraries used by all targets in this project
#include (LAMMPSCommonLibsSetup)
# setup macros
#include (LAMMPSMacros)
# setup MPI support
#include (LAMMPSMPISetup)

include (LAMMPSConfig)
include (LAMMPSConfigVersion)
include (LAMMPS_Targets)

#include (KokkosConfig)
#include (KokkosConfigCommon)
#include (KokkosConfigVersion)
#include (KokkosTargets)

#set(LAMMPS_LIB ${LAMMPS_ROOT}/_lammps${PYTHON_MODULE_EXTENSION})
set(LAMMPS_LIB ${LAMMPS_ROOT}/../../../../lib64/liblammps.so)

set(LAMMPS_LIBRARIES ${LAMMPS_LIB} ${LAMMPS_COMMON_LIBS})

endif (LAMMPS_FOUND)
