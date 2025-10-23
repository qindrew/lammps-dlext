# Global requirements
# -------------------

find_package(Git REQUIRED)


# Global variables
# ----------------

set(LAMMPS_URL "https://github.com/lammps/lammps.git")

if(NOT LAMMPS_ROOT)
    if(DEFINED ENV{LAMMPS_ROOT})
        set(LAMMPS_ROOT $ENV{LAMMPS_ROOT})
    elseif(CMAKE_PREFIX_PATH)
        find_path(LAMMPS_ROOT
            NAMES LAMMPS_Targets.cmake
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES LAMMPS
        )
        if("${LAMMPS_ROOT}" STREQUAL "LAMMPS_ROOT-NOTFOUND")
            message(FATAL_ERROR
                "Unable to find LAMMPS. Try setting the CMake "
                "variables LAMMPS_ROOT or CMAKE_PREFIX_PATH."
            )
        endif()
    endif()
endif()


# Utility functions
# -----------------

#     fetch_lammps
#
# Given a LAMMPS `tag` use CPM to retrieve the code for that release.
function(fetch_lammps tag)
    # We use lowercase lammps to avoid clashes with the main library
    CPMAddPackage(NAME lammps
        GIT_REPOSITORY  ${LAMMPS_URL}
        GIT_TAG         ${tag}
        GIT_SHALLOW     TRUE
        DOWNLOAD_ONLY   TRUE
    )
    if(lammps_ADDED)
        set(lammps_SOURCE_DIR ${lammps_SOURCE_DIR} PARENT_SCOPE)
    else()
        message(WARNING "Failed to download LAMMPS source")
    endif()
endfunction()

#     find_executable
#
# Looks for the IMPORTED_LOCATION of the first available IMPORTED_LOCATION_<config>
# for the given target and sets a variable `varname` on the parent scope.
function(find_executable target varname)
    get_target_property(var ${target} IMPORTED_LOCATION)
    if("${var}" STREQUAL "var-NOTFOUND")
        get_target_property(configs ${target} IMPORTED_CONFIGURATIONS)
        list(GET configs 0 config)
        get_target_property(var ${target} "IMPORTED_LOCATION_${config}")
    endif()
    set(${varname} ${var} PARENT_SCOPE)
endfunction()

#     find_lammps_cxx_compiler(path)
#
# Look for the  `nvcc_wrapper` shipped by lammps within the specified `path` and,
# if found, set its location to `LAMMPS_CXX_COMPILER` in the parent scope.
function(find_lammps_cxx_compiler path)
    get_filename_component(NVCC_WRAPPER "${path}/nvcc_wrapper" ABSOLUTE)
    if(EXISTS ${NVCC_WRAPPER})
        set(LAMMPS_CXX_COMPILER ${NVCC_WRAPPER} PARENT_SCOPE)
    endif()
endfunction()

#     get_lammps_tag(version)
#
# Given a LAMMPS `version` as reported to CMake or Python, sets `LAMMPS_tag` in the
# parent scope to the latest git tag matching this version within the LAMMPS repo.
function(get_lammps_tag version)
  message(STATUS "=== get_lammps_tag() called ===")
  message(STATUS "Input version string: '${version}'")
  message(STATUS "Initial LAMMPS_tag: '${LAMMPS_tag}' (if already defined)")

  # Honor user-provided tag
  if(DEFINED LAMMPS_tag AND NOT "${LAMMPS_tag}" STREQUAL "")
    message(STATUS "User provided LAMMPS_tag='${LAMMPS_tag}', skipping detection.")
    set(LAMMPS_tag "${LAMMPS_tag}" PARENT_SCOPE)
    return()
  endif()

  # Try to read tag from the binary
  execute_process(
    COMMAND ${LAMMPS_EXECUTABLE} -h
    RESULT_VARIABLE _ec
    OUTPUT_VARIABLE _help
    ERROR_QUIET
  )
  message(STATUS "LAMMPS_EXECUTABLE='${LAMMPS_EXECUTABLE}' returned code ${_ec}")
  if(_ec EQUAL 0)
    string(REGEX MATCH "Git info [^ ]+ / ([^)]+)" _ "${_help}")
    message(STATUS "LAMMPS -h output match: '${CMAKE_MATCH_1}'")
    if(NOT "${CMAKE_MATCH_1}" STREQUAL "" AND NOT "${CMAKE_MATCH_1}" STREQUAL "(unknown)")
      message(STATUS "Detected tag directly from LAMMPS binary: '${CMAKE_MATCH_1}'")
      set(LAMMPS_tag "${CMAKE_MATCH_1}" PARENT_SCOPE)
      return()
    endif()
  endif()

  # --- Robust version parsing: pull the first three integers we see ---
  message(STATUS "Attempting to parse version string for YYYY/MM/DD parts...")
  string(REGEX MATCHALL "[0-9]+" _nums "${version}")
  list(LENGTH _nums _nlen)
  message(STATUS "Extracted numbers (${_nlen}): '${_nums}'")

  if(_nlen LESS 3)
    message(WARNING "FindLAMMPSTools: cannot parse version '${version}'. Pass -DLAMMPS_tag=<tag>.")
    return()
  endif()

  list(GET _nums 0 _year)
  list(GET _nums 1 _mm)
  list(GET _nums 2 _dd)
  message(STATUS "Parsed year='${_year}', month='${_mm}', day='${_dd}'")

  # Normalize month/day to integers
  math(EXPR _mm_int "${_mm}")
  math(EXPR _dd_int "${_dd}")
  message(STATUS "Normalized to integers: month=${_mm_int}, day=${_dd_int}")

  # Compose pattern like *10Sep2025* expected by LAMMPS tags
  set(_MONTHS _ Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec)
  if(_mm_int LESS 1 OR _mm_int GREATER 12)
    message(WARNING "FindLAMMPSTools: month '${_mm}' out of range from version '${version}'.")
    return()
  endif()
  list(GET _MONTHS ${_mm_int} _mon)
  set(_pattern "*${_dd_int}${_mon}${_year}*")
  message(STATUS "Composed tag search pattern: '${_pattern}'")

  # Query git for tags
  message(STATUS "Running git ls-remote on '${LAMMPS_URL}' ...")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} ls-remote --tags --refs ${LAMMPS_URL} ${_pattern}
    RESULT_VARIABLE _gec
    OUTPUT_VARIABLE _tags
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "Git command exit code: ${_gec}")
  message(STATUS "Git tags output: '${_tags}'")

  string(REGEX MATCH ".*/(.+)$" _ "${_tags}")
  message(STATUS "Extracted tag candidate: '${CMAKE_MATCH_1}'")

  if(NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(LAMMPS_tag "${CMAKE_MATCH_1}" PARENT_SCOPE)
    message(STATUS "Final LAMMPS_tag set to: '${CMAKE_MATCH_1}'")
  else()
    message(WARNING "FindLAMMPSTools: no matching git tag for pattern '${_pattern}'. Pass -DLAMMPS_tag=<tag>.")
  endif()

  message(STATUS "=== get_lammps_tag() complete ===")
endfunction()

#     get_lammps_version(path)
#
# Tries loading the LAMMPSConfigVersion.cmake from within `path`,
# and sets LAMMPS_VERSION in the parent scope.
function(get_lammps_version path)
    if(EXISTS "${path}/LAMMPSConfigVersion.cmake")
        include("${path}/LAMMPSConfigVersion.cmake")
    endif()
    set(LAMMPS_VERSION "${PACKAGE_VERSION}" PARENT_SCOPE)
endfunction()

#     find_lammps()
#
# Macro equivalent to find_package(LAMMPS QUIET), but avoids looking for MPI,
# which is performed separately. It also looks for the NVCC wrapper that
# comes with LAMMPS and tries to find an appropriate git tag matching the
# LAMMPS version.
macro(find_lammps)
    include("${LAMMPS_ROOT}/LAMMPS_Targets.cmake")
    find_executable(LAMMPS::lmp "LAMMPS_EXECUTABLE")
    find_lammps_cxx_compiler("${LAMMPS_EXECUTABLE}/..")
    get_lammps_version(${LAMMPS_ROOT})
    get_lammps_tag(${LAMMPS_VERSION})
endmacro()

#     append_paths(list)
#
# Given a list and any number of file names, appends to the list
# the absolute paths to the provided files.
function(append_paths list)
    list(POP_FRONT ARGV)
    foreach(file ${ARGV})
        get_filename_component(path ${file} ABSOLUTE)
        set(${list} ${${list}} ${path})
    endforeach()
    set(${list} ${${list}} PARENT_SCOPE)
endfunction()

#    copy_target_property
#
# Given a source target `src` and a destination target `dst`, reads the value of the
# `property` in `src` and if found, sets it in `dst`.
function(copy_target_property src dst property)
    get_target_property(var ${src} ${property})
    if(NOT ("${var}" STREQUAL "var-NOTFOUND"))
        set_target_properties(${dst} PROPERTIES ${property} "${var}")
    endif()
endfunction()

#    copy_target_property_fallback
#
# Given a source target `src` and a destination target `dst`, reads the value of the
# `property` in `src` and only if not found, sets it in `dst` using the corresponding
# value from the configuration `config` provided.
function(copy_target_property_fallback src dst property config)
    get_target_property(var ${src} ${property})
    if("${var}" STREQUAL "var-NOTFOUND")
        get_target_property(var ${src} "${property}_${config}")
        set_target_properties(${dst} PROPERTIES ${property} "${var}")
    endif()
endfunction()

#    copy_target_property
#
# Given a source target `src` and a destination target `dst`, finds all the
# IMPORTED_CONFIGURATIONS in `src` and for each of them tries to extract the value of the
# corresponding `property` in `src`, and if found, sets it in `dst`.
function(copy_target_property_configs src dst property)
    get_target_property(configs ${src} IMPORTED_CONFIGURATIONS)
    list(LENGTH configs nconfigs)

    if("${configs}" STREQUAL "configs-NOTFOUND")
        return()
    endif()

    if(${nconfigs})
        foreach(config ${configs})
            copy_target_property(${src} ${dst} "${property}_${config}")
        endforeach()
        # TODO: fix when multiple configurations are present
        # (we're assuming the first one is appropriate)
        list(GET configs 0 config)
        copy_target_property_fallback(${src} ${dst} ${property} ${config})
    endif()
endfunction()

#     set_python_module_path
#
# Tries finding the path where the python lammps module is installed and sets the
# variable `PYLAMMPS_PATH` to that path, otherwise `PYLAMMPS_PATH` will be empty.
function(set_python_module_path)
    find_package(Python QUIET COMPONENTS Interpreter)
    if(NOT (Python_FOUND AND Python_Interpreter_FOUND))
        message(FATAL_ERROR
            "Could not find Python interpreter, make sure it is installed and enabled"
        )
    endif()
    set(find_lammps_script "
from __future__ import print_function;
import os
try:
    import lammps
    print(os.path.dirname(lammps.__file__), end='')
except:
    print('', end='')"
    )
    set(find_liblammps_script "
from __future__ import print_function;
import os
try:
    import lammps
    lmp = lammps.lammps(cmdargs='-log none -screen none'.split())
    print(lmp.lib._name, end='')
except:
    print('', end='')"
    )
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "${find_lammps_script}"
        OUTPUT_VARIABLE PYLAMMPS_PATH
    )
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "${find_liblammps_script}"
        OUTPUT_VARIABLE PYLAMMPS_LIBRARY
    )
    if("${PYLAMMPS_PATH}" STREQUAL "" OR "${PYLAMMPS_LIBRARY}" STREQUAL "")
        unset(PYLAMMPS_LIBRARY)
        find_library(PYLAMMPS_LIBRARY
            NAMES lammps
            HINTS ${Python_SITELIB}
            PATH_SUFFIXES lammps
        )
        if("${PYLAMMPS_LIBRARY}" STREQUAL "PYLAMMPS_LIBRARY-NOTFOUND")
            message(FATAL_ERROR "Unable to locate LAMMPS python module")
        else()
            get_filename_component(PYLAMMPS_PATH "${PYLAMMPS_LIBRARY}" DIRECTORY)
        endif()
    endif()
    set(PYLAMMPS_PATH "${PYLAMMPS_PATH}" PARENT_SCOPE)
    set(PYLAMMPS_LIBRARY "${PYLAMMPS_LIBRARY}" PARENT_SCOPE)
endfunction()


# Setup LAMMPS
# ------------

# We use find_lammps() first instead of find_package(LAMMPS) to avoid finding
# MPI which requires CXX enabled, but we want to enable CXX after looking for
# the NVCC compiler wrapper that comes with LAMMPS.
find_lammps()

message(STATUS "Found LAMMPS at ${LAMMPS_ROOT} (version ${LAMMPS_VERSION})")

fetch_lammps(${LAMMPS_tag})

if(NOT CMAKE_BUILD_TYPE)
    if(${LAMMPS_VERSION} GREATER 20190618)
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Type of build" FORCE)
    else()
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
    endif()
endif()

set(CMAKE_CXX_EXTENSIONS OFF CACHE FILEPATH "Use compiler extensions")
if(LAMMPS_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${LAMMPS_CXX_COMPILER} CACHE FILEPATH "C++ compiler")
endif()

enable_language(CXX)

find_package(LAMMPS REQUIRED)

if(TARGET LAMMPS::mpi_stubs)  # LAMMPS was built without MPI support
    target_include_directories(LAMMPS::mpi_stubs SYSTEM INTERFACE
        "${lammps_SOURCE_DIR}/src/STUBS"
    )
endif()

if(NOT LAMMPS_INSTALL_PREFIX)
    get_filename_component(LAMMPS_INSTALL_PREFIX "${LAMMPS_ROOT}/../../.." ABSOLUTE)
endif()

add_library(LAMMPS_src INTERFACE)
add_library(LAMMPS::src ALIAS LAMMPS_src)

target_include_directories(LAMMPS_src INTERFACE "${lammps_SOURCE_DIR}/src")

find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    message(STATUS "Kokkos support has been enabled (version ${Kokkos_VERSION})")
    target_include_directories(LAMMPS_src INTERFACE "${lammps_SOURCE_DIR}/src/KOKKOS")
else()
    message(STATUS
        "Kokkos support is not enabled. If you built LAMMPS with Kokkos,"
        "make sure that CMake if able to find it by providing the Kokkos_ROOT."
    )
endif()
