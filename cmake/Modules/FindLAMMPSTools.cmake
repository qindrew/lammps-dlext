# Global requirements
# -------------------

find_package(Git REQUIRED)


# Global variables
# ----------------

set(LAMMPS_URL "https://github.com/lammps/lammps.git")


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

#     map_version_to_tag
#
# Given a LAMMPS `version` as reported to CMake or Python, sets `LAMMPS_tag` in the
# parent scope to the latest git tag matching this version within the LAMMPS repo.
function(map_version_to_tag version)
    set(MONTHS _ Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec)

    string(REGEX MATCH "([0-9][0-9][0-9][0-9])([0-9][0-9])([0-9][0-9]).*" _ "${version}")
    set(year ${CMAKE_MATCH_1})
    list(GET MONTHS ${CMAKE_MATCH_2} month)
    math(EXPR day ${CMAKE_MATCH_3})
    set(pattern "*${day}${month}${year}*")

    execute_process(
        COMMAND ${GIT_EXECUTABLE} ls-remote --tags --refs ${LAMMPS_URL} ${pattern}
        RESULT_VARIABLE exit_code  # TODO: use this to check if it fails and inform the user
        OUTPUT_VARIABLE git_tags
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string(REGEX MATCH ".*/(.+)$" _ "${git_tags}")

    set(LAMMPS_tag "${CMAKE_MATCH_1}" PARENT_SCOPE)
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
# variable `LAMMPS_Python_PATH` to that path, otherwise `LAMMPS_Python_PATH` will
# be empty.
function(set_python_module_path)
    find_package(Python QUIET COMPONENTS Interpreter)
    if(NOT (Python_FOUND AND Python_Interpreter_FOUND))
        message(FATAL_ERROR
            "Could not find Python interpreter, make sure it is installed and enabled"
        )
    endif()
    set(FIND_LAMMPS_SCRIPT "
from __future__ import print_function;
import os
try:
    import lammps
    print(os.path.dirname(lammps.__file__), end='')
except:
    print('', end='')"
    )
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "${FIND_LAMMPS_SCRIPT}"
        OUTPUT_VARIABLE LAMMPS_Python_PATH
    )
    set(LAMMPS_Python_PATH "${LAMMPS_Python_PATH}" PARENT_SCOPE)
endfunction()


# Setup LAMMPS
# ------------

find_package(LAMMPS REQUIRED)
message(STATUS "Found LAMMPS at ${LAMMPS_ROOT} (version ${LAMMPS_VERSION})")

map_version_to_tag(${LAMMPS_VERSION})
fetch_lammps(${LAMMPS_tag})

if(TARGET LAMMPS::mpi_stubs)  # LAMMPS was built without MPI support
    target_include_directories(LAMMPS::mpi_stubs SYSTEM INTERFACE
        "${lammps_SOURCE_DIR}/src/STUBS"
    )
endif()

add_library(dlext::lammps SHARED IMPORTED)

foreach(property "IMPORTED_LOCATION" "IMPORTED_SONAME")
    copy_target_property(LAMMPS::lammps dlext::lammps ${property})
    copy_target_property_configs(LAMMPS::lammps dlext::lammps ${property})
endforeach()
copy_target_property(LAMMPS::lammps dlext::lammps INTERFACE_COMPILE_DEFINITIONS)
copy_target_property(LAMMPS::lammps dlext::lammps INTERFACE_LINK_LIBRARIES)

target_include_directories(dlext::lammps INTERFACE "${lammps_SOURCE_DIR}/src")

find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    message(STATUS "Kokkos support has been enabled (version ${Kokkos_VERSION})")
    target_include_directories(dlext::lammps INTERFACE "${lammps_SOURCE_DIR}/src/KOKKOS")
endif()
