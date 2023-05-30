cmake_minimum_required(VERSION 3.16..3.26)

## Setup project
project(dlext LANGUAGES C CXX)

set(PROJECT_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_MODULE_PATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Setup dependencies and extend LAMMPS interface
include(GNUInstallDirs)
include("${PROJECT_MODULE_PATH}/FetchCPM.cmake")
include("${PROJECT_MODULE_PATH}/FetchDLPack.cmake")
include("${PROJECT_MODULE_PATH}/FindLAMMPSTools.cmake")

if(NOT LAMMPS_INSTALL_PREFIX)
    get_filename_component(LAMMPS_INSTALL_PREFIX "${LAMMPS_ROOT}/../../.." ABSOLUTE)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${LAMMPS_INSTALL_PREFIX} CACHE PATH "" FORCE)
endif()

message(STATUS "Install plugin to: " ${CMAKE_INSTALL_PREFIX})

## Create the main library
add_library(${PROJECT_NAME} SHARED "")

if(Kokkos_FOUND AND (${Kokkos_VERSION} VERSION_GREATER_EQUAL "3.3.0"))
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
else()
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC dlext::lammps dlpack::dlpack)
if(Kokkos_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Kokkos::kokkos)
endif()

add_subdirectory(dlext)

## Install
install(TARGETS ${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
    FILES_MATCHING PATTERN "*.h"
)

## Create python module
option(BUILD_PYTHON_WRAPPER "Build Python wrapper" ON)

if(BUILD_PYTHON_WRAPPER)
    add_subdirectory(python)
endif()
