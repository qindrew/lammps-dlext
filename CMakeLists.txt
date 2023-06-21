cmake_minimum_required(VERSION 3.16..3.26)

## Setup project
project(dlext LANGUAGES C)  # CXX is activated within FindLAMMPSTools

set(PROJECT_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_MODULE_PATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Setup dependencies and extend LAMMPS interface
include(GNUInstallDirs)
# The second include requires the first one, and the rest depend on the second
include("${PROJECT_MODULE_PATH}/FetchCPM.cmake")
include("${PROJECT_MODULE_PATH}/FindLAMMPSTools.cmake")
include("${PROJECT_MODULE_PATH}/FetchDLPack.cmake")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${LAMMPS_INSTALL_PREFIX} CACHE PATH "" FORCE)
endif()

message(STATUS "Install plugin to: ${CMAKE_INSTALL_PREFIX}")

## Create the main library
add_library(${PROJECT_NAME} SHARED "")
add_library(${PROJECT_NAME}_headers INTERFACE)

if(Kokkos_FOUND AND (${Kokkos_VERSION} VERSION_GREATER_EQUAL "3.3.0"))
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
else()
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
endif()
target_link_libraries(${PROJECT_NAME}
    PUBLIC ${PROJECT_NAME}_headers LAMMPS_src LAMMPS::lammps dlpack::dlpack
)
if(Kokkos_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Kokkos::kokkos)
    target_compile_definitions(${PROJECT_NAME} PUBLIC LMP_KOKKOS)
endif()

target_include_directories(${PROJECT_NAME}_headers INTERFACE include)

# We will populate this list with the project source files
set("${PROJECT_NAME}_SOURCE_FILES" "")

add_subdirectory(dlext)

target_sources(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_SOURCE_FILES})

## Install
install(TARGETS ${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
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
