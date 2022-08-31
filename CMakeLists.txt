cmake_minimum_required(VERSION 3.16..3.22)

# Set-up project
project(dlext LANGUAGES C CXX)

# Try finding LAMMPS first from the current environment
find_package(LAMMPS 20220107 QUIET)

set(PROJECT_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_MODULE_PATH})

if(NOT LAMMPS_FOUND)
    find_package(LAMMPS 20220217 REQUIRED)
endif()

if(NOT LAMMPS_INSTALL_PREFIX)
    set(LAMMPS_INSTALL_PREFIX ${LAMMPS_ROOT})
endif()

if(NOT LAMMPS_LIBRARIES)
    set(LAMMPS_LIBRARIES LAMMPS::_lammps)
endif()

include(GNUInstallDirs)
include("${PROJECT_MODULE_PATH}/FetchCPM.cmake")
include("${PROJECT_MODULE_PATH}/FetchDLPack.cmake")

# Plugins must be built as shared libraries
if(NOT BUILD_SHARED_LIBS)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled LAMMPS")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${LAMMPS_INSTALL_PREFIX} CACHE PATH "" FORCE)
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(STATUS "Install plugin to: " ${CMAKE_INSTALL_PREFIX})

# Create the main library
add_library(${PROJECT_NAME} SHARED "")

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC ${LAMMPS_LIBRARIES} dlpack::dlpack)

add_subdirectory(dlext)

# Install
install(TARGETS ${PROJECT_NAME}
    DESTINATION ${LAMMPS_ROOT}
)

install(DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lammps/${PROJECT_NAME}"
    FILES_MATCHING PATTERN "*.h"
)

# Create python module
add_subdirectory(python)
