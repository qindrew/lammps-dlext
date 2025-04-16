include("${PROJECT_MODULE_PATH}/Fetchpybind11.cmake")

set_python_module_path()  # sets variable PYLAMMPS_PATH

set(pybind11_MODULE_NAME "_api")

pybind11_add_module(${pybind11_MODULE_NAME} MODULE "")

add_library(${pybind11_MODULE_NAME}::lammps SHARED IMPORTED)

set_target_properties(${pybind11_MODULE_NAME}::lammps PROPERTIES
    IMPORTED_LOCATION ${PYLAMMPS_LIBRARY}
)
copy_target_property(LAMMPS::lammps ${pybind11_MODULE_NAME}::lammps INTERFACE_COMPILE_DEFINITIONS)
copy_target_property(LAMMPS::lammps ${pybind11_MODULE_NAME}::lammps INTERFACE_LINK_LIBRARIES)

#if(Kokkos_FOUND AND (${Kokkos_VERSION} VERSION_GREATER_EQUAL "3.3.0"))
target_compile_features(${pybind11_MODULE_NAME} PRIVATE cxx_std_17)
#else()
#    target_compile_features(${pybind11_MODULE_NAME} PRIVATE cxx_std_11)
#endif()
target_link_libraries(${pybind11_MODULE_NAME}
    PRIVATE ${pybind11_MODULE_NAME}::lammps dlpack::dlpack
    PRIVATE ${PROJECT_NAME}::headers LAMMPS::src
)
target_sources(${pybind11_MODULE_NAME}
    PRIVATE ${${PROJECT_NAME}_SOURCE_FILES} lammps_dlext.cpp
)
#if(Kokkos_FOUND)
target_link_libraries(${pybind11_MODULE_NAME} PRIVATE Kokkos::kokkos)
target_compile_definitions(${pybind11_MODULE_NAME} PRIVATE LMP_KOKKOS)
#endif()

add_custom_command(TARGET ${pybind11_MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PYLAMMPS_LIBRARY} "${PYLAMMPS_LIBRARY}.0"
)

# Install the library
install(DIRECTORY ${PROJECT_NAME}
    DESTINATION ${PYLAMMPS_PATH}
)

install(TARGETS ${pybind11_MODULE_NAME}
    DESTINATION "${PYLAMMPS_PATH}/${PROJECT_NAME}"
)
