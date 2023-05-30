include("${PROJECT_MODULE_PATH}/Fetchpybind11.cmake")

set_python_module_path()  # sets variable LAMMPS_Python_PATH

set(pybind11_MODULE_NAME "dlpack_extension")

pybind11_add_module(${pybind11_MODULE_NAME} MODULE "")

target_compile_features(${pybind11_MODULE_NAME} PRIVATE cxx_std_11)
target_link_libraries(${pybind11_MODULE_NAME} PRIVATE ${PROJECT_NAME})
target_sources(${pybind11_MODULE_NAME} PRIVATE lammps_dlext.cpp)

# Install the library
install(DIRECTORY ${PROJECT_NAME}
    DESTINATION ${LAMMPS_Python_PATH}
)

install(TARGETS ${pybind11_MODULE_NAME}
    DESTINATION "${LAMMPS_Python_PATH}/${PROJECT_NAME}"
)
