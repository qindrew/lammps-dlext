set(pybind11_MODULE_NAME "dlpack_extension")

pybind11_add_module(${pybind11_MODULE_NAME} MODULE "")

target_compile_features(${pybind11_MODULE_NAME} PRIVATE cxx_std_11)
target_link_libraries(${pybind11_MODULE_NAME} PRIVATE ${PROJECT_NAME})
target_sources(${pybind11_MODULE_NAME} PRIVATE PyDLExt.cc Sampler.cc)

# add include path to pybind11 available from the virtual environment: $HOME/miniconda3/envs/pysages/include
target_include_directories(${pybind11_MODULE_NAME} PUBLIC ${CMAKE_INSTALL_PREFIX}/../../../../include)
target_include_directories(${pybind11_MODULE_NAME} PUBLIC ${CMAKE_INSTALL_PREFIX}/../../../../include/kokkos)
 
# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if(BUILD_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(
           ${pybind11_MODULE_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
        )
   endif()
   if(MPI_LINK_FLAGS)
       set_target_properties(
           ${pybind11_MODULE_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
        )
   endif()
endif()

set_target_properties(${pybind11_MODULE_NAME} PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT -DFFT_FFTW3 -DFFT_FFTW_THREADS -DKOKKOS_DEPENDENCE -DLAMMPS_FFMPEG -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64 -DLAMMPS_SMALLBIG -DLMP_KOKKOS -DLMP_PYTHON -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX -D_MPICC_H  -expt-extended-lambda -Wext-lambda-captures-this -arch=sm_60")


#fix_cudart_rpath(${pybind11_MODULE_NAME})

# Install the library
install(TARGETS ${pybind11_MODULE_NAME}
    DESTINATION "${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}"
)

install(DIRECTORY ${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)
