add_subdirectory(src)

target_include_directories(${PROJECT_NAME} PUBLIC include)

# this openmpi-x86_64 path is for my fedora box -- comment out for midway3
target_include_directories(${PROJECT_NAME} PUBLIC /usr/include/openmpi-x86_64)

install(DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lammps/${PROJECT_NAME}"
    FILES_MATCHING PATTERN "*.h"
)
