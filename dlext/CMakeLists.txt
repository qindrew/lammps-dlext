add_subdirectory(src)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC /usr/include/openmpi-x86_64)

#set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-expt-extended-lambda -Wext-lambda-captures-this -arch=sm_60")


install(DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lammps/${PROJECT_NAME}"
    FILES_MATCHING PATTERN "*.h"
)
