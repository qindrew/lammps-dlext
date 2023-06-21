add_subdirectory(src)

set("${PROJECT_NAME}_SOURCE_FILES" ${${PROJECT_NAME}_SOURCE_FILES} PARENT_SCOPE)

target_include_directories(${PROJECT_NAME}_headers INTERFACE include)

install(DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lammps/${PROJECT_NAME}"
    FILES_MATCHING PATTERN "*.h"
)
