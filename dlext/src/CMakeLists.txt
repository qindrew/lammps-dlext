append_paths("${PROJECT_NAME}_SOURCE_FILES" FixDLExt.cpp LAMMPSView.cpp)

set("${PROJECT_NAME}_SOURCE_FILES" ${${PROJECT_NAME}_SOURCE_FILES} PARENT_SCOPE)

# target_sources(${PROJECT_NAME}::src
#     PRIVATE LAMMPSView.cpp FixDLExt.cpp
# )
