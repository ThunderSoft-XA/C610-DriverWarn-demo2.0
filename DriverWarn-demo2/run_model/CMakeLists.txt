project (runmodel)


# add project source to lib and set alias of lib
add_library(${PROJECT_NAME} src/ai_inference.cpp)
add_library(sub::runmodel ALIAS ${PROJECT_NAME})

# include corresponding .h/.hpp header file  

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)