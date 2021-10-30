cmake_minimum_required (VERSION 2.8.8)
project (tf_example)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")
 
set(TF_ROOT "/home/thundersoft/software/tensorflow")

link_directories(./lib)
include_directories(
   ${TF_ROOT}
   ${TF_ROOT}/bazel-bin/tensorflow
   ${TF_ROOT}/tensorflow
   ) 
add_executable(tf_test  tflite_run.cpp) 
target_link_libraries(tf_test tensorflow_cc tensorflow_framework)