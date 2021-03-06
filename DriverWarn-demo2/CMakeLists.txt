cmake_minimum_required (VERSION 3.4.1)
set(CMAKE_VERBOSE_MAKEFILE on)
project (DriverWarn)


MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")  
      
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")  
    MESSAGE(STATUS "current platform: Linux ")  
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")  
    MESSAGE(STATUS "current platform: Windows")  
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")  
    MESSAGE(STATUS "current platform: FreeBSD")  
ELSE ()  
    MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")  
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")  
  
MESSAGE(STSTUS "###################################")  

# OpenCV Version: 3.4.3
find_package(PkgConfig REQUIRED)
set(OpenCV_DIR /usr/local/opencv/share/OpenCV)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV Include path: ${OpenCV_INCLUDE_DIRS}")

# for avoid missing glib header file
# glib-2.0, version 2.58.0
pkg_check_modules(GLIB REQUIRED glib-2.0)

# gstreamer
set(GSTREAMER_INCLUDE_DIRS "/usr/include/gstreamer-1.0;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include/glib-2.0;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include/glib-2.0/include;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include/gobject;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include/gst;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "/usr/lib/gstreamer-1.0/include;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "/usr/include/glib-2.0;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "/usr/lib/glib-2.0/include;")
set(GSTREAMER_LIBS  "gstapp-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstbase-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstrtsp-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstreamer-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstsdp-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gio-2.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gobject-2.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "glib-2.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstvideo-1.0;")
set(GSTREAMER_LIBRARY_DIRS  "/usr/lib/;")

message(STATUS "    GST DEBUG:###########################")
message(STATUS "    libraries: ${GSTREAMER_LIBS}")
message(STATUS "    include path: ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "    library path: ${GSTREAMER_LIBRARY_DIRS}")

#config
set(PARAM_CONF_LIBS "config;")
set(PARAM_CONF_LIBS_DIRS "${CMAKE_SOURCE_DIR}/libs;")
set(PARAM_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/config/include;")

#utils
# set(UTIL_LIBS "utils;")
# set(UTIL_LIBS_DIRS "${CMAKE_SOURCE_DIR}/libs;")
# set(UTIL_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/utils;")

#YUV
set(YUV_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/libyuv/include;")
set(YUV_LIBS  libyuv)
set(YUV_LIBRARY_DIRS  "${CMAKE_SOURCE_DIR}/libs/aarch/${YUV_LIBS};")

#Tensorflow
set(TENSORFLOW_INCLUDE "${CMAKE_SOURCE_DIR}/include/tensorflow;")
set(TFLITE_LIBS tflite)
set(TENSORFLOW_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/libs/aarch/${TFLITE_LIBS};")

aux_source_directory(${CMAKE_SOURCE_DIR} DIR_SRCS)

include_directories(
        ${CMAKE_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        # ${UTIL_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${YUV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/run_model/include
        ${TENSORFLOW_INCLUDE}
        ${TENSORFLOW_INCLUDE}/external/flat/include
        ${TENSORFLOW_INCLUDE}/external/absl


        )

ADD_SUBDIRECTORY(libyuv)
aux_source_directory(config/src CONFIG_SRC)

link_directories(
        ${OpenCV_LIBRARY_DIRS}
        ${GSTREAMER_LIBRARY_DIRS}
        ${YUV_LIBRARY_DIRS}
        ${TENSORFLOW_LIBRARY_DIRS}
        )

add_executable(calib_test
        ${CMAKE_SOURCE_DIR}/calibration/calibration.cpp
        )

add_executable(gst_test
        ${CONFIG_SRC}
        ${CMAKE_SOURCE_DIR}/src/gstcamera.cpp
        ${CMAKE_SOURCE_DIR}/run_model/src/ai_inference.cpp
        ${CMAKE_SOURCE_DIR}/test/gst_test.cpp
        )

add_executable(inference_test
        ${CONFIG_SRC}
        ${CMAKE_SOURCE_DIR}/run_model/src/tflite_blazeface.cpp
        ${CMAKE_SOURCE_DIR}/run_model/src/face_detection.cpp
        ${CMAKE_SOURCE_DIR}/run_model/src/ai_inference.cpp
        ${CMAKE_SOURCE_DIR}/test/inference_test.cpp
        )

add_executable(
        ${PROJECT_NAME}
        ${CONFIG_SRC}
        ${CMAKE_SOURCE_DIR}/src/gstcamera.cpp
        ${CMAKE_SOURCE_DIR}/run_model/src/ai_inference.cpp
        ${CMAKE_SOURCE_DIR}/src/central_agency.cpp

        ${CMAKE_SOURCE_DIR}/test/test.cpp
        )

add_executable(opencv_face_test
        ${CMAKE_SOURCE_DIR}/test/opencv_face_detection.cpp
        )

target_link_libraries(opencv_face_test

    ${OpenCV_LIBS}
)

target_link_libraries(
    gst_test
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBS}
    ${TFLITE_TARGET_NAME}
    ${CMAKE_SOURCE_DIR}/libs/aarch/libyuv/libyuv.so
    ${CMAKE_SOURCE_DIR}/libs/aarch/tflite/libtensorflow-lite.so
    evaluation_utils
    nnapi_delegate
    time
    memory_info
    pthread
    m
    z
)

target_link_libraries(
    calib_test
    ${OpenCV_LIBS}
)

target_link_libraries(
    inference_test
    ${OpenCV_LIBS}
    ${TFLITE_TARGET_NAME}
    ${CMAKE_SOURCE_DIR}/libs/aarch/libyuv/libyuv.so
    ${CMAKE_SOURCE_DIR}/libs/aarch/tflite/libtensorflow-lite.so
    evaluation_utils
    nnapi_delegate
    time
    memory_info
    pthread
    m
    z
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBS}
    ${YUV_DYNAMIC_LIBS}
    ${TFLITE_TARGET_NAME}
    ${CMAKE_SOURCE_DIR}/libs/aarch/libyuv/libyuv.so
    ${CMAKE_SOURCE_DIR}/libs/aarch/tflite/libtensorflow-lite.so
    evaluation_utils
    nnapi_delegate
    time
    memory_info
    pthread
    m
    z
)

# c++11 required
set(CMAKE_BUILD_TYPE "DEBUG")
# MESSAGE(${CMAKE_BINARY_DIR}/../)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lpthread  -lm -lrt -ljpeg -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -ljpeg -lstdc++")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -lstdc++")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib/)
