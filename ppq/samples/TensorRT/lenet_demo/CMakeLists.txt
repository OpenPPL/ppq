cmake_minimum_required(VERSION 3.1)
project(lenet)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(TARGET_NAME "lenet_int8")
set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

message("TENSORRT_INCLUDE:" ${TENSORRT_INCLUDE})

# specify the header file path
include_directories(${CMAKE_SOURCE_DIR}/common
                    /usr/local/cuda/include
                    /opt/TensorRT-8.4.1.5/include)
# specify the library file path
link_directories(/usr/local/cuda/lib64
                 /opt/TensorRT-8.4.1.5/lib)

file(GLOB MTF_SRC ${PROJECT_SOURCE_DIR}/common/*.h
                  ${PROJECT_SOURCE_DIR}/common/*.cpp)
add_library(trt_deploy_common ${MTF_SRC})

add_executable(${TARGET_NAME} lenet_int8.cpp)
target_link_libraries(${TARGET_NAME} nvinfer cudart trt_deploy_common)
