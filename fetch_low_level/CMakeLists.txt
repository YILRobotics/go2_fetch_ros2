cmake_minimum_required(VERSION 3.16)
project(fetch_low_level)

# Some workspace build scripts still pass this legacy option. The target uses
# only the CUDA runtime API and contains no .cu sources, so nvcc is unnecessary.
if(DEFINED CMAKE_CUDA_COMPILER)
  message(STATUS "Ignoring CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}; fetch_low_level uses C++ only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(fetch_interfaces REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
        $ENV{CONDA_PREFIX}/include
        $ENV{HOME}/miniconda3/envs/env_deploy/include
        $ENV{HOME}/miniconda3/envs/env_deploy/lib/python3.10/site-packages/tensorrt/include)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
        $ENV{CONDA_PREFIX}/lib
        $ENV{HOME}/miniconda3/envs/env_deploy/lib
        $ENV{HOME}/miniconda3/envs/env_deploy/lib/python3.10/site-packages/tensorrt_libs)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR
    "TensorRT development files not found. NvInfer.h=${TENSORRT_INCLUDE_DIR}, "
    "libnvinfer=${TENSORRT_LIBRARY}. Install the TensorRT development package "
    "or set TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARY explicitly.")
endif()

# This target uses the CUDA runtime API but does not compile CUDA source, so it
# must not require nvcc/FindCUDAToolkit. This also supports pip/Conda CUDA
# runtime installations used by the existing Python TensorRT node.
find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
  HINTS /usr/include /usr/local/cuda/include
        $ENV{CONDA_PREFIX}/include
        $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/include
        $ENV{HOME}/miniconda3/envs/env_deploy/lib/python3.10/site-packages/nvidia/cuda_runtime/include)
find_library(CUDA_RUNTIME_LIBRARY
  NAMES cudart libcudart.so.12
  HINTS /usr/lib /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
        $ENV{CONDA_PREFIX}/lib
        $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
        $ENV{HOME}/miniconda3/envs/env_deploy/lib/python3.10/site-packages/nvidia/cuda_runtime/lib)
if(NOT CUDA_RUNTIME_INCLUDE_DIR OR NOT CUDA_RUNTIME_LIBRARY)
  message(FATAL_ERROR
    "CUDA runtime development files not found. cuda_runtime_api.h=${CUDA_RUNTIME_INCLUDE_DIR}, "
    "libcudart=${CUDA_RUNTIME_LIBRARY}. Set CUDA_RUNTIME_INCLUDE_DIR and "
    "CUDA_RUNTIME_LIBRARY explicitly.")
endif()

add_executable(low_level_policy_node
  src/low_level_policy_node.cpp
  src/motor_crc.cpp)
target_include_directories(low_level_policy_node PRIVATE include)
target_include_directories(low_level_policy_node SYSTEM PRIVATE
  ${TENSORRT_INCLUDE_DIR} ${CUDA_RUNTIME_INCLUDE_DIR})
target_link_libraries(low_level_policy_node
  ${TENSORRT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
ament_target_dependencies(low_level_policy_node
  rclcpp rcl_interfaces geometry_msgs std_msgs unitree_go fetch_interfaces)

add_executable(high_level_policy_node_cpp
  src/high_level_policy_node.cpp)
target_include_directories(high_level_policy_node_cpp SYSTEM PRIVATE
  ${TENSORRT_INCLUDE_DIR} ${CUDA_RUNTIME_INCLUDE_DIR})
target_link_libraries(high_level_policy_node_cpp
  ${TENSORRT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
ament_target_dependencies(high_level_policy_node_cpp
  rclcpp rcl_interfaces geometry_msgs nav_msgs std_msgs tf2 tf2_ros
  unitree_go unitree_api visualization_msgs fetch_interfaces)

install(TARGETS low_level_policy_node high_level_policy_node_cpp
  DESTINATION lib/${PROJECT_NAME})
ament_package()
