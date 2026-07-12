cmake_minimum_required(VERSION 3.16)
project(fetch_policy)

# The policy uses the CUDA runtime API from C++; no CUDA source compilation is needed.
if(DEFINED CMAKE_CUDA_COMPILER)
  message(STATUS "Ignoring CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}; fetch_policy uses C++ only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(fetch_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(unitree_api REQUIRED)
find_package(unitree_go REQUIRED)
find_package(visualization_msgs REQUIRED)

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
    "libnvinfer=${TENSORRT_LIBRARY}. Set TENSORRT_INCLUDE_DIR and "
    "TENSORRT_LIBRARY explicitly if they are installed outside known paths.")
endif()

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

add_executable(high_level_policy_node_cpp src/high_level_policy_node.cpp)
target_include_directories(high_level_policy_node_cpp SYSTEM PRIVATE
  ${TENSORRT_INCLUDE_DIR} ${CUDA_RUNTIME_INCLUDE_DIR})
target_link_libraries(high_level_policy_node_cpp
  ${TENSORRT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
ament_target_dependencies(high_level_policy_node_cpp
  fetch_interfaces geometry_msgs nav_msgs rcl_interfaces rclcpp std_msgs
  tf2 tf2_ros unitree_api unitree_go visualization_msgs)

add_executable(low_level_policy_node
  src/low_level_policy_node.cpp
  src/motor_crc.cpp)
target_include_directories(low_level_policy_node PRIVATE include)
target_include_directories(low_level_policy_node SYSTEM PRIVATE
  ${TENSORRT_INCLUDE_DIR} ${CUDA_RUNTIME_INCLUDE_DIR})
target_link_libraries(low_level_policy_node
  ${TENSORRT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
ament_target_dependencies(low_level_policy_node
  fetch_interfaces geometry_msgs rcl_interfaces rclcpp std_msgs unitree_go)

install(TARGETS high_level_policy_node_cpp low_level_policy_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
