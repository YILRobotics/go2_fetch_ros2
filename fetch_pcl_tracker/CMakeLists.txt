cmake_minimum_required(VERSION 3.16)
project(fetch_pcl_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation sample_consensus search kdtree)

add_executable(cube_tracker_pcl_node src/cube_tracker_pcl_node.cpp)
target_include_directories(cube_tracker_pcl_node SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(cube_tracker_pcl_node Eigen3::Eigen ${PCL_LIBRARIES})
ament_target_dependencies(cube_tracker_pcl_node
  rclcpp sensor_msgs geometry_msgs nav_msgs std_msgs visualization_msgs
  tf2 tf2_ros tf2_sensor_msgs pcl_conversions)

install(TARGETS cube_tracker_pcl_node DESTINATION lib/${PROJECT_NAME})
ament_package()
