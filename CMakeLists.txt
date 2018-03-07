cmake_minimum_required(VERSION 2.8.3)
project(quadrobot_vision)

set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL 1.8 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  pcl_ros  
  quadrobot_msgs
)

file(GLOB msgs_to_mat_SRC
    "src/msgs_to_mat/*.h"
    "src/msgs_to_mat/*.cpp"
)
file(GLOB plane_detection_SRC
    "src/plane_detection/*.h"
    "src/plane_detection/*.cpp"
)
file(GLOB object_detection_SRC
    "src/object_detection/*.h"
    "src/object_detection/*.cpp"
)
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs quadrobot_msgs
  DEPENDS system_lib
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(msgs_to_mat ${msgs_to_mat_SRC})
add_executable(plane_detection ${plane_detection_SRC})
add_executable(object_detection ${object_detection_SRC})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
target_include_directories(msgs_to_mat PUBLIC
  src/msgs_to_mat
 )
target_include_directories(plane_detection PUBLIC
  src/plane_detection
 )
 target_include_directories(object_detection PUBLIC
  src/object_detection
 )
 target_link_libraries(msgs_to_mat
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
)
 target_link_libraries(plane_detection
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
)
target_link_libraries(object_detection
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

