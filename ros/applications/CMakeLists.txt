set( dir ${SOURCE_CODE_BASE_DIR}/ros/applications )

set( ros_path "/opt/ros/kinetic" CACHE STRING "path to ros, something like /opt/ros/kinetic" )
if( NOT EXISTS "${ros_path}")
    message( "${ros_path} not found; please specify ros path as something like /opt/ros/kinetic on your computer or set snark_build_ros to OFF" )
    set( ros_path "NOT FOUND" CACHE STRING "path to ros, something like /opt/ros/kinetic" )
endif( NOT EXISTS "${ros_path}")
link_directories( "${ros_path}/lib" )
include_directories( "${ros_path}/include" )

set( ros_libs roscpp roscpp_serialization rosbag rosbag_storage rostime rosconsole roslz4
              rosconsole_log4cxx rosconsole_backend_interface
              console_bridge cpp_common xmlrpcpp )

SET( GLOB stdmsgs_include ${dir}/std_msgs/*.h )

# source_group( points-to-ros FILES points-to-ros.cpp )
add_executable( points-to-ros points-to-ros.cpp )
target_link_libraries( points-to-ros ${ros_libs} ${snark_ALL_EXTERNAL_LIBRARIES} ${catkin_LIBRARIES} )

# source_group( points-from-ros FILES points-from-ros.cpp )
add_executable( points-from-ros points-from-ros.cpp )
target_link_libraries( points-from-ros snark_ros ${ros_libs} ${snark_ALL_EXTERNAL_LIBRARIES} ${catkin_LIBRARIES} )

add_executable( ros-multi-array ros-multi-array.cpp )
target_link_libraries( ros-multi-array snark_ros ${stdmsgs_include} ${ros_libs} ${snark_ALL_EXTERNAL_LIBRARIES} ${catkin_LIBRARIES} )

add_executable( ros-image ros-image.cpp )
target_link_libraries( ros-image snark_ros ${stdmsgs_include} ${ros_libs} snark_imaging ${snark_ALL_EXTERNAL_LIBRARIES} ${catkin_LIBRARIES} )

install( TARGETS points-to-ros points-from-ros ros-image ros-multi-array
         RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${snark_INSTALL_BIN_DIR}
         COMPONENT Runtime )

install( PROGRAMS ros-bag-to-bin ros-to-image DESTINATION ${CMAKE_INSTALL_PREFIX}/${snark_INSTALL_BIN_DIR} COMPONENT Runtime )

install( PROGRAMS ros-to-csv DESTINATION ${CMAKE_INSTALL_PREFIX}/${snark_INSTALL_BIN_DIR} COMPONENT Runtime )

install( PROGRAMS ros-from-csv DESTINATION ${CMAKE_INSTALL_PREFIX}/${snark_INSTALL_BIN_DIR} COMPONENT Runtime )
