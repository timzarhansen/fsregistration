cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(VTK_LEGACY_REMOVE)
set(CATKIN_ENABLE_TESTING OFF)

project(fsregistration)



find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3)


find_package(tf2 REQUIRED)
find_package(soft20 REQUIRED)
find_package(OpenCV 4.7 REQUIRED )
find_package(PCL 1.14 REQUIRED )

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

rosidl_generate_interfaces(fsregistration
        "msg/PotentialSolution2D.msg"

        "srv/RequestListPotentialSolution2D.srv"
        "srv/RequestOnePotentialSolution2D.srv"
        DEPENDENCIES builtin_interfaces
        DEPENDENCIES std_msgs
        DEPENDENCIES geometry_msgs
        DEPENDENCIES sensor_msgs
)


################################################################################
# generalhelpfullib
################################################################################

add_library(generalhelpfullib SHARED src/generalHelpfulTools.cpp)
target_include_directories(generalhelpfullib PUBLIC include)
ament_target_dependencies(generalhelpfullib tf2 Eigen3)

install(
        DIRECTORY include
        DESTINATION include
)

################################################################################
# Peak Algorithm
################################################################################

add_library(findpeaks INTERFACE)
target_include_directories(findpeaks INTERFACE
        find-peaks/include
)
add_library(UNION_FIND
        SHARED
        find-peaks/src/union_find.cpp
)
target_include_directories(UNION_FIND PRIVATE
        find-peaks/include
)
target_link_libraries(findpeaks INTERFACE UNION_FIND)

#include_directories(${PCL_INCLUDE_DIRS})
#
#link_directories(${PCL_LIBRARY_DIRS})
#
#add_definitions(${PCL_DEFINITIONS})

################################################################################
# executables
################################################################################

add_executable(registration3DVoxels src/registrationOf3DVoxels.cpp src/softRegistrationClass.cpp src/softCorrelationClass.cpp src/PeakFinder.cpp include/PeakFinder.h
        src/softCorrelationClass3D.cpp
        src/softRegistrationClass3D.cpp)
ament_target_dependencies(registration3DVoxels PCL)
target_link_libraries(registration3DVoxels fftw3  generalhelpfullib soft20::soft20 findpeaks ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_include_directories(registration3DVoxels PUBLIC  ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})





add_executable(registrationOfTwoImages src/registrationOfTwoImageScans.cpp src/softRegistrationClass.cpp src/softCorrelationClass.cpp src/PeakFinder.cpp include/PeakFinder.h)
target_link_libraries(registrationOfTwoImages fftw3 generalhelpfullib soft20::soft20 findpeaks ${OpenCV_LIBS})
target_include_directories(registrationOfTwoImages PUBLIC ${OpenCV_INCLUDE_DIRS})


add_executable(ros2ServiceRegistrationFS2D src/serviceRegistrationImage.cpp src/softRegistrationClass.cpp src/softCorrelationClass.cpp src/PeakFinder.cpp include/PeakFinder.h)
ament_target_dependencies(ros2ServiceRegistrationFS2D rclcpp geometry_msgs tf2_geometry_msgs cv_bridge)
target_link_libraries(ros2ServiceRegistrationFS2D soft20::soft20 generalhelpfullib findpeaks ${OpenCV_LIBS})
target_include_directories(ros2ServiceRegistrationFS2D PUBLIC ${OpenCV_INCLUDE_DIRS})
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(ros2ServiceRegistrationFS2D "${cpp_typesupport_target}")

add_executable(ros2ServiceRegistrationFS3D src/serviceRegistrationVoxel.cpp src/softRegistrationClass3D.cpp src/softCorrelationClass3D.cpp)
ament_target_dependencies(ros2ServiceRegistrationFS3D rclcpp geometry_msgs tf2_geometry_msgs cv_bridge)
target_link_libraries(ros2ServiceRegistrationFS3D soft20::soft20 generalhelpfullib findpeaks ${OpenCV_LIBS})
target_include_directories(ros2ServiceRegistrationFS3D PUBLIC ${OpenCV_INCLUDE_DIRS})
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(ros2ServiceRegistrationFS3D "${cpp_typesupport_target}")



#add_executable(servicecall src/callingServiceExample.cpp )
#ament_target_dependencies(servicecall rclcpp  cv_bridge tf2_geometry_msgs)
#target_link_libraries(servicecall soft20::soft20 generalhelpfullib findpeaks ${OpenCV_LIBS})
#target_include_directories(servicecall PUBLIC  ${OpenCV_INCLUDE_DIRS})
#rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
#target_link_libraries(servicecall "${cpp_typesupport_target}")


install(TARGETS
        ros2ServiceRegistrationFS2D
        ros2ServiceRegistrationFS3D
        #        guibluerov2
        DESTINATION lib/${PROJECT_NAME}
)


ament_export_dependencies(rosidl_default_runtime)
ament_package()


