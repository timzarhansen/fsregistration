cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CATKIN_ENABLE_TESTING OFF)
project(fms2d)



find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3)
find_package(tf2 REQUIRED)
find_package(soft20 REQUIRED)
find_package(OpenCV 4.7 REQUIRED )
find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces( fms2d
        "msg/PotentialSolution.msg"

        "srv/RequestListPotentialSolution.srv"
        "srv/RequestOnePotentialSolution.srv"
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


################################################################################
# 2D Peak Algorithm
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


################################################################################
# executables
################################################################################

add_executable(registrationOfTwoImages src/registrationOfTwoImageScans.cpp src/softRegistrationClass.cpp src/softCorrelationClass.cpp src/PeakFinder.cpp include/PeakFinder.h)
target_link_libraries(registrationOfTwoImages fftw3  generalhelpfullib soft20::soft20 findpeaks ${OpenCV_LIBS} fftw3)
target_include_directories(registrationOfTwoImages PUBLIC  ${OpenCV_INCLUDE_DIRS})



#add_executable(ros2ServiceRegistration src/serviceImage.cpp src/softRegistrationClass.cpp src/softCorrelationClass.cpp src/PeakFinder.cpp include/PeakFinder.h)
#target_link_libraries(ros2ServiceRegistration fftw3  generalhelpfullib soft20::soft20 findpeaks ${OpenCV_LIBS} fftw3)
#target_include_directories(ros2ServiceRegistration PUBLIC  ${OpenCV_INCLUDE_DIRS})
## create a library for registration

## create a ROS2 service for registration

ament_export_dependencies(rosidl_default_runtime)
ament_package()


