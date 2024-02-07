cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CATKIN_ENABLE_TESTING OFF)
project(fs2d)



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
find_package(rclcpp REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

rosidl_generate_interfaces( fs2d
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

install(
        DIRECTORY include
        DESTINATION include
)

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


add_executable(registrationOfTwoImages src/registrationOfTwoImageScans.cpp src/softRegistrationClass.cpp src/softCorrelationClass.cpp src/PeakFinder.cpp include/PeakFinder.h
        src/softCorrelationClass3D.cpp
        src/softRegistrationClass3D.cpp)
target_link_libraries(registrationOfTwoImages fftw3  generalhelpfullib soft20::soft20 findpeaks ${OpenCV_LIBS})
target_include_directories(registrationOfTwoImages PUBLIC  ${OpenCV_INCLUDE_DIRS})



add_executable(ros2ServiceRegistration src/serviceRegistrationImage.cpp src/softRegistrationClass.cpp src/softCorrelationClass.cpp src/PeakFinder.cpp include/PeakFinder.h)
ament_target_dependencies(ros2ServiceRegistration rclcpp geometry_msgs tf2_geometry_msgs cv_bridge)
target_link_libraries(ros2ServiceRegistration soft20::soft20 generalhelpfullib findpeaks ${OpenCV_LIBS})
target_include_directories(ros2ServiceRegistration PUBLIC  ${OpenCV_INCLUDE_DIRS})
#rosidl_target_interfaces(ros2ServiceRegistration
#        ${PROJECT_NAME} "rosidl_typesupport_cpp")
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(ros2ServiceRegistration "${cpp_typesupport_target}")

#add_executable(servicecall src/callingServiceExample.cpp )
#ament_target_dependencies(servicecall rclcpp  cv_bridge tf2_geometry_msgs)
#target_link_libraries(servicecall  ${OpenCV_LIBS} generalhelpfullib)
#target_include_directories(servicecall PUBLIC  ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(servicecall "${cpp_typesupport_target}")





install(TARGETS
        ros2ServiceRegistration
        #        guibluerov2
        DESTINATION lib/${PROJECT_NAME}
)



ament_export_dependencies(rosidl_default_runtime)
ament_package()


