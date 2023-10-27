# fs2d
fourie Soft 2D Registration Method
Currently FS2D hast two versions. First a direkt Image registration, where based on strings, two images can be read in and the registration can be calculated
The second version is a ROS2 Service, where you send two images and recieve an awnser.(See `msg` and `srv`)


# install
The idea is to have this as a ROS2 package, where you can just install the package for easy use(as a Service), and as a library.


Ros-Humble Desktop [here](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) 

OpenCV 

soft20 Library (can be done in a ros2 work space) [here](https://github.com/Zarbokk/soft20.git)

Eigen 3

cv_bridge


# first run
use the example Data for an initial test. There are two sonar images, which you can use for the first registration.
Just give the paths of the two images as arguments to the `registrationOfTwoImages` executable after compiling with CMake.



# more description to come:::