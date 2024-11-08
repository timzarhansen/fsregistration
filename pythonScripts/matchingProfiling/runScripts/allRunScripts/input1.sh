#! /bin/bash
source /opt/ros/humble/setup.bash
source /home/tim-external/ros_ws/install/setup.bash
ros2 run fsregistration ros2ServiceRegistrationFS3D &>/dev/null
cd /home/tim-external/ros_ws/src/fsregistration/pythonScripts/matchingProfiling/


./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 96 0.01 0.1 &>/dev/null
pid9=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 96 0.01 0.01 &>/dev/null
pid10=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 96 0.001 0.1 &>/dev/null
pid11=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 96 0.001 0.01 &>/dev/null
pid12=$!
#ros2 service list


wait $pid9 $pid10 $pid11 $pid12
