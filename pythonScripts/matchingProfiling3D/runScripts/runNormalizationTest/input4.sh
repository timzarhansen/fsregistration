#! /bin/bash
source /opt/ros/humble/setup.bash
source /home/tim-external/ros_ws/install/setup.bash
ros2 run fsregistration ros2ServiceRegistrationFS3D & >/dev/null 2>&1
cd /home/tim-external/ros_ws/src/fsregistration/pythonScripts/matchingProfiling/

./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 4 28 0.001 0.1 1 & >/dev/null 2>&1
pid1=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 8 56 0.001 0.01 0 & >/dev/null 2>&1
pid2=$!

wait $pid1 $pid2





