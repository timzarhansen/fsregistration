#! /bin/bash
source /opt/ros/humble/setup.bash
source /home/tim-external/ros_ws/install/setup.bash
ros2 run fsregistration ros2ServiceRegistrationFS3D &
cd /home/tim-external/ros_ws/src/fsregistration/pythonScripts/matchingProfiling/

./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 32 112 0.01 0.1 &
pid1=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 32 112 0.01 0.01 &
pid2=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 32 112 0.001 0.1 &
pid3=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 32 112 0.001 0.01 &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4





