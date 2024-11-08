#! /bin/bash
source /opt/ros/humble/setup.bash
source /home/tim-external/ros_ws/install/setup.bash
ros2 run fsregistration ros2ServiceRegistrationFS3D & >/dev/null 2>&1
cd /home/tim-external/ros_ws/src/fsregistration/pythonScripts/matchingProfiling/

./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 28 0.01 0.1 & >/dev/null 2>&1
pid1=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 28 0.01 0.01 & >/dev/null 2>&1
pid2=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 28 0.001 0.1 & >/dev/null 2>&1
pid3=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 28 0.001 0.01 & >/dev/null 2>&1
pid4=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 48 0.01 0.1 & >/dev/null 2>&1
pid5=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 48 0.01 0.01 & >/dev/null 2>&1
pid6=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 48 0.001 0.1 & >/dev/null 2>&1
pid7=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 48 0.001 0.01 & >/dev/null 2>&1
pid8=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8