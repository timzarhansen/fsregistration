#! /bin/bash
source /opt/ros/humble/setup.bash
source /home/tim-external/ros_ws/install/setup.bash
ros2 run fsregistration ros2ServiceRegistrationFS3D &>/dev/null
cd /home/tim-external/ros_ws/src/fsregistration/pythonScripts/matchingProfiling/

./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 0 4 24 0.01 0.1 &>/dev/null
pid1=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 0 4 24 0.01 0.01 &>/dev/null
pid2=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 0 4 24 0.001 0.1 &>/dev/null
pid3=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 0 4 24 0.001 0.01 &>/dev/null
pid4=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 0 8 56 0.01 0.1 &>/dev/null
pid5=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 0 8 56 0.01 0.01 &>/dev/null
pid6=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 0 8 56 0.001 0.1 &>/dev/null
pid7=$!
./testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 0 8 56 0.001 0.01 &>/dev/null
pid8=$!


wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8