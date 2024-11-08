#! /bin/bash
source /opt/ros/humble/setup.bash
source /home/tim-external/ros_ws/install/setup.bash
ros2 run fsregistration ros2ServiceRegistrationFS3D &
cd /home/tim-external/ros_ws/src/fsregistration/pythonScripts/matchingProfiling/

nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 0 4 24 0.01 0.1 &
pid1=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 0 4 24 0.01 0.01 &
pid2=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 0 4 24 0.001 0.1 &
pid3=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 0 4 24 0.001 0.01 &
pid4=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 0 8 56 0.01 0.1 &
pid5=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 0 8 56 0.01 0.01 &
pid6=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 0 8 56 0.001 0.1 &
pid7=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 0 8 56 0.001 0.01 &
pid8=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 112 0.01 0.1 &
pid9=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 112 0.01 0.01 &
pid10=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 112 0.001 0.1 &
pid11=$!
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 112 0.001 0.01 &
pid12=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 pid9=$! pid10=$! pid11=$! pid12=$!