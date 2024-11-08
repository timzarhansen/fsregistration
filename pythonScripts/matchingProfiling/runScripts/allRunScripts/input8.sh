#! /bin/bash
source /opt/ros/humble/setup.bash
source /home/tim-external/ros_ws/install/setup.bash
ros2 run fsregistration ros2ServiceRegistrationFS3D &
cd /home/tim-external/ros_ws/src/fsregistration/pythonScripts/matchingProfiling/

nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 28 0.01 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 28 0.01 0.01 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 28 0.001 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 28 0.001 0.01 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 48 0.01 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 48 0.01 0.01 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 48 0.001 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 48 0.001 0.01 &
