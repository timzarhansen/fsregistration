#!/bin/bash


cd /home/tim-external/ros_ws/src/fsregistration/pythonScripts/matchingProfiling3D

python3 testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml low train

python3 testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml low val



python3 testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml high train

python3 testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml high val



python3 testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml None train

python3 testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml None val


