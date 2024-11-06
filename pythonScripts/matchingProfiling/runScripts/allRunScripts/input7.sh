#! /bin/bash
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 24 0.01 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 24 0.01 0.01 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 24 0.001 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 32 1 8 24 0.001 0.01 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 56 0.01 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 56 0.01 0.01 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 56 0.001 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 64 1 16 56 0.001 0.01 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 1 32 96 0.01 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 1 32 96 0.01 0.01 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 1 32 96 0.001 0.1 &
nohup python3 testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 1 32 96 0.001 0.01 &



