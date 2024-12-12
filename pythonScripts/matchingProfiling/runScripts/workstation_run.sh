#! /bin/bash


for i in $(seq 1 6);
do


    entrypointString="./ros_ws/src/fsregistration/pythonScripts/matchingProfiling/runScripts/runNormalizationTest/input$i.sh"
    echo $entrypointString
    docker run --rm -t -i -d --ipc=host --name run$i \
    --entrypoint $entrypointString \
     -v /Users/timhansen/Documents/dataFolder/3dmatch:/home/tim-external/dataFolder/3dmatch:z \
     -v /Users/timhansen/Documents/ros_ws/cache/humble/build:/home/tim-external/ros_ws/build:z \
     -v /Users/timhansen/Documents/ros_ws/cache/humble/install:/home/tim-external/ros_ws/install:z \
     -v /Users/timhansen/Documents/ros_ws/cache/humble/log:/home/tim-external/ros_ws/log:z \
     -v /Users/timhansen/Documents/ros_ws/configFiles:/home/tim-external/ros_ws/configFiles:z \
     -v /Users/timhansen/Documents/MATLAB/matlabTestEnvironment:/home/tim-external/matlab:z \
     -v /Users/timhansen/Documents/ros_ws/src:/home/tim-external/ros_ws/src:z \
     fs3d_benchmark:latest
done



