#!/bin/bash

for sizePixel in 0.25 0.5 0.75 1.0; do
  for n in 128 256; do
    for skips in $(seq 1 10); do
        python3 testingSequence.py $n 1 0.1 $sizePixel $skips /home/tim-external/dataFolder/2019-01-10-14-36-48-radar-oxford-10k-partial2/radar
    done
  done
done

#python3 testingSequence.py 256 1 0.1 0.25 4 /home/tim-external/dataFolder/2019-01-10-14-36-48-radar-oxford-10k-partial/radar
#python3 testingSequence.py 256 1 0.1 0.25 6 /home/tim-external/dataFolder/2019-01-10-14-36-48-radar-oxford-10k-partial/radar

