#!/bin/bash

RESULTFOLDER='/sps/cms/chiron/CMSSW_12_1_0_pre5-16c-1'

for value in $(seq -f "%03g" 0 700)
do
name=$RESULTFOLDER/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_9000_$value.root
if [ ! -f "$name" ]; then
    echo "$name does not exist."
    echo "$value"
fi
done
echo All done

