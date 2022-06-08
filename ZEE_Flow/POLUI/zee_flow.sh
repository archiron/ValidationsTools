#!/bin/sh
# This file is called ./zee_flow.sh

echo "i : $1"
echo "chemin : $2"
echo "nb evts : $3"
echo "folder result : $4"
LOG_SOURCE=$2
echo "Step 1 in : $LOG_SOURCE"

name1="step1_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name2="step2_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name31="step3_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name32="step3_inDQM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name33="step3_inMINIAODSIM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name34="step3_inNANOEDMAODSIM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
#name4="DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_$(printf "%04d" $3)_$(printf "%03d" $1).root"
echo $name1
echo $name2
echo $name31
echo $name32
echo $name33
echo $name34

cd $4
ls
eval `scramv1 runtime -sh`

cmsRun $2/step1.py $1 $3
cmsRun $2/step2.py $1 $3
#rm $4/$name1
cmsRun $2/step3.py $1 $3
#rm $4/$name2
cmsRun $2/step4.py $1 $3
#rm $4/$name31
#rm $4/$name32
#rm $4/$name33
#rm $4/$name34

