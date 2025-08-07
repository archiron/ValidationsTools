#!/bin/sh
# This file is called ./zee_flow.sh for CCA computers.
###################
## CCA computers ##
###################

echo "nb : $1"
echo "chemin : $2"
echo "nb evts : $3"
echo "result folder : $4"
echo "initial SEED : $5"
echo "nb skip : $6"
echo ""

LOG_SOURCE=$2
echo "Work in : $LOG_SOURCE"

name2="step2_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name31="step3_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name32="step3_inDQM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name33="step3_inMINIAODSIM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name34="step3_inNANOEDMAODSIM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
echo $name2

cd $2
eval `scramv1 runtime -sh`
cd -

#cmsRun $2/step2.py $1 $2 $3 $4 $6
echo $name31
echo $name32
echo $name33
echo $name34
cmsRun $2/step3.py $1 $2 $3 $4 $6


