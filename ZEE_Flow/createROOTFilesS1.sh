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

name1="step1_$(printf "%04d" $3)_$(printf "%03d" $1).root"
echo $name1

cd $2
eval `scramv1 runtime -sh`
cd -

cmsRun $2/step1.py $1 $2 $3 $4 $5 $6

