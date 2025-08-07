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

cd $2
eval `scramv1 runtime -sh`
cd -

cmsRun $2/step4.py $1 $2 $3 $4 $6
