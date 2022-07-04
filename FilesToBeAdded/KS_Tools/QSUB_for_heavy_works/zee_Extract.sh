#!/bin/sh
# This file is called ./zee_flow.sh 11_3_0_pre4

if [ "$1" == "" ] 
then
	echo "zee_flow.sh has no argument"
	exit
fi

echo "chemin : $1"
echo "result folder : $2"

LOG_SOURCE=$1
echo "Step 1 in : $LOG_SOURCE"

#cd $LOG_SOURCE
cd $1
#source /afs/cern.ch/cms/cmsset_default.sh
eval `scramv1 runtime -sh`
cd -

#python3 $1/reduceSize.py
python3 $1/extractValues.py
