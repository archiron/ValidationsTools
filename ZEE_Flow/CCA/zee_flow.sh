#!/bin/sh
# This file is called ./zee_flow.sh 11_3_0_pre4

if [ "$1" == "" ] 
then
	echo "zee_flow.sh has no argument"
	exit
fi

echo "nb : $1"
echo "chemin : $2"
echo "nb evts : $3"
echo "result folder : $4"

LOG_SOURCE=$2
echo "Step 1 in : $LOG_SOURCE"

#cd $LOG_SOURCE
cd $2
#source /afs/cern.ch/cms/cmsset_default.sh
eval `scramv1 runtime -sh`
cd -

name1="step1_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name2="step2_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name31="step3_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name32="step3_inDQM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name33="step3_inMINIAODSIM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name34="step3_inNANOEDMAODSIM_$(printf "%04d" $3)_$(printf "%03d" $1).root"

cmsRun $2/step1.py $1 $2 $3

cmsRun $2/step2.py $1 $2 $3
rm $name1
cmsRun $2/step3.py $1 $2 $3
rm $name2

cmsRun $2/step4.py $1 $2 $3
rm $name31
rm $name32
rm $name33
rm $name34

cp *.root $4
cp DQM*.root $4
