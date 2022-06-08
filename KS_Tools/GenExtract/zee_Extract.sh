#!/bin/sh
# This file is called ./zee_flow.sh 11_3_0_pre4

if [ "$1" == "" ] 
then
	echo "zee_flow.sh has no argument"
	exit
fi

echo "chemin START : $1"
echo "chemin WORK : $2"
echo "result folder : $3"

#cd $LOG_SOURCE
cd $1
#source /afs/cern.ch/cms/cmsset_default.sh
eval `scramv1 runtime -sh`
cd -

cd $2
#python3 $2/reduceSize.py
#echo "executing $1/reduceSize.py"
#python3 $2/extractValues.py
#echo "executing $2/extractValues.py"
#python3 $2/extrGT.py
#echo "executing $1/extrGT.py"
python3 $2/createFiles_v2.py
echo "executing $2/createFiles_v2.py"
#python3 $2/zeeMapDiff.py
#echo "executing $2/zeeMapDiff.py"
#python3 $2/zeepValues.py
#echo "executing $2/zeepValues.py"

