#!/bin/sh
# This file is called ./extractValues.sh by extractValues_init.sh

if [ "$1" == "" ] 
then
	echo "zee_flow.sh has no argument"
	exit
fi

echo "chemin START : $1"
echo "chemin WORK : $2"
echo "chemin COMMON : $3"
echo "Check folder : $4"
echo "paths file : $5"
#echo "chemin LIB : $3"
#echo "result folder : $5"

#cd $LOG_SOURCE
cd $1
#source /afs/cern.ch/cms/cmsset_default.sh
#eval `scramv1 runtime -sh`
#cd -

#cd $2
echo "executing $4/checkRootFiles.py"
python3 $4/checkRootFiles.py $3 $4 $5
echo "executing $4/checkMapDiff.py"
python3 $4/checkMapDiff.py $3 $4 $5

#python3 $2/extrGT.py
#echo "executing $1/extrGT.py"
#python3 $2/createFiles_v2.py
#echo "executing $2/createFiles_v2.py"
#python3 $2/zeepValues.py
#echo "executing $2/zeepValues.py"

