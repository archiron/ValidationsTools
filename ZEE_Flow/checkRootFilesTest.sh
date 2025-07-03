#!/bin/sh
# This file is called ./extractValues.sh by extractValues_init.sh

if [ "$1" == "" ] 
then
	echo "zee_flow.sh has no argument"
	exit
fi

#echo "chemin START : $1"
#echo "chemin WORK : $2"
echo "chemin COMMON : $1"
echo "Check folder : $2"
echo "paths file : $3"

#cd $LOG_SOURCE
cd $1
#echo "executing $2/checkRootFiles.py"
#python3 $2/checkRootFiles.py $1 $2 $3
echo " "
echo "executing $2/checkMapDiff.py"
python3 $2/checkMapDiffTest.py $1 $2 $3
echo " "
#echo "executing $2/checkCreatedVsOfficial.py"
#python3 $2/checkCreatedVsOfficial.py $1 $2 $3
echo " "
#echo "executing $2/checkRootFilesvsRef.py"
#python3 $2/checkRootFilesvsRef.py $1 $2 $3

