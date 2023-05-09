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
echo "executing $4/checkRootFiles.py"
#python3 $4/checkRootFiles.py $3 $4 $5
echo " "
echo "executing $4/checkMapDiff.py"
#python3 $4/checkMapDiff.py $3 $4 $5
echo " "
echo "executing $4/checkCreatedVsOfficial.py"
python3 $4/checkCreatedVsOfficial.py $3 $4 $5


