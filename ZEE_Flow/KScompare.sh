#!/bin/sh
# This file is called ./extractValues.sh by extractValues_init.sh

if [ "$1" == "" ] 
then
	echo "zee_Extract.sh has no argument"
	exit
fi

echo "chemin START : $1"
echo "chemin WORK : $2"
echo "chemin COMMON : $3"
echo "paths file : $4"
#echo "chemin LIB : $3"
#echo "result folder : $5"

#cd $LOG_SOURCE
cd $1
#source /afs/cern.ch/cms/cmsset_default.sh
#eval `scramv1 runtime -sh`
#cd -

#cd $2
echo "executing $2/KScompare.py $3 $4"
python3 $2/KScompare.py $3 $4
echo " "
echo "executing $2/statpValues.py $3 $4"
python3 $2/statpValues.py $3 $4
echo " "
echo "executing $2/statConfiance.py $3 $4"
python3 $2/statConfiance.py $3 $4

#python3 $2/extrGT.py
#echo "executing $1/extrGT.py"
#python3 $2/createFiles_v2.py
#echo "executing $2/createFiles_v2.py"
#python3 $2/zeeMapDiff.py
#echo "executing $2/zeeMapDiff.py"

