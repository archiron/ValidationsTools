#!/bin/sh
# This file is called ./createFiles.sh by createFiles.sh

if [ "$1" == "" ] 
then
	echo "createFiles.sh has no argument"
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
#echo "executing $2/createFiles.py $3 $4"
#python3 $2/createFiles.py $3 $4
#echo "executing $2/createFiles_v2.py $3 $4"
#python3 $2/createFiles_v2.py $3 $4
echo "executing $2/createFiles_v3.py $3 $4"
python3 $2/createFiles_v3.py $3 $4
echo "executing $2/KShistos.py $3 $4"
python3 $2/KShistos.py $3 $4

