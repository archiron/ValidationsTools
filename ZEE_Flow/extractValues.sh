#!/bin/sh
# This file is called ./extractValues.sh by extractValues_init.sh

if [ "$1" == "" ] 
then
	echo "extractValues.sh has no argument"
	exit
fi

echo "chemin START : $1"
echo "chemin WORK : $2"
echo "chemin COMMON : $3"
echo "paths file : $4"

#cd $LOG_SOURCE
cd $1
#source /afs/cern.ch/cms/cmsset_default.sh
#eval `scramv1 runtime -sh`
#cd -

#cd $2
echo "executing $2/extractValues.py $3 $4"
python3 $2/extractValues.py $3 $4
echo " "
echo "executing $2/extractNewFilesValues.py $3 $4"
python3 $2/extractNewFilesValues.py $3 $4

