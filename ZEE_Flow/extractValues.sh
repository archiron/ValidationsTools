#!/bin/sh
# This file is called ./extractValues.sh by extractValues_init.sh

if [ "$1" == "" ] 
then
	echo "extractValues.sh has no argument"
	exit
fi

echo "chemin START : $1"
echo "chemin WORK : $2"
echo "chemin LIB : $3"
echo "chemin COMMON : $4"
echo "result folder : $5"

#cd $LOG_SOURCE
cd $1
#source /afs/cern.ch/cms/cmsset_default.sh
#eval `scramv1 runtime -sh`
#cd -

#cd $2
echo "executing $2/extractValues.py"
python3 $2/extractValues.py $3 $4 $5
echo "executing $2/extractNewFilesValues.py $3 $4 $5"
python3 $2/extractNewFilesValues.py $3 $4 $5

