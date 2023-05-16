#!/bin/sh
# This file is called ./generateAE.sh by generateAE_init.sh

if [ "$1" == "" ] 
then
	echo "generateAE.sh has no argument"
	exit
fi

echo "chemin WORK : $1" # AE_SOURCE
echo "chemin COMMON : $2"
echo "paths file : $3"
echo "timeFolder : $4"

cd $1
cd -

echo "executing $1/resumeAE.py $2 $3 $4"
time python3 $1/resumeAE.py $2 $3 $4
