#!/bin/sh
# This file is called ./initAE.sh by generateAE_init.sh

if [ "$1" == "" ] 
then
	echo "initAE.sh has no argument"
	exit
fi

echo "chemin WORK : $1" # AE_SOURCE
echo "chemin COMMON : $2"
echo "paths file : $3"

cd $1
cd -

echo "executing $1/initAE.py $2 $3"
time python3 $1/initAE.py $2 $3
