#!/bin/sh
# This file is called ./generateAE.sh by generateAE_init.sh

if [ "$1" == "" ] 
then
	echo "generateAE.sh has no argument"
	exit
fi

echo "chemin WORK : $1" # AE_SOURCE
echo "chemin COMMON : $2"
echo "script name : $3"
echo "paths file : $4"
echo "nb datasets : $5"
echo "dataset : $6"
echo "option : $7"

cd $1
cd -

echo "executing $1/$3 $2 $4 $5 $6 $7"
time python3 $1/$3 $2 $4 $5 $6 $7

