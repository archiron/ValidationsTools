#!/bin/sh
# This file is called ./generateAE.sh by generateAE_init.sh

if [ "$1" == "" ] 
then
	echo "generateAE.sh has no argument"
	exit
fi

echo "chemin START : $1"
echo "chemin WORK : $2" # AE_SOURCE
echo "chemin COMMON : $3"
echo "script name : $4"
echo "paths file : $5"
echo "option : $6"

cd $1
cd -

echo "executing $2/$4 $3 $5 $6"
python3 $2/$4 $3 $5 $6

