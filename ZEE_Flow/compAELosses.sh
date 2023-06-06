#!/bin/sh
# This file is called ./compAELosses.sh by compAELosses_init.sh

if [ "$1" == "" ] 
then
	echo "compLossesAE.sh has no argument"
	exit
fi

echo "chemin WORK : $1" # AE_SOURCE
echo "chemin COMMON : $2"
echo "paths file : $3"

cd $1
cd -

echo "executing $1/compAELosses.py $2 $34"
time python3 $1/compAELosses.py $2 $3
