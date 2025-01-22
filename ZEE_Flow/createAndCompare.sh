#!/bin/sh
# This file is called ./createAndCompare.sh by createAndCompare.sh

if [ "$1" == "" ] 
then
	echo "createAndCompare.sh has no argument"
	exit
fi

echo "chemin START : $1"
echo "chemin WORK : $2"
echo "chemin COMMON : $3"
echo "paths file : $4"

cd $1

echo "executing $2/createAndCompare.py $3 $4"
python3 $2/createAndCompare.py $3 $4
