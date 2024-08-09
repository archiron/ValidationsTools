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

cd $1

echo "executing $2/extractValues.py $3 $4"
python3 -m cProfile $2/extractValues.py $3 $4
echo " "
#echo "executing $2/extractNewFilesValues.py $3 $4" # only use if you have AE pictures to create (time very long)
#python3 $2/extractNewFilesValues.py $3 $4

