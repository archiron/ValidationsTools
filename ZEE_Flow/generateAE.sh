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
echo "option : $7" #cpu/gpu
echo "timeFolder : $8"

cd $1
cd -

module load Programming_Languages/python/3.9.1
source /pbs/home/c/chiron/private/ValidationsTools/ValidationsTools/bin/activate 

echo "executing $1/$3 $2 $4 $5 $6 $7 $8"
time python3 $1/$3 $2 $4 $5 $6 $7 $8
echo "executing $1/postAE.py $2 $4 $6"
time python3 $1/postAE.py $2 $4 $6

deactivate

