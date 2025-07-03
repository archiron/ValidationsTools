#!/bin/sh
# This file is called ./createFiles.sh by createFiles.sh

if [ "$1" == "" ] 
then
	echo "createFiles.sh has no argument"
	exit
fi

echo "chemin START : $1"
echo "chemin WORK : $2"
echo "chemin COMMON : $3"
echo "paths file : $4"

cd $1

#echo "executing $2/createFiles_v4.py $3 $4"
#python3 $2/createFiles_v4.py $3 $4

echo "executing $2/KShistos.py $3 $4"
python3 $2/KShistos.py $3 $4

#echo "executing $2/KSvalidation.py $3 $4"
#python3 $2/KSvalidation.py $3 $4

### OLD ###
#echo "executing $2/createFiles_v3.py $3 $4"
#python3 $2/createFiles_v3.py $3 $4

#echo "=====**********=====\n"
#python3 $2/KSvalidation-4.py $3 $4

