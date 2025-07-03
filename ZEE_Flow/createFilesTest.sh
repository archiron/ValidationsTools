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
echo "mode : $5"
echo "histo name : $6"

cd $1

echo "executing $2/createFilesTest.py $3 $4 $5 $6"
#python3 $2/createFilesTest.py $3 $4
#python3 $2/createFilesTest_v2.py $3 $4
#time python3 $2/createFilesTest_v3.py $3 $4
#python3 $2/createFilesTest_v3.py $3 $4 $5
#python3 $2/createFilesTest_v4.py $3 $4 $5
python3 $2/createFilesTest_v5.py $3 $4 $5 $6


