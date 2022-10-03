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
echo "paths file : $4"
#echo "chemin LIB : $3"
#echo "result folder : $5"

#module load Programming_Languages/python/3.9.1
#source /pbs/home/c/chiron/private/ValidationsTools/ValidationsTools/bin/activate 

cd $1
#source /afs/cern.ch/cms/cmsset_default.sh
#eval `scramv1 runtime -sh`
cd -

#cd $2
#echo "executing $2/AEGeneration.py $3 $4"
#python3 $2/AEGeneration.py $3 $4
#echo " "
echo "executing $2/lossValuesVsKS.py $3 $4"
python3 $2/lossValuesVsKS.py $3 $4
#echo " "
#echo "executing $2/zeeKSvsAEComp.py $3 $4"
#python3 $2/zeeKSvsAEComp.py $3 $4

