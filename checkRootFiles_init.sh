#!/bin/sh
# This file is called . extractValues_init.sh

JobName="chichi_serial_job_test" # for slurm
output="chichi_%j.log" # for slurm

declare -a readarray

aa=$PWD
echo "actual path : $aa"

STR=$aa
Choice='Local'
for SUB in 'llr' 'pbs'
do
  if [[ "$STR" == *"$SUB"* ]]; then
    echo "It's $SUB there.";
    Choice=${SUB^^};
  fi
done

echo "Choice is : $Choice"

FileName="paths$Choice.py"
echo $FileName
readarray toto -t array < CommonFiles/$FileName
N=${#toto[@]}
echo "N= $N"

LOG_SOURCE="${toto[15]}"
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE=${LOG_SOURCE//\"}
LOG_OUTPUT="${toto[16]}"
LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
LOG_OUTPUT=${LOG_OUTPUT//\"}
RESULTFOLDER="${toto[17]}"
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
LOG_KS_SOURCE="${toto[18]}"
LOG_KS_SOURCE=${LOG_KS_SOURCE//LOG_KS_SOURCE=}
LOG_KS_SOURCE=${LOG_KS_SOURCE//\"}
LIB_SOURCE="${toto[19]}"
LIB_SOURCE=${LIB_SOURCE//LIB_SOURCE=}
LIB_SOURCE=${LIB_SOURCE//\"}
COMMON_SOURCE="${toto[20]}"
COMMON_SOURCE=${COMMON_SOURCE//COMMON_SOURCE=}
COMMON_SOURCE=${COMMON_SOURCE//\"}
DATA_SOURCE="${toto[21]}"
DATA_SOURCE=${DATA_SOURCE//DATA_SOURCE=}
DATA_SOURCE=${DATA_SOURCE//\"}
CHECK_SOURCE="${toto[22]}"
CHECK_SOURCE=${CHECK_SOURCE//CHECK_SOURCE=}
CHECK_SOURCE=${CHECK_SOURCE//\"}

echo "LOG_SOURCE : $LOG_SOURCE"
echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "RESULTFOLDER : $RESULTFOLDER"
echo "LOG_KS_SOURCE : $LOG_KS_SOURCE"
echo "LIB_SOURCE : $LIB_SOURCE"
echo "COMMON_SOURCE : $COMMON_SOURCE"
echo "DATA_SOURCE : $DATA_SOURCE"
echo "CHECK_SOURCE : $CHECK_SOURCE"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    #cd -
    /opt/exp_soft/cms/t3/t3submit -8c -long checkRootFiles.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $CHECK_SOURCE $FileName
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    sbatch -L sps -n 8 --mem=8000 -J $JobName -o $output checkRootFiles.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $CHECK_SOURCE $FileName
fi

echo "END"

