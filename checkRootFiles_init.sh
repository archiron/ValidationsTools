#!/bin/sh
# This file is called . checkRootFiles_init.sh

JobName="checkRootFiles_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/checkRootFiles_%j.log" # for slurm

declare -a readarray

aa=$PWD
echo "actual path : $aa"

STR=$aa
Choice='Local'
for SUB in 'llr' 'pbs' 'cern'
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

LOG_SOURCE="$aa/${toto[15]}"
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE=${LOG_SOURCE//\"}
LOG_OUTPUT="$aa/${toto[16]}"
LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
LOG_OUTPUT=${LOG_OUTPUT//\"}
RESULTFOLDER="${toto[17]}"
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
LOG_KS_SOURCE="$aa/${toto[18]}"
LOG_KS_SOURCE=${LOG_KS_SOURCE//LOG_KS_SOURCE=}
LOG_KS_SOURCE=${LOG_KS_SOURCE//\"}
LIB_SOURCE="$aa/${toto[19]}"
LIB_SOURCE=${LIB_SOURCE//LIB_SOURCE=}
LIB_SOURCE=${LIB_SOURCE//\"}
COMMON_SOURCE="$aa/${toto[20]}"
COMMON_SOURCE=${COMMON_SOURCE//COMMON_SOURCE=}
COMMON_SOURCE=${COMMON_SOURCE//\"}
DATA_SOURCE="$aa/${toto[21]}"
DATA_SOURCE=${DATA_SOURCE//DATA_SOURCE=}
DATA_SOURCE=${DATA_SOURCE//\"}
CHECK_SOURCE="$aa/${toto[22]}"
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
    source /opt/exp_soft/llr/root/v6.24.04-el7-gcc9xx-py370/etc/init.sh
    cd $LOG_SOURCE
    #/opt/exp_soft/cms/t3/t3submit -8c -long checkRootFiles.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $CHECK_SOURCE $FileName
    #/opt/exp_soft/cms/t3/t3submit -8c -short checkRootFiles.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $CHECK_SOURCE $FileName
    /opt/exp_soft/cms/t3/t3submit -8c -reserv checkRootFiles.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $CHECK_SOURCE $FileName
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    module load Programming_Languages/python/3.9.1
    module load Compilers/gcc/9.3.1
    module load DataManagement/xrootd/4.8.1
    module load Analysis/root/6.24.06
    sbatch -L sps -n 8 --mem=16000 -t 4-0:0:0 -J $JobName -o $output checkRootFiles.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $CHECK_SOURCE $FileName
fi

echo "END"

