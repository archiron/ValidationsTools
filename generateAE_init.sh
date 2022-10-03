#!/bin/sh
# This file is called . generateAE_init.sh

JobName="generateAE_serial_job_test" # for slurm
output="generateAE_%j.log" # for slurm

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
#LOG_OUTPUT="${toto[15]}"
#LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
#LOG_OUTPUT=${LOG_OUTPUT//\"}
RESULTFOLDER="${toto[15]}"
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
LOG_KS_SOURCE="$aa/${toto[18]}"
LOG_KS_SOURCE=${LOG_KS_SOURCE//LOG_KS_SOURCE=}
LOG_KS_SOURCE=${LOG_KS_SOURCE//\"}
#LIB_SOURCE="${toto[19]}"
#LIB_SOURCE=${LIB_SOURCE//LIB_SOURCE=}
#LIB_SOURCE=${LIB_SOURCE//\"}
COMMON_SOURCE="$aa/${toto[20]}"
COMMON_SOURCE=${COMMON_SOURCE//COMMON_SOURCE=}
COMMON_SOURCE=${COMMON_SOURCE//\"}
LOG_AE_SOURCE="$aa/${toto[23]}"
LOG_AE_SOURCE=${LOG_AE_SOURCE//LOG_AE_SOURCE=}
LOG_AE_SOURCE=${LOG_AE_SOURCE//\"}

echo "LOG_SOURCE : $LOG_SOURCE"
echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "RESULTFOLDER : $RESULTFOLDER"
echo "LOG_AE_SOURCE : $LOG_AE_SOURCE"
echo "LIB_SOURCE : $LIB_SOURCE"
echo "COMMON_SOURCE : $COMMON_SOURCE"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    source /opt/exp_soft/llr/root/v6.24.04-el7-gcc9xx-py370/etc/init.sh
    cd $LOG_SOURCE
    #/opt/exp_soft/cms/t3/t3submit -8c -short generateAE.sh $LOG_SOURCE $LOG_AE_SOURCE $COMMON_SOURCE $FileName
    #/opt/exp_soft/cms/t3/t3submit -8c -long generateAE.sh $LOG_SOURCE $LOG_AE_SOURCE $COMMON_SOURCE $FileName
    /opt/exp_soft/cms/t3/t3submit -8c -reserv generateAE.sh $LOG_SOURCE $LOG_AE_SOURCE $COMMON_SOURCE $FileName
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    module load Programming_Languages/python/3.9.1
    source /pbs/home/c/chiron/private/ValidationsTools/ValidationsTools/bin/activate 
    cd $LOG_SOURCE
    #eval `scramv1 runtime -sh`
    sbatch -L sps -n 4 --mem=16000 -J $JobName -o $output generateAE.sh $LOG_SOURCE $LOG_AE_SOURCE $COMMON_SOURCE $FileName
    deactivate
fi

echo "END"

