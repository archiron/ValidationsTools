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

LOG_SOURCE="${toto[13]}"
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE=${LOG_SOURCE//\"}
LOG_OUTPUT="${toto[14]}"
LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
LOG_OUTPUT=${LOG_OUTPUT//\"}
RESULTFOLDER="${toto[15]}"
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
LOG_KS_SOURCE="${toto[15]}"
LOG_KS_SOURCE=${LOG_KS_SOURCE//LOG_KS_SOURCE=}
LOG_KS_SOURCE=${LOG_KS_SOURCE//\"}
echo "LOG_SOURCE : $LOG_SOURCE"
echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "RESULTFOLDER : $RESULTFOLDER"
echo "LOG_KS_SOURCE : $LOG_KS_SOURCE"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    #cd -
    /opt/exp_soft/cms/t3/t3submit -8c -long $LOG_KS_SOURCE/extractValues.sh $LOG_SOURCE $LOG_KS_SOURCE $RESULTFOLDER
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    sbatch -L sps -n 8 --mem=8000 -J $JobName -o $output $LOG_KS_SOURCE/extractValues.sh $LOG_SOURCE $LOG_KS_SOURCE $RESULTFOLDER
fi

echo "END"

