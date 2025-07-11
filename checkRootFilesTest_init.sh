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
COMMON_SOURCE="$aa/${toto[20]}"
COMMON_SOURCE=${COMMON_SOURCE//COMMON_SOURCE=}
COMMON_SOURCE=${COMMON_SOURCE//\"}
CHECK_SOURCE="$aa/${toto[22]}"
CHECK_SOURCE=${CHECK_SOURCE//CHECK_SOURCE=}
CHECK_SOURCE=${CHECK_SOURCE//\"}

echo "LOG_SOURCE : $LOG_SOURCE"
echo "COMMON_SOURCE : $COMMON_SOURCE"
echo "CHECK_SOURCE : $CHECK_SOURCE"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    source /opt/exp_soft/llr/root/v6.32-el9-gcc13xx-py3124/etc/init.sh
    cd $LOG_SOURCE
    /opt/exp_soft/cms/t3/t3submit -8c -long -name checkMap checkRootFilesTest.sh $COMMON_SOURCE $CHECK_SOURCE $FileName # -mail chiron@llr.in2p3.fr 
    #/opt/exp_soft/cms/t3/t3submit -8c -short checkRootFilesTest.sh $COMMON_SOURCE $CHECK_SOURCE $FileName
    #/opt/exp_soft/cms/t3/t3submit -8c -reserv checkRootFilesTest.sh $COMMON_SOURCE $CHECK_SOURCE $FileName
    #. checkRootFilesTest.sh $COMMON_SOURCE $CHECK_SOURCE $FileName # $LOG_SOURCE $LOG_KS_SOURCE 
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    module load Programming_Languages/python/3.9.1
    module load Compilers/gcc/9.3.1
    module load DataManagement/xrootd/4.8.1
    module load Analysis/root/6.24.06
    sbatch -L sps -n 3 --mem=8000 -t 4-0:0:0 -J $JobName -o $output checkRootFiles.sh $COMMON_SOURCE $CHECK_SOURCE $FileName
fi

cd $aa
echo "END"

