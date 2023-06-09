#!/bin/sh
# This file is called . generateAE_init.sh

JobName="generateAE_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/generateAE_%j.log" # for slurm

timeFolder="$(date +"%Y%m%d-%H%M%S")"

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

RESULTFOLDER="${toto[15]}"
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
LOG_KS_SOURCE="$aa/${toto[18]}"
LOG_KS_SOURCE=${LOG_KS_SOURCE//LOG_KS_SOURCE=}
LOG_KS_SOURCE=${LOG_KS_SOURCE//\"}

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
    module purge
    module use /opt/exp_soft/vo.gridcl.fr/software/modules/
    module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7
    module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles
    module load torch/1.5.0-py37-cuda9.2
    module load python/3.7.0
    cd $LOG_SOURCE
    options="-reserv" # -short -long -reserv

    cd $LOG_AE_SOURCE
    cd -

    echo "executing $LOG_AE_SOURCE/AEGenerationGPU.py $COMMON_SOURCE $FileName $var ${arrLine[1]} 'gpu' $timeFolder"
    time python3 $LOG_AE_SOURCE/AEGenerationGPU.py $COMMON_SOURCE $FileName 'gpu' $timeFolder

fi
echo "END"
