#!/bin/sh
# This file is called . generateAE_init.sh

JobName="resumeAE_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/resumeAE_%j.log" # for slurm

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
    module purge
    module use /opt/exp_soft/vo.gridcl.fr/software/modules/
    module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7
    module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles
    module load torch/1.5.0-py37-nocuda
    module load python/3.7.0
    cd $LOG_SOURCE
    options="-reserv" # -short -long -reserv
    /opt/exp_soft/cms/t3/t3submit -8c $options resumeAE.sh $LOG_AE_SOURCE $COMMON_SOURCE $FileName '20230522-175054' # -mail chiron@llr.in2p3.fr 
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    module load Programming_Languages/python/3.9.1
    source /pbs/home/c/chiron/private/ValidationsTools/ValidationsTools/bin/activate 
    cd $LOG_SOURCE

    sbatch -L sps -n 4 --mem=16000 -t 0-6:0:0 -J $JobName -o $output resumeAE.sh $LOG_AE_SOURCE $COMMON_SOURCE $FileName '20230525-091823'
    deactivate
fi

echo "END"

