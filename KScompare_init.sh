#!/bin/sh
# This file is called . KSCompare_init.sh

JobName="KSCompare_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/KSCompare_%j.log" # for slurm

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
RESULTFOLDER="${toto[17]}"
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

echo "LOG_SOURCE : $LOG_SOURCE"
echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "RESULTFOLDER : $RESULTFOLDER"
echo "LOG_KS_SOURCE : $LOG_KS_SOURCE"
echo "LIB_SOURCE : $LIB_SOURCE"
echo "COMMON_SOURCE : $COMMON_SOURCE"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    #module purge
    #source /usr/share/Modules/init/sh
    #module use /opt/exp_soft/vo.gridcl.fr/software/modules/
    #module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7
    #module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles
    #module load python/3.7.0
    module reset
    source /usr/share/Modules/init/sh
    module use /opt/exp_soft/vo.gridcl.fr/software/modules/
    module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el9
    
    module load python/3.12.4
    #module load compilers/gcc/11.x.x
    source /opt/exp_soft/llr/root/v6.32-el9-gcc13xx-py3124/etc/init.sh
    #source /opt/exp_soft/llr/root/v6.24.04-el7-gcc9xx-py370/etc/init.sh

    cd $LOG_SOURCE
    #/opt/exp_soft/cms/t3/t3submit -8c -long KScompare.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName # -mail chiron@llr.in2p3.fr 
    /opt/exp_soft/cms/t3/t3submit -8c -short KScompare.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName
    #/opt/exp_soft/cms/t3/t3submit -8c -reserv KScompare.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    module load Programming_Languages/python/3.9.1
    module load Compilers/gcc/9.3.1
    module load DataManagement/xrootd/4.8.1
    module load Analysis/root/6.24.06
    sbatch -L sps -n 2 --mem=8000 -t 4-0:0:0 -J $JobName -o $output KScompare.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName
fi

echo "END"

