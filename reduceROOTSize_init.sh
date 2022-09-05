#!/bin/sh
# This file is called . reduceROOTSize.sh

JobName="chichi_serial_job_test" # for slurm
output="chichi_%j.log" # for slurm

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

FileName1="rootValues.py"
echo $FileName1
readarray toor -t array < CommonFiles/$FileName1
N1=${#toor[@]}
echo "N1= $N1"

Nbegin="${toor[15]}"
Nbegin=${Nbegin//Nbegin = } # WARNING : "Nbegin = " MUST be written in the same form as in rootValues.py
Nbegin=${Nbegin//\"}
Nbegin=${Nbegin::-1} # remove last char (\r\n)
Nend="${toor[16]}"
Nend=${Nend//Nend = }
Nend=${Nend//\"}
Nend=${Nend::-1}
NB_EVTS="${toor[17]}"
NB_EVTS=${NB_EVTS//NB_EVTS = }
NB_EVTS=${NB_EVTS//\"}
NB_EVTS=${NB_EVTS::-1}
echo "Nbegin : $Nbegin=="
echo "Nend : $Nend=="
echo "NB_EVTS : $NB_EVTS=="

FileName="paths$Choice.py"
echo $FileName
readarray toto -t array < CommonFiles/$FileName
N=${#toto[@]}
echo "N= $N"
#echo ${toto[@]}

LOG_SOURCE="$aa/${toto[15]}"
#LOG_SOURCE=$aa
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE=${LOG_SOURCE//\"}
#LOG_SOURCE="${LOG_SOURCE}/${release}/src/Kolmogorov"
LOG_OUTPUT="$aa/${toto[16]}"
LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
LOG_OUTPUT=${LOG_OUTPUT//\"}
RESULTFOLDER="${toto[17]}"
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
echo "LOG_SOURCE : $LOG_SOURCE"
echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "RESULTFOLDER : $RESULTFOLDER"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    source /opt/exp_soft/llr/root/v6.24.04-el7-gcc9xx-py370/etc/init.sh
    cd $LOG_SOURCE
    for i in $(eval echo "{$Nbegin..$Nend}") 
    do
      #/opt/exp_soft/cms/t3/t3submit -8c -long reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      #/opt/exp_soft/cms/t3/t3submit -8c -short reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      /opt/exp_soft/cms/t3/t3submit -8c -reserv reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
    done
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    module load Programming_Languages/python/3.9.1
    module load Compilers/gcc/9.3.1
    module load DataManagement/xrootd/4.8.1
    module load Analysis/root/6.24.06
    for i in $(eval echo "{$Nbegin..$Nend}")
    do
      sbatch -L sps -n 8 --mem=8000 -J $JobName -o $output reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
    done
fi

echo "END"

