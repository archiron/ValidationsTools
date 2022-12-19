#!/bin/sh
# This file is called . createROOTFiles.sh

JobName="createROOTFiles_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/createROOTFiles_%j.log" # for slurm

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

# Get local Release used
for item in `ls -drt ZEE_Flow/*/` 
do
  printf "   %s\n" $item
  release=$item
done
release=${release%?}
#release=${release//\/}
#release=${release//ZEE_Flow/}
echo $release

#LOG_SOURCE="${toto[15]}"
LOG_SOURCE=$aa
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE=${LOG_SOURCE//\"}
LOG_SOURCE="${LOG_SOURCE}/${release}/src/Kolmogorov"
LOG_OUTPUT="$aa/${toto[16]}"
LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
LOG_OUTPUT=${LOG_OUTPUT//\"}
RESULTFOLDER="${toto[17]}"
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
RESULTFOLDER=$(printf $RESULTFOLDER)
RESULTAPPEND=$(printf "/%04d" $NB_EVTS)
RESULTFOLDER="${RESULTFOLDER}${RESULTAPPEND}"
echo "LOG_SOURCE : $LOG_SOURCE"
echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "RESULTFOLDER : $RESULTFOLDER"

mkdir -p $RESULTFOLDER

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    for i in $(eval echo "{$Nbegin..$Nend}") 
    do
      #/opt/exp_soft/cms/t3/t3submit -8c -long createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      #/opt/exp_soft/cms/t3/t3submit -8c -short createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      /opt/exp_soft/cms/t3/t3submit -8c -reserv createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
    done
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    for i in 641 701 714 809 822 838 982 987 991 #$(eval echo "{$Nbegin..$Nend}")
    do
      sbatch -L sps -n 8 --mem=16000 -t 4-0:0:0 -J $JobName -o $output createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
    done
fi

echo "END"

