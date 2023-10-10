#!/bin/sh
# This file is called . createROOTFiles.sh

JobName="createROOTFiles_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/createROOTFiles_%j.log" # for slurm

declare -a readarray

aa=$PWD
echo "actual path : $aa"

STR=$aa
Choice='Local'
for SUB in 'llr' 'pbs' 'cern' 'kins'
do
  if [[ "$STR" == *"$SUB"* ]]; then
    echo "It's $SUB there.";
    Choice=${SUB^^};
  fi
done
echo "Choice is : $Choice"
if [[ "$Choice" == "KINS" ]] 
  then
      Choice='LLR'
fi
echo "Choice is : $Choice"

FileName1="rootValues.py"
echo $FileName1
readarray toor -t array < $aa/CommonFiles/$FileName1
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
echo "Nbegin : $Nbegin =="
echo "Nend : $Nend =="
echo "NB_EVTS : $NB_EVTS =="

FileName2="sources.py"
echo $FileName2
readarray releases -t array < $aa/CommonFiles/$FileName2
release="${releases[17]}"
tt=${release//input_ref_file = \'DQM_V}
tt=${tt//\'}
tt=${tt//__DQMIO.root}
tt=${tt/__C/ C}
read -a strarr1 <<< "$tt"
vv=${strarr1[1]}
vv=${vv/-/ }
read -a strarr2 <<< "$vv"
release=${strarr2[0]}
echo "release = $release"

FileName="paths$Choice.py"
echo $FileName
readarray toto -t array < $aa/CommonFiles/$FileName
N=${#toto[@]}
echo "N= $N"

# Get local (LAST) Release used
#for item in `ls -drt ZEE_Flow/*/` 
#do
#  printf "   %s\n" $item
#  release=$item
#done
#release=${release%?}
#release=${release//\/}
#release=${release//ZEE_Flow/}
#echo $release

#LOG_SOURCE="${toto[15]}"
LOG_SOURCE=$aa
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE=${LOG_SOURCE//\"}
LOG_SOURCE="${LOG_SOURCE}/ZEE_Flow/${release}/src/Kolmogorov"
#LOG_OUTPUT="$aa/${toto[16]}"
#LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
#LOG_OUTPUT=${LOG_OUTPUT//\"}
RESULTFOLDER="${toto[17]}"
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
RESULTFOLDER=$(printf $RESULTFOLDER)
RESULTRELEASE=$(printf "/%s" $release)
RESULTAPPEND=$(printf "/%04d" $NB_EVTS)
RESULTFOLDER="${RESULTFOLDER}${RESULTAPPEND}${RESULTRELEASE}"
#echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "LOG_SOURCE : $LOG_SOURCE"
echo "RESULTFOLDER : $RESULTFOLDER"

mkdir -p $RESULTFOLDER
initialSEED=123456 # must be "now" or an integer such as 123456

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    for i in $(eval echo "{$Nbegin..$Nend}") 
    do
      #/opt/exp_soft/cms/t3/t3submit -8c -long createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      #/opt/exp_soft/cms/t3/t3submit -8c -short createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      /opt/exp_soft/cms/t3/t3submit -8c -reserv createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER $initialSEED
    done
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    for i in $(eval echo "{$Nbegin..$Nend}") # 35
    do
      sbatch -L sps -n 8 --mem=8000 -t 0-6:0:0 -J $JobName -o $output createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER $initialSEED
    done
fi

echo "END"


