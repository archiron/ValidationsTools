#!/bin/sh
# This file is called ./zee_flow.sh 11_3_0_pre4
Nbegin=0
Nend=2
NB_EVTS=9000

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
#echo ${toto[@]}

#for (( j=0; j<${N}; j++ ));
#do
#  printf "Current index %d with value %s" $j "${toto[$j]}"
#done
LOG_SOURCE="${toto[13]}"
LOG_OUTPUT="${toto[14]}"
RESULTFOLDER="${toto[15]}"
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE=${LOG_SOURCE//\"}
LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
LOG_OUTPUT=${LOG_OUTPUT//\"}
RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
RESULTFOLDER=${RESULTFOLDER//\"}
echo "LOG_SOURCE : $LOG_SOURCE"
echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "RESULTFOLDER : $RESULTFOLDER"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    cd $LOG_SOURCE
    #ls $LOG_SOURCE
    eval `scramv1 runtime -sh`
    #cd -
    for i in $(eval echo "{$Nbegin..$Nend}") 
    do
      #qsub -l sps=1 -P P_cmsf -pe multicores 8 -q mc_long -o $LOG_OUTPUT zee_flow.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      /opt/exp_soft/cms/t3/t3submit -8c -long zee_flow.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
    done
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    cd $LOG_SOURCE
    eval `scramv1 runtime -sh`
    for i in $(eval echo "{$Nbegin..$Nend}")
    do
      #/opt/exp_soft/cms/t3_tst/t3submit -8c -long zee_flow.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      #/opt/exp_soft/cms/t3/t3submit -8c -short zee_flow.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      sbatch -L sps zee_flow.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
    done
fi

echo "END"

