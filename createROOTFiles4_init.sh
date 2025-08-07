#!/bin/sh
# This file is called with : . createROOTFiles.sh

JobName="createROOTFiles_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/createROOTFiles_%j.log" # for slurm

declare -a readarray

aa=$PWD
echo "actual path : $aa"

STR=$aa
Choice='Local'
for SUB in 'llr' 'pbs' 'cern' # llr: LLR lab, pbs: CC Lyon facility, cern: CERN lxplus
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
N_skip="${toor[18]}"
N_skip=${N_skip//N_skip = }
N_skip=${N_skip//\"}
N_skip=${N_skip::-1}
echo "Nbegin : $Nbegin =="
echo "Nend : $Nend =="
echo "NB_EVTS : $NB_EVTS =="
echo "N_skip : $N_skip =="

FileName2="filesSources.py"
echo $FileName2
readarray releases -t array < CommonFiles/$FileName2
release="${releases[19]}"
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
readarray toto -t array < CommonFiles/$FileName
N=${#toto[@]}
echo "N= $N"

LOG_SOURCE=$aa
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE1=${LOG_SOURCE//\"}
LOG_SOURCE="${LOG_SOURCE1}/ZEE_Flow/${release}/src/Kolmogorov"
RESULTFOLDER="/data_CMS/cms/chiron/ROOT_Files/"
RESULTRELEASE=$(printf "/%s" $release)
RESULTAPPEND=$(printf "/%04d" $NB_EVTS)
RESULTFOLDER="${RESULTFOLDER}${RESULTRELEASE}"
echo "LOG_SOURCE : $LOG_SOURCE"
echo "RESULTFOLDER : $RESULTFOLDER"

mkdir -p $RESULTFOLDER
initialSEED=123456 # must be "now" or an integer such as 123456

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    cd "${LOG_SOURCE1}/ZEE_Flow/"
    
    for i in 1
    do
      echo "==> [$i]"
      #/opt/exp_soft/cms/t3/t3submit -8c -long -name "CMSSW_15_0_0_pre3" createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER $initialSEED
      . createROOTFilesS4.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER $initialSEED $N_skip & > ~/RoutFichier.log 2>&1 #&
      #/opt/exp_soft/cms/t3/t3submit -8c -short createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER $initialSEED
      #. createROOTFiles2.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER $initialSEED #&
      #/opt/exp_soft/cms/t3/t3submit -8c -reserv createROOTFiles.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER $initialSEED
    done
fi

cd $aa
echo "END"


