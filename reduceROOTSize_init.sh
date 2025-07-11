#!/bin/sh
# This file is called . reduceROOTSize.sh

JobName="reduceROOT_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/reduceROOT_%j.log" # for slurm

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

FileName2="filesSources.py"
echo $FileName2
readarray releases -t array < CommonFiles/$FileName2
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
echo "Nbegin : $Nbegin =="
echo "Nend : $Nend =="
echo "NB_EVTS : $NB_EVTS =="

FileName="paths$Choice.py"
echo $FileName
readarray toto -t array < CommonFiles/$FileName
N=${#toto[@]}
echo "N= $N"

LOG_SOURCE="$aa/${toto[15]}"
#LOG_SOURCE=$aa
LOG_SOURCE=${LOG_SOURCE//LOG_SOURCE=}
LOG_SOURCE=${LOG_SOURCE//\"}
#LOG_SOURCE="${LOG_SOURCE}/${release}/src/Kolmogorov"
LOG_OUTPUT="$aa/${toto[16]}"
LOG_OUTPUT=${LOG_OUTPUT//LOG_OUTPUT=}
LOG_OUTPUT=${LOG_OUTPUT//\"}
#RESULTFOLDER="${toto[17]}"
#RESULTFOLDER=${RESULTFOLDER//RESULTFOLDER=}
#RESULTFOLDER=${RESULTFOLDER//\"}
#RESULTFOLDER=$(printf $RESULTFOLDER)
RESULTFOLDER="/data_CMS/cms/chiron/ROOT_Files/"
RESULTAPPEND=$(printf "/%04d" $NB_EVTS)
RESULTRELEASE=$(printf "/%s" $release)
#RESULTFOLDER="${RESULTFOLDER}${RESULTAPPEND}${RESULTRELEASE}"
RESULTFOLDER="${RESULTFOLDER}${RESULTRELEASE}"
echo "LOG_SOURCE : $LOG_SOURCE"
echo "LOG_OUTPUT : $LOG_OUTPUT"
echo "RESULTFOLDER : $RESULTFOLDER"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    module reset
    source /usr/share/Modules/init/sh
    module use /opt/exp_soft/vo.gridcl.fr/software/modules/
    module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el9
    module load python/3.12.4 # torch included !
    source /opt/exp_soft/llr/root/v6.32-el9-gcc13xx-py3124/etc/init.sh
    cd $LOG_SOURCE
    for i in $(eval echo "{$Nbegin..$Nend}") 
    do
      echo "==> $i"
      #/opt/exp_soft/cms/t3/t3submit -8c -long reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER & # -mail chiron@llr.in2p3.fr 
      #/opt/exp_soft/cms/t3/t3submit -8c -short reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
      . reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER &
      #/opt/exp_soft/cms/t3/t3submit -8c -reserv reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
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
      sbatch -L sps -n 8 --mem=8000 -t 0-2:0:0 -J $JobName -o $output reduceROOTSize.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
    done
fi

cd $aa
echo "END"


