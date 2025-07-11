#!/bin/sh
# This file is called . createFiles_init.sh

JobName="createFiles_serial_job_test" # for slurm
output="/sps/cms/chiron/TEMP/createFiles_%j.log" # for slurm

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

readarray datasets -t array < ChiLib/HistosConfigFiles/ElectronMcSignalHistos.txt # $Chilib_path
N2=${#datasets[@]}
echo "nb lines in datasets= $N2"
var=0
for line in "${datasets[@]}"
do
  if [[ $line == *"ElectronMcSignalValidator/"* ]]; then
    #echo $line
    arrLine=(${line//dator/ })
    #echo "${arrLine[1]} - $var"
    let "var++"
  fi
done
echo "nb datasets in datasets= $var"
let "var--"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"
    module reset
    source /usr/share/Modules/init/sh
    module use /opt/exp_soft/vo.gridcl.fr/software/modules/
    module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el9
    
    module load python/3.12.4
    source /opt/exp_soft/llr/root/v6.32-el9-gcc13xx-py3124/etc/init.sh

    cd $LOG_SOURCE
    #/opt/exp_soft/cms/t3/t3submit -8c -long -name "KS_stats_1000 v9" createFilesTest.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName "b"
    #. createFilesTest.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName "i"
    #/opt/exp_soft/cms/t3/t3submit -8c -short createFilesTest.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName
    #/opt/exp_soft/cms/t3/t3submit -8c -reserv createFilesTest.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName

    var=0
    for line in "${datasets[@]}"
    do
      if [[ $line == *"ElectronMcSignalValidator/"* ]]; then
        #echo $line
        arrLine=(${line//dator/ })
        echo "${arrLine[1]}"
        #/opt/exp_soft/cms/t3/t3submit -8c -long -name "KS_stats_1000 v9" createFilesTest.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName "b"
        . createFilesTest.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName "i"  ${arrLine[1]} &
        let "var++"
        if [ $var == 2 ]
        then
          break   # break the for loop
        fi

      fi
    done
fi
echo "END"

