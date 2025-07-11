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
COMMON_SOURCE="$aa/${toto[20]}"
COMMON_SOURCE=${COMMON_SOURCE//COMMON_SOURCE=}
COMMON_SOURCE=${COMMON_SOURCE//\"}
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
echo "LOG_AE_SOURCE : $LOG_AE_SOURCE"
echo "COMMON_SOURCE : $COMMON_SOURCE"

readarray datasets -t array < ChiLib/HistosConfigFiles/ElectronMcSignalHistos.txt # $Chilib_path
N2=${#datasets[@]}
echo "nb lines in datasets= $N2"
var=0
for line in "${datasets[@]}"
do
  if [[ $line == *"ElectronMcSignalValidator/"* ]]; then
    arrLine=(${line//dator/ })
    let "var++"
  fi
done
echo "nb datasets in datasets= $var"
let "var--"
timeFolder2="$(date +"%Y%m%d-%H%M%S")_V2"

if [[ "$Choice" == "LLR" ]] 
  then
    echo "LLR"

    module reset
    source /usr/share/Modules/init/sh
    module use /opt/exp_soft/vo.gridcl.fr/software/modules/
    module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el9
    module load python/3.12.4 # torch included !
    source /opt/exp_soft/llr/root/v6.32-el9-gcc13xx-py3124/etc/init.sh

    echo " cd $LOG_SOURCE ==="
    cd $LOG_SOURCE
    options="-short" # -short -long -reserv

    # computing of the values
    echo '#############################################'
    echo '########## computing of the values ##########'
    echo '#############################################'
    #/opt/exp_soft/cms/t3/t3submit -8c -$options extractValuesTest.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName
    #. extractValuesTest.sh $LOG_SOURCE $LOG_KS_SOURCE $COMMON_SOURCE $FileName
    
    # initialization of the temps folders
    echo '#########################################################'
    echo '########## initialization of the temps folders ##########'
    echo '#########################################################'
    #/opt/exp_soft/cms/t3/t3submit -8c $options initAE.sh $LOG_AE_SOURCE $COMMON_SOURCE $FileName # -mail chiron@llr.in2p3.fr 
    #. initAE.sh $LOG_AE_SOURCE $COMMON_SOURCE $FileName
    echo "==="
    process_id=$!
    echo "PID: $process_id"
    wait $process_id

    # computation for each histo
    echo '################################################'
    echo '########## computation for each histo ##########'
    echo '################################################'
    for line in "${datasets[@]}"
    do
      if [[ $line == *"ElectronMcSignalValidator/"* ]]; then
        #echo '########## ' ${arrLine[1]} ' ##########'
        arrLine=(${line//dator/ })
        #/opt/exp_soft/cms/t3/t3submit -8c $options generateAETest.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGenerationTest_V2.py $FileName $var ${arrLine[1]} 'cpu' $timeFolder # $LOG_SOURCE  # -mail chiron@llr.in2p3.fr 
        . generateAETest.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGenerationTest_V2.py $FileName $var ${arrLine[1]} 'cpu' $timeFolder #&
      fi
    done
    
    # post operations of the temps folders
    echo '#########################################################'
    echo '########## post operation of the temps folders ##########'
    echo '#########################################################'
    #/opt/exp_soft/cms/t3/t3submit -8c $options postAE.sh $LOG_AE_SOURCE $COMMON_SOURCE $FileName # -mail chiron@llr.in2p3.fr 
    #. postAE.sh $LOG_AE_SOURCE $COMMON_SOURCE $FileName

    deactivate
fi

echo "END"

