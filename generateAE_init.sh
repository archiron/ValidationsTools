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

readarray datasets -t array < ChiLib/HistosConfigFiles/ElectronMcSignalHistos.txt # $Chilib_path
N2=${#datasets[@]}
echo "nb lines in datasets= $N2"
var=0
for line in "${datasets[@]}"
do
  if [[ $line == *"ElectronMcSignalValidator/"* ]]; then
    echo $line
    arrLine=(${line//dator/ })
    echo "${arrLine[1]} - $var"
    let "var++"
  fi
done
let "var--"
echo "nb datasets in datasets= $var"

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
    #/opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGeneration.py $FileName ''
    #/opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE resumeAE.py $FileName ''
    #/opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V1.py $FileName 'even'
    #/opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V1.py $FileName 'odd'
    #/opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V1.py $FileName 'both' # odd + even
    #/opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V2.py $FileName ''
    #/opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V3.py $FileName 'cpu'
    #/opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V4.py $FileName 'cpu' # $LOG_SOURCE 
    for line in "${datasets[@]}"
    do
      if [[ $line == *"ElectronMcSignalValidator/"* ]]; then
        #echo $line
        arrLine=(${line//dator/ })
        echo "${arrLine[1]}"
        /opt/exp_soft/cms/t3/t3submit -8c $options generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGeneration.py $FileName $var ${arrLine[1]} 'cpu' $timeFolder # $LOG_SOURCE 
      fi
    done
elif [[ "$Choice" == "PBS" ]] 
  then
    echo "PBS"
    module load Programming_Languages/python/3.9.1
    source /pbs/home/c/chiron/private/ValidationsTools/ValidationsTools/bin/activate 
    cd $LOG_SOURCE
    #sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGeneration.py $FileName ''
    #sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE resumeAE.py $FileName ''
    #sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V1.py $FileName 'even'
    #sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V1.py $FileName 'odd'
    #sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V1.py $FileName 'both'
    #sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V2.py $FileName ''
    #sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V3.py $FileName 'cpu'
    #sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGen-V4.py $FileName 'cpu' # $LOG_SOURCE 
    for line in "${datasets[@]}"
    do
      if [[ $line == *"ElectronMcSignalValidator/"* ]]; then
        #echo $line
        arrLine=(${line//dator/ })
        echo "${arrLine[1]}"
        sbatch -L sps -n 4 --mem=16000 -t 4-0:0:0 -J $JobName -o $output generateAE.sh $LOG_AE_SOURCE $COMMON_SOURCE AEGeneration.py $FileName $var ${arrLine[1]} 'cpu' $timeFolder # $LOG_SOURCE 
      fi
    done
    deactivate
fi

echo "END"

