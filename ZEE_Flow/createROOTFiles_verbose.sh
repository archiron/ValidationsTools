#!/bin/sh
# This file is called ./zee_flow.sh for CCA computers.
###################
## CCA computers ##
###################

echo "nb : $1"
echo "chemin : $2"
echo "nb evts : $3"
echo "result folder : $4"
echo "initial SEED : $5"
echo ""

LOG_SOURCE=$2
echo "Step 1 in : $LOG_SOURCE"

name1="step1_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name2="step2_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name31="step3_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name32="step3_inDQM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name33="step3_inMINIAODSIM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
name34="step3_inNANOEDMAODSIM_$(printf "%04d" $3)_$(printf "%03d" $1).root"
#name4="DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_$(printf "%04d" $3)_$(printf "%03d" $1).root"
echo $name1
echo $name2
echo $name31
echo $name32
echo $name33
echo $name34

cd $2
eval `scramv1 runtime -sh`
cd -
ls
pwd
option1="ZEE_14TeV_TuneCP5_cfi --beamspot DBrealistic --conditions auto:phase1_2024_realistic --datatier GEN-SIM --era Run3_2024 --eventcontent FEVTDEBUG --fileout file:step1.root --geometry DB:Extended --nStreams 2 --nThreads 8 --number 10 --python_filename step_1_cfg.py --relval 9000,100 --step GEN,SIM"
cmsDriver.py $option1
#ls
option2="step2 --conditions auto:phase1_2024_realistic --datatier GEN-SIM-DIGI-RAW --era Run3_2024 --eventcontent FEVTDEBUGHLT --filein file:step1.root --fileout file:step2.root --geometry DB:Extended --nStreams 2 --nThreads 8 --number 10 --python_filename step_2_cfg.py --step DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2024"
cmsDriver.py $option2
ls
#rm step1.root
option3="step3 --conditions auto:phase1_2024_realistic --datatier GEN-SIM-RECO,MINIAODSIM,NANOAODSIM,DQMIO --era Run3_2024 --eventcontent RECOSIM,MINIAODSIM,NANOEDMAODSIM,DQM --filein file:step2.root --fileout file:step3.root --geometry DB:Extended --nStreams 2 --nThreads 8 --number 10 --python_filename step_3_cfg.py --step RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,NANO,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@nanoAODDQM"
cmsDriver.py $option3
#rm step2.root

option4="step4 --conditions auto:phase1_2024_realistic --era Run3_2024 --filein file:step3_inDQM.root --fileout file:step4.root --filetype DQM --geometry DB:Extended --mc --nStreams 2 --number 10 --python_filename step_4_cfg.py --scenario pp --step HARVESTING:@standardValidation+@standardDQM+@ExtraHLT+@miniAODValidation+@miniAODDQM+@nanoAODDQM"
cmsDriver.py $option4
#rm step3*.root
ls
mv DQM*.root $4
