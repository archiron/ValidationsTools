#!/bin/sh
# This file is called . /zee_Extract_init.sh

### SLURM VARIATION ###

LOG_SOURCE_WORK='/pbs/home/c/chiron/private/KS_Tools/GenExtract/'
LOG_SOURCE_START='/pbs/home/c/chiron/private/ZEE_Flow/CMSSW_12_1_0_pre5/src/'

LOG_OUTPUT='/sps/cms/chiron/TEMP/'
RESULTFOLDER='/sps/cms/chiron/CMSSW_12_1_0_pre5-16c-11/'

cd $LOG_SOURCE_START
echo "LOG_SOURCE_WORK : $LOG_SOURCE_WORK"
echo "LOG_SOURCE_START : $LOG_SOURCE_START"

sbatch -L sps zee_Extract.sh $LOG_SOURCE_START $LOG_SOURCE_WORK $RESULTFOLDER

