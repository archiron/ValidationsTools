#!/bin/sh
# This file is called ./zee_flow_init.sh

### QSUB VARIATION ###

LOG_SOURCE_WORK='/pbs/home/c/chiron/private/KS_Tools/GenExtract/'
LOG_SOURCE_START='/pbs/home/c/chiron/private/ZEE_Flow/CMSSW_12_1_0_pre5/src/'

LOG_OUTPUT='/sps/cms/chiron/TEMP/'
RESULTFOLDER='/sps/cms/chiron/CMSSW_12_1_0_pre5-16c-5'

cd $LOG_SOURCE_START
echo "LOG_SOURCE_WORK : $LOG_SOURCE_WORK"
echo "LOG_SOURCE_START : $LOG_SOURCE_START"

#source /afs/cern.ch/cms/cmsset_default.sh
eval `scramv1 runtime -sh`
cd -
qsub -l sps=1 -P P_cmsf -pe multicores 4 -q mc_long -o $LOG_OUTPUT extractValues.sh $LOG_SOURCE_START $LOG_SOURCE_WORK $RESULTFOLDER

