#!/bin/sh
# This file is called ./zee_flow_init.sh

LOG_SOURCE='/pbs/home/c/chiron/private/KS_Tools/GenExtract/'
LOG_OUTPUT='/sps/cms/chiron/TEMP/'
RESULTFOLDER='/sps/cms/chiron/CMSSW_12_1_0_pre5-16c-1'

echo "LOG_SOURCE : $LOG_SOURCE"

cd $LOG_SOURCE
#source /afs/cern.ch/cms/cmsset_default.sh
eval `scramv1 runtime -sh`
cd -
qsub -l sps=1 -P P_cmsf -pe multicores 16 -q mc_long -o $LOG_OUTPUT zee_Extract.sh $LOG_SOURCE $RESULTFOLDER

