#!/bin/sh
# This file is called ./zee_flow_init.sh

LOG_SOURCE=$PWD 
NB_EVTS=9000
RESULTFOLDER="/data_CMS/cms/chiron/HGCAL/TEST_12_1_0_pre5bb"

echo "LOG_SOURCE : $LOG_SOURCE"

cd $LOG_SOURCE
eval `scramv1 runtime -sh`
ls

for (( i=102; i<150; i++ )) # only for step1
do
   /opt/exp_soft/cms/t3_tst/t3submit -8c -long zee_flow.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
   #/opt/exp_soft/cms/t3/t3submit -8c -short zee_flow.sh $i $LOG_SOURCE $NB_EVTS $RESULTFOLDER
done

 
