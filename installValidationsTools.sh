#!/bin/shchichi_serial_job_test
# This file is called . installValidationsTools.sh

Release="CMSSW_12_1_0_pre5"

# tester si on est bien dans ValidationsTools
git clone https://github.com/archiron/ChiLib_CMS_Validation ChiLib

cd ZEE_Flow
cmsrel $Release $RElease
cd $Release/src
mkdir Kolmogorov
cd ../../

aa=$PWD
echo "actual path : $aa"

STR=$aa
Choice='Local'
for SUB in 'llr' 'pbs'
do
  if [[ "$STR" == *"$SUB"* ]]; then
    echo "It's $SUB there.";
    Choice=${SUB^^};
  fi
done

if [[ "$Choice" == "pbs" ]]; then
    $Choice="CCA";
fi
if [[ "$Choice" == "llr" ]]; then
    $Choice="LLR";
fi
echo "Choice is : $Choice"

cp $Choice/* $Release/src/Kolmogorov

cd ../ # back to /ValidationsTools

echo "End of installation"

