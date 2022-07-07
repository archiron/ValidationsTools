#!/bin/shchichi_serial_job_test
# This file is called . installValidationsTools.sh

Release="CMSSW_12_1_0_pre5"

# tester si on est bien dans ValidationsTools
git clone https://github.com/archiron/ChiLib_CMS_Validation ChiLib

cd ZEE_Flow
cmsrel $Release $Release
cd $Release/src
mkdir Kolmogorov
cd ../../
chmod 755 ZEE_Flow/$release/src/Kolmogorov/*.sh

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

if [[ "$Choice" == "PBS" ]]; then
    Choice="CCA";
fi
echo "Choice is : $Choice"

cp $Choice/* $Release/src/Kolmogorov

cd ../ # back to /ValidationsTools

echo "End of installation"

