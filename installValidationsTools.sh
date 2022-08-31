#!/bin/sh
# This file is called . installValidationsTools.sh

Release="CMSSW_12_1_0_pre5"

# tester si on est bien dans ValidationsTools
echo "Cloning ChiLib"
git clone https://github.com/archiron/ChiLib_CMS_Validation ChiLib

cd ZEE_Flow
echo "installing $Release"
cmsrel $Release $Release
cd $Release/src
mkdir Kolmogorov
cd ../../
chmod 755 ZEE_Flow/$release/src/Kolmogorov/*.sh

aa=$PWD
echo "actual path : $aa"

STR=$aa
Choice='Local'
for SUB in 'llr' 'pbs' 'cern'
do
  if [[ "$STR" == *"$SUB"* ]]; then
    echo "It's $SUB here.";
    Choice=${SUB^^};
  fi
done

if [[ "$Choice" == "PBS" ]]; then
    Choice="CCA";
fi
echo "Choice is : $Choice"

echo "Copying files into Kolmogorov folder"
cp $Choice/* $Release/src/Kolmogorov

cd ../ # back to /ValidationsTools

echo "End of installation"

