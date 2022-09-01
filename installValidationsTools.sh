#!/bin/sh
# This file is called . installValidationsTools.sh nomRelease
# if nomRelease is empty the default Release is used.

defaultRelease="CMSSW_12_1_0_pre5"
Release=$defaultRelease
if [[ $# -eq 1 ]]; then
  Release=$1
fi
echo "use the $Release release"

# tester si on est bien dans ValidationsTools
echo "Cloning ChiLib"
git clone https://github.com/archiron/ChiLib_CMS_Validation ChiLib

cd ZEE_Flow
chmod 755 *.sh
echo "installing $Release"
cmsrel $Release $Release
cd $Release/src
mkdir Kolmogorov
cd ../../

echo "Copying files into Kolmogorov folder"
cp createROOTFiles.sh $Release/src/Kolmogorov
cp step*.py $Release/src/Kolmogorov

cd ../ # back to /ValidationsTools

echo "End of installation"

