# ValidationsTools

this repository get all the validations tools such as Kolmogorov-Smirnov (KS Tools) or AutoEncoders Tools (AE).

- git clone https://github.com/archiron/ValidationsTools ValidationsTools 
- cd ValidationsTools 
- git clone https://github.com/archiron/ChiLib_CMS_Validation ChiLib 

Launching the ROOT files creation :
createROOTFiles.sh  

When ALL ROOT files are created, launch :
reduceROOTSize.sh 

When all the ROOT files are created and reduced, we need to creates 1 file per histo with all the histo values of each ROOT file.
It is the first job of the extractValues[_init].sh scripts.
