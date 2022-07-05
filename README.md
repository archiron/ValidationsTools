# ValidationsTools

this repository get all the validations tools such as Kolmogorov-Smirnov (KS Tools) or AutoEncoders Tools (AE).

### Launching the ROOT files creation :
- git clone https://github.com/archiron/ValidationsTools ValidationsTools 
- cd ValidationsTools 
- git clone https://github.com/archiron/ChiLib_CMS_Validation ChiLib<br><br>
After this, one need to create the environment for the ROOT files creation. This is done with :
- cd ZEE_Flow
- cmsrel CMSSW_12_1_0_pre5 CMSSW_12_1_0_pre5
- cd CMSSW_12_1_0_pre5/src
- mkdir Kolmogorov
- depending of the site you are working on (the CC or the LLR site), you have to copy the ZZ_Flow/CCA or ZEE_Flow/LLR files onto the CMSSW_12_1_0_pre5/src/Kolmogorov folder.

### Launching the ROOT files creation :
return to the top of the Tools (i.e. ValidationsTools), then launch :
. createROOTFiles.sh  <br>
this script launch the creation of the ROOT files, using the step[1-4].py scripts.
<br>You can use own but they have to be similar to the existing ones (to be added later, waht to add ?).
<br> later create a link to an explanation of the common files used for the ROOT files creation (CommonFiles/path...).

### Reducing size :
When ALL ROOT files are created, launch (always from the top folder) :
. reduceROOTSize.sh <br>
this will reduce the size of the ROOT files (typically from 150/200 Mo to 1.5/2 Mo), keeping the name of the file.

### Extracting values : 
When all the ROOT files are created and reduced, we need to creates 1 file per histo with all the histo values of each ROOT file.
It is the first job of the extractValues[_init].sh scripts.
