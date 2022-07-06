# ValidationsTools

this repository get all the validations tools such as Kolmogorov-Smirnov (KS Tools) or AutoEncoders Tools (AE).

### Launching the installation :
- git clone https://github.com/archiron/ValidationsTools ValidationsTools 
- cd ValidationsTools <br>
then launch : . installValidationsTools.sh<br>
This install the library (ChiLib), the release file env (cmsrel $Release $Release) and copy the ZEE_Flow/CCA or ZEE_Flow/LLR files (depending of the site you are working on - the CC or the LLR site) onto the $Release/src/Kolmogorov folder. $Release is the release you want working with.

### Launching the ROOT files creation :
From the top of the Tools (i.e. ValidationsTools), then launch :
. createROOTFiles.sh  <br>
This script launch the creation of the ROOT files, using the step[1-4].py scripts.
<br>You can use own but they have to be similar to the existing ones (to be added later, waht to add ?).
<br> later create a link to an explanation of the common files used for the ROOT files creation (CommonFiles/path...).

### Reducing size :
When ALL ROOT files are created, launch (always from the top folder) :
. reduceROOTSize.sh <br>
this will reduce the size of the ROOT files (typically from 150/200 Mo to 1.5/2 Mo), keeping the name of the file.

### Extracting values : 
When all the ROOT files are created and reduced, we need to creates 1 file per histo with all the histo values for each ROOT file.
It is the first job of the extractValues[_init].sh scripts.

It can be launched with : .  extractValues_init.sh<br>
and create into the RESULTFOLDER (see 4) a lot of text files. All ROOT files are read, and then for each histo, the values of the histo curve is stored into an array, ROOT file after ROOT file.<br>
Once we have all the ROOT files read, the array is stored into a text file dedicated to this histo.<br>
