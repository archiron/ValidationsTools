# ValidationsTools

this repository get all the validations tools such as Kolmogorov-Smirnov (KS Tools) or AutoEncoders Tools (AE).
KS : talk about first dev (1 release vs 1 reference) to (1 reference vs lot of releases). pValues.
From the values of an histogram, we can construct an integrated cumulative curve into one we can take the maximum of the difference between 2 consecutives values. Repeating this operation with a lot of ROOT files (200, 500 or 1000) we can construct a KS curve.
From this curve we can extract a pValue to obtain an idea of the validity of the histo and then the release.

AE : analyze on a reference and comparison with a lot of releases. The precedent explanation was made for one release vs 1 reference. Keeping the reference we can compare with a lot of releases (here about 12).
From the ROOT files presented above we can train an autoencoder and predict the result of a given entry for one release. Doing this for a tenth of releases we can obtain a comparison, function of the releases, for the pValues or the differences at the end of the AE.

All thoses files are located into a folder named ValidationsTools which contains at least 5 folders :
- Doc : containing this documentation,
- ZEE_Flow : folder used for the creation of the ROOT files,
- KS_Tools : folder used for the KS pictures and files,
- AutoEncoder : containing the autoencoder tools and results,
- DATA : empty at the installation. Contain the « official » ROOT files, i.e. the releases ROOT files used for the comparison mentioned above.

### Launching the installation :
- git clone https://github.com/archiron/ValidationsTools ValidationsTools 
- cd ValidationsTools <br>
- chmod 755 *.sh
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

It can be launched with : .  extractValues_init.sh and create into the RESULTFOLDER a lot of text files. 
<br>All ROOT files are read, and then for each histo, the values of the histo curve is stored into an array, ROOT file after ROOT file.<br>
Once we have all the ROOT files read, the array is stored into a text file dedicated to this histo.<br>

### Checking ROOT files :
. checkRootFiles_init.sh

### Create some files and pictures
. createFiles_init.sh

### Comparing pValues for all added ROOT files :
. KScompare_init.sh

### Generating AE pictures :
. generateAE_init.sh

# Notes

RESULTFOLDER : path where you want the created ROOT files are located. It also contain the text files for each histo.
