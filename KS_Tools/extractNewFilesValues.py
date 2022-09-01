#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# extractNewFilesValues: create file with values for each histo 
# for different releases of the added ROOT files
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

from sys import argv

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

#ROOT.gSystem.Load("libFWCoreFWLite.so")
#ROOT.gSystem.Load("libDataFormatsFWLite.so")
#ROOT.FWLiteEnabler.enable()

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # LIB path
    print("step 4 - arg. 2 :", sys.argv[2]) # COMMON files path
    print("step 4 - arg. 3 :", sys.argv[3]) # RESULTFOLDER
    resultPath = sys.argv[3]
else:
    print("rien")
    resultPath = ''

Chilib_path = sys.argv[1]
sys.path.append(Chilib_path)
Common_path = sys.argv[2]
sys.path.append(Common_path)

from default import *
from controlFunctions import *
from graphicFunctions import getHisto, getHistoConfEntry, fill_Snew2, fill_Snew
from sources import *

import numpy as np

# get the branches for ElectronMcSignalHistos.txt
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = []
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

print("func_ExtractNewFilesVaues")

# these line for daltonians !
#seaborn.set_palette('colorblind')

dataPath = Common_path.replace('CommonFiles', 'DATA')
dataPath = checkFolderName(dataPath)
print(dataPath)

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)
folder += '{:03d}'.format(nbFiles)
folder = checkFolderName(folder)
print('folder après check : %s' % folder)
checkFolder(folder)

# get list of the added ROOT files
rootFolderName = dataPath # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')

print('we use the files :')
for item in rootFilesList:
    print('%s' % item)

N_histos = len(branches)
print('N_histos : %d' % N_histos)

KS_resume = folder + "branchesHistos_NewFiles.txt"
print("KSname 0 : %s" % KS_resume)
wKS_ = open(KS_resume, 'w')

tic = time.time()

for i in range(0, N_histos): # 1 histo for debug
        
    for file in rootFilesList:
        # extract release version
        fil = file.split('__')[2]
        fil = fil.split('-')[0]
        inputFile = dataPath + file
        rFile = ROOT.TFile(inputFile)
        h1 = getHisto(rFile, tp_1)
        print(fil + '/' + branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])

        d = getHistoConfEntry(histo_1)
        #print("d = {}".format(d))

        texttoWrite = fil + "," + branches[i] + ","
        s_new = fill_Snew2(d, histo_1)
        
        for elem in s_new:
            texttoWrite += str(elem) + ","
        texttoWrite = texttoWrite[:-1] # remove last char
        texttoWrite += '\n'
        wKS_.write(texttoWrite)

wKS_.close()
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

