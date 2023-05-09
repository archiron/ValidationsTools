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

import os,sys,shutil
import importlib
import importlib.machinery
import importlib.util
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

from sys import argv

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 2 :", sys.argv[2]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
else:
    print("rien")

print("\nextractNewFilesValues")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('pathChiLib : {:s}'.format(pathChiLib))
print('pathCommonFiles : {:s}'.format(pathCommonFiles))
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

import default as dfo
from default import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import getHisto, getHistoConfEntry, fill_Snew2 #, fill_Snew
from sources import *

import numpy as np

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

print("func_ExtractNewFilesValues")
pathNb_evts = pathBase + '/' + str(NB_EVTS)
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))

# these line for daltonians !
#seaborn.set_palette('colorblind')

pathDATA = pathCommonFiles.replace('CommonFiles', 'DATA') # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
pathDATA = checkFolderName(pathDATA)
print(pathDATA)

# get list of generated ROOT files
rootFilesList_0 = getListFiles(pathNb_evts, 'root')
if (len(rootFilesList_0) ==0 ):
    print('there is no generated ROOT files')
    exit()
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)
pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathNb_files = pathCase + '{:03d}'.format(nbFiles)
print('folder après check : %s' % pathNb_files)
checkFolder(pathNb_files)

# get list of the added ROOT files
rootFilesList = getListFiles(pathDATA, 'root')

print('we use the files :')
print('there is ' + '{:03d}'.format(len(rootFilesList)) + ' added ROOT files')
if (len(rootFilesList) == 0):
    print('no added ROOT files to work with. Existing.')
    exit()
for item in rootFilesList:
    print('%s' % item)

N_histos = len(branches)
print('N_histos : %d' % N_histos)

KS_resume = pathNb_files + "branchesHistos_NewFiles.txt"
print("KSname 0 : %s" % KS_resume)
wKS_ = open(KS_resume, 'w')

tic = time.time()

for i in range(0, N_histos): # 1 histo for debug
        
    for file in rootFilesList:
        # extract release version
        fil = file.split('__')[2]
        fil = fil.split('-')[0]
        inputFile = pathDATA + file
        rFile = ROOT.TFile(inputFile)
        h1 = getHisto(rFile, tp_1)
        print(fil + '/' + branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])

        if (histo_1):
            print('%s OK' % branches[i])
            d = getHistoConfEntry(histo_1)
            #print("d = {}".format(d))

            texttoWrite = fil + "," + branches[i] + ","
            s_new = fill_Snew2(d, histo_1)
        
            for elem in s_new:
                texttoWrite += str(elem) + ","
            texttoWrite = texttoWrite[:-1] # remove last char
            texttoWrite += '\n'
            wKS_.write(texttoWrite)
        else:
            print('%s KO' % branches[i])

wKS_.close()
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

