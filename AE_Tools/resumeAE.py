#!/usr/bin/env python
# coding: utf-8

################################################################################
# AEGeneration : create a KS comparison (max diff) between the original curve 
# and the predicted one for different egamma validation releases.
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import datetime, time
import sys, os
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 2 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np
## WARNING pbm with torch
import torch
from torch.utils import data
from torch.nn.functional import normalize

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nAE Generation")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(Chilib_path))
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as dfo
from default import *
from rootValues import NB_EVTS
from defaultStd import *
from autoEncoders import *
from controlFunctions import *
from graphicAutoEncoderFunctions import *

from DecisionBox import *
DB = DecisionBox()

arrayValues = pd.DataFrame()
pValues1 = pd.DataFrame()
pValues2 = pd.DataFrame()
pValues3 = pd.DataFrame()
diff1 = []
rels = []
histos = []

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

nbBranches = len(branches) # [0:8]
print('there is {:03d} datasets'.format(nbBranches))

resultPath += '/' + str(NB_EVTS)
resultPath = checkFolderName(resultPath)
print('resultPath : {:s}'.format(resultPath))

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder = resultPath + checkFolderName(dfo.folder)
data_dir = folder + '/{:03d}'.format(nbFiles)
print('data_dir path : {:s}'.format(data_dir))
data_res = data_dir + '/AE_RESULTS/'
print('data_res path : {:s}'.format(data_res))

t = datetime.datetime.today()
timeFolder = time.strftime("%Y%m%d-%H%M%S")
timeFolder = '20221108-101911'

####### Loss prediction #######
folderName = data_res + createAEfolderName(hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, useHL3, useHL4, latent_size) # , timeFolder, nbFiles, branches[i]
checkFolder(folderName)
print('\nComplete folder name : {:s}'.format(folderName))

loopMaxValue = nbBranches #25 # nbBranches
for i in range(0, loopMaxValue):
    print('{:s}\n'.format(branches[i]))
    df = []
    fileName = "/predLossesValues_" + branches[i] + ".txt"
    Name = folderName + '/' + branches[i] + '/' + timeFolder + fileName
    if Path(Name).exists():
        print('{:s} exist'.format(Name))
        df = pd.read_csv(Name, header=None)
        print(df.head())
        print('end of %s' % branches[i])
        rels = df[1].to_numpy()
        arrayValues[branches[i]] = df[0]
    else:
        print('{:s} does not exist'.format(Name))
        continue

####### Pictures creation #######
folderPictures = folderName + '/Pictures/' + timeFolder
checkFolder(folderPictures)

#sortedRels = rels.sort() # sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
sortedRels = sorted(rels)
print(sortedRels)

print(arrayValues.head())
lignes, col = arrayValues.shape
print('[{:d}, {:d}]'.format(lignes, col))
aa = arrayValues.to_numpy()

for ind in range(0, lignes):
    fileName = folderPictures + '/Summary_' + sortedRels[ind].strip() + '_releaseEvolution.png'
    print(fileName)
    createSimplePicture(sortedRels[ind], aa[ind], ['nb of values', 'pred losses values'], fileName) #, branches)
fileName = folderPictures + '/Summary_AllReleases_Evolution.png'
createComplexPicture(sortedRels, aa, ['nb of values', 'pred losses values'], fileName)
fileName = folderPictures + '/Summary_AllReleases_EvolutionArray.png'
createComplexPicture2(sortedRels, aa, ['nb of values', 'pred losses values'], fileName, branches)

####### pValues #######
folderKS = data_dir 
checkFolder(folderKS)
folderPictures = folderKS + '/KS/'
checkFolder(folderPictures)
print('\KS folder name : {:s}'.format(folderKS))
loopMaxValue = len(sortedRels)
for i in range(0, loopMaxValue):
    print('{:s}\n'.format(sortedRels[i]))
    miniRel = sortedRels[i].strip()
    miniRel = miniRel[6:]
    fileName = "/histo_pValues_" + miniRel + "_v2.txt"
    Name = folderKS + fileName
    if Path(Name).exists():
        print('{:s} exist'.format(Name))
        df = pd.read_csv(Name, header=None)
        print(df.head())
        print('end of %s' % sortedRels[i])
        histos = df[0].to_numpy()
        pValues1[sortedRels[i]] = df[1]
        pValues2[sortedRels[i]] = df[2]
        pValues3[sortedRels[i]] = df[3]
    else:
        print('{:s} does not exist'.format(Name))
        continue

print(histos)
print(pValues1.head())
#print(pValues2.head())
#print(pValues3.head())
lignes, col = pValues1.shape
print('[{:d}, {:d}]'.format(lignes, col))
aa1 = pValues1.to_numpy().transpose()
aa2 = pValues2.to_numpy().transpose()
aa3 = pValues3.to_numpy().transpose()

for ind in range(0,loopMaxValue):
    fileName = folderPictures + '/Summary_pV1_releaseEvolution.png'
    createComplexPicture2(sortedRels, aa1, ['nb of values', 'pValues 1'], fileName, histos)
    fileName = folderPictures + '/Summary_pV2_releaseEvolution.png'
    createComplexPicture2(sortedRels, aa2, ['nb of values', 'pValues 2'], fileName, histos)
    fileName = folderPictures + '/Summary_pV3_releaseEvolution.png'
    createComplexPicture2(sortedRels, aa3, ['nb of values', 'pValues 3'], fileName, histos)


####### Differences #######
folderKS = data_dir #+ '/KS/'
checkFolder(folderKS)
folderPictures = folderKS + '/KS/'
checkFolder(folderPictures)
print('\KS folder name : {:s}'.format(folderKS))
differences = pd.DataFrame()
loopMaxValue = len(sortedRels)
for i in range(0, loopMaxValue):
    histos = []
    diff1 = []
    print('{:s}\n'.format(sortedRels[i]))
    miniRel = sortedRels[i].strip()
    miniRel = miniRel[6:]
    fileName = "/histo_differences_KScurve_" + miniRel + "__950_v2.txt"
    Name = folderKS + fileName
    if Path(Name).exists():
        print('{:s} exist'.format(Name))
        diff_file = open(Name, 'r')
        lines = diff_file.readlines()
        print('end of %s' % sortedRels[i])
        for elem in lines:
            a, b = elem.split(' : ')
            #print(a, b)
            histos.append(a)
            diff1.append(float(b))
        differences[sortedRels[i]] = diff1
    else:
        print('{:s} does not exist'.format(Name))
        continue
print(differences.head())
aa1 = differences.to_numpy().transpose()

fileName = folderPictures + '/Summary_differences_releaseEvolution.png'
createComplexPicture2(sortedRels, aa1, ['nb of values', 'differences'], fileName, histos)

print('end')

