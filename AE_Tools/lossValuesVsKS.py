#! /usr/bin/env python
#-*-coding: utf-8 -*-

##################################################################################
# lossValuesVsKS: a tool to compare Kolmogorov-Smirnov values (max diff & pValues)
# with AutoEncoder for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
##################################################################################

from genericpath import exists
from multiprocessing import Condition
import os,sys
import time
import importlib
import importlib.machinery
import importlib.util

#import seaborn # only with cmsenv on cca.in2p3.fr

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

print("func_CreateLossVsKSComp")

# these line for daltonians !
#seaborn.set_palette('colorblind')

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

from default import *
from defaultStd import *
from autoEncoders import *
from controlFunctions import *
from graphicAutoEncoderFunctions import *

arrayKSValues = []
rels = []

# get the branches for ElectronMcSignalHistos.txt
branches = []
branchPath = Chilib_path + "HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, branchPath)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

N_histos = len(branches)
print('there is {:03d} histos : '.format(N_histos))

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder = resultPath + checkFolderName(folder)
data_dir = folder + '/{:03d}'.format(nbFiles)
print('data_dir path : {:s}'.format(data_dir))
data_res = data_dir + '/AE_RESULTS/'
print('data_res path : {:s}'.format(data_res))

# get list of added ROOT files
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList)) + ' added ROOT files')
for item in rootFilesList:
    #print('%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])
sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
for elem in sortedRels:
    print(elem)

# get list of text files
pathKSFiles = data_dir
print('KS path : %s' % pathKSFiles)
KSlistFiles = []
tmp = getListFiles(pathKSFiles, 'txt')
for elem in tmp:
    if (elem[5:10] == '_diff'): # to keep only histo_differences_KScurves files
        KSlistFiles.append(elem)
print(KSlistFiles, len(KSlistFiles))
    
for item in KSlistFiles:
    print('file : %s' % item)
    aa = item[26:-9]
    fileName = pathKSFiles + '/' + item
    file1 = open(fileName, 'r')
    bb = file1.readlines()
    for elem in bb:
        tmp = []
        cc = elem.split(' : ')
        tmp = [cc[0], aa, float(cc[1][:-1])]
        arrayKSValues.append(tmp)
sortedArrayKSValues = sorted(arrayKSValues, key = lambda x: x[0]) # gives an array with releases sorted
#for elem in sortedArrayKSValues:
#    print(elem)

if (len(KSlistFiles) != len(rootFilesList)):
    print('you must have the same number of KS files than releases')
    exit()
else:
    print('we have the same number of KS files than releases')

    tic = time.time()

# create the folder name for saving the picture
folderName = data_res + createAEfolderName(hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, useHL3, useHL4, latent_size) # , nbFiles, branches[i]
print('\nComplete folder name : {:s}'.format(folderName))
checkFolder(folderName)

i = 0
loopMaxValue = N_histos # N_histos
for i in range(0, loopMaxValue):
    fileName = folderName + 'predLossesValues_' + branches[i] + '.txt' # penser à virer le multi
    #affiche = colorText('{:03d}/{:03d}'.format(i,loopMaxValue-1), "green")
    print(fileName)

'''
data_res = 'RESULTS'
rootFolderName = 'DATASETS/NewFiles'

        if exists(fileName):
            print('predLosses OK')
        else:
            print('predLosses KO')
            continue

        file2 = open(fileName, 'r')
        dd = file2.readlines()
        arrLoss = []
        for elem in dd:
            ee = elem.split(', ')
            arrLoss.append([ee[1][:-1], float(ee[0])])
        file2.close()
        sortedArrLoss = sorted(arrLoss, key = lambda x: x[0]) # gives an array with releases sorted
        #print('sortedLoss : ', sortedArrLoss)
        print('sortedArrLoss OK')

        KSLoss = []
        for elem in sortedArrayKSValues:
            if elem[0] == branches[i]:
                #print(elem,elem[1], elem[2])
                KSLoss.append([elem[1], elem[2]])
        sortedKSLoss = sorted(KSLoss, key = lambda x: x[0]) # gives an array with releases sorted
        #print('sortedKSLoss : ', sortedKSLoss)
        if (len(sortedKSLoss) > 0):
            print('sortedKSLoss OK')
        else:
            print('sortedKSLoss KO')
            continue

        labels = []
        Val1 = []
        Val2 = []
        print('arr ok')
        for j in range(0, len(sortedRels)):
            #print(j, sortedRels[j], [sortedArrLoss[j][1], sortedKSLoss[j][1]])
            Val1.append(sortedArrLoss[j][1])
            Val2.append(sortedKSLoss[j][1])
            labels.append(sortedRels[j][1])

        #print(Val1)
        #print(Val2)
        #print(labels)

        #print(branches[i])
        fileName = folderName + 'compLossesValuesVsKS_' + branches[i] + '.png'
        title = r"$\bf{" + branches[i] + "}$" + ' : Losses values vs KS values as function of releases.'
        createCompLossesPicture2Axis(labels, Val1, Val2, fileName, title)
        i += 1
'''

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("End !")
