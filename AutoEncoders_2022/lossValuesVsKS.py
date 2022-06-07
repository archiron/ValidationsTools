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

#import seaborn # only with cmsenv on cca.in2p3.fr

from graph import *
from controlFunctions import *
from defaultStd import *
from default import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

def func_CreateLossVsKSComp(branches, nbFiles):
    data_res = 'RESULTS'

    arrayKSValues = []
    rels = []

    print("func_CreateLossVsKSComp")
    N_histos = len(branches)
    print('N_histos : %d' % N_histos)

    # get list of text files
    pathKSFiles = 'DATASETS/KS' + '/{:03d}'.format(nbFiles)
    print('KS path : %s' % pathKSFiles)
    KSlistFiles = []
    tmp = getListFiles(pathKSFiles, 'txt')
    for elem in tmp:
        if (elem[6:10] == 'diff'): # to keep only histo_differences_KScurves files
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
    for elem in sortedArrayKSValues:
        print(elem)

    # get list of ROOT files
    rootFolderName = 'DATASETS/NewFiles'
    rootFilesList = getListFiles(rootFolderName, 'root')
    #print('we use the files :')
    for item in rootFilesList:
        #print('%s' % item)
        b = (item.split('__')[2]).split('-')
        rels.append([b[0], b[0][6:]])
    sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
    #for elem in sortedRels:
    #    print(elem)

    if (len(KSlistFiles) != len(rootFilesList)):
        print('you must have the same number of KS files than releases')
        exit()

    tic = time.time()

    i = 0
    loopMaxValue = N_histos # N_histos
    for i in range(0, loopMaxValue):
        # create the folder name for saving the picture
        folderName = data_res+"/HL_1.{:03d}".format(hidden_size_1) + "_HL_2.{:03d}".format(hidden_size_2)
        if useHL3 == 1:
            folderName += "_HL_3.{:03d}".format(hidden_size_3)
        if useHL4 == 1:
            folderName += "_HL_3.{:03d}".format(hidden_size_3) # if we kept useHL3=0, add HL3 & HL4 !
            folderName += "_HL_4.{:03d}".format(hidden_size_4)
        folderName += "_LT.{:02d}".format(latent_size) + '/' + "{:04d}".format(nbFiles)
        folderName += '/' + branches[i] + '/'
        affiche = colorText('{:03d}/{:03d}'.format(i,loopMaxValue-1), "green")
        print('\n{:s} : {:s}'.format(affiche, folderName))
        checkFolder(folderName)

        fileName = folderName + 'predLossesValues_' + branches[i] + '.txt' # penser à virer le multi
        print(fileName)
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

        '''print(Val1)
        print(Val2)
        print(labels)'''

        #print(branches[i])
        fileName = folderName + 'compLossesValuesVsKS_' + branches[i] + '.png'
        title = r"$\bf{" + branches[i] + "}$" + ' : Losses values vs KS values as function of releases.'
        createCompLossesPicture2Axis(labels, Val1, Val2, fileName, title)
        i += 1
    
    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc-tic))

    # print nb of red/green lines

    return

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    branchPath = "DATASETS/ElectronMcSignalHistos.txt"
    branches = getBranches(tp_1, branchPath)
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    # nb of files to be used

    func_CreateLossVsKSComp(branches, nbFiles)  # create the KS files from histos datas

    print("Fin !")

