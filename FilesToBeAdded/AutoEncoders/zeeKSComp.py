#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# zeeKSComp: create one file per release with max diff for each histo
# for different egamma validation releases
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

from genericpath import exists
import os,sys
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('agg')

#Chilib_path = '../../ChiLib'
Chilib_path = '/pbs/home/c/chiron/private/KS_Tools/ChiLib_CMS_Validation'
sys.path.append(Chilib_path)
import default as dfo
from default import *
from sources import *
from graph import *
from controlFunctions import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

def func_CompareKS(br):
    rels = []

    print("func_Extract")
    dfo.folderName = checkFolderName(dfo.folderName)
    dfo.folder = checkFolderName(dfo.folder)
    
    N_histos = len(br)
    print('N_histos : %d' % N_histos)
    
    # create folder 
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST: # the folder did not exist
                raise  # raises the error again
        print('Creation of %s release folder\n' % folder)
    else:
        print('Folder %s already created\n' % folder)

    # get list of files
    #rootFolderName = '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
    LOG_SOURCE_WORK = '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles/'
    rootFilesList = getListFiles(LOG_SOURCE_WORK, 'root')
    print('we use the files :')
    for item in rootFilesList:
        print('%s' % item)
        b = (item.split('__')[2]).split('-')
        rels.append([b[0], b[0][6:]])
    sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

    tic = time.time()

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for elem in sortedRels:
        print(elem)
        rel = elem[1]

        # get the KS file datas
        KS_diffName = dfo.folder + '{:d}'.format(nbFiles) + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
        pValue_Name = dfo.folder + '{:d}'.format(nbFiles) + "/histo_pValues" + "_" + rel + ".txt"
        if exists(KS_diffName):
            print('%s existe'%KS_diffName)
        else:
            print('%s n\'existe pas'%KS_diffName)
        if exists(KS_diffName):
            print('%s existe'%pValue_Name)
        else:
            print('%s n\'existe pas'%pValue_Name)

        wKS0 = open(KS_diffName, 'r').readlines()
        wKS1 = open(pValue_Name, 'r').readlines()
        print(len(wKS0))
        print(len(wKS1))
        tmpArr1 = []
        tmpArr2 = []
        for line in wKS0:
            #print(len(line))
            aa = line.split(' : ')
            tmpArr1.append(aa[0])
            tmpArr2.append(float(aa[1][:-1]))
        df1['index'] = tmpArr1
        df1[rel] = tmpArr2

        tmpArr1 = []
        tmpArr2 = []
        tmpArr3 = []
        tmpArr4 = []
        for line in wKS1:
            #print(len(line))
            aa = line.split(', ')
            tmpArr1.append(aa[0])
            tmpArr2.append(float(aa[1]))
            tmpArr3.append(float(aa[2]))
            tmpArr4.append(float(aa[3]))#[:-1]
        df2['index'] = tmpArr1
        df2[rel+'_pV1'] = tmpArr2
        df2[rel+'_pV2'] = tmpArr3
        df2[rel+'_pV3'] = tmpArr4 

    print(df1.head(5))
    print()
    print(df2.head(5))
    
    labels = list(df1)[1:]
    print(labels)
    (N_histos, _) = df1.shape
    ########################################################################
    # Il faut verifier que chaque colonne a la meme hauteur que les autres !
    # Il faut extraire les bonnes colonnes !
    ########################################################################
    for ind in df1.index:
        print(ind)
        a = df1.iloc[ind].to_numpy()
        branch = a[0]
        print(branch)
        val = list(a[1:])
        #print(val)
        pictureName = dfo.folder + 'KS/comparison_KS_values_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
        print(pictureName)
        title = r"$\bf{" + branch + "}$" + ' : Comparison of KS values as function of releases.'
        createCompLossesPicture(labels,val, pictureName, title)
        #if ind == 2:
        #    break

    for ind in df2.index:
        print(ind)
        a = df2.iloc[ind].to_numpy()
        branch = a[0]
        print(branch)
        val = list(a[1:])
        #print(val)
        pictureName = dfo.folder + 'KS/comparison_pValues_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
        print(pictureName)
        title = r"$\bf{" + branch + "}$" + ' : Comparison of KS pValues as function of releases.'
        createCompPValuesPicture(labels,val, pictureName, title)
        #if ind == 2:
        #    break

    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc-tic))

    return

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
    branches = getBranches(tp_1, source)
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    func_CompareKS(branches)  # create the KS files from histos datas

    print("Fin !")

