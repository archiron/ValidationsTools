#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# zeeKSCompare: create one file per release with max diff for each histo
# for different egamma validation releases
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

from genericpath import exists
import os,sys
import imp, importlib
import importlib.machinery
import importlib.util
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 2 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 4 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import matplotlib

matplotlib.use('agg')

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("func_Extract")

#blu = imp.load_source(filePaths, commonPath+filePaths)
#print('DATA_SOURCE : %s' % blu.DATA_SOURCE)
#resultPath = blu.RESULTFOLDER # checkFolderName(blu.RESULTFOLDER)
#print('result path : {:s}'.format(resultPath))

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
sys.path.append(Chilib_path)
sys.path.append(commonPath)

sys.path.append(Chilib_path)
import default as dfo
from default import *
from sources import *
from graphicAutoEncoderFunctions import *
from controlFunctions import *

folder = checkFolderName(dfo.folder)
resultPath = checkFolderName(resultPath)

# get the branches for ElectronMcSignalHistos.txt
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

rels = []

N_histos = len(branches)
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

# get list of the added ROOT files
rootFolderName = blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('we use the files :')
for item in rootFilesList:
    print('%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])
sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)
folder += '{:03d}'.format(nbFiles)

tic = time.time()

df1 = pd.DataFrame()
df2 = pd.DataFrame()

for elem in sortedRels:
    print(elem)
    rel = elem[1]

    # get the KS file datas
    KS_diffName = folder + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "_v2.txt"
    pValue_Name = folder + "/histo_pValues" + "_" + rel + "_v2.txt"
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
    pictureName = folder + 'KS/comparison_KS_values_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : Comparison of KS values as function of releases.'
    createCompLossesPicture(labels,val, pictureName, title)
    #if ind == 2:
    #    break

'''
    for ind in df2.index:
        print(ind)
        a = df2.iloc[ind].to_numpy()
        branch = a[0]
        print(branch)
        val = list(a[1:])
        #print(val)
        pictureName = folder + 'KS/comparison_pValues_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
        print(pictureName)
        title = r"$\bf{" + branch + "}$" + ' : Comparison of KS pValues as function of releases.'
        createCompPValuesPicture(labels,val, pictureName, title)
        #if ind == 2:
        #    break
'''
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))


