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

from sys import argv

#import seaborn # only with cmsenv on cca.in2p3.fr

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
    print("step 4 - arg. 2 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 4 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
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

Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(Chilib_path))
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as dfo
from default import *
from controlFunctions import *
from sources import *
from graphicAutoEncoderFunctions import *
from graphicFunctions import getHisto, getHistoConfEntry, fill_Snew2, fill_Snew

resultPath = checkFolderName(resultPath)
folder = resultPath + checkFolderName(dfo.folder)

# get the branches for ElectronMcSignalHistos.txt
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

rels = []
tmp_branches = []
nb_ttl_histos = []

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
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('we use the files :')
for item in rootFilesList:
    tmp_branch = []
    nbHistos = 0
    print('%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])
    f_root = ROOT.TFile(rootFolderName + item)
    h1 = getHisto(f_root, tp_1)
    for i in range(0, N_histos): # 1 N_histos histo for debug
        histo_1 = h1.Get(branches[i])
        d = getHistoConfEntry(histo_1)
        s_tmp = fill_Snew2(d, histo_1)
        #s_tmp = fill_Snew(histo_1)
        if (s_tmp.min() < 0.):
            print('pbm whith histo %s, min < 0' % branches[i])
        elif (np.floor(s_tmp.sum()) == 0.):
            print('pbm whith histo %s, sum = 0' % branches[i])
        else:
            nbHistos += 1
            tmp_branch.append(branches[i])
    nb_ttl_histos.append(nbHistos)
    tmp_branches.append(tmp_branch)

print('nb_ttl_histos : ', nb_ttl_histos)
if(len(set(nb_ttl_histos))==1):
    print('All elements are the same with value {:d}.'.format(nb_ttl_histos[0]))
else:
    print('All elements are not the same.')
    print('nb ttl of histos : ' , nb_ttl_histos)
newBranches = optimizeBranches(tmp_branches)

if (len(branches) != len(newBranches)):
    print('len std branches : {:d}'.format(len(branches)))
    print('len new branches : {:d}'.format(len(newBranches)))
    branches = newBranches
    N_histos = len(branches)
print('N_histos : %d' % N_histos)

sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)
folder += '{:03d}'.format(nbFiles)
folder = checkFolderName(folder)
print('folder après check : %s' % folder)
checkFolder(folder)

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
        print(aa[0])
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
    
folder += 'KS'
folder =checkFolderName(folder)
print('folder après check : %s' % folder)
checkFolder(folder)

labels = list(df1)[1:]
print(labels)
(N_histos, _) = df1.shape

for ind in df1.index:
    print(ind)
    a = df1.iloc[ind].to_numpy()
    branch = a[0]
    print(branch)
    val = list(a[1:])
    #print(val)
    pictureName = folder + 'comparison_KS_values_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : Comparison of KS diff values as function of releases.'
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
    pictureName = folder + 'comparison_pValues_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : Comparison of KS pValues as function of releases.'
    createCompPValuesPicture(labels,val, pictureName, title)
    #if ind == 2:
    #    break

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))


