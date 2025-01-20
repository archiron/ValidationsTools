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
import importlib
import importlib.machinery
import importlib.util
import time

from sys import argv

#import seaborn # only with cmsenv on cca.in2p3.fr

#argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal # ROOT.kBreak # 
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = True
#argv.remove( '-b-' )

from ROOT import gROOT
root_version = ROOT.gROOT.GetVersion()

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 2 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 4 :", sys.argv[2]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
else:
    print("rien")
    pathBase = ''

import pandas as pd
import matplotlib
import numpy as np

print('PANDAS     version : {}'.format(pd.__version__))
print('PYTHON     version : {}'.format(sys.version))
print("NUMPY      version : {}".format(np.__version__))
print('MATPLOTLIB version : {}'.format(matplotlib.__version__))
print("ROOT      version : {}".format(root_version))

matplotlib.use('agg')

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("func_Extract")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(pathChiLib))
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

import validationsDefault as dfo
from validationsDefault import *
from rootValues import NB_EVTS
from controlFunctions import *
from filesSources import *
from graphicAutoEncoderFunctions import createCompLossesPicture, createCompPValuesPicture, createCompPValuesPicture2
from graphicFunctions import Graphic

gr = Graphic()

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathROOTFiles = blo.pathROOT + "/" + release
pathROOTFiles = checkFolderName(pathROOTFiles)
print('pathROOTFiles : {:s}'.format(pathROOTFiles))

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

rels = []
tmp_branches = []
nb_ttl_histos = []

N_histos = len(branches)
print('N_histos : %d' % N_histos)
    
# create folder 
if not os.path.exists(pathCase):
    try:
        os.makedirs(pathCase)
    except OSError as e:
        if e.errno != errno.EEXIST: # the folder did not exist
            raise  # raises the error again
    print('Creation of %s release folder\n' % pathCase)
else:
    print('Folder %s already created\n' % pathCase)

# get list of the added ROOT files
pathDATA = pathLIBS + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(pathDATA, 'root')
print('we use the files :')
for item in rootFilesList:
    tmp_branch = []
    nbHistos = 0
    print('%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])
    f_root = ROOT.TFile(pathDATA + item)
    h1 = gr.getHisto(f_root, tp_1)
    for i in range(0, N_histos): # 1 N_histos histo for debug
        histo_1 = h1.Get(branches[i])
        if (histo_1):
            print('%s OK' % branches[i])
            d = gr.getHistoConfEntry(histo_1)
            s_tmp = gr.fill_Snew2(d, histo_1)
            #s_tmp = fill_Snew(histo_1)
            if (s_tmp.min() < 0.):
                print('pbm whith histo %s, min < 0' % branches[i])
            elif (np.floor(s_tmp.sum()) == 0.):
                print('pbm whith histo %s, sum = 0' % branches[i])
            else:
                nbHistos += 1
                tmp_branch.append(branches[i])
        else:
            print('%s KO' % branches[i])
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
rootFilesList_0 = getListFiles(pathROOTFiles, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)
pathNb_files = pathCase + '{:03d}'.format(nbFiles)
pathNb_files = checkFolderName(pathNb_files)
print('pathNb_files après check : %s' % pathNb_files)
checkFolder(pathNb_files)
pathKS = pathNb_files + 'KS'
pathKS =checkFolderName(pathKS)
print('folder KS après check : %s' % pathKS)
checkFolder(pathKS)

tic = time.time()

df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()
df_ttl1 = []
df_ttl4 = []
df_ttl5 = []

for elem in sortedRels:
    print(elem)
    rel = elem[1]

    # get the KS file datas
    KS_diffName = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
    KS_diffName_std = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "-std.txt"
    KS_diffName_mean = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "-mean.txt"
    pValue_Name = pathNb_files + "/histo_pValues" + "_" + rel + ".txt"
    if exists(KS_diffName):
        print('%s existe'%KS_diffName)
    else:
        print('%s n\'existe pas'%KS_diffName)
    if exists(KS_diffName):
        print('%s existe'%pValue_Name)
    else:
        print('%s n\'existe pas'%pValue_Name)

    wKS0 = open(KS_diffName, 'r').readlines()
    wKS4 = open(KS_diffName_std, 'r').readlines()
    wKS5 = open(KS_diffName_mean, 'r').readlines()
    wKS1 = open(pValue_Name, 'r').readlines()
    sum0 = 0.
    sum4 = 0.
    sum5 = 0.

    print(len(wKS0))
    print(len(wKS4))
    print(len(wKS5))
    print(len(wKS1))
    
    # diff
    tmpArr1 = []
    tmpArr2 = []
    

    nbLines = 0
    for line in wKS0:
        #print(len(line))
        aa = line.split(' : ')
        print(aa[0])
        tmpArr1.append(aa[0])
        tmpArr2.append(float(aa[1][:-1]))
        sum0 += float(aa[1][:-1])
        #print('{:s} : {:f}'.format(aa[0], sum0))
        nbLines += 1
    df1['index'] = tmpArr1
    df1[rel] = tmpArr2
    df_ttl1.append([rel, sum0/nbLines])

    tmpArr1 = []
    tmpArr2 = []
    nbLines = 0
    for line in wKS4:
        #print(len(line))
        aa = line.split(' : ')
        print(aa[0])
        tmpArr1.append(aa[0])
        tmpArr2.append(float(aa[1][:-1]))
        sum4 += float(aa[1][:-1])
        nbLines += 1
    df4['index'] = tmpArr1
    df4[rel] = tmpArr2
    df_ttl4.append([rel, sum4/nbLines])

    tmpArr1 = []
    tmpArr2 = []
    nbLines = 0
    for line in wKS5:
        #print(len(line))
        aa = line.split(' : ')
        print(aa[0])
        tmpArr1.append(aa[0])
        tmpArr2.append(float(aa[1][:-1]))
        sum5 += float(aa[1][:-1])
        nbLines += 1
    df5['index'] = tmpArr1
    df5[rel] = tmpArr2
    df_ttl5.append([rel, sum5/nbLines])

    # pValues
    tmpArr1 = []
    tmpArr2 = []
    tmpArr3 = []
    tmpArr4 = []
    tmpArr5 = []
    tmpArr6 = []
    for line in wKS1:
        #print(len(line))
        aa = line.split(', ')
        tmpArr1.append(aa[0])
        tmpArr2.append(float(aa[1]))
        tmpArr3.append(float(aa[2]))
        tmpArr4.append(float(aa[3]))#[:-1]
        tmpArr5.append(float(aa[4]))
        tmpArr6.append(float(aa[5]))#[:-1]
    df2['index'] = tmpArr1
    df2[rel+'_pV1'] = tmpArr2
    df2[rel+'_pV2'] = tmpArr3
    df2[rel+'_pV3'] = tmpArr4 
    df2[rel+'_pV4'] = tmpArr5
    df2[rel+'_pV5'] = tmpArr6 
    df3['index'] = tmpArr1
    df3[rel+'_pV1'] = tmpArr2
    df3[rel+'_pV4'] = tmpArr5
    df3[rel+'_pV5'] = tmpArr6 

print(df1.head(5))
print()
print(df4.head(5))
print()
print(df5.head(5))
print()
print(df2.head(5))
print()
print(df_ttl1)
print()
print(df_ttl4)
print()
print(df_ttl5)
print()
    
labels = list(df1)[1:]
print('labels')
print(labels)
(N_histos, _) = df1.shape

for ind in df1.index:
    print(ind)
    a = df1.iloc[ind].to_numpy()
    branch = a[0]
    print(branch)
    val = list(a[1:])
    #print(val)
    pictureName = pathKS + 'comparison_KS_values_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : KS cum diff values vs releases.'
    createCompLossesPicture(labels,val, pictureName, title)

for ind in df4.index:
    print(ind)
    a = df4.iloc[ind].to_numpy()
    branch = a[0]
    print(branch)
    val = list(a[1:])
    #print(val)
    pictureName = pathKS + 'comparison_KS_values_' + branch + '_{:03d}'.format(nbFiles) +'-std.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : KS std diff values vs releases.'
    createCompLossesPicture(labels,val, pictureName, title)

for ind in df5.index:
    print(ind)
    a = df5.iloc[ind].to_numpy()
    branch = a[0]
    print(branch)
    val = list(a[1:])
    #print(val)
    pictureName = pathKS + 'comparison_KS_values_' + branch + '_{:03d}'.format(nbFiles) +'-mean.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : KS mean diff values vs releases.'
    createCompLossesPicture(labels,val, pictureName, title)

for ind in df2.index:
    print('df2 : {:d}'.format(ind))
    a = df2.iloc[ind].to_numpy()
    branch = a[0]
    print(branch)
    val = list(a[1:])
    print(val)
    pictureName = pathKS + 'comparison_pValues_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : KS pValues vs releases.'
    createCompPValuesPicture2(labels,val, pictureName, title)

for ind in df3.index:
    print('df3 : {:d}'.format(ind))
    a = df3.iloc[ind].to_numpy()
    branch = a[0]
    print(branch)
    val = list(a[1:])
    print(val)
    pictureName = pathKS + 'comparison_pValues_' + branch + '_{:03d}'.format(nbFiles) +'_b.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : KS pValues vs releases.'
    createCompPValuesPicture(labels,val, pictureName, title)

# histo complet recapitulatif
lab = []
val = []
for elem in df_ttl1:
    lab.append(elem[0])
    val.append(elem[1])
print(val)
pictureName = pathKS + 'comparison_KS_values_total_cum_{:03d}'.format(nbFiles) +'.png' # 
print(pictureName)
title = r"$\bf{total}$" + ' : KS cum diff values vs releases.'
createCompLossesPicture(lab,val, pictureName, title)

lab = []
val = []
for elem in df_ttl4:
    lab.append(elem[0])
    val.append(elem[1])
print(val)
pictureName = pathKS + 'comparison_KS_values_total_std_{:03d}'.format(nbFiles) +'.png' # 
print(pictureName)
title = r"$\bf{total}$" + ' : KS std diff values vs releases.'
createCompLossesPicture(lab,val, pictureName, title)

lab = []
val = []
for elem in df_ttl5:
    lab.append(elem[0])
    val.append(elem[1])
print(val)
pictureName = pathKS + 'comparison_KS_values_total_mean_{:03d}'.format(nbFiles) +'.png' # 
print(pictureName)
title = r"$\bf{total}$" + ' : KS mean diff values vs releases.'
createCompLossesPicture(lab,val, pictureName, title)

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("Fin !")
