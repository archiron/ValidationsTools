#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# createFiles: create file for Kolmogorov-Smirnov maximum diff
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

# lines below are only for func_Extract
from sys import argv

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # Check files path
    print("step 4 - arg. 2 :", sys.argv[2]) # Check Folder
    print("step 4 - arg. 3 :", sys.argv[3]) # FileName for paths
    commonPath = sys.argv[1]
    workPath=sys.argv[2][:-6]
    filePaths = sys.argv[3]
    print('common path : {:s}'.format(commonPath))
    print('file path : {:s}'.format(filePaths))
    print('work path : {:s}'.format(workPath))
else:
    print("rien")
    resultPath = ''


import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\ncheckCreatedVsOfficial")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as dfo
from default import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import getHisto, getHistoConfEntry, fill_Snew2 #, fill_Snew
from graphicAutoEncoderFunctions import GraphicKS
from DecisionBox import DecisionBox
from sources import *

resultPath += '/' + str(NB_EVTS)
resultPath = checkFolderName(resultPath)
print('resultPath : {:s}'.format(resultPath))
resultPath = checkFolderName(resultPath)
folder = resultPath + checkFolderName(dfo.folder)

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

DB = DecisionBox()
grKS = GraphicKS()
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

# get list of added ROOT files for comparison
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('we use the files :')
for item in rootFilesList:
    tmp_branch = []
    nbHistos = 0
    print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:], item])
    f_root = ROOT.TFile(rootFolderName + item)
    h_rel = getHisto(f_root, tp_1)
    for i in range(0, N_histos): # 1 N_histos histo for debug
        histo_rel = h_rel.Get(branches[i])
        if (histo_rel):
            print('%s OK' % branches[i])
            d = getHistoConfEntry(histo_rel)
            s_tmp = fill_Snew2(d, histo_rel)
            #s_tmp = fill_Snew(histo_rel)
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

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder += '{:03d}'.format(nbFiles)
folder = checkFolderName(folder)
checkFolder(folder)
folderKS = folder + 'Check'
folderKS =checkFolderName(folderKS)
checkFolder(folderKS)
print('')

source_dest = folder + "/ElectronMcSignalHistos.txt"
shutil.copy2(source, source_dest)

#sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

# get the "reference" root file datas
f_KSref = ROOT.TFile(rootFolderName + input_ref_file)
print('we use the %s file as KS reference' % input_ref_file)
h_KSref = getHisto(f_KSref, tp_1)
print(h_KSref)

tic = time.time()

for i in range(0, N_histos): # 1 N_histos histo for debug
    print(branches[i]) # print histo name
    
    histo_KSref = h_KSref.Get(branches[i])
    if (histo_KSref):
        print('%s OK' % branches[i])
        name = resultPath + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
        print('\n%d - %s' %(i, name))
        df = pd.read_csv(name)
    
        # check the values data
        cols = df.columns.values
        n_cols = len(cols)
        print('nb of columns for histos : %d' % n_cols)
        cols_entries = cols[6::2]
        df_entries = df[cols_entries]

        df_GetEntries = df['nbBins']

        # get nb of columns & rows for histos
        (Nrows, Ncols) = df_entries.shape
        print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))
        df_entries = df_entries.iloc[:, 1:Ncols-1]
        (Nrows, Ncols) = df_entries.shape
        print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))
        
        histo_KSref = h_KSref.Get(branches[i])
        s_KSref = []
        for entry in histo_KSref:
            s_KSref.append(entry)
        s_KSref = np.asarray(s_KSref)
        s_KSref = s_KSref[1:-1]
        Ntot_h_KSref = histo_KSref.GetEntries()
        #print(s_KSref)

        # redefining Nrows
        rowArray = [10, 50, 100]
        for Nrows in rowArray:
            #Nrows = 50
            print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))
            #print(df_entries.iloc[0,:])

            # create the datas for the p-Value graph
            # by comparing all curves between them. (KS 1)
            nb1 = 0
            totalDiff = []
            for k in range(0,Nrows-1):
                for l in range(k+1, Nrows):
                    nb1 += 1
                    series0 = df_entries.iloc[k,:]
                    series1 = df_entries.iloc[l,:]
                    totalDiff.append(DB.diffMAXKS3(series0, series1)[0]) # 9000, 9000
            print('ttl nb1 of couples 1 : %d' % nb1)
            totalDiff_KS = totalDiff.copy()
            for k in range(0,Nrows):
                nb1 += 1
                series0 = df_entries.iloc[k,:]
                totalDiff_KS.append(DB.diffMAXKS3(series0, s_KSref)[0]) # 9000, 9000

            print('ttl nb1 of couples 1 : %d' % nb1)

            # Kolmogoroff-Smirnov curve 1
            seriesTotalDiff1 = pd.DataFrame(totalDiff, columns=['KSDiff'])
            print('TD.min 1 : %f' % (seriesTotalDiff1.values.min()))
            print('TD.max 1 : %f' % (seriesTotalDiff1.values.max()))
            seriesTotalDiff1_KS = pd.DataFrame(totalDiff_KS, columns=['KSDiff'])
            print('TD.min 1 : %f' % (seriesTotalDiff1_KS.values.min()))
            print('TD.max 1 : %f' % (seriesTotalDiff1_KS.values.max()))

            count, division = np.histogram(seriesTotalDiff1[~np.isnan(seriesTotalDiff1)], bins=nbins)
            div_min = np.amin(division)
            div_max = np.amax(division)
            fileName = folderKS + '/KS-ttlDiff_v1_' + branches[i] + '_{:03d}'.format(Nrows)+ '.png'
            grKS.createSimpleKSttlDiffPicture(totalDiff, nbins, 'KS diff. v1', fileName)
            count_KS, division_KS = np.histogram(seriesTotalDiff1_KS[~np.isnan(seriesTotalDiff1_KS)], bins=nbins)
            div_min_KS = np.amin(division_KS)
            div_max_KS = np.amax(division_KS)
            fileName_KS = folderKS + '/KS-ttlDiff_v2_' + branches[i] + '_{:03d}'.format(Nrows)+ '.png'
            grKS.createSimpleKSttlDiffPicture(totalDiff_KS, nbins, 'KS diff. v2', fileName_KS)

            x= []
            x_KS = []
            for j in range(0, len(division)-1):
                x.append((division[j] + division[j+1])/2.)
            for j in range(0, len(division_KS)-1):
                x_KS.append((division_KS[j] + division_KS[j+1])/2.)

            legende = ['KS v1', 'KS v2']
            fileName_comp = folderKS + '/KS-ttlDiff_comp_' + branches[i] + '_{:03d}'.format(Nrows) + '.png'
            grKS.createSimpleCompKSttlDiffPicture(x, count, x_KS, count_KS, legende, 'KS diff comparison', fileName_comp)

            plt.close('all')

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("Fin !\n")

