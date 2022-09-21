#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# checkRootFiles: generate a summary of all ROOT files and all histos
# for egamma validation comparison 
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import sys
import importlib

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv
from xml.sax.handler import DTDHandler

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
    print("step 4 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 2 :", sys.argv[2]) # Check Folder
    print("step 5 - arg. 3 :", sys.argv[3]) # FileName for paths
    commonPath = sys.argv[1]
    workPath=sys.argv[2][:-6]
    filePaths = sys.argv[3]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )

resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))
Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(Chilib_path))
dataPath = workPath + '/' + blo.DATA_SOURCE
print('DATA_SOURCE : %s' % dataPath)

sys.path.append(Chilib_path)
sys.path.append(commonPath)
sys.path.append(dataPath)

import default as dfo
from controlFunctions import *
from graphicFunctions import getHisto, getHistoConfEntry, fill_Snew, fill_Snew2
from DecisionBox import DecisionBox
from default import *
from sources import *

#folder = resultPath + checkFolderName(dfo.folder)
folder = checkFolderName(dfo.folder)
resultPath = checkFolderName(resultPath)
dataPath = checkFolderName(dataPath)

# get the branches for ElectronMcSignalHistos.txt
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = []
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

print("func_Extract")
resultPath = checkFolderName(resultPath)    
DB = DecisionBox()

N_histos = len(branches)
print('N_histos : %d' % N_histos)

# get the list of the generated ROOT files
fileList = getListFiles(resultPath, 'root') # get the list of the root files in the folderName folder
fileList.sort()
print('list of the generated ROOT files')
print('there is ' + '{:03d}'.format(len(fileList)) + ' ROOT files')
nbFiles = change_nbFiles(len(fileList), nbFiles)
fileList = fileList[0:nbFiles]
#print('file list :')
#print(fileList)

# PASS 1 - tests on min/max
print('\n\nPASS 1')
nb_ttl_histos = []
tmp_branches = []
for elem in fileList:
    tmp_branch = []
    nbHistosPass1 = 0
    
    input_file = resultPath + str(elem.split()[0])
    name_1 = input_file.replace(resultPath, '').replace('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_', '').replace('.root', '')
    print('\n %s - name_1 : %s' % (input_file, name_1))
    #print('\n %s - name_1 : %s' % (input_file, colorText(name_1, 'lightyellow')))
    f_root = ROOT.TFile(input_file)
    h1 = getHisto(f_root, tp_1)
    #h1.ls() # OK

    for i in range(0, N_histos): # 1 N_histos histo for debug
        #print('\n' + branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])
        '''
        s_new = []
        for entry in histo_1:
            s_new.append(entry)
        s_new = np.asarray(s_new)
        s_new = s_new[1:-1]
        '''
        s_new = fill_Snew(histo_1)
        Ntot_h1 = histo_1.GetEntries()
        
        # print min/max for the new curve
        #print('min : %f - max : %f' % (s_new.min(), s_new.max()))
        if (s_new.min() < 0.):
            print('pbm whith histo %s, min < 0' % branches[i])
        elif (np.floor(s_new.sum()) == 0.):
            print('pbm whith histo %s, sum = 0' % branches[i])
        else:
            nbHistosPass1 += 1
            tmp_branch.append(branches[i])

    nb_ttl_histos.append(nbHistosPass1)
    tmp_branches.append(tmp_branch)
    print('there {:03d} histos for PASS 1a'.format(nbHistosPass1))
    if (nbHistosPass1 != N_histos): 
        print('Warning ! a few histos cannot be usable.')

if(len(set(nb_ttl_histos))==1):
    print('All elements are the same with value {:d} for PASS 1a.'.format(nb_ttl_histos[0]))
else:
    print('All elements are not the same for PASS 1a.')
    print(nb_ttl_histos)
    print(tmp_branches)
    newBranches = optimizeBranches(tmp_branches)

# get list of added ROOT files for comparison
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('\nlist of the added ROOT files')
print('we use the files :')

nb_ttl_histos = []
tmp_branches = []
for item in rootFilesList:
    tmp_branch = []
    nbHistosPass1 = 0
    
    print('%s' % item)
    f_root = ROOT.TFile(rootFolderName + item)
    h1 = getHisto(f_root, tp_1)

    for i in range(0, N_histos): # 1 N_histos histo for debug
        #print('\n' + branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])
        '''
        s_new = []
        for entry in histo_1:
            s_new.append(entry)
        s_new = np.asarray(s_new)
        s_new = s_new[1:-1]
        '''
        s_new = fill_Snew(histo_1)
        Ntot_h1 = histo_1.GetEntries()

        # print min/max for the new curve
        #print('min : %f - max : %f' % (s_new.min(), s_new.max()))
        if (s_new.min() < 0.):
            print('pbm whith histo %s, min < 0' % branches[i])
        elif (np.floor(s_new.sum()) == 0.):
            print('pbm whith histo %s, sum = 0' % branches[i])
        else:
            nbHistosPass1 += 1
            tmp_branch.append(branches[i])

    nb_ttl_histos.append(nbHistosPass1)
    tmp_branches.append(tmp_branch)
    print('there is {:03d} histos for PASS 1b'.format(nbHistosPass1))
    if (nbHistosPass1 != N_histos): 
        print('Warning ! a few histos cannot be usable.')

if(len(set(nb_ttl_histos))==1):
    print('All elements are the same with value {:d} for PASS 1b.'.format(nb_ttl_histos[0]))
else:
    print('All elements are not the same for PASS 1b.')
    print(nb_ttl_histos)
    print(tmp_branches)
    newBranches = optimizeBranches(tmp_branches)

# PASS 2 - tests on min/max with another way for entry
print('\n\nPASS 2')
nb_ttl_histos = []
tmp_branches = []
for elem in fileList:
    tmp_branch = []
    nbHistosPass2 = 0
    
    input_file = resultPath + str(elem.split()[0])
    name_1 = input_file.replace(resultPath, '').replace('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_', '').replace('.root', '')
    print('\n %s - name_1 : %s' % (input_file, name_1))
    #print('\n %s - name_1 : %s' % (input_file, colorText(name_1, 'lightyellow')))
    f_root = ROOT.TFile(input_file)
    h1 = getHisto(f_root, tp_1)

    for i in range(0, N_histos): # 1 N_histos histo for debug
        #print('\n' + branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])
            
        d = getHistoConfEntry(histo_1)
        #print("d = {}".format(d))

        '''
        s_new = []
        ii = 0
        if (d==1):
            for entry in histo_1:
                s_new.append(entry)
        else:
            for entry in histo_1:
                if ((histo_1.GetBinEntries(ii) == 0.) and (entry == 0.)):
                    s_new.append(0.)
                elif ((histo_1.GetBinEntries(ii) == 0.) and (entry != 0.)):
                    s_new.append(1.e38)
                    print('========================================',ii,entry,histo_1.GetBinEntries(ii))
                else:
                    s_new.append(entry/histo_1.GetBinEntries(ii))
                ii+=1
        s_new = np.asarray(s_new)
        s_new = s_new[1:-1]
        '''
        s_new = fill_Snew2(d, histo_1)
        Ntot_h1 = histo_1.GetEntries()
        
        # print min/max for the new curve
        #print('min : %f - max : %f' % (s_new.min(), s_new.max()))
        if (s_new.min() < 0.):
            print('pbm whith histo %s, min < 0' % branches[i])
        elif (np.floor(s_new.sum()) == 0.):
            print('pbm whith histo %s, sum = 0' % branches[i])
        else:
            nbHistosPass2 += 1
            tmp_branch.append(branches[i])

    nb_ttl_histos.append(nbHistosPass2)
    tmp_branches.append(tmp_branch)
    print('there {:03d} histos for PASS 2'.format(nbHistosPass2))
    if (nbHistosPass2 != N_histos): 
        print('Warning ! a few histos cannot be usable.')

if(len(set(nb_ttl_histos))==1):
    print('All elements are the same with value {:d} for PASS 2a.'.format(nb_ttl_histos[0]))
else:
    print('All elements are not the same for PASS 2a.')
    print(nb_ttl_histos)
    print(tmp_branches)
    newBranches = optimizeBranches(tmp_branches)

# get list of added ROOT files for comparison
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('\nlist of the added ROOT files')
print('we use the files :')

nb_ttl_histos = []
tmp_branches = []
for item in rootFilesList:
    tmp_branch = []
    nbHistosPass2 = 0
    
    print('%s' % item)
    f_root = ROOT.TFile(rootFolderName + item)
    h1 = getHisto(f_root, tp_1)

    for i in range(0, N_histos): # 1 N_histos histo for debug
        #print('\n' + branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])

        d = getHistoConfEntry(histo_1)
        #print("d = {}".format(d))

        '''
        s_new = []
        ii = 0
        if (d==1):
            for entry in histo_1:
                s_new.append(entry)
        else:
            for entry in histo_1:
                if ((histo_1.GetBinEntries(ii) == 0.) and (entry == 0.)):
                    s_new.append(0.)
                elif ((histo_1.GetBinEntries(ii) == 0.) and (entry != 0.)):
                    s_new.append(1.e38)
                    print('========================================',ii,entry,histo_1.GetBinEntries(ii))
                else:
                    s_new.append(entry/histo_1.GetBinEntries(ii))
                ii+=1
        s_new = np.asarray(s_new)
        s_new = s_new[1:-1]
        '''
        s_new = fill_Snew2(d, histo_1)
        Ntot_h1 = histo_1.GetEntries()
        
        # print min/max for the new curve
        #print('min : %f - max : %f' % (s_new.min(), s_new.max()))
        if (s_new.min() < 0.):
            print('pbm whith histo %s, min < 0' % branches[i])
        elif (np.floor(s_new.sum()) == 0.):
            print('pbm whith histo %s, sum = 0' % branches[i])
        else:
            nbHistosPass2 += 1
            tmp_branch.append(branches[i])

    nb_ttl_histos.append(nbHistosPass2)
    tmp_branches.append(tmp_branch)
    print('there is {:03d} histos for PASS 2b'.format(nbHistosPass2))
    if (nbHistosPass2 != N_histos): 
        print('Warning ! a few histos cannot be usable.')

if(len(set(nb_ttl_histos))==1):
    print('All elements are the same with value {:d} for PASS 2b.'.format(nb_ttl_histos[0]))
else:
    print('All elements are not the same for PASS 2b.')
    print(nb_ttl_histos)
    print(tmp_branches)
    newBranches = optimizeBranches(tmp_branches)

# PASS 3 - tests on diffMAXKS
print('\n\nPASS 3')
#input_rel_file = file
f_rel = ROOT.TFile(rootFolderName + input_rel_file)
print('we use the %s file as new release' % input_rel_file)
h1 = getHisto(f_rel, tp_1)

if (ind_reference == -1): 
    ind_reference = np.random.randint(0, nbFiles)
print('reference ind. : %d' % ind_reference)

for i in range(0, N_histos): # 1 N_histos histo for debug
    name = resultPath + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
    print('\n%d - %s' %(i, name))
    df = pd.read_csv(name)
    #print('\n' + branches[i]) # print histo name

    histo_1 = h1.Get(branches[i])
    s_new = []
    for entry in histo_1:
        s_new.append(entry)
    s_new = np.asarray(s_new)
    s_new = s_new[1:-1]
    Ntot_h1 = histo_1.GetEntries()
    #print('nb of entries : {:f}'.format(Ntot_h1))

    # check the values data
    #print(df.head(5))
    cols = df.columns.values
    n_cols = len(cols)
    #print('nb of columns for histos : %d' % n_cols)
    cols_entries = cols[6::2]
    df_entries = df[cols_entries]
    #print(df_entries.head(15))#

    # nbBins (GetEntries())
    df_GetEntries = df['nbBins']

    # get nb of columns & rows for histos
    (Nrows, Ncols) = df_entries.shape
    #print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))
    df_entries = df_entries.iloc[:, 1:Ncols-1]
    (Nrows, Ncols) = df_entries.shape
    print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))

    # create the datas for the p-Value graph
    # by comparing all curves between them. (KS 1)
    nb1 = 0
    totalDiff = []
    for k in range(0,Nrows-1):
        for l in range(k+1, Nrows):
            nb1 += 1
            series0 = df_entries.iloc[k,:]
            series1 = df_entries.iloc[l,:]     
            sum0 = df_GetEntries[k]
            sum1 = df_GetEntries[l]
            totalDiff.append(DB.diffMAXKS(series0, series1, sum0, sum1)[0]) # 9000, 9000

    print('ttl nb1 of couples 1 : %d' % nb1)

    # create the datas for the p-Value graph
    # by comparing 1 curve with the others.
    # Get a random histo as reference (KS 2)
    series_reference = df_entries.iloc[ind_reference,:]
    nbBins_reference = df_GetEntries[ind_reference]
    #print('nb bins reference : %d' % nbBins_reference)
    nb2 = 0
    totalDiff2 = []
    for k in range(0,Nrows-0):
        if (k != ind_reference):
            nb2 += 1
            series0 = df_entries.iloc[k,:]
            sum0 = df_GetEntries[k]
            totalDiff2.append(DB.diffMAXKS(series0, series_reference, sum0, nbBins_reference)[0]) # 9000, 9000

    print('ttl nb of couples 2 : %d' % nb2)
        
    # create the datas for the p-Value graph
    # by comparing the new curve with the others.
    # Get the new as reference (KS 3)
    
    nb3 = 0
    totalDiff3 = []
    for k in range(0,Nrows-0):
        nb3 += 1
        series0 = df_entries.iloc[k,:]
        sum0 = df_GetEntries[k]
        totalDiff3.append(DB.diffMAXKS(series0, s_new, sum0, Ntot_h1)[0])

    print('ttl nb of couples 3 : %d' % nb3)





print("Fin !")
