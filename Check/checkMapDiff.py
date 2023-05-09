#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# checkMapDiff : a tool to generate a map of the Kolmogorov-Smirnov diff values
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys
import importlib
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv
#from xml.sax.handler import DTDHandler

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 2 :", sys.argv[2]) # Check Folder
    print("step 5 - arg. 3 :", sys.argv[3]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    pathLIBS=sys.argv[2][:-6]
    filePaths = sys.argv[3]
else:
    print("rien")
    pathBase = ''

import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )

pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))
pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(pathChiLib))
pathDATA = pathLIBS + '/' + blo.DATA_SOURCE
print('DATA_SOURCE : %s' % pathDATA)

sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)
sys.path.append(pathDATA)

import default as dfo
from default import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import getHisto, getHistoConfEntry, fill_Snew2
from DecisionBox import DecisionBox
from sources import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

pathNb_evts = pathBase + '/' + str(NB_EVTS)
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathDATA = checkFolderName(pathDATA)

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

print("checkMapDiff")
DB = DecisionBox()

N_histos = len(branches)
print('N_histos : %d' % N_histos)
    
# get the list of the generated ROOT files
fileList = getListFiles(pathNb_evts, 'root') # get the list of the root files in the folderName folder
fileList.sort()
print('list of the generated ROOT files')
print('there is ' + '{:03d}'.format(len(fileList)) + ' ROOT files')
nbFiles = change_nbFiles(len(fileList), nbFiles)
fileList = fileList[0:nbFiles]
#print('file list :')
#print(fileList)

pathNb_files = pathCase + '{:03d}'.format(nbFiles)
pathNb_files = checkFolderName(pathNb_files)
print('folder après check : %s' % pathNb_files)
checkFolder(pathNb_files)
pathCheck = pathNb_files + '/Check'
pathCheck =checkFolderName(pathCheck)
print('folder après check : %s' % pathCheck)
checkFolder(pathCheck)

# get the "new" root file datas
f_rel = ROOT.TFile(pathDATA + input_rel_file)
h1 = getHisto(f_rel, tp_1)

print('we use the %s file as reference' % input_ref_file)
print('we use the %s file as new release' % input_rel_file)

if (ind_reference == -1): 
    ind_reference = np.random.randint(0, nbFiles)
print('reference ind. : %d' % ind_reference)

tic = time.time()

for i in range(0, N_histos): # 1 histo for debug
    histo_1 = h1.Get(branches[i])
    if (histo_1):
        #print('%s OK' % branches[i])
        name = pathNb_evts + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
        print('\n%d - %s' %(i, name))
        df = pd.read_csv(name)
        
        #print(branches[i]) # print histo name
    
        d = getHistoConfEntry(histo_1)
        #print("d = {}".format(d))
    
        ii=0
        #s_new = fill_Snew(histo_1)
        s_new = fill_Snew2(d, histo_1)
        Ntot_h1 = histo_1.GetEntries()

        # check the values & errors data
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

        xlist = np.linspace(0, Nrows-1, Nrows)
        X, Y = np.meshgrid(xlist, xlist)
        # create the datas for the p-Value graph
        # by comparing all curves between them. (KS 1)
        nb1 = 0
        totalDiff1 = np.zeros((Nrows,Nrows))
        for k in range(0,Nrows-1):
            for l in range(k+1, Nrows):
                nb1 += 1
                series0 = df_entries.iloc[k,:]
                series1 = df_entries.iloc[l,:]     
                sum0 = df_GetEntries[k]
                sum1 = df_GetEntries[l]
                totalDiff1[k][l] = DB.diffMAXKS(series0, series1, sum0, sum1)[0] # 9000, 9000
        #print(totalDiff1)
        print('ttl nb1 of couples 1 : %d' % nb1)
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(X, Y, totalDiff1)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Filled Contours Plot')
        ax.set_xlabel('file number')
        ax.set_ylabel('file number')
        fig.savefig(pathCheck + '/map-ttlDiff_1_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()

        # print 1 line
        Z = totalDiff1[int(Nrows/2)]
        XX = np.arange(0,Nrows) # [0,1,2,3,4]
        fig,ax=plt.subplots(1,1)
        #ax.plot(XX, Z) # only line
        ax.plot(XX, Z, 'ro') # only points
        #ax.plot(XX, Z, 'ro-') # line with points
        ax.set_title('one line Plot')
        ax.set_xlabel('file number')
        ax.set_ylabel('diff values')
        fig.savefig(pathCheck + '/line-ttlDiff_1_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()

        # create the datas for the p-Value graph
        # by comparing 1 curve with the others.
        # Get a random histo as reference (KS 2)
            #ind_reference = np.random.randint(0, Nrows)
            #print('reference ind. : %d' % ind_reference)
        series_reference = df_entries.iloc[ind_reference,:]
        nbBins_reference = df_GetEntries[ind_reference]
        print('nb bins reference : %d' % nbBins_reference)
        nb2 = 0
        totalDiff2 = np.zeros((Nrows,Nrows))
        for k in range(0,Nrows-0):
            if (k != ind_reference):
                nb2 += 1
                series0 = df_entries.iloc[k,:]
                sum0 = df_GetEntries[k]
                totalDiff2[k][l] = DB.diffMAXKS(series0, series_reference, sum0, nbBins_reference)[0] # 9000, 9000
        #print(totalDiff2)
        print('ttl nb of couples 2 : %d' % nb2)
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(X, Y, totalDiff2)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Filled Contours Plot')
        ax.set_xlabel('file number')
        ax.set_ylabel('file number')
        fig.savefig(pathCheck + '/map-ttlDiff_2_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()
    
        # create the datas for the p-Value graph
        # by comparing the new curve with the others.
        # Get the new as reference (KS 3)
        nb3 = 0
        totalDiff3 = np.zeros((Nrows,Nrows))
        for k in range(0,Nrows-0):
            nb3 += 1
            series0 = df_entries.iloc[k,:]
            sum0 = df_GetEntries[k]
            totalDiff3[k][l] = DB.diffMAXKS(series0, s_new, sum0, Ntot_h1)[0]
        #print(totalDiff3)
        print('ttl nb of couples 3 : %d' % nb3)
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(X, Y, totalDiff3)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Filled Contours Plot')
        ax.set_xlabel('file number')
        ax.set_ylabel('file number')
        fig.savefig(pathCheck + '/map-ttlDiff_3_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()
    else:
        print('%s KO' % branches[i])
    
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

