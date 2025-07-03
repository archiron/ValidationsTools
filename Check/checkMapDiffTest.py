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

import sys
import importlib
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv
#from xml.sax.handler import DTDHandler

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal # ROOT.kBreak # 
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = True

root_version = ROOT.gROOT.GetVersion()

print('PYTHON     version : {}'.format(sys.version))
print("ROOT       version : {}".format(root_version))

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

import validationsDefault as dfo
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import Graphic
from DecisionBox import DecisionBox
from validationsDefault import *
from filesSources import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathDATA = checkFolderName(pathDATA)
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

print("checkMapDiff")
gr = Graphic()
gr.initRoot()
DB = DecisionBox()

N_histos = len(branches)
print('N_histos : %d' % N_histos)
    
# get the list of the generated ROOT files
fileList = getListFiles(pathROOTFiles, 'root') # get the list of the root files in the folderName folder
fileList.sort()
print('list of the generated ROOT files')
print('there is ' + '{:03d}'.format(len(fileList)) + ' ROOT files')
nbFiles = change_nbFiles(len(fileList), nbFiles)
nbFiles = 2000 # TEMP
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
AddedRootFolderName = os.path.join(pathLIBS, blo.DATA_SOURCE + '/Run3/RECO/') # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
print('pathDATA : {:s}'.format(AddedRootFolderName))
f_rel = ROOT.TFile(AddedRootFolderName + input_rel_file)
h1 = gr.getHisto(f_rel, tp_1)

print('we use the %s file as reference' % input_ref_file)
print('we use the %s file as new release' % input_rel_file)

if (ind_reference == -1): 
    ind_reference = np.random.randint(0, nbFiles)
print('reference ind. : %d' % ind_reference)

tic = time.time()

for i in range(0, N_histos): # 1 N_histos histo for debug
    histo_1 = h1.Get(branches[i])
    if (histo_1):
        #print('%s OK' % branches[i])
        name = pathROOTFiles + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
        print('\n%d - %s' %(i, name))
        df = pd.read_csv(name)
        
        #print(branches[i]) # print histo name
    
        d = gr.getHistoConfEntry(histo_1)
        #print("d = {}".format(d))
    
        ii=0
        #s_new = gr.fill_Snew(histo_1)
        s_new = gr.fill_Snew2(d, histo_1)
        Ntot_h1 = histo_1.GetEntries()

        # check the values & errors data
        cols = df.columns.values
        n_cols = len(cols)
        print('nb of columns for histos : %d' % n_cols)
        cols_entries = cols[7::2]
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
            for lj in range(k+1, Nrows):
                nb1 += 1
                series0 = df_entries.iloc[k,:]
                series1 = df_entries.iloc[lj,:]
                totalDiff1[k][lj] = DB.diffMAXKS3(series0, series1)[0]
        #print(totalDiff1)
        print('ttl nb1 of couples 1 : %d' % nb1)
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(X, Y, totalDiff1)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Filled Contours Plot : ' + str(Ncols))
        ax.set_xlabel('file number')
        ax.set_ylabel('file number')
        fig.savefig(pathCheck + '/map-ttlDiff_1_Test_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()

        # print 1 line
        Z = totalDiff1[int(Nrows/2)]
        XX = np.arange(0,Nrows) # [0,1,2,3,4]
        fig,ax=plt.subplots(1,1)
        #ax.plot(XX, Z) # only line
        ax.plot(XX, Z, 'ro') # only points
        #ax.plot(XX, Z, 'ro-') # line with points
        ax.set_title('one line Plot : ' + str(Ncols))
        ax.set_xlabel('file number')
        ax.set_ylabel('diff values')
        fig.savefig(pathCheck + '/line-ttlDiff_1_Test_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()
    else:
        print('%s KO' % branches[i])
    
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

