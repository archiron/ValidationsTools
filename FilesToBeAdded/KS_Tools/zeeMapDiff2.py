#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# zeeExtract_MP: a tool to generate Kolmogorov-Smirnov values/pictures
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys
import time

import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

from sys import argv

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()

#import seaborn # only with cmsenv on cca.in2p3.fr

sys.path.append('../ChiLib_CMS_Validation')
Chilib_path = '/pbs/home/c/chiron/private/KS_Tools/ChiLib_CMS_Validation'
sys.path.append(Chilib_path)
import default as df
from default import *
from sources import *
from graphicFunctions import getHisto
from DecisionBox import DecisionBox

# these line for daltonians !
#seaborn.set_palette('colorblind')

def checkFolderName(folderName):
    if folderName[-1] != '/':
        folderName += '/'
    return folderName

def changeColor(color):
    # 30:noir ; 31:rouge; 32:vert; 33:orange; 34:bleu; 35:violet; 36:turquoise; 37:blanc
    # other references at https://misc.flogisoft.com/bash/tip_colors_and_formatting
    if (color == 'black'):
        return '[30m'
    elif (color == 'red'):
        return '[31m'
    elif (color == 'green'):
        return '[32m'
    elif (color == 'orange'):
        return '[33m'
    elif (color == 'blue'):
        return '[34m'
    elif (color == ''):
        return '[35m'
    elif (color == 'purple'):
        return '[36m'
    elif (color == 'turquoise'):
        return '[37m'
    elif (color == 'lightyellow'):
        return '[93m'
    else:
        return '[30m'

def colorText(sometext, color):
    return '\033' + changeColor(color) + sometext + '\033[0m'

def getBranches(t_p):
    b = []
    source = open(Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt", "r")
    for ligne in source:
        if t_p in ligne:
            #print("source : %s" % ligne)
            tmp = ligne.split(" ", 1)
            #print(tmp[0].replace(t_p + "/", ""))
            b.append(tmp[0].replace(t_p + "/", ""))
    source.close()
    return b

def cleanBranches(branches):
    #if (branches[i] == 'h_ele_seedMask_Tec'): # temp (pbm with nan)
    #if re.search('OfflineV', branches[i]): # temp (pbm with nbins=81 vs nbins=80)
    toBeRemoved = ['h_ele_seedMask_Tec'] # , 'h_ele_convRadius', 'h_ele_PoPtrue_golden_barrel', 'h_ele_PoPtrue_showering_barrel'
    for ele in toBeRemoved:
        if ele in branches:
            branches.remove(ele)

def func_CreateMap(br, nbFiles):
    DB = DecisionBox()
    print("func_Extract")
    df.folderName = checkFolderName(df.folderName)
    df.folder = checkFolderName(df.folder)

    branches = br
    N_histos = len(branches)
    print('N_histos : %d' % N_histos)
    
    # nb of bins for sampling
    nbins = 100 
    
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

    ##### TEMP #####
    LOG_SOURCE_WORK='/pbs/home/c/chiron/private/KS_Tools/GenExtract/'
    # get the "new" root file datas
    f_rel = ROOT.TFile(LOG_SOURCE_WORK + input_rel_file)
    #f_rel = ROOT.TFile(input_rel_file)

    print('we use the %s file as reference' % input_ref_file)
    print('we use the %s file as new release' % input_rel_file)

    ind_reference = 1#99 # np.random.randint(0, nbFiles)
    print('reference ind. : %d' % ind_reference)

    tic = time.time()

    for i in range(0, N_histos): # 1 histo for debug
        
        h1 = getHisto(f_rel, tp_1)
        print(branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])
        ii=0
        s_new = []
        for entry in histo_1:
            s_new.append(entry)
            ii += 1
        s_new = np.asarray(s_new)
        s_new = s_new[1:-1]
        Ntot_h1 = histo_1.GetEntries()
        
        name = df.folderName + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + "_0_lite.txt"
        print('\n%d - %s' %(i, name))
        df = pd.read_csv(name)

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

        #xlist = np.linspace(0, Nrows-1, Nrows)
        xlist = np.linspace(0, 249, 250)
        X, Y = np.meshgrid(xlist, xlist)
        # create the datas for the p-Value graph
        # by comparing all curves between them. (KS 1)
        nb1 = 0
        totalDiff1 = np.zeros((250,250))
        for k in range(0,249):
            for l in range(k+1, 250):
                nb1 += 1
                series0 = df_entries.iloc[k+400,:]
                series1 = df_entries.iloc[l+400,:]     
                sum0 = df_GetEntries[k+400]
                sum1 = df_GetEntries[l+400]
                totalDiff1[k][l] = DB.diffMAXKS(series0, series1, sum0, sum1)[0] # 9000, 9000
        #print(totalDiff1)
        print('ttl nb1 of couples 1 : %d' % nb1)
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(X, Y, totalDiff1)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Filled Contours Plot')
        ax.set_xlabel('file number')
        ax.set_ylabel('file number')
        fig.savefig(df.folder + '/map-ttlDiff_1_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()

        # print 1 line
        Z = totalDiff1[75]
        XX = np.arange(0,250) # [0,1,2,3,4]
        fig,ax=plt.subplots(1,1)
        ax.plot(XX, Z)
        ax.set_title('one line Plot')
        ax.set_xlabel('file number')
        ax.set_ylabel('diff values')
        fig.savefig(df.folder + '/line-ttlDiff_1_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()

        '''
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
        fig.savefig(df.folder + '/map-ttlDiff_2_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
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
        fig.savefig(df.folder + '/map-ttlDiff_3_' + '_{:03d}'.format(nbFiles) + '_' + branches[i] + '.png')
        fig.clf()
        '''
    
    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc-tic))

    return

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    branches = getBranches(tp_1)
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    # nb of files to be used
    nbFiles = 750

    func_CreateMap(branches[0:5], nbFiles) # create the KS files from histos datas for datasets
    #func_CreateMap(branches, nbFiles)  # create the KS files from histos datas

    print("Fin !")

