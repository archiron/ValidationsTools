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

import os,sys, re
import time

import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv
from os import listdir
from os.path import isfile, join

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()

sys.path.append('../ChiLib_CMS_Validation')
import default as df
from graphicFunctions import getHisto
from default import *
from DecisionBox import DecisionBox
from sources import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

def checkFolderName(folderName):
    if folderName[-1] != '/':
        folderName += '/'
    return folderName

def getListFiles(path):
    #print('path : %s' % path)
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith(".root")] # keep only root files
    #print(onlyfiles)
    return onlyfiles

def getBranches(t_p):
    b = []
    source = open("../ChiLib_CMS_Validation/HistosConfigFiles/ElectronMcSignalHistos.txt", "r")
    for ligne in source:
        if t_p in ligne:
            #print(ligne)
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

def func_CreateKS(br, nbFiles):
    print("func_Extract")
    df.folderName = checkFolderName(df.folderName)
    df.folder = checkFolderName(df.folder)

    branches = br
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

    # get the "new" root file datas
    f_rel = ROOT.TFile(input_rel_file)

    # get the "reference" root file datas
    f_ref = ROOT.TFile(input_ref_file)

    print('we use the %s file as reference' % input_ref_file)
    print('we use the %s file as new release' % input_rel_file)

    KS_resume = df.folder + "branchesHistos_RelRef.txt"
    print("KSname 0 : %s" % KS_resume)
    wKS_ = open(KS_resume, 'w')

    tic = time.time()

    for i in range(0, N_histos): # 1 histo for debug
        name = df.folderName + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + "_0_lite.txt"
        print('\n%d - %s' %(i, name))
        
        h1 = getHisto(f_rel, tp_1)
        print(branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])
        s_new = []
#        e_new = []
        texttoWrite = "new " + branches[i] + ","
        for entry in histo_1:
            s_new.append(entry)
#            e_new.append(histo_1.GetBinError(ii))
        s_new = np.asarray(s_new)
        s_new = s_new[1:-1]
#        e_new = e_new[1:-1]
        for elem in s_new:
            texttoWrite += str(elem) + ","
        texttoWrite = texttoWrite[:-1] # remove last char
        texttoWrite += '\n'
        wKS_.write(texttoWrite)

        h2 = getHisto(f_ref, tp_1)
        print(branches[i]) # print histo name
        histo_2 = h2.Get(branches[i])
        s_old = []
#        e_old = []
        texttoWrite = "old " + branches[i] + ","
        for entry in histo_2:
            s_old.append(entry)
#            e_old.append(histo_2.GetBinError(ii))
        s_old = np.asarray(s_old)
        s_old = s_old[1:-1]
#        e_old = e_old[1:-1]
        for elem in s_old:
            texttoWrite += str(elem) + ","
        texttoWrite = texttoWrite[:-1] # remove last char
        texttoWrite += '\n'
        wKS_.write(texttoWrite)

    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc-tic))

    return

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    branches = getBranches(tp_1)
    print(branches[0:10])
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    nbFiles = 200

    func_CreateKS(branches, nbFiles)  # create the KS files from histos datas

    print("Fin !")

