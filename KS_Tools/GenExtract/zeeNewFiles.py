#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# zeeNewFiles: create files with values for each histo for different releases
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
from graphicFunctions import getHisto
from controlFunctions import *
from default import *
from sources import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

def getHistoConfEntry(h1):
    d = 1

    if ( h1.InheritsFrom("TH2") ):
        print('TH2')
    elif ( h1.InheritsFrom("TProfile") ):
        #print('TProfile')
        d = 0
    elif ( h1.InheritsFrom("TH1")): # TH1
        print('TH1')
    else:
        print("don't know")

    return d

def func_CreateKS(br, nbFiles):
    print("func_Extract")
    folderName = '/home/arnaud/cernbox/DEV_PYTHON/AutoEncoder/2022/DATASETS/NewFiles'
    folderName = '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
    folderName = checkFolderName(folderName)
    folder = folderName

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

    # get list of files
    rootFilesList = getListFiles(folderName, 'root')

    print('we use the files :')
    for item in rootFilesList:
        print('%s' % item)

    KS_resume = folder + "branchesHistos_NewFiles.txt"
    print("KSname 0 : %s" % KS_resume)
    wKS_ = open(KS_resume, 'w')

    tic = time.time()

    for i in range(0, N_histos): # 1 histo for debug
        
        for file in rootFilesList:
            # extract release version
            fil = file.split('__')[2]
            fil = fil.split('-')[0]
            #print(fil)
            inputFile = 'DATA/NewFiles/' + file
            #print(inputFile)
            rFile = ROOT.TFile(inputFile)
            h1 = getHisto(rFile, tp_1)
            print(branches[i]) # print histo name
            histo_1 = h1.Get(branches[i])
            s_new = []
#            e_new = []

            d = getHistoConfEntry(histo_1)
            #print("d = {}".format(d))

            ii = 0
            texttoWrite = fil + "," + branches[i] + ","
            if (d==1):
                for entry in histo_1:
                    s_new.append(entry)
#                    e_new.append(histo_1.GetBinError(ii))
            else:
                print('GLOBOS')
                for entry in histo_1:
                    if ((histo_1.GetBinEntries(ii) == 0.) and (entry == 0.)):
                        s_new.append(0.)
                    elif ((histo_1.GetBinEntries(ii) == 0.) and (entry != 0.)):
                        s_new.append(1.e38)
                        print('=======================================================',ii,entry,histo_1.GetBinEntries(ii))
                    else:
                        s_new.append(entry/histo_1.GetBinEntries(ii))
                    ii+=1
            
            s_new = np.asarray(s_new)
            s_new = s_new[1:-1]
#            e_new = e_new[1:-1]
            for elem in s_new:
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
    source = open("../ChiLib_CMS_Validation/HistosConfigFiles/ElectronMcSignalHistos.txt", "r")
    branches = getBranches(tp_1, source)
    print(branches[0:10])
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    nbFiles = 950

    func_CreateKS(branches, nbFiles)  # create the KS files from histos datas

    print("Fin !")

