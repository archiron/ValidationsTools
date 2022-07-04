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
from default import *
from sources import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

def changeDirectory(rootFile, path):
    """
    Change the current directory (ROOT.gDirectory) by the corresponding (rootFile,pathSplit)
    module from cmdLineUtils.py
    """
    rootFile.cd()
    theDir = ROOT.gDirectory.Get(path)
    if not theDir:
        print("Directory %s does not exist." % path)
    else:
        theDir.cd()
    return 0

def checkLevel(f_rel, f_out, path0, listkeys, nb):
    inPath = 'DQMData/Run 1/EgammaV'
    print('\npath : %s' % path0)
    if path0 != "":
        path0 += '/'
    for elem in listkeys:
        #print('%d == checkLevel : %s' % (nb, elem.GetTitle()))
        if (elem.GetClassName() == "TDirectoryFile"):
            path = path0 + elem.GetName()
            if (nb >= 3 and re.search(inPath, path)):
                print('\npath : %s' % path)
                f_out.mkdir(path)
            tmp = f_rel.Get(path).GetListOfKeys()
            checkLevel(f_rel, f_out, path, tmp, nb+1)
        elif (elem.GetClassName() == "TTree"):
            #print('------ TTree')
            src = f_rel.Get(path0)
            cloned = src.CloneTree()
            #f_out.WriteTObject(cloned, elem.GetName())
            if (nb >= 3 and re.search(inPath, path0)):
                changeDirectory(f_out, path0[:-1])
                cloned.Write()
        elif (elem.GetClassName() != "TDirectory"):
            #print('copy %s object into %s path' % (elem.GetName(), path0[:-1]))
            #f_out.WriteTObject(elem.ReadObj(), elem.GetName())#:"DQMData/Run 1/EgammaV"
            if (nb >= 3 and re.search(inPath, path0)):
                changeDirectory(f_out, path0[:-1])
                elem.ReadObj().Write()

def getListFiles(path):
    #print('path : %s' % path)
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith(".root")] # keep only root files
    #print(onlyfiles)
    return onlyfiles

def func_ReduceSize(nbFiles):
    print("func_ReduceSize")
    
    fileList = getListFiles(folderName) # get the list of the root files in the folderName folder
    fileList.sort()
    print('there is %d files' % len(fileList))
    fileList = fileList[0:nbFiles]
    print('file list :')
    print(fileList)
    print('-- end --')

    for elem in fileList:
        input_file = folderName + str(elem.split()[0])
        print('\n %s' % input_file)

        f_rel = ROOT.TFile(input_file, "UPDATE")
        racine = input_file.split('.')
        f_out = TFile(racine[0] + 'b.' + racine[1], 'recreate')
        t2 = f_rel.GetListOfKeys()
        print(racine[0] + 'b.' + racine[1])
        checkLevel(f_rel, f_out, "", t2, 0)

        f_out.Close()
        f_rel.Close()

    return

if __name__=="__main__":

    # nb of files to be used
    nbFiles = 750

    func_ReduceSize(nbFiles)

    print("Fin !")

