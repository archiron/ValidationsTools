#! /usr/bin/env python
#-*-coding: utf-8 -*-

# MUST be launched with the cmsenv cmd after a cmsrel cmd !!

import os,sys,subprocess
#import urllib2
import re

import pandas as pd
import numpy as np
import matplotlib

import matplotlib.dates as md
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

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    branches = getBranches(tp_1)
    print(branches[0:10])
    print(branches[5])
    #branches = branches[0:60]

    N_histos = len(branches)
    print('N_histos : %d' % N_histos)
    for i in range(0, N_histos): # 1 histo for debug
        if re.search('OfflineV', branches[i]): # temp (pbm with nbins=81 vs nbins=80)
            print(branches[i])

    print("Fin !")

