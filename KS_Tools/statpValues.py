#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# statValues: a tool to extract/generate pictutes from p-Values
# generated from createFiles_v2 tool for egamma validation comparison.
# display the nb of histograms as a function of the pValue value.
# Asked by Florian.
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

#import seaborn # only with cmsenv on cca.in2p3.fr

from sys import argv

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 2 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 4 :", sys.argv[2]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS=sys.argv[1][:-12]
else:
    print("rien")
    pathBase = ''

# these line for daltonians !
#seaborn.set_palette('colorblind')

print('statpValue')

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

import default as df
from default import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import createHistoPicture

######## ===== COMMON LINES ===== ########
pathNb_evts = pathBase + '/' + str(NB_EVTS)
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
######## ===== COMMON LINES ===== ########
pathCase = pathNb_evts + checkFolderName(df.folder)

# get list of generated ROOT files
rootFilesList_0 = getListFiles(pathNb_evts, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

pathNb_files = pathCase + '{:03d}'.format(nbFiles)
pathNb_files = checkFolderName(pathNb_files)
print('folder après check : %s' % pathNb_files)
checkFolder(pathNb_files)
pathKS = pathNb_files + 'KS'
pathKS =checkFolderName(pathKS)
print('folder KS après check : %s' % pathKS)
checkFolder(pathKS)

rels = []
# get list of the added ROOT files
pathDATA = pathLIBS + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(pathDATA, 'root')
print('we use the files :')
for item in rootFilesList:
    print('%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])

sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

for elem in sortedRels:
    print(elem)
    rel = elem[1]
    KS_pValues = pathNb_files + 'histo_pValues_' + rel + '.txt'
    print("KSname 2 : %s" % KS_pValues)
    wKSp = open(KS_pValues, 'r')

    histo1= TH1F('KS 1', 'KS 1 : ' + rel + '', 100,0.,1.)
    histo2= TH1F('KS 2', 'KS 2 : ' + rel + '', 100,0.,1.)
    histo3= TH1F('KS 3', 'KS 3 : ' + rel + '', 100,0.,1.)

    v = wKSp.readlines()
    for elem in v:
        print(elem)
        a = elem.split(',')
        print(a)
        histo1.Fill(float(a[1])) # pvalue1
        histo2.Fill(float(a[2])) # pvalue2
        histo3.Fill(float(a[3])) # pvalue3
    
    createHistoPicture(histo1, pathKS + 'KS_1_' + rel + '.png')
    createHistoPicture(histo2, pathKS + 'KS_2_' + rel + '.png')
    createHistoPicture(histo3, pathKS + 'KS_3_' + rel + '.png')

print("Fin !")

