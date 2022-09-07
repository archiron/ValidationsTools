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
import imp, importlib
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

#ROOT.gSystem.Load("libFWCoreFWLite.so")
#ROOT.gSystem.Load("libDataFormatsFWLite.so")
#ROOT.FWLiteEnabler.enable()

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 2 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 4 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
else:
    print("rien")
    resultPath = ''

# these line for daltonians !
#seaborn.set_palette('colorblind')

print('statpValue')

#blu = imp.load_source(filePaths, commonPath+filePaths)
#print('DATA_SOURCE : %s' % blu.DATA_SOURCE)
#resultPath = blu.RESULTFOLDER # checkFolderName(blu.RESULTFOLDER)
#print('result path : {:s}'.format(resultPath))

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(Chilib_path))
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as df
from default import *
from graphicFunctions import createHistoPicture
from controlFunctions import *

folder = checkFolderName(df.folder)
resultPath = checkFolderName(resultPath)

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder += '{:03d}'.format(nbFiles)
folder = checkFolderName(folder)
print('folder après check : %s' % folder)
checkFolder(folder)
folder += 'KS'
folder =checkFolderName(folder)
print('folder après check : %s' % folder)
checkFolder(folder)

rels = []
# get list of the added ROOT files
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('we use the files :')
for item in rootFilesList:
    print('%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])

sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

for elem in sortedRels:
    print(elem)
    rel = elem[1]
    KS_pValues = folder + 'histo_pValues_' + rel + '_v2.txt'
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
    
    createHistoPicture(histo1, folder + 'KS_1_' + rel + '.png')
    createHistoPicture(histo2, folder + 'KS_2_' + rel + '.png')
    createHistoPicture(histo3, folder + 'KS_3_' + rel + '.png')

print("Fin !")

