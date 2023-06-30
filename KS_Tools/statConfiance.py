#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# statConfiance: a tool to extract/generate pictutes from confiance values
# generated from createFiles_v2 tool for egamma validation comparison.
# display the nb of histograms.
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

print('statConfiance')

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
from functions import Tools

tl = Tools()

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

pathDBox = pathKS + 'DBox'
pathDBox =checkFolderName(pathDBox)
checkFolder(pathDBox)

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

stats = []
# get the .txt files into the DBox folder
textFilesList = getListFiles(pathDBox, 'txt')
for item in textFilesList:
    #print('%s' % item)
    shN = ''
    test = True
    fileName = pathDBox + '/' + item
    f1 = open(fileName, 'r')
    for line in f1:
        if (line[0:34] == '<td> <p><b>confiance : </b></p><p>'):
            tmp1 = line.replace('<td> <p><b>confiance : </b></p><p>', "")
            tmp1 = tmp1.replace('<br>', "")
            tmp1 = tmp1.replace('<p>', "")
            tmp1 = tmp1.replace('</p>', "")
            tmp1 = tmp1.replace('\n', "")
            tmp1 = tmp1.split('coeff ')
            print(tmp1)
            c1 = tmp1[1].replace('1 : ', "")
            if ('nan' in c1):
                test = False
            c2 = tmp1[2].replace('2 : ', "")
            if ('nan' in c2):
                test = False
            c3 = tmp1[3].replace('3 : ', "")
            if ('nan' in c3):
                test = False
            print('c1 : {:s}, c2 : {:s}, c3 : {:s}'.format(c1, c2, c3))
        if (line[0:30] == '<td><div><a href="KS-ttlDiff_3'):
            tmp1 = line.replace('<td><div><a href="KS-ttlDiff_3_', "")
            tmp2 = tmp1.split(".png")
            shN =  tl.shortHistoName_0(tmp2[0])
            print('{:s}'.format(shN))
    if ( (shN != '') and test):
        tmp3 = fileName.split(shN)
        release = tmp3[1][1:].replace('.txt', "")
        print(release)
        stats.append([shN, release, c1, c2, c3])

#sortedStats = sorted(stats, key = lambda x: x[0]) # gives an array with releases sorted
for elem in stats:
    print(elem)

N = len(stats)
print('N : {:d}'.format(N))
print('rels : {:d}'.format(len(sortedRels)))

stats2 = []
for elem in sortedRels:
    C1 = 0 # sum of all c1 for a release
    C2 = 0 # sum of all c1 for a release
    C3 = 0 # sum of all c1 for a release

    #print(elem)
    nb1 = 0
    nb2 = 0
    nb3 = 0
    rel = elem[1]
    for item in stats:
        if (rel == item[1]):
            nb1 += 1
            nb2 += 1
            nb3 += 1
            C1 += float(item[2])
            C2 += float(item[3])
            C3 += float(item[4])
    stats2.append([rel, C1/nb1, C2/nb2, C3/nb3])

for elem in stats2:
    print(elem)

print("Fin !")

