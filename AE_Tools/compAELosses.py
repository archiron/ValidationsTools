#!/usr/bin/env python
# coding: utf-8

################################################################################
# compAELosses.py : create an AE comparison of the losses for each layer scheme 
# and for different egamma validation releases.
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import datetime, time
import sys, os
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

if len(sys.argv) > 1:
    print(sys.argv)
    print("resumeAE - arg. 0 :", sys.argv[0]) # name of the script
    print("resumeAE - arg. 1 :", sys.argv[1]) # COMMON files path
    print("resumeAE - arg. 2 :", sys.argv[2]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
else:
    print("rien")
    pathBase = ''

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nAE Generation")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(pathChiLib))
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

import default as dfo
from default import *
from rootValues import NB_EVTS
from controlFunctions import *

pathNb_evts = pathBase + '/' + str(NB_EVTS)
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))

pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathNb_files = pathCase + '/{:03d}'.format(nbFiles)
print('pathNb_files path : {:s}'.format(pathNb_files))

pathKS = pathNb_files + '/KS/'
checkFolder(pathKS)
print('\KS folder name : {:s}'.format(pathKS))

t = datetime.datetime.today()
tic= time.time()

lDirsKS = os.listdir(pathKS)
l1 = []
for name in lDirsKS:
    if os.path.isdir(os.path.join(pathKS, name)):
        if (name[0:2] == 'HL'):
            #print('OK %s' % name)
            l1.append(os.path.join(pathKS, name))
        #else:
        #    print(name[0:2])

l2 = []
for name in l1:
    #print(name)
    ldir1 = os.listdir(name)
    for name2 in ldir1:
        #print(name2)
        if os.path.isdir(os.path.join(name, name2)):
            #print(name2)
            l2.append(os.path.join(name, name2))

l3 = []
lossCompAEName = os.path.join(pathKS, 'lossCompAE.txt')
file = open(lossCompAEName, 'w')
percent = 0.05
for name in l2:
    #print(name)
    ldir2 = os.listdir(name)
    for name2 in ldir2:
        #print(os.path.join(name, name2))
        nb_green = 0
        nb_red = 0
        if (not os.path.isdir(os.path.join(name, name2))):
            #print(name2[0:4])
            if (name2[0:4] == 'loss'):
                #print(name2)
                l3.append(os.path.join(name, name2))
                lossName = os.path.join(name, name2)
                loss_file = open(lossName, 'r')
                print(lossName)
                lines = loss_file.readlines()
                #print(lines)
                loss_file.close()
                tmp1 = list(map(float, lines))
                for elem in tmp1:
                    if (elem <= percent):
                        nb_green += 1
                    else:
                        nb_red += 1
                # split name
                scheme = name.split('/')[11]
                timeFolder = name.split('/')[12]
                print('scheme : %s - timeFolder : %s' % (scheme, timeFolder))
                # split name2
                n2 = name2.split('.')[0]
                n2 = n2[7:]
                print('%s' % n2)
                print('green : %d - red : %d' % (nb_green, nb_red))
                file.write('%s : %s : %s : green : %d - red : %d\n' % (scheme, timeFolder, n2, nb_green, nb_red))
        else:
            print('%s KO' % name2)
    print('')

file.close()

#for name in l3:
#    print(name)

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print('end')

