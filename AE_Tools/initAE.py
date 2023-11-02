#!/usr/bin/env python
# coding: utf-8

################################################################################
# initAE : create the condition for the initialization of the AEGeneration.py 
# and copy the necessary files into.
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
#  module use /opt/exp_soft/vo.gridcl.fr/software/modules/
#  module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7
#  module load torch/1.5.0-py37-nocuda
# into the AE_Tools folder, launch :
#  python3 initAE.py ~/PYTHON/ValidationsTools/CommonFiles/ pathsLLR.py timeFolder

################################################################################

import datetime, time
import sys, os, shutil
import importlib
import importlib.machinery
import importlib.util

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

if len(sys.argv) > 1:
    print(sys.argv)
    print("initAE - arg. 0 :", sys.argv[0]) # name of the script
    print("initAE - arg. 1 :", sys.argv[1]) # COMMON files path
    print("initAE - arg. 2 :", sys.argv[2]) # FileName for paths
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
pathOutput = blo.LOG_OUTPUT
#pathOutput = '/home/llr/info/chiron_u/public/tmp' # LLR
print('output path : {:s}'.format(pathOutput))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(pathChiLib))
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

import default as dfo
from default import *
from rootValues import NB_EVTS
from defaultStd import *
from controlFunctions import *
from sources import *

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

# get the branches for ElectronMcSignalHistos.txt
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

nbBranches = len(branches) # [0:8]
print('there is {:03d} datasets'.format(nbBranches)) # 263 branches

pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))

# get list of generated ROOT files
rootPath = pathNb_evts # for CC
#rootPath = "/data_CMS/cms/chiron/ROOT_Files/CMSSW_12_1_0_pre5-ROOTFiles_0950" # LLR
rootFilesList_0 = getListFiles(rootPath, 'root')
print('there is ' + '{:04d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathNb_files = pathCase + '/{:03d}'.format(nbFiles)
print('pathNb_files path : {:s}'.format(pathNb_files))
#data_res = pathNb_files + '/AE_RESULTS/'
#print('data_res path : {:s}'.format(data_res))

t = datetime.datetime.today()
tic= time.time()

loopMaxValue = nbBranches #25 # nbBranches
for i in range(0, loopMaxValue):
    print('{:s}\n'.format(branches[i]))
    Name = pathOutput + '/' + branches[i] 
    print('{:s}\n'.format(Name))
    if not os.path.exists(Name): # create folder Name
        os.makedirs(Name) # create folder Name
    else:
        print('exist {:s}\n'.format(Name))
    
    original = pathNb_files + '/branchesHistos_NewFiles.txt'
    target = Name + '/branchesHistos_NewFiles.txt'
    shutil.copyfile(original, target)
    
    KSlistFiles = []
    tmp = getListFiles(pathNb_files, 'txt')
    for elem in tmp:
        if (elem[5:10] == '_diff'): # to keep only histo_differences_KScurves files
            KSlistFiles.append(elem)
    for item in KSlistFiles:
        print('file : %s' % item)
        original = pathNb_files + '/' + item
        target = Name + '/' + item
        shutil.copyfile(original, target)

    # ChiLib
    NameChiLib = pathOutput + '/ChiLib'
    print('{:s}\n'.format(NameChiLib))
    if not os.path.exists(NameChiLib): # create folder Name
        os.makedirs(NameChiLib) # create folder Name
    else:
        print('exist {:s}\n'.format(NameChiLib))

    tmp = getListFiles(pathChiLib, 'py')
    for item in tmp:
        print('file : %s' % item)
        original = pathChiLib + '/' + item
        target = NameChiLib + '/' + item
        shutil.copyfile(original, target)

    # Commons
    NameCommons = pathOutput + '/CommonFiles'
    print('{:s}\n'.format(NameCommons))
    if not os.path.exists(NameCommons): # create folder Name
        os.makedirs(NameCommons) # create folder Name
    else:
        print('exist {:s}\n'.format(NameCommons))

    tmp = getListFiles(pathCommonFiles, 'py')
    for item in tmp:
        print('file : %s' % item)
        original = pathCommonFiles + '/' + item
        target = NameCommons + '/' + item
        shutil.copyfile(original, target)


toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print('end')

