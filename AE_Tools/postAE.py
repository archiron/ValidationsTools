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
#  python3 postAE.py ~/PYTHON/ValidationsTools/CommonFiles/ pathsLLR.py timeFolder

################################################################################

import time
import sys, os, shutil
import importlib
import importlib.machinery
import importlib.util

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

if len(sys.argv) > 1:
    print(sys.argv)
    print("postAE - arg. 0 :", sys.argv[0]) # name of the script
    print("postAE - arg. 1 :", sys.argv[1]) # COMMON files path
    print("postAE - arg. 2 :", sys.argv[2]) # FileName for paths
    print("postAE - arg. 3 :", sys.argv[3]) # dataset name
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
    branch = sys.argv[3][1:]
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
pathOutput = blo.LOG_OUTPUT
print('output path : {:s}'.format(pathOutput))

tic= time.time()

print('{:s}\n'.format(branch))
Name = pathOutput + '/' + branch 
print('{:s}\n'.format(Name))
if os.path.exists(Name): # 
    #os.removedirs(Name) # remove folder Name
    shutil.rmtree(Name) # remove folder Name

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print('end')

