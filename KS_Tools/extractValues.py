#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# extreactValues: generate a file for each histo with the values of all ROOT files
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import sys

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

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # LIB path
    print("step 4 - arg. 2 :", sys.argv[2]) # COMMON files path
    print("step 4 - arg. 3 :", sys.argv[3]) # RESULTFOLDER
    resultPath = sys.argv[3]
else:
    print("rien")
    resultPath = ''

Chilib_path = sys.argv[1]
sys.path.append(Chilib_path)
Common_path = sys.argv[2]
sys.path.append(Common_path)

from default import *
from controlFunctions import *
from graphicFunctions import getHisto

# get the branches for ElectronMcSignalHistos.txt
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = []
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

# nb of files to be used
nbFiles = 200

print("func_Extract")
resultPath = checkFolderName(resultPath)    
wr = []
histos = {}

# get the branches for ElectronMcSignalHistos.txt
#branches += ["h_recEleNum", "h_scl_ESFrac_endcaps", "h_scl_sigietaieta", "h_ele_PoPtrue_endcaps", "h_ele_PoPtrue", "h_scl_bcl_EtotoEtrue_endcaps", "h_scl_bcl_EtotoEtrue_barrel", "h_ele_Et"]
#branches += ["h_recEleNum"]
for leaf in branches:
    histos[leaf] = []
    
fileList = getListFiles(resultPath) # get the list of the root files in the folderName folder
fileList.sort()
print('there is %d files' % len(fileList))
fileList = fileList[0:nbFiles]
print('file list :')
print(fileList)
print('-- end --')

for elem in fileList:
    input_file = resultPath + str(elem.split()[0])
    name_1 = input_file.replace(resultPath, '').replace('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_', '').replace('.root', '')
    print('\n %s - name_1 : %s' % (input_file, name_1))
    #print('\n %s - name_1 : %s' % (input_file, colorText(name_1, 'lightyellow')))
    
    f_root = ROOT.TFile(input_file) # 'DATA/' + 
    h1 = getHisto(f_root, tp_1)
    #h1.ls() # OK
    for leaf in branches:
        print("== %s ==" % leaf)
        temp_leaf = []
        histo = h1.Get(leaf)

        temp_leaf.append(histo.GetMean()) # 0
        temp_leaf.append(histo.GetMeanError()) # 2
        temp_leaf.append(histo.GetStdDev()) # 6
        temp_leaf.append(histo.GetEntries()) # 6b

        temp_leaf.append(name_1) # 7
        #print('temp_leaf : %s' % temp_leaf)
            
        texttoWrite = ''
        i=0
        for entry in histo:
            #print(i,entry)
            texttoWrite += 'b_' + '{:03d}'.format(i) + ',c_' + '{:03d},'.format(i)
            temp_leaf.append(entry) # b_
            temp_leaf.append(histo.GetBinError(i)) # c_
            i+=1
        print('there is %d entries' % i)
        texttoWrite = texttoWrite[:-1] # remove last char
        temp_leaf.append(texttoWrite) # end
        histos[leaf].append(temp_leaf)

#print histos into histo named files
i_leaf = 0
for leaf in branches:
    fileName = resultPath + 'histo_' + str(leaf) + '_' + '{:03d}'.format(nbFiles) + '.txt'
    print('fileName : %s' % fileName)
    wr.append(open(fileName, 'w'))
    nb_max = len(histos[leaf][0]) - 1
    print("== %s == nb_max : %d" % (leaf, nb_max))
    wr[i_leaf].write('evol,Mean,MeanError,StdDev,nbBins,name,')
    wr[i_leaf].write(str(histos[leaf][0][nb_max]))
    wr[i_leaf].write('\n')
    #'''
    for i_file in range(0, len(fileList)):
        texttoWrite = str(i_file) + ','
        wr[i_leaf].write(texttoWrite) 
        for i in range(0, nb_max-1):
            #print('i : %d' % i)
            wr[i_leaf].write(str(histos[leaf][i_file][i]))
            wr[i_leaf].write(',')
        wr[i_leaf].write(str(histos[leaf][i_file][nb_max-1]))
        texttoWrite = '\n'
        wr[i_leaf].write(texttoWrite) 
    wr[i_leaf].close()
    i_leaf +=1
    #'''

print("Fin !")

