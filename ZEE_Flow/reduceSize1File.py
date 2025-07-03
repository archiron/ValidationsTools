#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# reduceSize1File: a tool to reduce the size of the ROOT files, keeping only
# the used branches.
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys, re

# lines below are only for func_Extract
from sys import argv
from os import listdir
from os.path import isfile, join

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal # ROOT.kBreak # 
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = True

#from ROOT import *

sys.path.append('../ChiLib')

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

def checkLevel(f_rel, f_out, path0, listkeys, nb, inPath):
    #inPath = 'DQMData/Run 1/Info'
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
            checkLevel(f_rel, f_out, path, tmp, nb+1, inPath)
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

if len(sys.argv) > 1:
    print(sys.argv)
    print("reduce - arg. 0 :", sys.argv[0]) # name of the script
    print("reduce - arg. 1 :", sys.argv[1]) # index
    #print("reduce - arg. 2 :", sys.argv[2]) # path
    print("reduce - arg. 2 :", sys.argv[2]) # nb of events
    print("reduce - arg. 3 :", sys.argv[3]) # RESULTFOLDER
    ind = int(sys.argv[1])
    resultPath = sys.argv[3]
    max_number = int(sys.argv[2])
else:
    print("reduce - rien")
    ind = 0
    resultPath = ''
    max_number = 10 # number of events

print("func_ReduceSize")
#input_file = resultPath + '/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_' + '%0004d'%max_number + '_' + '%003d'%ind + '.root'
release = resultPath.split('/')[-1]
#print('release : {:s}'.format(release))
input_file = resultPath + '/DQM_V0001_R000000001__RelValZEE_14__' + release + '__RECO_' + '%0004d'%max_number + '_' + '%003d'%ind + '.root'
racine = input_file.split('.')
output_file = racine[0] + 'b.' + racine[1]

print('\n %s' % input_file)
print('\n %s' % output_file)

paths = ['DQMData/Run 1/EgammaV', 'DQMData/Run 1/Info']

f_rel = ROOT.TFile(input_file, "UPDATE")
f_out = ROOT.TFile(output_file, 'recreate')
t2 = f_rel.GetListOfKeys()
print(racine[0] + 'b.' + racine[1])
for elem in paths:
    checkLevel(f_rel, f_out, "", t2, 0, elem)

f_out.Close()
f_rel.Close()

tmp_file = resultPath + '/tmp' + '%0004d'%max_number + '_' + '%003d'%ind + '.root'
print('\n %s' % tmp_file)
print('move input_file to tmp_file')
os.rename(input_file, tmp_file) # mv input_file -> tmp_file
print('move output_file to input_file')
os.rename(output_file, input_file) # mv output_file -> input_file
print('delete tmp_file')
os.remove(tmp_file) # remove input_file

print("Fin !")

