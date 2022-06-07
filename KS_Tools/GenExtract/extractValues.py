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

sys.path.append('../ChiLib_CMS_Validation')
import default as df
from default import *
from sources import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

def checkFolderName(folderName):
    if folderName[-1] != '/':
        folderName += '/'
    return folderName

def getHisto(file, tp):
    #t1 = file.Get("DQMData")
    #t2 = t1.Get("Run 1")
    #t3 = t2.Get("EgammaV")
    #t4 = t3.Get("Run summary")
    #t5 = t4.Get(tp)
    path = 'DQMData/Run 1/EgammaV/Run summary/' + tp
    t_path = file.Get(path)
    return t_path # t5

def changeColor(color):
    # 30:noir ; 31:rouge; 32:vert; 33:orange; 34:bleu; 35:violet; 36:turquoise; 37:blanc
    # other references at https://misc.flogisoft.com/bash/tip_colors_and_formatting
    if (color == 'black'):
        return '[30m'
    elif (color == 'red'):
        return '[31m'
    elif (color == 'green'):
        return '[32m'
    elif (color == 'orange'):
        return '[33m'
    elif (color == 'blue'):
        return '[34m'
    elif (color == ''):
        return '[35m'
    elif (color == 'purple'):
        return '[36m'
    elif (color == 'turquoise'):
        return '[37m'
    elif (color == 'lightyellow'):
        return '[93m'
    else:
        return '[30m'

def colorText(sometext, color):
    return '\033' + changeColor(color) + sometext + '\033[0m'

def getListFiles(path):
    #print('path : %s' % path)
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith(".root")] # keep only root files
    #print(onlyfiles)
    return onlyfiles

def getBranches(t_p):
    b = []
    #source = open("../ChiLib_CMS_Validation/HistosConfigFiles/ElectronMcSignalHistos.txt", "r")
    source = open("/pbs/home/c/chiron/private/KS_Tools/ChiLib_CMS_Validation/HistosConfigFiles/ElectronMcSignalHistos.txt", "r")
    for ligne in source:
        if t_p in ligne:
            #print(ligne)
            tmp = ligne.split(" ", 1)
            #print(tmp[0].replace(t_p + "/", ""))
            b.append(tmp[0].replace(t_p + "/", ""))
    source.close()
    return b

def cleanBranches(branches):
    #if (branches[i] == 'h_ele_seedMask_Tec'): # temp (pbm with nan)
    #if re.search('OfflineV', branches[i]): # temp (pbm with nbins=81 vs nbins=80)
    toBeRemoved = ['h_ele_seedMask_Tec'] # , 'h_ele_convRadius', 'h_ele_PoPtrue_golden_barrel', 'h_ele_PoPtrue_showering_barrel'
    for ele in toBeRemoved:
        if ele in branches:
            branches.remove(ele)

def func_Extract(br, nbFiles): # read files
    print("func_Extract")
    df.folderName = checkFolderName(df.folderName)    
    branches = []
    wr = []
    histos = {}
        

    # get the branches for ElectronMcSignalHistos.txt
    #branches += ["h_recEleNum", "h_scl_ESFrac_endcaps", "h_scl_sigietaieta", "h_ele_PoPtrue_endcaps", "h_ele_PoPtrue", "h_scl_bcl_EtotoEtrue_endcaps", "h_scl_bcl_EtotoEtrue_barrel", "h_ele_Et"]
    #branches += ["h_recEleNum"]
    branches = br
    for leaf in branches:
        histos[leaf] = []
    
    fileList = getListFiles(df.folderName) # get the list of the root files in the folderName folder
    fileList.sort()
    print('there is %d files' % len(fileList))
    fileList = fileList[0:nbFiles]
    print('file list :')
    print(fileList)
    print('-- end --')

    for elem in fileList:
        input_file = df.folderName + str(elem.split()[0])
        name_1 = input_file.replace(df.folderName, '').replace('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_', '').replace('.root', '')
        print('\n %s - name_1 : %s' % (input_file, colorText(name_1, 'lightyellow')))
        
        f_root = ROOT.TFile(input_file) # 'DATA/' + 
        h1 = getHisto(f_root, tp_1)
        #h1.ls()
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
        wr.append(open(df.folderName + 'histo_' + str(leaf) + '_' + '{:03d}'.format(nbFiles) + '_0_lite.txt', 'w'))
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
    return

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    branches = getBranches(tp_1)
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    # nb of files to be used
    nbFiles = 200

    func_Extract(branches, nbFiles) # create file with histo datas.

    print("Fin !")

