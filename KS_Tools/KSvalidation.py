#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# createFiles: create file for Kolmogorov-Smirnov maximum diff
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys
import importlib
import importlib.machinery
import importlib.util

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
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
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nKSvalidation")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as dfo
from default import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import *
from graphicFunctions import Graphic
from DecisionBox import DecisionBox 
from sources import *
from functions import Tools
from valEnv_default import env_default

sys.path.append(os.getcwd()) # path where you work

DB = DecisionBox()
tl = Tools()
gr = Graphic()
valEnv_d = env_default()

resultPath += '/' + str(NB_EVTS)
resultPath = checkFolderName(resultPath)
print('resultPath : {:s}'.format(resultPath))
resultPath = checkFolderName(resultPath)
#folder = checkFolderName(dfo.folder)
folder = resultPath + checkFolderName(dfo.folder)

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

rels = []
tmp_branches = []
nb_ttl_histos = []

N_histos = len(branches)
print('N_histos : %d' % N_histos)

# create folder 
if not os.path.exists(folder):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST: # the folder did not exist
            raise  # raises the error again
    print('Creation of %s release folder\n' % folder)
else:
    print('Folder %s already created\n' % folder)

# get list of added ROOT files for comparison
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('we use the files :')
for item in rootFilesList:
    tmp_branch = []
    nbHistos = 0
    print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:], item])
    f_root = ROOT.TFile(rootFolderName + item)
    h_rel = getHisto(f_root, tp_1)
    for i in range(0, N_histos): # 1 N_histos histo for debug
        histo_rel = h_rel.Get(branches[i])
        if (histo_rel):
            print('%s OK' % branches[i])
            d = getHistoConfEntry(histo_rel)
            s_tmp = fill_Snew2(d, histo_rel)
            #s_tmp = fill_Snew(histo_rel)
            if (s_tmp.min() < 0.):
                print('pbm whith histo %s, min < 0' % branches[i])
            elif (np.floor(s_tmp.sum()) == 0.):
                print('pbm whith histo %s, sum = 0' % branches[i])
            else:
                nbHistos += 1
                tmp_branch.append(branches[i])
        else:
            print('%s KO' % branches[i])
    nb_ttl_histos.append(nbHistos)
    tmp_branches.append(tmp_branch)

print('nb_ttl_histos : ', nb_ttl_histos)
if(len(set(nb_ttl_histos))==1):
    print('All elements are the same with value {:d}.'.format(nb_ttl_histos[0]))
else:
    print('All elements are not the same.')
    print('nb ttl of histos : ' , nb_ttl_histos)
newBranches = optimizeBranches(tmp_branches)

if (len(branches) != len(newBranches)):
    print('len std branches : {:d}'.format(len(branches)))
    print('len new branches : {:d}'.format(len(newBranches)))
    branches = newBranches
    N_histos = len(branches)
print('N_histos : %d' % N_histos)

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder += '{:03d}'.format(nbFiles)
folder = checkFolderName(folder)
checkFolder(folder)
folderNB = folder
folderKS = folder + 'KS'
folderKS =checkFolderName(folderKS)
checkFolder(folderKS)

sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

folderDBox = folderKS + 'DBox'
folderDBox =checkFolderName(folderDBox)
checkFolder(folderDBox)

f_KSref = ROOT.TFile(rootFolderName + input_ref_file)
print('we use the %s file as KS reference' % input_ref_file)
h_KSref = getHisto(f_KSref, tp_1)
Release = input_ref_file.split('-')[0]
Release = Release.split('__')[2]
shortRelease = Release[6:] # get the KS reference release without the "CMSSW_"

# get histoConfig file
histoArray = []
hCF = Chilib_path + '/HistosConfigFiles/ElectronMcSignalHistos.txt'
fCF = open(hCF, 'r') # histo Config file
for line in fCF:
    if (len(line) != 0):
        if "/" in line:
            histoArray.append(line)
            #print(line)

for i in range(0, N_histos): # 1 N_histos histo for debug len(histoArray)
    print(histoArray[i]) # print histo name
    
    short_histo_name, short_histo_names, histo_positions = tl.shortHistoName(histoArray[i])
    histo_rel = h_rel.Get(short_histo_names[0])
    if (histo_rel):
        print('%s OK' % short_histo_names[0])
        ycFlag = False
        print('\nshort histo name : {:s}'.format(short_histo_names[0]))
        
        for elem in sortedRels:
            print(elem)
            rel = elem[1]
            file = elem[2]

            gif_name = folderDBox + short_histo_names[0] + '_' + rel + ".gif"
            short_png_name = short_histo_names[0] + '_' + rel+ "_n.png"
            png_name = folderDBox + short_png_name # for DB yellow curves
            short_png_cumul_name = short_histo_names[0] + '_' + rel+ "_cum.png"
            png_cumul_name = folderDBox + short_png_cumul_name # for DB yellow curves
            config_target_name = folderDBox + 'definitions_' + rel+ ".txt"

            # get the "new" root file datas
            input_rel_file = file
            f_rel = ROOT.TFile(rootFolderName + input_rel_file)
            h1 = getHisto(f_rel, tp_1)
            histo_1 = h1.Get(short_histo_names[0]) #
            histo_2 = h_KSref.Get(short_histo_names[0]) #

            if ( i == 0 ): # we create the config file only once !
                datas = []
                datas.append('ZEE_14') # we only work with ZEE_14
                datas.append('RECO') # only RECO. PU later
                datas.append(shortRelease)
                datas.append(str(input_ref_file)) # ROOT KS reference file
                datas.append('RECO') # only RECO. PU later
                datas.append(rel)
                datas.append(str(input_rel_file)) # ROOT file to compare
                datas.append(Release)
                datas.append('CMSSW_' + rel)
                datas.append('')
                datas.append('ElectronMcSignalHistos.txt') # can be other histo file
                tl.createDefinitionsFile(datas, config_target_name)
                DB.generateExplanation(folderDBox)

            fHisto = open(folderDBox + short_histo_name + '_' + rel + '.txt', 'w') # web page
            fHisto.write('<table border="1" bordercolor=green cellpadding="2" style="margin-left:auto;margin-right:auto">' + '\n')
            KS_Path1 = folderDBox.replace('/sps/cms/chiron/Validations/', 'https://llrvalidation.in2p3.fr/') # /sps/cms/chiron/Validations/ /data_CMS/cms/chiron/Validations/
            KS_Path0 = folderNB # 
            KS_values_1 = DB.decisionBox1(short_histo_names[0], histo_1, histo_2, KS_Path0, rel) # , shortRelease
            KS_values_2 = DB.decisionBox2(short_histo_names[0], histo_1, histo_2, KS_Path0, rel) # , shortRelease
            KS_values_3 = DB.decisionBox3(short_histo_names[0], histo_1, histo_2, KS_Path0, rel) # , shortRelease

            if (len(KS_values_1) > 5):
                yellowCurves = [ KS_values_1[5] ]
                yellowCurvesCum = [ KS_values_1[6] ]
                ycFlag = True
        
            print('ycFlag : {:s}/{:s} : {:d}'.format(short_histo_names[0], rel, ycFlag))
            gr.PictureChoice(histo_1, histo_2, histo_positions[1], histo_positions[2], gif_name, 0) # 

            if ycFlag:
                gr.PictureChoice_DB( histo_1, histo_2, histo_positions[1], histo_positions[2], png_name, 0, yellowCurves)
                gr.PictureChoice_DB3(histo_1, histo_2, histo_positions[1], histo_positions[2], png_cumul_name, 0, yellowCurvesCum)

                percentage = 0.05
                #if ( KS_values_1[4] >= percentage ):
                if ( KS_values_3[1] >= percentage ):
                    DB_picture = valEnv_d.imageOK() # color = 'green'
                else:
                    DB_picture = valEnv_d.imageKO() # color = 'red'
            KS_V = [KS_values_1, KS_values_2, KS_values_3]
            #KS_V = [KS_values_1]
            Names = [short_histo_name, gif_name, short_histo_names[0], short_png_name, short_png_cumul_name]
            DB.DBwebPage2(fHisto, Names, KS_V, DB_picture, KS_Path1, ycFlag) # , KS_Path0, rel, shortReference, self.webURL, self.shortWebFolder, dataSetFolder

        fHisto.close()

print("Fin !\n")
