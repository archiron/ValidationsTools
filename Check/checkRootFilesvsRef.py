#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# checkRootFiles: generate a summary of all ROOT files and all histos
# for egamma validation comparison 
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

from re import split
import sys
import importlib

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv
from xml.sax.handler import DTDHandler

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal # ROOT.kBreak # 
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = True

root_version = ROOT.gROOT.GetVersion()

print('PYTHON     version : {}'.format(sys.version))
print("ROOT       version : {}".format(root_version))

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 2 :", sys.argv[2]) # Check Folder
    print("step 5 - arg. 3 :", sys.argv[3]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    pathLIBS=sys.argv[2][:-6]
    filePaths = sys.argv[3]
else:
    print("rien")
    pathBase = ''

import pandas as pd
import numpy as np

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )

pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))
pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(pathChiLib))
pathDATA = pathLIBS + '/' + blo.DATA_SOURCE
print('DATA_SOURCE : %s' % pathDATA)

sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)
sys.path.append(pathDATA)

import validationsDefault as dfo
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import Graphic
from graphicAutoEncoderFunctions import createCompLossesPicture4
from DecisionBox import DecisionBox
from validationsDefault import *
from filesSources import *

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s} from {:s}'.format(release, input_ref_file))

#folder = pathBase + checkFolderName(dfo.folder)
pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathDATA = checkFolderName(pathDATA)
pathROOTFiles = blo.pathROOT + "/" + release
pathROOTFiles = checkFolderName(pathROOTFiles)
print('pathROOTFiles : {:s}'.format(pathROOTFiles))

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

print("checkRootFilesvsRef")
#pathNb_evts = checkFolderName(pathNb_evts)    

gr = Graphic()
gr.initRoot()
DB = DecisionBox()

N_histos = len(branches)
print('N_histos : %d' % N_histos)

# get the list of the generated ROOT files
fileList = getListFiles(pathROOTFiles, 'root') # get the list of the root files in the folderName folder
fileList.sort()
print('list of the generated ROOT files')
print('there is ' + '{:03d}'.format(len(fileList)) + ' ROOT files')
nbFiles = change_nbFiles(len(fileList), nbFiles)
fileList = fileList[0:nbFiles]
#print('file list :')
#print(fileList)

tmp_branches2 = []
nb_ttl_histos = []
nb_ttl_histos2 = []
rels2 = []
tmpSource1 = []

ii = 0
for elem in fileList:
    if (ii == 500):
        continue
    tmp_branch = []
    nbHistos = 0
    input_file = pathROOTFiles + str(elem.split()[0])
    b = (elem.split('__')[2]).split('-')
    print(colorText(b[0], 'blue'), colorText(b[0][6:], 'green'), elem)
    rels2.append([b[0], b[0][6:], elem])

    name_1 = input_file.replace(pathROOTFiles, '').replace('DQM_V0001_R000000001__RelValZEE_14__'+release+'__RECO_', '').replace('.root', '')
    #print('\n %s - name_1 : %s' % (input_file, name_1))
    print('\n %s - name_1 : %s' % (input_file, colorText(name_1, 'blue')))
    f_root = ROOT.TFile(input_file)
    h_rel = gr.getHisto(f_root, tp_1)
    #h1.ls() # OK

    for i in range(0, N_histos): # 1 N_histos histo for debug
        histo_rel = h_rel.Get(branches[i])
        if (histo_rel):
            d = gr.getHistoConfEntry(histo_rel)
            s_tmp = gr.fill_Snew2(d, histo_rel)
            if (s_tmp.min() < 0.):
                print('pbm whith histo %s, min < 0' % branches[i])
            elif (np.floor(s_tmp.sum()) == 0.):
                print('pbm whith histo %s, sum = 0' % branches[i])
            else:
                nbHistos += 1
                tmp_branch.append(branches[i])
        else:
            print('%s KO' % branches[i])
    nb_ttl_histos2.append(nbHistos)
    tmp_branches2.append(tmp_branch)
    ii += 1
    tmpSource1.append(1)

print('nb_ttl_histos : ', nb_ttl_histos2)
if(len(set(nb_ttl_histos))==1):
    print('All elements are the same with value {:d}.'.format(nb_ttl_histos[0]))
else:
    print('All elements are not the same.')
    print('nb ttl of histos : ' , nb_ttl_histos)
newBranches2 = optimizeBranches(tmp_branches2)

if (len(branches) != len(newBranches2)):
    print('len std branches : {:d}'.format(len(branches)))
    print('len new branches : {:d}'.format(len(newBranches2)))
    branches = newBranches2
    N_histos = len(branches)

print('N_histos : %d' % N_histos)

sortedRels2 = rels2 # keep releases order
# get the "reference" root file datas
f_KSref = ROOT.TFile(pathDATA + '/Run3/RECO/' + input_ref_file)
print('we use the %s file as KS reference' % f_KSref)

h_KSref = gr.getHisto(f_KSref, tp_1)
print(h_KSref)

diffTab2 = pd.DataFrame()
print(diffTab2)
toto = []
for i in range(0, N_histos):#, N_histos-1 range(N_histos - 1, N_histos):  # 1 N_histos histo for debug
    #print('histo : {:s}'.format(branches[i])) # print histo name
    r_rels2 = []
    
    histo_rel = h_rel.Get(branches[i])
    if (histo_rel):
        print('%s OK' % branches[i])

        # create the datas for the p-Value graph
        # by comparing 1 curve with the others.
        histo_KSref = h_KSref.Get(branches[i])
        s_KSref = []
        for entry in histo_KSref:
            s_KSref.append(entry)
        s_KSref = np.asarray(s_KSref)
        s_KSref = s_KSref[1:-1]
        print('s_KSref has {:d} elements'.format(len(s_KSref)))

        #print('\nWorking with sorted rels\n')
        ind_rel = 0
        diffValues = []
        diffValues2 = []
        i_2 = 0
        for elem in sortedRels2:
            #print(elem)
            rel = elem[1]
            file = elem[2]
            # get the "new" root file datas
            f_rel = ROOT.TFile(pathROOTFiles + file)
            #print('we use the {:s} file as new release '.format(file))

            h_rel = gr.getHisto(f_rel, tp_1)
            histo_rel = h_rel.Get(branches[i])

            s_new = []
            for entry in histo_rel:
                s_new.append(entry)
            s_new = np.asarray(s_new)
            s_new = s_new[1:-1]
            if (len(s_KSref) != len(s_new)):
                print('pbm whith histo %s, lengths are not the same' % branches[i])
                continue

            if (s_new.min() < 0.):
                print('pbm whith histo %s, min < 0' % branches[i])
                continue
            if (np.floor(s_new.sum()) == 0.):
                print('pbm whith histo %s, sum = 0' % branches[i])
                continue
                
            # diff max between new & old
            diffMax0, posMax0, sDKS = DB.diffMAXKS3(s_KSref, s_new)

            diffValues.append(diffMax0)
            diffValues2.append(diffMax0)
            name_1 = split('_', file.replace(pathROOTFiles, '').replace('DQM_V0001_R000000001__RelValZEE_14__'+release+'__RECO_', '').replace('.root', ''))[1]
            #print("name_1 %s" % name_1)
            r_rels2.append(str(rel) + '_' + name_1)
            ind_rel += 1
            i_2 += 1
        
        toto.append(diffValues)
        lab = r_rels2
        val = diffValues
        val2 = diffValues2
        print('il y a {:d} points dans les valeurs'.format(len(val)))
        print('il y a {:d} points dans les labels'.format(len(lab)))
        #pictureName = rootfil + '/pngs/maxDiff_comparison_' + branches[i] + '_3.png' # 
        #print(pictureName)
        #title = 'KS cum diff values vs releases. ' + branches[i]
        #createCompLossesPicture4(lab,val,val2, pictureName, title, 'Releases', 'max diff')
    else:
        print('%s KO' % branches[i])
diffTab2 = pd.DataFrame(toto, columns=r_rels2)
globos = diffTab2.mean(axis=0, numeric_only=True)

# generate pictures
dt = globos#.head(50)
lab = list(dt.index.values)
val1 = globos.to_list()

#pictureName = pathROOTFiles + '/pngs/maxDiff_comparison_values_2.png' # 
#print(pictureName)
#title = r"$\bf{total}$" + ' : KS cum diff values vs releases.'
#createCompLossesPicture(lab,val1, pictureName, title, 'Releases', 'max diff')

val2 = []
i = 0
for item in tmpSource1:
    if (item == 1):
        val2.append(val1[i])
    else:
        val2.append(np.nan)
    i += 1
pictureName = pathROOTFiles + '/maxDiff_ROOT_comparison_values_3.png' # 
print(pictureName)
title = r"$\bf{total}$" + ' : KS cum diff values vs releases.'
createCompLossesPicture4(lab, val1, val2, pictureName, title, 'Releases', 'max diff')
