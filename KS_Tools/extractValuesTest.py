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
import importlib
import importlib.machinery
import importlib.util
import numpy as np

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

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
    print("step 4 - arg. 2 :", sys.argv[2]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
else:
    print("rien")

print("\nextractValues")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('pathChiLib : {:s}'.format(pathChiLib))
print('pathCommonFiles : {:s}'.format(pathCommonFiles))
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

from validationsDefault import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import Graphic
from filesSources import *

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

######## ===== COMMON LINES ===== ########
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########
N_histos = len(branches)
print('N_histos : %d' % N_histos)

print("func_Extract")

pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
pathROOTFiles = blo.pathROOT + "/" + release
pathROOTFiles = checkFolderName(pathROOTFiles)
print('pathROOTFiles : {:s}'.format(pathROOTFiles))

gr = Graphic()
gr.initRoot()
wr = []
histos = {}
tmp_branches = []
nb_ttl_histos = []

# get the branches for ElectronMcSignalHistos.txt
#branches += ["h_recEleNum", "h_scl_ESFrac_endcaps", "h_scl_sigietaieta", "h_ele_PoPtrue_endcaps", "h_ele_PoPtrue", "h_scl_bcl_EtotoEtrue_endcaps", "h_scl_bcl_EtotoEtrue_barrel", "h_ele_Et"]
#branches += ["h_recEleNum"]
for leaf in branches:
    histos[leaf] = []

# get list of the added ROOT files for comparison
pathDATA = os.path.join(pathLIBS, blo.DATA_SOURCE + '/Run3/RECO/') # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
print('pathDATA : {:s}'.format(pathDATA))
rootFilesList = getListFiles(pathDATA, 'root')
rootFilesList2 = []
rootList2 = os.path.join(pathLIBS, blo.DATA_SOURCE + '/Values/rootSourcesRelValZEE_14mcRun3RECO/rootSourcesRelValZEE_14mcRun3RECO.txt')

sourceList = open(rootList2, "r")
for ligne in sourceList:
    t_ligne = ligne.replace('_0.txt', '.root')
    t_ligne = t_ligne.replace('_1.txt', '.root')
    rootFilesList2.append(t_ligne.rstrip())
compteur = 0
for item in rootFilesList2:
    print('\n{:2d} : {:s}'.format(compteur, item))
    compteur += 1
rootFilesList3 = []
for item in rootFilesList2: 
    if item not in rootFilesList3: 
        rootFilesList3.append(item)
        print(item)
print('Root files List have {:d} files'.format(len(rootFilesList3)))

print('we use the files :')
print('there is ' + '{:03d}'.format(len(rootFilesList3)) + ' added ROOT files')
if (len(rootFilesList3) == 0):
    print('no added ROOT files to work with. Existing.')
    exit()
for item in rootFilesList3:
    print('%s' % item)
fileList = rootFilesList3[0:nbFiles]
print('file list :')
print(fileList)

for item in rootFilesList3:
    tmp_branch = []
    nbHistos = 0
    print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    #rels.append([b[0], b[0][6:], item])
    name = os.path.join(pathDATA, item)
    print('{:s} : {:d}'.format(item, len(item)))
    print('name : {:s}'.format(name))
    f_root = ROOT.TFile(name)
    h_rel = gr.getHisto(f_root, tp_1)
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
    nb_ttl_histos.append(nbHistos)
    tmp_branches.append(tmp_branch)

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

nbFiles = change_nbFiles(len(fileList), nbFiles)

for elem in fileList:
    input_file = pathDATA + str(elem.split()[0])
    name_1 = input_file.replace(pathDATA, '').replace('DQM_V0001_R000000001__RelValZEE_14__CMSSW_14_1_0__RECO_', '').replace('.root', '')
    #print('\n %s - name_1 : %s' % (input_file, name_1))
    print('\n {:s} - name_1 : {:s}'.format(input_file, colorText(name_1, 'green')))
    tmp_0 = name_1.split('__')[2]
    tmp_1 = tmp_0.split('-')
    release = tmp_1[0]
    tmp_2 = tmp_1[1].split('_',1)
    GT = tmp_2[0]
    extent = tmp_2[1]
    print(' {:s} : {:s} - {:s}'.format(colorText(release, 'blue'), colorText(GT, 'green'), colorText(extent, 'green')))

    f_root = ROOT.TFile(input_file) # 
    h1 = gr.getHisto(f_root, tp_1)
    for leaf in branches:
        print("== %s ==" % leaf)
        temp_leaf = []
        histo = h1.Get(leaf)
        if (histo):
            d = gr.getHistoConfEntry(histo)

            temp_leaf.append(histo.GetMean()) # 0
            temp_leaf.append(histo.GetMeanError()) # 2
            temp_leaf.append(histo.GetStdDev()) # 6
            temp_leaf.append(histo.GetEntries()) # 6b
            
            #temp_leaf.append(name_1) # 7
            #print('temp_leaf : %s' % temp_leaf)

            temp_leaf.append(release)
            temp_leaf.append(GT)
                
            texttoWrite = ''
            i=0
            for entry in histo:
                #print(i,entry)
                texttoWrite += 'b_' + '{:03d}'.format(i) + ',c_' + '{:03d},'.format(i)
                if ( d == 1):
                    temp_leaf.append(entry) # b_
                else :
                    if ((histo.GetBinEntries(i) == 0.) and (entry == 0.)):
                        temp_leaf.append(0.)
                    elif ((histo.GetBinEntries(i) == 0.) and (entry != 0.)):
                        temp_leaf.append(1.e38)
                        print('========================================',i,entry,histo.GetBinEntries(i))
                    else:
                        temp_leaf.append(entry/histo.GetBinEntries(i))
                temp_leaf.append(histo.GetBinError(i)) # c_
                i+=1
            #print('there is %d entries' % i)
            texttoWrite = texttoWrite[:-1] # remove last char
            temp_leaf.append(texttoWrite) # end
            histos[leaf].append(temp_leaf)
        else:
            print('%s KO' % leaf)


#print histos into histo named files
i_leaf = 0
for leaf in branches:
    histo = h1.Get(leaf)
    if (histo):
        fileName = pathROOTFiles + 'histo_' + str(leaf) + '_' + '{:03d}'.format(nbFiles) + '.txt'
        print('{:02d} fileName : {:s}'.format(i_leaf, fileName))
        wr.append(open(fileName, 'w'))
        nb_max = len(histos[leaf][0]) - 1
        #print("== %s == nb_max : %d" % (leaf, nb_max))
        wr[i_leaf].write('evol,Mean,MeanError,StdDev,nbBins,release,GT,')
        wr[i_leaf].write(str(histos[leaf][0][nb_max]))
        wr[i_leaf].write('\n')
        
        for i_file in range(0, len(fileList)):
            texttoWrite = str(i_file) + ','
            wr[i_leaf].write(texttoWrite) 
            for i in range(0, nb_max-1):
                wr[i_leaf].write(str(histos[leaf][i_file][i]))
                wr[i_leaf].write(',')
            wr[i_leaf].write(str(histos[leaf][i_file][nb_max-1]))
            texttoWrite = '\n'
            wr[i_leaf].write(texttoWrite) 
        wr[i_leaf].close()
        i_leaf +=1

print("Fin !")

