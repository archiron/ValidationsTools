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

import os,sys, re
import time

import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

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
from graphicFunctions import getHisto
from default import *
from DecisionBox import DecisionBox
from sources import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

def getHistoConfEntry(h1):
    d = 1

    if ( h1.InheritsFrom("TH2") ):
        print('TH2')
    elif ( h1.InheritsFrom("TProfile") ):
        print('TProfile')
        d = 0
    elif ( h1.InheritsFrom("TH1")): # TH1
        print('TH1')
    else:
        print("don't know")

    return d

def checkFolderName(folderName):
    if folderName[-1] != '/':
        folderName += '/'
    return folderName

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
            checkLevel(f_rel, f_out, path, tmp, nb+1)
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

def getListFiles(path):
    #print('path : %s' % path)
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith(".root")] # keep only root files
    #print(onlyfiles)
    return onlyfiles

def getBranches(t_p):
    b = []
    source = open("../ChiLib_CMS_Validation/HistosConfigFiles/ElectronMcSignalHistos.txt", "r")
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

def diffR2(s0,s1):
    s0 = np.asarray(s0) # if not this, ind is returned as b_00x instead of int value
    s1 = np.asarray(s1)
    #print('s0[%d] - s1[%d]' %(len(s0), len(s1)))
    N = len(s0)
    #print('diffR2 : %d' % N)
    R0 = 0.
    for i in range(0, N):
        t0 = s0[i]- s1[i]
        R0 += t0 * t0
    return R0/N

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
            d = getHistoConfEntry(histo)
            print("d = {}".format(d))

            temp_leaf.append(histo.GetMean()) # 0
            temp_leaf.append(histo.GetMeanError()) # 2
            temp_leaf.append(histo.GetStdDev()) # 6
            temp_leaf.append(histo.GetEntries()) # 6b

            temp_leaf.append(name_1) # 7
            #print('temp_leaf : %s' % temp_leaf)
            
            texttoWrite = ''
            i=0
            if (d == 1):
                for entry in histo:
                    #print(i,entry)
                    texttoWrite += 'b_' + '{:03d}'.format(i) + ',c_' + '{:03d},'.format(i)
                    temp_leaf.append(entry) # b_
                    temp_leaf.append(histo.GetBinError(i)) # c_
                    i+=1
            else:
                for entry in histo:
                    print(i,entry)
                    #print("%d/%d : %f - %f - %f") % (i, histo.GetXaxis().GetNbins()+2, entry, histo.GetBinError(i), histo.GetBinEntries(i))
                    print(i, histo.GetXaxis().GetNbins()+2, entry, histo.GetBinError(i), histo.GetBinEntries(i))
                    texttoWrite += 'b_' + '{:03d}'.format(i) + ',c_' + '{:03d},'.format(i)
                    if ((histo.GetBinEntries(i) == 0.) and (entry == 0.)):
                        temp_leaf.append(0.)
                    elif ((histo.GetBinEntries(i) == 0.) and (entry != 0.)):
                        temp_leaf.append(1.e38)
                    else:
                        temp_leaf.append(entry/histo.GetBinEntries(i)) # b_
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

def func_ReduceSize(nbFiles):
    print("func_ReduceSize")
    df.folderName = checkFolderName(df.folderName)
    
    fileList = getListFiles(df.folderName) # get the list of the root files in the folderName folder
    fileList.sort()
    print('there is %d files' % len(fileList))
    fileList = fileList[0:nbFiles]
    print('file list :')
    print(fileList)
    print('-- end --')

    for elem in fileList:
        input_file = df.folderName + str(elem.split()[0])
        print('\n %s' % input_file)

        paths = ['DQMData/Run 1/EgammaV', 'DQMData/Run 1/Info']

        f_rel = ROOT.TFile(input_file, "UPDATE")
        racine = input_file.split('.')
        f_out = TFile(racine[0] + 'b.' + racine[1], 'recreate')
        t2 = f_rel.GetListOfKeys()
        print(racine[0] + 'b.' + racine[1])
        for elem in paths:
            checkLevel(f_rel, f_out, "", t2, 0, elem)

        f_out.Close()
        f_rel.Close()

    return

def func_CreateKS(br, nbFiles):
    DB = DecisionBox()
    print("func_Extract")
    df.folderName = checkFolderName(df.folderName)
    df.folder = checkFolderName(df.folder)

    branches = br
    N_histos = len(branches)
    print('N_histos : %d' % N_histos)
    
    # nb of bins for sampling
    nbins = 100 
    
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

    # get the "new" root file datas
    f_rel = ROOT.TFile(input_rel_file)

    # get the "reference" root file datas
    f_ref = ROOT.TFile(input_ref_file)

    print('we use the %s file as reference' % input_ref_file)
    print('we use the %s file as new release' % input_rel_file)

    nb_red1 = 0
    nb_green1 = 0
    nb_red2 = 0
    nb_green2 = 0
    nb_red3 = 0
    nb_green3 = 0

    KS_diffName = df.folder + "histo_differences_KScurve.txt"
    print("KSname 1 : %s" % KS_diffName)
    wKS0 = open(KS_diffName, 'w')

    KS_resume = df.folder + "histo_resume.txt"
    print("KSname 0 : %s" % KS_resume)
    wKS_ = open(KS_resume, 'w')

    KS_pValues = df.folder + "histo_pValues.txt"
    print("KSname 2 : %s" % KS_pValues)
    wKSp = open(KS_pValues, 'w')

    ind_reference = 1#99 # np.random.randint(0, nbFiles)
    print('reference ind. : %d' % ind_reference)

    tic = time.time()

    for i in range(0, N_histos): # 1 histo for debug
        name = df.folderName + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + "_0_lite.txt"
        print('\n%d - %s' %(i, name))
        df = pd.read_csv(name)
        
        h1 = getHisto(f_rel, tp_1)
        #print("h1")
        #print(h1)
        print(branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])
        ii=0
        s_new = []
        e_new = []
        for entry in histo_1:
            #print("%d/%d : %s - %s") % (ii, histo_1.GetXaxis().GetNbins(), entry, histo_1.GetBinError(i))
            s_new.append(entry)
            e_new.append(histo_1.GetBinError(ii))
            ii += 1
        s_new = np.asarray(s_new)
        s_new = s_new[1:-1]
        e_new = e_new[1:-1]
        #print(s_new)
        #print(e_new)
        Ntot_h1 = histo_1.GetEntries()

        h2 = getHisto(f_ref, tp_1)
        #print("h2")
        #print(h2)
        print(branches[i]) # print histo name
        histo_2 = h2.Get(branches[i])
        ii=0
        s_old = []
        e_old = []
        for entry in histo_2:
            #print("%d/%d : %s - %s") % (ii, histo_2.GetXaxis().GetNbins(), entry, histo_2.GetBinError(i))
            s_old.append(entry)
            e_old.append(histo_2.GetBinError(ii))
            ii += 1
        s_old = np.asarray(s_old)
        s_old = s_old[1:-1]
        e_old = e_old[1:-1]
        #print(s_old)
        #print(e_old)
        Ntot_h2 = histo_2.GetEntries()
        print('Ntot_h1 : %d - Ntot_h2 : %d' % (Ntot_h1, Ntot_h2))

        # print min/max for the new curve
        print('\n##########')
        print('min : %f' % s_new.min())
        print('max : %f' % s_new.max())
        print('###########\n')
        if (s_new.min() < 0.):
            print('pbm whith histo %s, min < 0' % branches[i])
            continue
        if (np.floor(s_new.sum()) == 0.):
            print('pbm whith histo %s, sum = 0' % branches[i])
            continue

        # create file for KS curve
        KSname1 = df.folder + "histo_" + branches[i] + "_KScurve1.txt"
        KSname2 = df.folder + "histo_" + branches[i] + "_KScurve2.txt"
        KSname3 = df.folder + "histo_" + branches[i] + "_KScurve3.txt"
        print("KSname 1 : %s" % KSname1)
        print("KSname 2 : %s" % KSname2)
        print("KSname 3 : %s" % KSname3)
        wKS1 = open(KSname1, 'w')
        wKS2 = open(KSname2, 'w')
        wKS3 = open(KSname3, 'w')

        # check the values & errors data
        #print(df.head(5))
        cols = df.columns.values
        n_cols = len(cols)
        print('nb of columns for histos : %d' % n_cols)
        cols_entries = cols[6::2]
        df_entries = df[cols_entries]
        #print(df_entries.head(15))#
        #stop

        # nbBins (GetEntries())
        df_GetEntries = df['nbBins']

       # get nb of columns & rows for histos
        (Nrows, Ncols) = df_entries.shape
        print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))
        df_entries = df_entries.iloc[:, 1:Ncols-1]
        (Nrows, Ncols) = df_entries.shape
        print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))

        # create the datas for the p-Value graph
        # by comparing all curves between them. (KS 1)
        nb1 = 0
        totalDiff = []
        for k in range(0,Nrows-1):
            for l in range(k+1, Nrows):
                nb1 += 1
                series0 = df_entries.iloc[k,:]
                series1 = df_entries.iloc[l,:]     
                sum0 = df_GetEntries[k]
                sum1 = df_GetEntries[l]
                totalDiff.append(DB.diffMAXKS(series0, series1, sum0, sum1)[0]) # 9000, 9000

        print('ttl nb1 of couples 1 : %d' % nb1)

        # create the datas for the p-Value graph
        # by comparing 1 curve with the others.
        # Get a random histo as reference (KS 2)
            #ind_reference = np.random.randint(0, Nrows)
            #print('reference ind. : %d' % ind_reference)
        series_reference = df_entries.iloc[ind_reference,:]
        nbBins_reference = df_GetEntries[ind_reference]
        print('nb bins reference : %d' % nbBins_reference)
        #print(series_reference)
        nb2 = 0
        totalDiff2 = []
        for k in range(0,Nrows-0):
            if (k != ind_reference):
                nb2 += 1
                series0 = df_entries.iloc[k,:]
                sum0 = df_GetEntries[k]
                totalDiff2.append(DB.diffMAXKS(series0, series_reference, sum0, nbBins_reference)[0]) # 9000, 9000

        print('ttl nb of couples 2 : %d' % nb2)
        #stop
    
        # create the datas for the p-Value graph
        # by comparing the new curve with the others.
        # Get the new as reference (KS 3)
        #print("s_new : ")
        #print(s_new)
        nb3 = 0
        totalDiff3 = []
        for k in range(0,Nrows-0):
            nb3 += 1
            series0 = df_entries.iloc[k,:]
            sum0 = df_GetEntries[k]
            totalDiff3.append(DB.diffMAXKS(series0, s_new, sum0, Ntot_h1)[0])

        print('ttl nb of couples 3 : %d' % nb3)
    
        # plot some datas (in fact doing nothing but creating fig)
        plt_entries = df_entries.plot(kind='line')
        fig = plt_entries.get_figure()
        fig.clf()
        # create the integrated curve
        curves = []
        for k in range(0,Nrows):
            series0 = df_entries.iloc[k,:]
            curves = DB.funcKS(series0)
            plt.plot(curves)
        fig.savefig(df.folder + '/cumulative_curve_' + branches[i] + '.png')
        fig.clf()
    
        # ================================ #
        # create the mean curve of entries #
        # ================================ #
        mean_df_entries = df_entries.mean()
        mean_sum = mean_df_entries.sum()
        
        diffMax1, posMax1 = DB.diffMAXKS(mean_df_entries, s_new, mean_sum, Ntot_h1)
        diffMax2, posMax2 = DB.diffMAXKS(series_reference, s_new, nbBins_reference, Ntot_h1)
        diffMax3, posMax3 = DB.diffMAXKS(s_new, s_old, Ntot_h1, Ntot_h2)
        print("diffMax1 : %f - posMax1 : %f" % (diffMax1, posMax1))
        print("diffMax2 : %f - posMax2 : %f" % (diffMax2, posMax2))
        print("diffMax3 : %f - posMax3 : %f" % (diffMax3, posMax3))
        print('Ntot_h1 : %d - Ntot_h2 : %d' % (Ntot_h1, Ntot_h2))

        # diff max between new & old
        diffMax0, posMax0, sDKS = DB.diffMAXKS2(s_old, s_new, Ntot_h2, Ntot_h1)
        print("diffMax0 : %f - posMax0 : %f" % (diffMax0, posMax0))
        wKS0.write('%s : %e\n' % (branches[i], diffMax0))
        print(s_new[0:8])
        print(s_old[0:8])
        print(sDKS[0:8]) # diff

        yellowCurve1 = mean_df_entries
        yellowCurve2 = series_reference
        yellowCurve3 = s_new
        yellowCurveCum1 = DB.funcKS(mean_df_entries) #  cumulative yellow curve
        yellowCurveCum2 = DB.funcKS(series_reference)
        yellowCurveCum3 = DB.funcKS(s_new)

        # Kolmogoroff-Smirnov curve
        seriesTotalDiff1 = pd.DataFrame(totalDiff, columns=['KSDiff'])
        KSDiffname1 = df.folder + '/KSDiffValues_1_' + branches[i] + '.csv'
        df.to_csv(KSDiffname1)
        plt_diff_KS1 = seriesTotalDiff1.plot.hist(bins=nbins, title='KS diff. 1')
        print('\ndiffMin0/sTD.min 1 : %f/%f' % (diffMax0, seriesTotalDiff1.values.min()))
        print('\ndiffMax0/sTD.max 1 : %f/%f' % (diffMax0, seriesTotalDiff1.values.max()))
        if (diffMax0 >= seriesTotalDiff1.values.max()):
            color1 = 'r'
            nb_red1 += 1
            x1 = seriesTotalDiff1.values.max()
        elif (diffMax0 <= seriesTotalDiff1.values.min()):
            color1 = 'g'
            nb_green1 += 1
            x1 = seriesTotalDiff1.values.min()
        else:
            color1 = 'g'
            nb_green1 += 1
            x1 = diffMax0
        print('x1 : %f' % x1)
        ymi, yMa = plt_diff_KS1.get_ylim()
        plt_diff_KS1.vlines(x1, ymi, 0.9*yMa, color=color1, linewidth=4)
        fig = plt_diff_KS1.get_figure()
        fig.savefig(df.folder + '/KS-ttlDiff_1_' + branches[i] + '.png')
        fig.clf()
        count, division = np.histogram(seriesTotalDiff1[~np.isnan(seriesTotalDiff1)], bins=nbins)
        div_min = np.amin(division)
        div_max = np.amax(division)
        KSDiffHistoname1 = df.folder + '/KSDiffHistoValues_1_' + branches[i] + '.csv'
        wKSDiff1 = open(KSDiffHistoname1, 'w')
        wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in count))
        wKSDiff1.write('\n')
        wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in division))
        wKSDiff1.write('\n')
        wKSDiff1.close()

        # Get the max of the integral
        I_max = DB.integralpValue(division, count, 0.)
        print('\nMax. integral 1 : %0.4e for nbins=%d' % (I_max, nbins))
        # print the min/max values of differences
        print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x1))
        #stop
        pValue = DB.integralpValue(division, count, diffMax0)
        #print(division)
        #print(count)
        # save the KS curves
        wKS1.write('%e, %d\n' % (I_max, nbins))
        wKS1.write('%e, %e\n' % (div_min, div_max))
        wKS1.write(' '.join("{:10.04e}".format(x) for x in count))
        wKS1.write('\n')
        wKS1.write(' '.join("{:10.04e}".format(x) for x in division))
        wKS1.write('\n')
        wKS1.write(' '.join("{:10.04e}".format(x) for x in yellowCurve1 )) # average (mean) curve
        wKS1.write('\n')
        wKS1.write(' '.join("{:10.04e}".format(x) for x in yellowCurveCum1 ))
        wKS1.write('\n')
        wKS1.close()

        # Kolmogoroff-Smirnov curve 2
        seriesTotalDiff2 = pd.DataFrame(totalDiff2, columns=['KSDiff'])
        plt_diff_KS2 = seriesTotalDiff2.plot.hist(bins=nbins, title='KS diff. 2')
        print('\ndiffMin0/sTD.min 1 : %f/%f' % (diffMax0, seriesTotalDiff2.values.min()))
        print('\ndiffMax0/sTD.max 2 : %f/%f' % (diffMax0, seriesTotalDiff2.values.max()))
        if (diffMax0 >= seriesTotalDiff2.values.max()):
            color2 = 'r'
            nb_red2 += 1
            x2 = seriesTotalDiff2.values.max()
        elif (diffMax0 <= seriesTotalDiff2.values.min()):
            color2 = 'g'
            nb_green2 += 1
            x2 = seriesTotalDiff2.values.min()
        else:
            color2 = 'g'
            nb_green2 += 1
            x2 = diffMax0
        print('x2 : %f' % x2)
        ymi, yMa = plt_diff_KS2.get_ylim()
        plt_diff_KS2.vlines(x2, ymi, 0.9*yMa, color=color2, linewidth=4)
        fig = plt_diff_KS2.get_figure()
        fig.savefig(df.folder + '/KS-ttlDiff_2_' + branches[i] + '.png')
        fig.clf()
        count, division = np.histogram(seriesTotalDiff2, bins=nbins)
        div_min = np.amin(division)
        div_max = np.amax(division)
        KSDiffHistoname2 = df.folder + '/KSDiffHistoValues_2_' + branches[i] + '.csv'
        wKSDiff2 = open(KSDiffHistoname2, 'w')
        wKSDiff2.write(' '.join("{:10.04e}".format(x) for x in count))
        wKSDiff2.write('\n')
        wKSDiff2.write(' '.join("{:10.04e}".format(x) for x in division))
        wKSDiff2.write('\n')
        wKSDiff2.close()
    
        # Get the max of the integral
        I_max2 = DB.integralpValue(division, count, 0.)
        print('\nMax. integral 2 : %0.4e for nbins=%d' % (I_max2, nbins))
        # print the min/max values of differences
        print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x2))
        pValue2 = DB.integralpValue(division, count, diffMax0)
        #print(division)
        #print(count)
        # save the KS curves
        wKS2.write('%e, %d\n' % (I_max2, nbins))
        wKS2.write('%e, %e\n' % (div_min, div_max))
        wKS2.write(' '.join("{:10.04e}".format(x) for x in count))
        wKS2.write('\n')
        wKS2.write(' '.join("{:10.04e}".format(x) for x in division))
        wKS2.write('\n')
        wKS2.write(' '.join("{:10.04e}".format(x) for x in yellowCurve2 )) # random curve
        wKS2.write('\n')
        wKS2.write(' '.join("{:10.04e}".format(x) for x in yellowCurveCum2 ))
        wKS2.write('\n')
        wKS2.close()

        # Kolmogoroff-Smirnov curve 3
        seriesTotalDiff3 = pd.DataFrame(totalDiff3, columns=['new'])
        plt_diff_KS3 = seriesTotalDiff3.plot.hist(bins=nbins, title='KS diff. 3')
        print('\ndiffMin0/sTD.min 3 : %f/%f' % (diffMax0, seriesTotalDiff3.values.min()))
        print('diffMax0/sTD.max 3 : %f/%f' % (diffMax0, seriesTotalDiff3.values.max()))
        if (diffMax0 >= seriesTotalDiff3.values.max()):
            color3 = 'r'
            nb_red3 += 1
            x3 = seriesTotalDiff3.values.max()
        elif (diffMax0 <= seriesTotalDiff3.values.min()):
            color3 = 'g'
            nb_green3 += 1
            x3 = seriesTotalDiff3.values.min()
        else:
            color3 = 'g'
            nb_green3 += 1
            x3 = diffMax0
        print('x3 : %f' % x3)
        ymi, yMa = plt_diff_KS3.get_ylim()
        plt_diff_KS3.vlines(x3, ymi, 0.9*yMa, color=color3, linewidth=4)
        fig = plt_diff_KS3.get_figure()
        fig.savefig(df.folder + '/KS-ttlDiff_3_' + branches[i] + '.png')
        fig.clf()
        count, division = np.histogram(seriesTotalDiff3, bins=nbins)
        div_min = np.amin(division)
        div_max = np.amax(division)
        KSDiffHistoname3 = df.folder + '/KSDiffHistoValues_3_' + branches[i] + '.csv'
        wKSDiff3 = open(KSDiffHistoname3, 'w')
        wKSDiff3.write(' '.join("{:10.04e}".format(x) for x in count))
        wKSDiff3.write('\n')
        wKSDiff3.write(' '.join("{:10.04e}".format(x) for x in division))
        wKSDiff3.write('\n')
        wKSDiff3.close()
    
        # Get the max of the integral
        I_max3 = DB.integralpValue(division, count, 0.)
        print('\nMax. integral 3 : %0.4e for nbins=%d' % (I_max3, nbins))
        # print the min/max values of differences
        print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x3))
        pValue3 = DB.integralpValue(division, count, diffMax0)
        #print(division)
        #print(count)
        # print the p-Value
        print('\nunormalized p_Value : %0.4e for nbins=%d' % (pValue, nbins))
        print('normalized p_Value : %0.4e for nbins=%d' % (pValue/I_max, nbins))
        # print the p-Value 2
        print('\nunormalized p_Value : %0.4e for nbins=%d' % (pValue2, nbins))
        print('normalized p_Value : %0.4e for nbins=%d' % (pValue2/I_max2, nbins))
        # print the p-Value 3
        print('\nunormalized p_Value : %0.4e for nbins=%d' % (pValue3, nbins))
        print('normalized p_Value : %0.4e for nbins=%d' % (pValue3/I_max3, nbins))

        wKSp.write('%s, %e, %e, %e\n' % (branches[i], pValue/I_max, pValue2/I_max2, pValue3/I_max3))

        plt.close('all')
        # save the KS curves
        wKS3.write('%e, %d\n' % (I_max3, nbins))
        wKS3.write('%e, %e\n' % (div_min, div_max))
        wKS3.write(' '.join("{:10.04e}".format(x) for x in count))
        wKS3.write('\n')
        wKS3.write(' '.join("{:10.04e}".format(x) for x in division))
        wKS3.write('\n')
        wKS3.write(' '.join("{:10.04e}".format(x) for x in yellowCurve3 )) # new curve
        wKS3.write('\n')
        wKS3.write(' '.join("{:10.04e}".format(x) for x in yellowCurveCum3 ))
        wKS3.write('\n')
        wKS3.close()

        '''R1 = diffR2(mean_df_entries, s_new)
        R2 = diffR2(series_reference, s_new)
        R3 = diffR2(s_new, s_old)
        print('R1 = %f [mean, new]' % R1)
        print('R2 = %f [ref, new]' % R2)
        print('R3 = %f [old, new]' % R3)'''

    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc-tic))

    # print nb of red/green lines
    print('KS 1 : %d red - %d green' % (nb_red1, nb_green1))
    print('KS 2 : %d red - %d green' % (nb_red2, nb_green2))
    print('KS 3 : %d red - %d green' % (nb_red3, nb_green3))
    nb_red = nb_red1 + nb_red2 + nb_red3
    nb_green = nb_green1 + nb_green2 + nb_green3
    print('KS ttl : %d red - %d green' % (nb_red, nb_green))
    wKS_.write('KS 1 : %d red - %d green\n' % (nb_red1, nb_green1))
    wKS_.write('KS 2 : %d red - %d green\n' % (nb_red2, nb_green2))
    wKS_.write('KS 3 : %d red - %d green\n' % (nb_red3, nb_green3))
    wKS_.write('KS ttl : %d red - %d green\n' % (nb_red, nb_green))

    return

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    branches = getBranches(tp_1)
    print(branches[0:10])
    #branches = branches[25:35]
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    # nb of files to be used
    nbFiles = 200

    #func_ReduceSize(nbFiles)
    
    #func_Extract(branches[0:5], nbFiles) # create file with histo datas.
    func_Extract(branches, nbFiles) # create file with histo datas.

    #func_CreateKS(branches[0:5], nbFiles) # create the KS files from histos datas for datasets
    #func_CreateKS(branches, nbFiles)  # create the KS files from histos datas

    print("Fin !")

