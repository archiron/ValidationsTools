#! /usr/bin/env python
#-*-coding: utf-8 -*-

import os,sys,subprocess
#import urllib2
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('agg') # to not display plt.show when enabled
from matplotlib import pyplot as plt

from sys import argv
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal
argv.remove( '-b-' )

from ROOT import * 

def getHisto(file, tp):
    #t1 = file.Get("DQMData")
    #t2 = t1.Get("Run 1")
    #t3 = t2.Get("EgammaV")
    #t4 = t3.Get("Run summary")
    #t5 = t4.Get(tp)
    path = 'DQMData/Run 1/EgammaV/Run summary/' + tp
    t_path = file.Get(path)
    return t_path # t5

def initRoot():
    initRootStyle()

def initRootStyle():
    eleStyle = ROOT.TStyle("eleStyle","Style for electron validation")
    eleStyle.SetCanvasBorderMode(0)
    eleStyle.SetCanvasColor(kWhite)
    eleStyle.SetCanvasDefH(600)
    eleStyle.SetCanvasDefW(800)
    eleStyle.SetCanvasDefX(0)
    eleStyle.SetCanvasDefY(0)
    eleStyle.SetPadBorderMode(0)
    eleStyle.SetPadColor(kWhite)
    eleStyle.SetPadGridX(False)
    eleStyle.SetPadGridY(False)
    eleStyle.SetGridColor(0)
    eleStyle.SetGridStyle(3)
    eleStyle.SetGridWidth(1)
    eleStyle.SetOptStat(1)
    eleStyle.SetPadTickX(1)
    eleStyle.SetPadTickY(1)
    eleStyle.SetHistLineColor(1)
    eleStyle.SetHistLineStyle(0)
    eleStyle.SetHistLineWidth(2)
    eleStyle.SetEndErrorSize(2)
    eleStyle.SetErrorX(0.)
    eleStyle.SetTitleColor(1, "XYZ")
    eleStyle.SetTitleFont(42, "XYZ")
    eleStyle.SetTitleXOffset(1.0)
    eleStyle.SetTitleYOffset(1.0)
    eleStyle.SetLabelOffset(0.005, "XYZ") # numeric label
    eleStyle.SetTitleSize(0.05, "XYZ")
    eleStyle.SetTitleFont(22,"X")
    eleStyle.SetTitleFont(22,"Y")
    eleStyle.SetPadBottomMargin(0.13) # 0.05
    eleStyle.SetPadLeftMargin(0.15)
    eleStyle.SetPadRightMargin(0.2) 
    eleStyle.SetMarkerStyle(21)
    eleStyle.SetMarkerSize(0.8)
    #eleStyle.SetLegendFont(42)
    #eleStyle.SetLegendTextSize(0.)
    eleStyle.cd()
    ROOT.gROOT.ForceStyle()
    
def launchTest(args):
    printFlag = False

    #tools = rt()
    #graph = ug()

    rootFilesPath = "DATA"
    picturesPath = "Temp_Images"

    #fileName1 = rootFilesPath + '/' + "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_9000_new.root"
    #fileName2 = rootFilesPath + '/' + "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_9000_ref.root"
    fileName1 = rootFilesPath + '/' + "DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_1_0_pre2-121X_mcRun3_2021_realistic_v1-v1__DQMIO.root"
    fileName2 = rootFilesPath + '/' + "DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre6-120X_mcRun3_2021_realistic_v4-v1__DQMIO.root"

    histoPath = "DQMData/Run 1/EgammaV/Run summary/ElectronMcSignalValidator"

    dataset = "h_recEleNum"
    #datasets = 'h_ele_PhiMnPhiTrueVsEta_pfx'
    #dataset = "h_ele_PoPtrueVsEta_pfx"
    #dataset = "h_recSeedNum"
    
    histoFile = "ElectronMcSignalHistos.txt"
    f_histoTxt = open(histoFile, 'r')
    model = "ElectronMcSignalValidator/" # tp_1
    tp_1 = "ElectronMcSignalValidator"
    
    for elem in f_histoTxt.readlines():
        if model in elem:
            elem = elem.replace(model, '')
            elem = " ".join(elem.split())
            AAA = elem.split(' ')
            if dataset == AAA[0]:
                print('histo_path \t scaled \t err \t eol \t eoc \t divide \t num \t denom')
                print(''.join("{}\t".format(x) for x in AAA))
                BBB = AAA
    isScaled = BBB[1]
    isErr = BBB[2]
    print('scaled : %s' % isScaled)
    print('err    : %s' % isErr)
 
    f_rel = ROOT.TFile(fileName1)
    h1 = getHisto(f_rel, tp_1)
    print("h1")
    h_1 = h1.Get(dataset)

    i = 0
    s1 = []
    e1 = []
    b1 = []
    r1 = []
    for entry in h_1:
        print("%d/%d : %s - %s - %s") % (i, h_1.GetXaxis().GetNbins(), entry, h_1.GetBinError(i), h_1.GetBinEntries(i))
        s1.append(entry)
        e1.append(h_1.GetBinError(i))
        b1.append(h_1.GetBinEntries(i))
        i += 1

    f_ref = ROOT.TFile(fileName2)
    h2 = getHisto(f_ref, tp_1)
    print("h2")
    h_2 = h2.Get(dataset)

    i = 0
    s2 = []
    e2 = []
    b2 = []
    r2 = []
    for entry in h_2:
        print("%d/%d : %s - %s - %s") % (i, h_2.GetXaxis().GetNbins(), entry, h_2.GetBinError(i), h_2.GetBinEntries(i))
        s2.append(entry)
        e2.append(h_2.GetBinError(i))
        b2.append(h_2.GetBinEntries(i))
        i += 1

    L = len(s1)
    print('L : %d' % L)
    for i in range(1, L):
        r1.append(s1[i] / b1[i])
        r2.append(s2[i] / b2[i])
        print('%d/%d : %f - %f' % (i, L, s1[i] / b1[i], s2[i] / b2[i]))

if __name__ == "__main__":

    launchTest(sys.argv)
    print('fin')
