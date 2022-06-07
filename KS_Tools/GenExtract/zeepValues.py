#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# zeepValues: a tool to extract/generate pictutes from p-Values
# generated from zeeExtract tools.
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

from default import *

#import seaborn # only with cmsenv on cca.in2p3.fr

#argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
#argv.remove( '-b-' )

from ROOT import *

ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()

# these line for daltonians !
#seaborn.set_palette('colorblind')
import default as df

def checkFolderName(folderName):
    if folderName[-1] != '/':
        folderName += '/'
    return folderName

def createHistoPicture(histo1, filename):
    cnv = TCanvas(str(id), "canvas")
    print('createPicture')
    color1 = ROOT.kRed #
    color0 = ROOT.kBlack
    color2 = ROOT.kBlue

    cnv.SetCanvasSize(960, 600)

    cnv.Clear()
    histo1.Draw()
    histo1.SetLineWidth(3)
    histo1.SetStats(1)
    #enderHisto(histo1)
    gPad.Update()
    statBox1 = histo1.GetListOfFunctions().FindObject("stats")
    histo1.SetLineColor(color0)
    histo1.SetMarkerColor(color1)
    statBox1.SetTextColor(color2)
    statBox1.SetFillStyle(0);
    statBox1.SetY1NDC(0.800)
    gPad.Update()

    cnv.Draw()
    cnv.Update()
    
    cnv.SaveAs(filename)

    return
    
if __name__=="__main__":
    df.folderName = checkFolderName(df.folderName)
    df.folder = checkFolderName(df.folder)

    KS_pValues = df.folder + "histo_pValues.txt"
    print("KSname 2 : %s" % KS_pValues)
    wKSp = open(KS_pValues, 'r')

    histo1= TH1F('KS 1', 'KS 1', 100,0.,1.)
    histo2= TH1F('KS 2', 'KS 2', 100,0.,1.)
    histo3= TH1F('KS 3', 'KS 3', 100,0.,1.)

    v = wKSp.readlines()
    for elem in v:
        print(elem)
        a = elem.split(',')
        print(a)
        histo1.Fill(float(a[1])) # pvalue1
        histo2.Fill(float(a[2])) # pvalue2
        histo3.Fill(float(a[3])) # pvalue3
    
    createHistoPicture(histo1, df.folder + 'KS_1.png')
    createHistoPicture(histo2, df.folder + 'KS_2.png')
    createHistoPicture(histo3, df.folder + 'KS_3.png')
    
    print("Fin !")

