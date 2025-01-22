#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# filesSources.py : list of ROOT files to be used with zeeExtract tools
# for egamma validation comparison                              
# 
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

# definition '-' : not selected
#            '+' : selected
#            '*' : selected AND reference

# get the "sources" root file datas
rootSources = [
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_5_0_pre4-124X_mcRun3_2022_realistic_v10-v1__DQMIO.root'],
['*', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_5_0_pre5-125X_mcRun3_2022_realistic_v3-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_6_0_pre2-125X_mcRun3_2022_realistic_v3-v2__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_6_0_pre3-125X_mcRun3_2022_realistic_v3-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_6_0_pre4-125X_mcRun3_2022_realistic_v4-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_0_0_pre3-130X_mcRun3_2022_realistic_v2-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_0_0_pre4-130X_mcRun3_2022_realistic_v2-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_1_0_pre2-131X_mcRun3_2022_realistic_v1-v3__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_1_0_pre3-131X_mcRun3_2022_realistic_v2-v2__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_1_0_pre4-131X_mcRun3_2022_realistic_v3-v1__DQMIO.root'],
['+', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_2_0_pre3-131X_mcRun3_2023_realistic_v8-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_3_0_pre1-131X_mcRun3_2023_realistic_v9-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_3_0_pre2-132X_mcRun3_2023_realistic_v2-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_3_0_pre3-132X_mcRun3_2023_realistic_v4-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_13_3_0_pre5-133X_mcRun3_2023_realistic_v2-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_14_0_0_pre1-133X_mcRun3_2023_realistic_v3-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_14_0_0_pre2-133X_mcRun3_2024_realistic_v5_STD_2024_noPU-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_14_0_0_pre3-140X_mcRun3_2024_realistic_v1_STD_2024_noPU-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_14_1_0_pre2-140X_mcRun3_2024_realistic_v7_STD_2024_noPU-v1__DQMIO.root'],
['-', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_14_1_0_pre7-140X_mcRun3_2024_realistic_v21_STD_RegeneratedGS_2024_noPU-v1__DQMIO.root'],
['+', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_14_2_0_pre4-140X_mcRun3_2024_realistic_v26_STD_RecylcedGS_2024_noPU-v1__DQMIO.root'],
['+', 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_15_0_0_pre1-140X_mcRun3_2024_realistic_v26_STD_RecylcedGS_2024_noPU-v1__DQMIO.root'],

]


