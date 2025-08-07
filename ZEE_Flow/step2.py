# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --conditions auto:phase1_2025_realistic --datatier GEN-SIM-DIGI-RAW --era Run3_2025 --eventcontent FEVTDEBUGHLT --filein file:step1.root --fileout file:step2.root --geometry DB:Extended --nStreams 1 --nThreads 8 --no_exec --number 10 --pileup Run3_Flat55To75_PoissonOOTPU --pileup_input das:/RelValMinBias_14TeV/CMSSW_15_0_0_pre3-142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/GEN-SIM --python_filename step_2_cfg.py --step DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2025
import FWCore.ParameterSet.Config as cms
import sys

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 2 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 2 - arg. 1 :", sys.argv[1]) # index
    print("step 2 - arg. 2 :", sys.argv[2]) # path of the script ($LOG_SOURCE)
    print("step 2 - arg. 3 :", sys.argv[3]) # nb of evts
    print("step 2 - arg. 4 :", sys.argv[4]) # path of output
    print("step 2 - arg. 5 :", sys.argv[5]) # nb skip
    ind = int(sys.argv[1])
    max_number = int(sys.argv[3])
    outputPath = sys.argv[4]
    nb_skip = int(sys.argv[5])
else:
    print("step 2 - rien")
    ind = 0
    path = ''
    max_number = 10 # number of events
    outputPath = ''
    nb_skip = 0

max_skipped = (ind - 1) * nb_skip
print('max_number={:d} maxEvents={:d} skipEvents={:d}'.format(max_number, nb_skip, max_skipped))

process = cms.Process('HLT',Run3_2025)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_Run3_Flat55To75_PoissonOOTPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nb_skip),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    #fileNames = cms.untracked.vstring('file:step1.root'),
    fileNames = cms.untracked.vstring('file:' + outputPath + '/step1_' + '%0004d'%max_number + '_' + '000d.root'),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_genParticles_*_*',
        'drop *_genParticlesForJets_*_*',
        'drop *_kt4GenJets_*_*',
        'drop *_kt6GenJets_*_*',
        'drop *_iterativeCone5GenJets_*_*',
        'drop *_ak4GenJets_*_*',
        'drop *_ak7GenJets_*_*',
        'drop *_ak8GenJets_*_*',
        'drop *_ak4GenJetsNoNu_*_*',
        'drop *_ak8GenJetsNoNu_*_*',
        'drop *_genCandidatesForMET_*_*',
        'drop *_genParticlesForMETAllVisible_*_*',
        'drop *_genMetCalo_*_*',
        'drop *_genMetCaloAndNonPrompt_*_*',
        'drop *_genMetTrue_*_*',
        'drop *_genMetIC5GenJs_*_*'
    ),
    skipEvents = cms.untracked.uint32(max_skipped),
    #eventsToProcess = cms.untracked.VEventRange('1:38-1:40'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    holdsReferencesToDeleteEarly = cms.untracked.VPSet(),
    makeTriggerResults = cms.obsolete.untracked.bool,
    modulesToCallForTryToContinue = cms.untracked.vstring(),
    modulesToIgnoreForDeleteEarly = cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    #fileName = cms.untracked.string('file:step2.root'),
    fileName = cms.untracked.string('file:' + outputPath + '/step2_' + '%0004d'%max_number + '_' + '%003d'%ind + 'd.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.input.fileNames = cms.untracked.vstring(['/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/130000/1e5ada39-6756-477b-998d-275a28df5271.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/130000/34a08dd7-52ec-4961-87c1-4db63e916208.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/130000/3b537009-296a-444d-88f9-2c60338c1643.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/130000/4058fa2c-d628-4dd3-a423-d5b78978b763.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/130000/5aa492bb-ab75-44e3-8860-f8fef483ba4c.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/130000/71344887-b8d4-406a-bae8-0616a829c7d1.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/130000/ba7ca052-22cf-48d3-8895-9c35e7369408.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/08daff31-93a1-4d91-a759-198954e88fbb.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/1bee0177-4bb6-4cf0-b0b8-443c6d89a9ad.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/20f133ae-e02c-4469-a247-26e6f28f95f6.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/2ea4988d-7ba8-4578-b283-81c30b726d33.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/3e8a26b8-cb94-4c9c-9a1b-5b24a60135be.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/41205b88-ea8d-4d04-a49c-bf60551c1b4e.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/4eb20029-29ad-4a49-9b65-2d9ebb041581.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/544a887a-c13c-42be-8a18-c4e2b5625681.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/61aa92a0-aa56-40af-b145-396ace399d63.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/62c080e8-5020-4569-8f44-03dba783e444.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/6d3e2904-8109-4ef5-9869-b02be190023b.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/865ff88e-ac25-472f-975d-d4667fad2a0d.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/8a04f5aa-982d-4e05-9784-05d8329f4f15.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/9208288f-00b1-4240-aa13-8e1c6e461fc7.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/92e94444-bea1-401a-b2ff-2b4eafeceb44.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/a9a76ac2-8146-4673-886a-46366b3b94cb.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/b1776699-5b8d-49df-ac63-2daa6f702546.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/b1bed26e-16ba-4f56-b1dc-0a917cae6c73.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/cf63b0b5-344a-4d18-9f31-feef14bbd5ed.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/d2311c67-c410-4da8-9a1f-e78d45221bc1.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/dee4322c-69c7-4d57-85bc-77e06ce2efcb.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2560000/df38d39a-c193-4d98-bdc0-51c3df382385.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/09c3a76a-f20f-479c-afdb-115fa148c3cf.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/0db89750-0ec7-48f6-927a-cfdac82ff95b.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/1e270410-0e45-46d8-821c-a2000a2f87a7.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/3a8aa24e-037a-4ba7-ad31-d67634d6220a.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/5c36b7e8-741c-4a46-a623-9fcb39e7c6cb.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/5d4e6ae5-065c-4fbe-823f-be98c66ddbe6.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/636f000a-71ad-4722-9f60-3108e144ee73.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/7ee96048-01f2-4b5e-a376-8680a91408f9.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/9ad81d09-5024-4872-b0b1-7fe29eef96d4.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/9c887908-ac0c-4f92-9e4e-456fa68f5613.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/f34cbafa-4b58-425c-bc7c-5193ef2877a8.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2810000/f47925a2-d684-4c01-aa9c-b2ee4c6ead93.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/2120fb51-1260-4ddf-80ed-ff6fff380005.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/226528e0-c1e2-4528-bbbf-406085ce5948.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/30eb5daa-d38f-4a1b-a199-0c03fc63ee9b.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/5f6600c3-9917-4439-9db9-795c6b543aa5.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/61126f7f-a00d-4808-a17f-5cd4cfc1a632.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/63565900-8021-4877-ae00-459fef141d7d.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/7c768e36-22fa-47a1-9a17-3b37b97613f0.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/91d0c6ef-9a52-44ff-948f-84b35b759209.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/ba382ed3-33c6-46f6-a755-c9710496f00a.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/bd81f826-8b47-4a91-88fb-089d66f796e7.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/cc919b53-139f-4da4-ba8c-641066371a1a.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/cd4ef18e-7009-43c7-9613-5211d4158832.root', '/store/mc/CMSSW_15_0_0_pre3/RelValMinBias_14TeV/GEN-SIM/142X_mcRun3_2025_realistic_v5_STD_MinBias_2025_GenSim-v1/2820000/d3764303-9611-41fd-a06f-dadd5a5e1eb8.root'])
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2025_realistic', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
# process.schedule imported from cff in HLTrigger.Configuration
process.schedule.insert(0, process.digitisation_step)
process.schedule.insert(1, process.L1simulation_step)
process.schedule.insert(2, process.digi2raw_step)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads = 8
process.options.numberOfStreams = 1

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
