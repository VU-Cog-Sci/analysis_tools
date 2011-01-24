#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import * 
from RetinotopicMappingSession import *
from ..circularTools import *
import matplotlib.cm as cm

class RetinotopicRemappingSession(RetinotopicMappingSession):
	def runQC(self, rois = ['V1','V2','V3']):
		"""
		Quality control for this session would mean to have a look at SNR in different areas
		runQC assumes a list of masks is in place in the processed/mri/masks folder and runs separate analyses for each of these ROIs
		"""
		fs = 10
		
		if len(self.conditionDict['polar']) > 0:
			# polar files
			rawInputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[pC], postFix = ['mcf']) for pC in self.scanTypeDict['epi_bold']]
			distilledInputFileNames = [os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[pC]), self.runList[pC].condition) for pC in self.scanTypeDict['epi_bold']]
			distilledInputFileNamesFull = [os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['polar'][0]]), 'polar'),os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['polar'][0]]), 'eccen')]
			
			for (pd, rd) in zip(distilledInputFileNames, rawInputFileNames):
				# setting up the data files
				rdIm = NiftiImage(rd)
				pdIm = NiftiImage(pd)
				
				pp = PdfPages(self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][rawInputFileNames.index(rd)]], extension = '.pdf' ))
				
				for roi in rois:
					for hemi in ['lh','rh']:
						maskFileName = self.runFile(stage = 'processed/mri/masks', base = hemi + '.' + roi )
						
						# run directly on the results of retmapping, that is, the polar.nii.gz and eccen.nii.gz files 
						# mask the input files:
						imO = ImageMaskingOperator(pdIm, maskObject = maskFileName)
						roiStatData = imO.applySingleMask(flat = True)		# F value is 3, noise SD is 6, phase value is 9
						
						fig = pl.figure()
						s = fig.add_subplot(321)
						pl.hist(roiStatData[3], color='r', alpha = 0.25, normed = True, bins = np.linspace(0,10,25), rwidth = 0.5 )
						s.set_title('F values in ' + roi + ' ' + hemi + '\n' + str(rd.split('/')[-3:]), fontsize=fs)
						s.axis([0, 10, 0, 0.5])
						s = fig.add_subplot(323)
						pl.hist(roiStatData[9], color='g', alpha = 0.25, normed = True, bins = np.linspace(-pi,pi,25), rwidth = 0.5)
						s.set_title('phase values in ' + roi, fontsize=fs)
						s.axis([-pi, pi, 0, 0.5])
						s = fig.add_subplot(325)
						pl.hist(roiStatData[6], color='b', alpha = 0.25, normed = True, bins = np.linspace(50,250,25), rwidth = 0.5)
						s.set_title('noise SD values in ' + roi, fontsize=fs)
						s.axis([50, 250, 0, 0.02])
						
						# run on the raw periodic data, that is, the polar.nii.gz and eccen.nii.gz files 
						imO = ImageMaskingOperator(rdIm, maskObject = maskFileName)
						roiRawData = imO.applySingleMask(flat = True)
						self.logger.debug('masked raw data from roi %s in hemi %s is %s', roi, hemi, str(roiRawData.shape) )
						# the best voxel in this file:
						bestVoxArray = roiStatData[3] == roiStatData[3].max()
						# show timecourse of run in this best voxel
						s = fig.add_subplot(222)
						s.plot( np.arange(rdIm.data.shape[0]) * rdIm.rtime, roiRawData[:,bestVoxArray] )
						s.set_title('best voxel time course in ' + roi + ' ' + hemi + '\n' + str(rd.split('/')[-3:]), fontsize=fs)
						pl.xlabel('time [s]')
						s = fig.add_subplot(224)
						ftRoiRD = np.abs(np.fft.fft(roiRawData, axis = 0)).mean(axis = -1)
						ftticks = np.fft.fftfreq(roiRawData.shape[0], d = rdIm.rtime )
						s.set_title('power spectrum ' + roi + ' ' + hemi, fontsize=fs)
						s.plot( ftticks[1:floor(ftRoiRD.shape[0]/2.0)], ftRoiRD[1:floor(ftRoiRD.shape[0]/2.0)] )
						pl.xlabel('frequency [Hz]')
					
						pp.savefig()
					
				pp.close()
#		pl.show()
	
	def primaryEyeDataAnalysis(self):
		"""
		Take the eye movement data for the runs in this session
		"""
		for ri in self.scanTypeDict['epi_bold']:
			self.runList[ri].eyeOp = ASLEyeOperator( inputObject = self.runList[ri].eyeLinkFile )
			self.runList[ri].eyeOp.firstPass(132, 8, TR = 2.0, makeFigure = True, figureFileName = os.path.join( self.runFile(stage = 'processed/eye', run = self.runList[ri], extension = '.pdf') ))
	
	def createFunctionalMask(self, exclusionThreshold = 2.0, maskFrame = 0):
		"""
		Take the eccen F-values, use as a mask, and take out the F-value mask of the peripheral fixation condition
		results in creation of a mask file which can be accessed later
		"""
		# F-value mask from eccen experiment
		eccenFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen.nii.gz')
		fixPeripheryFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['fix_periphery'][0]]), 'polar.nii.gz')
		imO = ImageMaskingOperator(inputObject = eccenFile, maskObject = fixPeripheryFile, thresholds = [exclusionThreshold])
		# change the first frame of the mask and input data (-log p-value) to its absolute value
		imO.maskData[0] = np.abs(imO.maskData[0])
		imO.inputData[0] = np.abs(imO.inputData[0])
		maskedDataArray = imO.applySingleMask(whichMask = maskFrame, maskThreshold = exclusionThreshold, nrVoxels = False, maskFunction = '__lt__', flat = False)
		maskImage = NiftiImage(maskedDataArray)
		maskImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen_mask-' + str(exclusionThreshold) + '.nii.gz')
		maskImage.save()
		# F-value mask from polar - fix map experiment
		polarFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'polar.nii.gz')
		fixPeripheryFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['fix_periphery'][0]]), 'polar.nii.gz')
		imO = ImageMaskingOperator(inputObject = polarFile, maskObject = fixPeripheryFile, thresholds = [exclusionThreshold])
		# change the first frame of the mask and input data (-log p-value) to its absolute value
		imO.maskData[0] = np.abs(imO.maskData[0])
		imO.inputData[0] = np.abs(imO.inputData[0])
		maskedDataArray = imO.applySingleMask(whichMask = maskFrame, maskThreshold = exclusionThreshold, nrVoxels = False, maskFunction = '__lt__', flat = False)
		maskImage = NiftiImage(maskedDataArray)
		maskImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'polar_mask-' + str(exclusionThreshold) + '.nii.gz')
		maskImage.save()
	
	def collectConditionFiles(self, add_eccen = False):
		"""
		Returns the highest-level selfreqavg output files for the conditions in conditionDict
		"""
		images = []
		for condition in self.conditionDict.keys():
			thisConditionFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[condition][0]]), 'polar.nii.gz')
			images.append(NiftiImage(thisConditionFile))
		if add_eccen:
			self.eccenImage = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen.nii.gz'))
			images.append(NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen.nii.gz')))
		return images
		
	def collectRunFiles(self):
		"""
		Returns the lower-level selfreqavg output files for the conditions in conditionDict
		"""
		images = []
		for condition in self.conditionDict.keys():
			images.append([])
			for c in self.conditionDict[condition]:
				thisConditionFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[c]), 'polar.nii.gz')
				images[-1].append(NiftiImage(thisConditionFile))
		return images
		
	
	def maskFiles(self, dataFiles, maskFile = None, maskThreshold = 5.0, maskFrame = 0, nrVoxels = False, flat = False):
		# anatomical or statistical mask?
		if os.path.split(maskFile)[0].split('/')[-1] == 'anat':
			if os.path.split(maskFile)[-1][:2] in ['lh','rh']:	# single hemisphere roi
				pass
			else:	# combine two anatomical rois
				lhFile = os.path.join(os.path.split(maskFile)[0], 'lh.' + os.path.split(maskFile)[1])
				rhFile = os.path.join(os.path.split(maskFile)[0], 'rh.' + os.path.split(maskFile)[1])
				maskFile = NiftiImage(NiftiImage(lhFile).data + NiftiImage(rhFile).data)
		# if not anatomical we assume it's just fine.
		maskedConditionFiles = []
		for f in dataFiles:
			imO = ImageMaskingOperator(f, maskObject = maskFile, thresholds = [maskThreshold])
			if flat == True:
				maskedConditionFiles.append(imO.applySingleMask(whichMask = maskFrame, maskThreshold = maskThreshold, nrVoxels = nrVoxels, maskFunction = '__gt__', flat = flat))
			else:
				maskedConditionFiles.append(NiftiImage(imO.applySingleMask(whichMask = maskFrame, maskThreshold = maskThreshold, nrVoxels = nrVoxels, maskFunction = '__gt__', flat = flat)))
		return maskedConditionFiles
	
	def conditionDataForRegions(self, regions = [['V1','V2','V3'], ['V3AB','V4']], maskFile = 'polar_mask-2.0.nii.gz', maskThreshold = 4.0, nrVoxels = False, add_eccen = True ):
		"""
		Produce phase-phase correlation plots across conditions.
		['rh.V1', 'lh.V1', 'rh.V2', 'lh.V2', 'rh.V3', 'lh.V3', 'rh.V3AB', 'lh.V3AB', 'rh.V4', 'lh.V4']
		['V1', 'V2', 'V3', 'V3AB', 'V4']
		['pericalcarine','lateraloccipital','lingual','fusiform','cuneus','precuneus','inferiorparietal', 'superiorparietal']
		"""
		self.rois = regions
		maskedFiles = self.maskFiles(dataFiles = self.collectConditionFiles(add_eccen = add_eccen), maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), maskFile ), maskThreshold = maskThreshold, maskFrame = 0)
		maskedConditionData = []
		for roi in regions:
			if roi.__class__.__name__ == 'str': 
				thisRoiData = self.maskFiles(dataFiles = maskedFiles, maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat/'), roi + '.nii.gz' ), maskThreshold = 0.0, maskFrame = 0, nrVoxels = nrVoxels, flat = True)
			elif roi.__class__.__name__ == 'list':
				print roi
				allRoisData = []
				for r in roi:
					allRoisData.append( np.array( self.maskFiles(dataFiles = maskedFiles, maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat/'), r + '.nii.gz' ), maskThreshold = 0.0, maskFrame = 0, nrVoxels = nrVoxels, flat = True) ) )
				
				thisRoiData = np.dstack(allRoisData)
			maskedConditionData.append(thisRoiData)
		self.maskedConditionData = maskedConditionData
		self.logger.debug('masked roi data shape is ' + str(len(self.maskedConditionData)) + ' ' + str(len(self.maskedConditionData[0])) + ' ' + str(self.maskedConditionData[0][0].shape))
	
		
	def runDataForRegions(self, regions = [['V1','V2','V3'], ['V3AB','V4']], maskFile = 'polar_mask-1.5.nii.gz', maskThreshold = 4.0, nrVoxels = False ):
		"""
		Produce phase-phase correlation plots across conditions.
		['rh.V1', 'lh.V1', 'rh.V2', 'lh.V2', 'rh.V3', 'lh.V3', 'rh.V3AB', 'lh.V3AB', 'rh.V4', 'lh.V4']
		['V1', 'V2', 'V3', 'V3AB', 'V4']
		['pericalcarine','lateraloccipital','lingual','fusiform','cuneus','precuneus','inferiorparietal', 'superiorparietal']
		"""
		self.rois = regions
		runFiles = self.collectRunFiles()
		maskedRunData = []
		for (i,condition) in zip(range(len(self.conditionDict.keys())),self.conditionDict.keys()):
			maskedRunData.append([])
			maskedFiles = self.maskFiles(dataFiles = runFiles[i], maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), maskFile ), maskThreshold = maskThreshold, maskFrame = 3)
			for roi in regions:
				thisRoiData = self.maskFiles(dataFiles = maskedFiles, maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat/'), roi + '.nii.gz' ), maskThreshold = 0.0, maskFrame = 0, nrVoxels = nrVoxels, flat = True)
				maskedRunData[-1].append(thisRoiData)
		self.maskedRunData = maskedRunData
		self.logger.debug('masked roi data shape is ' + str(len(self.maskedRunData)) + ' ' + str(len(self.maskedRunData[0])) + ' ' + str(len(self.maskedRunData[1][0]))+ ' ' + str(len(self.maskedRunData[0][0][0])))
	
	def phasePhasePlots(self, nrBins = 10):
		if not hasattr(self, 'maskedConditionData'):
			self.conditionDataForRegions()
		self.logger.debug('masked roi data shape is ' + str(len(self.maskedConditionData)) + ' ' + str(len(self.maskedConditionData[0])) + ' ' + str(self.maskedConditionData[0][0].shape))
		from itertools import combinations
		f = pl.figure(figsize = (1.5 * len(self.maskedConditionData), 1.5 *len(list(combinations(range(len(self.conditionDict)),2)))))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1
		combs = list(combinations(range(len(self.conditionDict)),2))
		self.histoResults = np.zeros((len(combs),len(self.maskedConditionData), nrBins, nrBins))
		for (c,comb) in zip(range(len(combs)), combs):
			for i in range(len(self.maskedConditionData)):
				sbp = f.add_subplot(len(list(combinations(range(len(self.conditionDict)),2))),len(self.maskedConditionData),plotNr)
				summedArray = - ( self.maskedConditionData[i][comb[0]][9] + self.maskedConditionData[i][comb[1]][9] == 0.0 )
				# pl.scatter(self.maskedConditionData[i][comb[0]][9][summedArray], self.maskedConditionData[i][comb[1]][9][summedArray], c = 'g',  alpha = 0.1)
				self.histoResults[c,i] = np.histogram2d(self.maskedConditionData[i][comb[0]][9][summedArray], self.maskedConditionData[i][comb[1]][9][summedArray], bins = nrBins, range = [[-pi,pi],[-pi,pi]], normed = True)[0]
				pl.imshow(self.histoResults[c,i], cmap=cm.gray)
				sbp.set_title(str(self.rois[i]) + '\t\t\t\t', fontsize=11)
				sbp.set_ylabel(self.conditionDict.keys()[comb[1]], fontsize=9)
				sbp.set_xlabel(self.conditionDict.keys()[comb[0]], fontsize=9)
				plotNr += 1
		self.combs = combs
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phasePhasePlots.pdf' ))
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phasePhasePlots.npy' ), self.histoResults)
	
	def phaseDistributionPlots(self):
		if not hasattr(self, 'maskedConditionData'):
			self.conditionDataForRegions()
		from itertools import combinations
		f = pl.figure(figsize = (10,10))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1
		for condition in range(len(self.conditionDict)):
			for i in range(len(self.maskedConditionData)):
				sbp = f.add_subplot(len(self.conditionDict),len(self.maskedConditionData),plotNr)
				summedArray = - ( self.maskedConditionData[i][condition][9] == 0.0 )
				# pl.scatter(self.maskedConditionData[i][comb[0]][9][summedArray], self.maskedConditionData[i][comb[1]][9][summedArray], c = 'g',  alpha = 0.1)
				pl.hist(self.maskedConditionData[i][condition][9][summedArray], normed = True)
#				pl.hist(self.maskedConditionData[i][condition][9], normed = True)
				sbp.set_title(str(self.rois[i]), fontsize=10)
				sbp.set_ylabel(self.conditionDict.keys()[condition], fontsize=10)
#				sbp.set_xlabel(self.conditionDict.keys()[comb[0]], fontsize=10)
				plotNr += 1
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phaseDistributionPlots.pdf' ))
	
	def significanceSignificancePlots(self):
		if not hasattr(self, 'maskedConditionData'):
			self.conditionDataForRegions()
		from itertools import combinations
		f = pl.figure(figsize = (10,10))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1
		for comb in combinations(range(len(self.conditionDict)),2):
			for i in range(len(self.maskedConditionData)):
				sbp = f.add_subplot(len(list(combinations(range(len(self.conditionDict)),2))),len(self.maskedConditionData),plotNr)
				summedArray = - ( self.maskedConditionData[i][comb[0]][0] + self.maskedConditionData[i][comb[1]][0] == 0.0 )
				pl.scatter(self.maskedConditionData[i][comb[0]][0][summedArray], self.maskedConditionData[i][comb[1]][0][summedArray], c = 'r', alpha = 0.1)
				sbp.set_title(str(self.rois[i]), fontsize=10)
#				if i == 1:
#					sbp.set_ylabel(self.conditionDict.keys()[cond1], fontsize=10)
#					sbp.set_xlabel(self.conditionDict.keys()[cond2], fontsize=10)
				sbp.axis([-10,10,-10,10])
				plotNr += 1
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'significanceSignificancePlots.pdf' ))
	
	def fitPhaseDifferences(self, comparisons = [['sacc_map','fix_map'],['sacc_map','remap'],['sacc_map','fix_periphery']], maskThreshold = 4.0, nrVoxels = False, runBootstrap = False, nrBootstrapRepetitions = 1000):
		if not hasattr(self, 'maskedConditionData'):
			self.conditionDataForRegions( maskThreshold = maskThreshold, nrVoxels = nrVoxels )
		
		fitResults = []
		allPhaseDiffs = []
		phaseHists = []
		f = pl.figure(figsize = (7,12))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		plotNr = 1		
		
		if not runBootstrap:
			try:
				allBootstrapResults = np.load(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitPhaseDifferences.npy' ))
			except OSError:
				runBootstrap = True
				allBootstrapResults = np.zeros((len(self.maskedConditionData), len(comparisons), nrBootstrapRepetitions, 2))
		else:
			allBootstrapResults = np.zeros((len(self.maskedConditionData), len(comparisons), nrBootstrapRepetitions, 2))
			
		for i in range(len(self.maskedConditionData)):
			fitResults.append([])
			allPhaseDiffs.append([])
			phaseHists.append([])
			for (c,cond) in zip(range(len(comparisons)), comparisons):
				sbp = f.add_subplot(len(self.maskedConditionData),3,plotNr)
				cond1 = self.conditionDict.keys().index(cond[0])
				cond2 = self.conditionDict.keys().index(cond[1])
				summedArray = - ( self.maskedConditionData[i][cond1][0] + self.maskedConditionData[i][cond2][0] == 0.0 )
				diffs = circularDifference(self.maskedConditionData[i][cond1][9][summedArray], self.maskedConditionData[i][cond2][9][summedArray])
				diffs.sort()
				[mu, kappa] =  fitVonMises(-diffs)
				fitResults[-1].append([mu, kappa])
				allPhaseDiffs[-1].append(diffs)
				phaseHists[-1].append(np.histogram(diffs, bins = 50, range = (-pi, pi), normed = True)[0])
				pl.hist(diffs, range = (-pi,pi), normed = True, bins = 50, color = ['r','g','b'][c], histtype = 'stepfilled', alpha = 0.15)
				pl.plot(np.linspace(-pi,pi,100), scipy.stats.vonmises.pdf(mu, kappa, np.linspace(pi,-pi,100)), ['r-','g-','b-'][c])
				sbp.set_title(str(self.rois[i]), fontsize=10)
				sbp.set_ylabel(self.conditionDict.keys()[cond1] + ' - ' + self.conditionDict.keys()[cond2], fontsize=10)
				
				if runBootstrap:
					allBootstrapResults[i,c] = bootstrapVonMisesFits( diffs, nrRepetitions = nrBootstrapRepetitions )
				
				sbp = f.add_subplot(len(self.maskedConditionData),3,plotNr+1)
				pl.hist(allBootstrapResults[i,c][:,0], range = (-pi,pi), normed = True, bins = 25, color = ['r','g','b'][c], histtype = 'stepfilled', alpha = 0.15)
				sbp.set_xlabel('means of bootstrap fits', fontsize=10)
				
				sbp = f.add_subplot(len(self.maskedConditionData),3,plotNr+2)
				sds = allBootstrapResults[i,c][:,1]
				sds.sort()
				pl.plot(sds.cumsum(), c = ['r','g','b'][c], alpha = 0.15)
				sbp.set_xlabel('kappa parameters of bootstrap fits', fontsize=10)
				sbp.axis([0,nrBootstrapRepetitions,0,20 * pi])
				
				# some cursory analysis of the bootstrap results.
				# how many of the bootstrap simulations resulted in a 0.0 mean which is not realistic?
				noMeans = (allBootstrapResults[i,c][:,0] == 0.0)
				self.logger.debug('amount of zero outcomes for ' + str(cond) + ' : ' + str( (allBootstrapResults[i,c][:,0] == 0.0).sum()/allBootstrapResults[i,c][:,0].shape[0] ))
				# what is the mean of the means?
				self.logger.debug('mean mean of bootstrap fits: ' + str(circularMean(allBootstrapResults[i,c][-noMeans,0])))
				
			phaseHists[-1] = np.array(phaseHists[-1])
			plotNr += 3	
		
		if runBootstrap:
			np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitPhaseDifferences.npy' ), allBootstrapResults)
			
		self.bootstrapResults = allBootstrapResults
		self.fitResults = np.array(fitResults)
		self.allPhaseDiffs = allPhaseDiffs
		self.phaseHists = np.array(phaseHists)
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitPhaseDifferences.pdf' ))
		
	def collapsePhaseDifferences(self, comparisons = [['sacc_map','fix_map'],['sacc_map','remap'],['sacc_map','fix_periphery']], maskThreshold = 4.0, nrVoxels = False):
		""""""
		if not hasattr(self, 'maskedConditionData'):
			self.conditionDataForRegions( maskThreshold = maskThreshold, nrVoxels = nrVoxels )
		collapsedPhaseDiffs = np.zeros((len(self.maskedConditionData),len(comparisons)))
		f = pl.figure(figsize = (7,12))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		plotNr = 1		
		for i in range(len(self.maskedConditionData)):
			sbp = f.add_subplot(len(self.maskedConditionData),1,plotNr)
			for (c,cond) in zip(range(len(comparisons)), comparisons):
				cond1 = self.conditionDict.keys().index(cond[0])
				cond2 = self.conditionDict.keys().index(cond[1])
				summedArray = - ( self.maskedConditionData[i][cond1][0] + self.maskedConditionData[i][cond2][0] == 0.0 )
				diffs = circularDifference(self.maskedConditionData[i][cond1][9][summedArray], self.maskedConditionData[i][cond2][9][summedArray])
				collapsedPhaseDiffs[i,c] = 1.0 - (np.abs(diffs).mean() / (pi/2.0))
			
			pl.bar(np.arange(0,3), collapsedPhaseDiffs[i])
			sbp.set_title(str(self.rois[i]), fontsize=10)
			sbp.set_ylabel(self.conditionDict.keys()[cond1] + ' - ' + self.conditionDict.keys()[cond2], fontsize=10)
			plotNr += 1
			
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsed.pdf' ))
		self.collapsedPhaseDiffs = collapsedPhaseDiffs
	
	def phaseDifferencesPerPhase(self, comparisons = [['fix_map','sacc_map'],['fix_map','remap'],['fix_map','fix_periphery']], baseCondition = 'fix_map', binSize = 10, maskThreshold = 3.0 ):
		self.conditionDataForRegions(add_eccen = True, maskThreshold = maskThreshold )
		
		if not hasattr(self, 'phasePhaseHistogramDict'):
			self.phasePhaseHistogramDict = {}
		
		f = pl.figure(figsize = (10,10))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1		
		outputData = []
		for cond in comparisons:
			cond1 = self.conditionDict.keys().index(cond[0])
			cond2 = self.conditionDict.keys().index(cond[1])
			outputData.append([])
			for i in range(len(self.maskedConditionData)):
				sbp = f.add_subplot(len(comparisons),len(self.maskedConditionData),plotNr)
				summedArray = - ( self.maskedConditionData[i][cond1][0] + self.maskedConditionData[i][cond2][0] == 0.0 )
				# base phase data based on eccen which is the last data file in maskedConditionData
				if baseCondition == 'eccen':
					baseData = self.maskedConditionData[i][-1][9][summedArray]
				else:
				 	baseData = self.maskedConditionData[i][self.conditionDict.keys().index(baseCondition)][9][summedArray]
				circDiffData = circularDifference(self.maskedConditionData[i][cond1][9][summedArray],self.maskedConditionData[i][cond2][9][summedArray])
				s = f.add_subplot(len(comparisons),len(self.maskedConditionData),plotNr)
				s.set_title(str(self.rois[i]), fontsize=8)
				s.set_xlabel('phase difference', fontsize=12)
				s.set_ylabel(baseCondition + ' phase', fontsize=12)
				s.set_xticks([-pi,-pi/2.0,0,pi/2.0,pi])
				s.set_xticklabels(['-$\pi$','-$\pi/2$','0','$\pi/2$','$\pi$'])
				s.set_yticks([-pi,-pi/2.0,0,pi/2.0,pi])
				s.set_yticklabels(['-$\pi$','-$\pi/2$','0','$\pi/2$','$\pi$'])
				histData = np.histogram2d(baseData,circDiffData, [np.linspace(-pi,pi,binSize),np.linspace(-pi,pi,binSize)])[0]
				pl.imshow(histData, extent = (-pi,pi,-pi,pi))
				plotNr += 1
				outputData[-1].append(histData)
		self.phasePhaseHistogramDict.update( {baseCondition: outputData} )
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phaseDifferencesPerPhase.pdf' ))
		return outputData

	def collapsePhaseDifferencesPerPhase(self, comparisons = [['sacc_map','fix_map'],['sacc_map','remap'],['sacc_map','fix_periphery']], baseCondition = 'fix_map', binSize = 0.5, nrBins = 100, maskThreshold = 4.0 ):
		self.conditionDataForRegions(add_eccen = True, maskThreshold = maskThreshold )

		if not hasattr(self, 'collapsedPhaseDiffDict'):
			self.collapsedPhaseDiffDict = {}

		f = pl.figure(figsize = (10,10))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		plotNr = 1
		
		outputData = np.zeros((len(self.rois), len(comparisons), nrBins))
		for i in range(len(self.rois)):
			sbp = f.add_subplot(len(self.rois),1,plotNr)
			sbp = f.add_subplot(len(self.maskedConditionData),1,plotNr)
			for (j, cond) in zip(range(len(comparisons)), comparisons):
				cond1 = self.conditionDict.keys().index(cond[0])
				cond2 = self.conditionDict.keys().index(cond[1])
				summedArray = - ( self.maskedConditionData[i][cond1][0] + self.maskedConditionData[i][cond2][0] == 0.0 )
				# base phase data based on eccen which is the last data file in maskedConditionData
				if baseCondition == 'eccen':
					baseData = self.maskedConditionData[i][-1][9][summedArray]
					# the eccen data would do better with a factor 0.22 offset - from tksurfer inspection of phase data
					offset = 0.5 * 2 * pi
					baseData = np.fmod(baseData + 3 * pi + offset, 2 * pi) - pi
				else:
				 	baseData = self.maskedConditionData[i][self.conditionDict.keys().index(baseCondition)][9][summedArray]
				circDiffData = circularDifference(self.maskedConditionData[i][cond1][9][summedArray],self.maskedConditionData[i][cond2][9][summedArray])
				diffs = (1.0 - (np.abs(circDiffData) / (pi/2.0)) )
				diffs = np.tile(diffs, 3) 
				baseData = np.concatenate((baseData - 2*pi, baseData, baseData + 2*pi))
				res = np.array([diffs[((baseData > (ph - binSize/2.0)) * (baseData <= (ph + binSize/2.0)))].mean() for ph in np.linspace(-pi,pi, nrBins)])
				outputData[i,j] = res
				
				sbp.set_title(str(self.rois[i]), fontsize=8)
				pl.plot(np.linspace(-pi,pi,nrBins), outputData[i,j], ['r-','g-','b-'][j])
			plotNr += 1
		self.collapsedPhaseDiffDict.update( {baseCondition: outputData} )
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsedPhaseDiff.pdf' ))
		return outputData
		
	def collapsePhaseDifferencesHorVer(self, comparisons = [['sacc_map','fix_map'],['sacc_map','remap'],['sacc_map','fix_periphery']], baseCondition = 'fix_map', nrBins = 6, maskThreshold = 4.0 ):
		self.conditionDataForRegions(add_eccen = True, maskThreshold = maskThreshold )

		if not hasattr(self, 'collapsedPhaseDiffDict'):
			self.collapsedPhaseDiffDict = {}

		f = pl.figure(figsize = (10,10))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		plotNr = 1

		outputData = np.zeros((len(self.rois), len(comparisons), nrBins))
		for i in range(len(self.rois)):
			sbp = f.add_subplot(len(self.rois),1,plotNr)
			sbp = f.add_subplot(len(self.maskedConditionData),1,plotNr)
			for (j, cond) in zip(range(len(comparisons)), comparisons):
				cond1 = self.conditionDict.keys().index(cond[0])
				cond2 = self.conditionDict.keys().index(cond[1])
				summedArray = - ( self.maskedConditionData[i][cond1][0] + self.maskedConditionData[i][cond2][0] == 0.0 )
				# base phase data based on eccen which is the last data file in maskedConditionData
				if baseCondition == 'eccen':
					baseData = self.maskedConditionData[i][-1][9][summedArray]
				else:
				 	baseData = self.maskedConditionData[i][self.conditionDict.keys().index(baseCondition)][9][summedArray]
				# horizontal / vertical index
				baseData = np.arctan(np.abs(np.tan(baseData))) / (pi/2.0)
				circDiffData = circularDifference(self.maskedConditionData[i][cond1][9][summedArray],self.maskedConditionData[i][cond2][9][summedArray])
				diffs = (1.0 - (np.abs(circDiffData) / (pi/2.0)) )
				
				res = np.array([diffs[((baseData > ph) * (baseData <= (ph + 1.0/nrBins)))].mean() for ph in np.linspace(0,1, nrBins, endpoint = False)])
				outputData[i,j] = res

				sbp.set_title(str(self.rois[i]), fontsize=8)
				pl.plot(np.linspace(-pi,pi,nrBins), outputData[i,j], ['r-','g-','b-'][j])
			plotNr += 1
		self.collapsedPhaseDiffDict.update( {baseCondition + '_HV': outputData} )

		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsedPhaseDiff.pdf' ))
		return outputData

	
	def combinationsPhaseDifferences(self, comparisons = [['fix_map','sacc_map'],['fix_map','remap'],['fix_map','fix_periphery']], maskThreshold = 3.0, nrVoxels = False):
		self.runDataForRegions( maskThreshold = maskThreshold )
		fitResults = []
		
		f = pl.figure(figsize = (7,12))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		plotNr = 1
		for i in range(len(self.rois)):
			sbp = f.add_subplot(len(self.rois),1,plotNr)
			fitResults.append([])
			for (j, cond) in zip(range(len(comparisons)), comparisons):
				cond1 = self.conditionDict.keys().index(cond[0])
				cond2 = self.conditionDict.keys().index(cond[1])
				combinations = np.concatenate([[np.sort(np.array([a,b])) for a in range(len(self.conditionDict[cond[0]]))] for b in range(len(self.conditionDict[cond[1]]))])
				# unique 2d array
				combinations = np.array(tuple(set(map(tuple, combinations.tolist()))))
				allCircularDifferencesThisRoi = []
				for c in combinations:
					allCircularDifferencesThisRoi.append(circularDifference(self.maskedRunData[cond1][i][c[0]][9], self.maskedRunData[cond2][i][c[1]][9]))
				diffs = np.concatenate(allCircularDifferencesThisRoi).ravel()
				# allCircularDifferencesThisRoi = np.array(allCircularDifferencesThisRoi)
				diffs = diffs[diffs != 0.0]
				[mu, kappa] =  fitVonMises(-diffs)
				fitResults[-1].append([mu, kappa])
				pl.hist(diffs, range = (-pi,pi), normed = True, bins = 25, color = ['r','g','b'][j], histtype = 'stepfilled', alpha = 0.15)
				pl.plot(np.linspace(-pi,pi,100), scipy.stats.vonmises.pdf(mu, kappa, np.linspace(pi,-pi,100)), ['r-','g-','b-'][j])
				sbp.set_title(str(self.rois[i]), fontsize=10)
				sbp.set_ylabel(self.conditionDict.keys()[cond1] + ' - ' + self.conditionDict.keys()[cond2] + ' ' + str(diffs.shape[0]), fontsize=10)
			plotNr += 1	
		self.combinationFitResults = np.array(fitResults)
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'combinationsPhaseDifferences.pdf' ))
		
		
		
		
		