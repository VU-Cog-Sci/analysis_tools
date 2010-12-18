#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import * 
from RetinotopicMappingSession import *


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
	
	def createFunctionalMask(self, exclusionThreshold = 2.0, maskFrame = 3):
		"""
		Take the eccen F-values, use as a mask, and take out the F-value mask of the peripheral fixation condition
		results in creation of a mask file which can be accessed later
		"""
		# F-value mask from eccen experiment
		eccenFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen.nii.gz')
		fixPeripheryFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['fix_periphery'][0]]), 'polar.nii.gz')
		imO = ImageMaskingOperator(inputObject = eccenFile, maskObject = fixPeripheryFile, thresholds = [exclusionThreshold])
		maskedDataArray = imO.applySingleMask(whichMask = maskFrame, maskThreshold = exclusionThreshold, nrVoxels = False, maskFunction = '__lt__', flat = False)
		maskImage = NiftiImage(maskedDataArray)
		maskImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen_mask-' + str(exclusionThreshold) + '.nii.gz')
		maskImage.save()
		# F-value mask from polar - fix map experiment
		polarFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'polar.nii.gz')
		fixPeripheryFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['fix_periphery'][0]]), 'polar.nii.gz')
		imO = ImageMaskingOperator(inputObject = polarFile, maskObject = fixPeripheryFile, thresholds = [exclusionThreshold])
		maskedDataArray = imO.applySingleMask(whichMask = maskFrame, maskThreshold = exclusionThreshold, nrVoxels = False, maskFunction = '__lt__', flat = False)
		maskImage = NiftiImage(maskedDataArray)
		maskImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'polar_mask-' + str(exclusionThreshold) + '.nii.gz')
		maskImage.save()
	
	def collectConditionFiles(self):
		"""
		Returns the highest-level selfreqavg output files for the conditions in conditionDict
		"""
		images = []
		for condition in self.conditionDict.keys():
			thisConditionFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[condition][0]]), 'polar.nii.gz')
			images.append(NiftiImage(thisConditionFile))
		return images
	
	def maskConditionFiles(self, conditionFiles, maskFile = None, maskThreshold = 4.0, maskFrame = 0, flat = False):
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
		for conditionFile in conditionFiles:
			imO = ImageMaskingOperator(conditionFile, maskObject = maskFile, thresholds = [maskThreshold])
			if flat == True:
				maskedConditionFiles.append(imO.applySingleMask(whichMask = maskFrame, maskThreshold = maskThreshold, nrVoxels = False, maskFunction = '__gt__', flat = flat))
			else:
				maskedConditionFiles.append(NiftiImage(imO.applySingleMask(whichMask = maskFrame, maskThreshold = maskThreshold, nrVoxels = False, maskFunction = '__gt__', flat = flat)))
		return maskedConditionFiles
	
	def dataForRegions(self, regions = ['V1', 'V2', 'V3', 'V3AB', 'V4','inferiorparietal', 'superiorparietal','precuneus','lingual','fusiform','lateraloccipital'], maskFile = 'polar_mask-2.0.nii.gz', maskThreshold = 3.0 ):
		"""
		Produce phase-phase correlation plots across conditions.
		['rh.V1', 'lh.V1', 'rh.V2', 'lh.V2', 'rh.V3', 'lh.V3', 'rh.V3AB', 'lh.V3AB', 'rh.V4', 'lh.V4']
		"""
		self.rois = regions
		maskedFiles = self.maskConditionFiles(conditionFiles = self.collectConditionFiles(), maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), maskFile ), maskThreshold = maskThreshold, maskFrame = 3)
		maskedRoiData = []
		for roi in regions:
			thisRoiData = self.maskConditionFiles(conditionFiles = maskedFiles, maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat/'), roi + '.nii.gz' ), maskThreshold = 0.0, maskFrame = 0, flat = True)
			maskedRoiData.append(thisRoiData)
		self.maskedRoiData = maskedRoiData
		
	def phasePhasePlots(self):
		if not hasattr(self, 'maskedRoiData'):
			self.dataForRegions()
		self.logger.debug('masked roi data shape is ' + str(len(self.maskedRoiData)) + ' ' + str(len(self.maskedRoiData[0])) + ' ' + str(self.maskedRoiData[0][0].shape))
		from itertools import combinations
		f = pl.figure(figsize = (1.5 * len(self.maskedRoiData), 1.5 *len(list(combinations(range(len(self.conditionDict)),2)))))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1
		for comb in combinations(range(len(self.conditionDict)),2):
			for i in range(len(self.maskedRoiData)):
				sbp = f.add_subplot(len(list(combinations(range(len(self.conditionDict)),2))),len(self.maskedRoiData),plotNr)
				summedArray = - ( self.maskedRoiData[i][comb[0]][9] + self.maskedRoiData[i][comb[1]][9] == 0.0 )
				# pl.scatter(self.maskedRoiData[i][comb[0]][9][summedArray], self.maskedRoiData[i][comb[1]][9][summedArray], c = 'g',  alpha = 0.1)
				pl.hexbin(self.maskedRoiData[i][comb[0]][9][summedArray], self.maskedRoiData[i][comb[1]][9][summedArray], gridsize = 10)
				sbp.set_title(self.rois[i] + '\t\t\t\t', fontsize=11)
				sbp.set_ylabel(self.conditionDict.keys()[comb[1]], fontsize=9)
				sbp.set_xlabel(self.conditionDict.keys()[comb[0]], fontsize=9)
				plotNr += 1
	
	def phaseDistributionPlots(self):
		if not hasattr(self, 'maskedRoiData'):
			self.dataForRegions()
		self.logger.debug('masked roi data shape is ' + str(len(self.maskedRoiData)) + ' ' + str(len(self.maskedRoiData[0])) + ' ' + str(self.maskedRoiData[0][0].shape))
		from itertools import combinations
		f = pl.figure(figsize = (10,10))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1
		for condition in range(len(self.conditionDict)):
			for i in range(len(self.maskedRoiData)):
				sbp = f.add_subplot(len(self.conditionDict),len(self.maskedRoiData),plotNr)
				summedArray = - ( self.maskedRoiData[i][condition][9] == 0.0 )
				# pl.scatter(self.maskedRoiData[i][comb[0]][9][summedArray], self.maskedRoiData[i][comb[1]][9][summedArray], c = 'g',  alpha = 0.1)
				pl.hist(self.maskedRoiData[i][condition][9][summedArray])
				sbp.set_title(self.rois[i], fontsize=10)
				sbp.set_ylabel(self.conditionDict.keys()[condition], fontsize=10)
#				sbp.set_xlabel(self.conditionDict.keys()[comb[0]], fontsize=10)
				plotNr += 1
	
	def significanceSignificancePlots(self):
		if not hasattr(self, 'maskedRoiData'):
			self.dataForRegions()
		self.logger.debug('masked roi data shape is ' + str(len(self.maskedRoiData)) + ' ' + str(len(self.maskedRoiData[0])) + ' ' + str(self.maskedRoiData[0][0].shape))
		from itertools import combinations
		f = pl.figure(figsize = (10,10))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1
		for comb in combinations(range(len(self.conditionDict)),2):
			for i in range(len(self.maskedRoiData)):
				sbp = f.add_subplot(len(list(combinations(range(len(self.conditionDict)),2))),len(self.maskedRoiData),plotNr)
				summedArray = - ( self.maskedRoiData[i][comb[0]][0] + self.maskedRoiData[i][comb[1]][0] == 0.0 )
				pl.scatter(self.maskedRoiData[i][comb[0]][0][summedArray], self.maskedRoiData[i][comb[1]][0][summedArray], c = 'r', alpha = 0.1)
				sbp.set_title(self.rois[i], fontsize=10)
				sbp.set_ylabel(self.conditionDict.keys()[comb[1]], fontsize=10)
				sbp.set_xlabel(self.conditionDict.keys()[comb[0]], fontsize=10)
				sbp.axis([-10,10,-10,10])
				plotNr += 1

		
		
		
		
		
		
