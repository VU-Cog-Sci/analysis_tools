#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import * 
from RetinotopicMappingSession import *
from ..other_scripts.circularTools import *
import matplotlib.cm as cm
from pylab import *
import pandas as pd
import pickle

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
			print os.path.splitext(self.runList[ri].eyeLinkFile)[-1]
			print self.runList[ri].eyeLinkFile
			if os.path.splitext(self.runList[ri].eyeLinkFile)[-1] != '.edf':
				self.runList[ri].eyeOp = ASLEyeOperator( inputObject = self.runList[ri].eyeLinkFile )
				self.runList[ri].eyeOp.firstPass(132, 8, TR = 2.0, makeFigure = True, figureFileName = os.path.join( self.runFile(stage = 'processed/eye', run = self.runList[ri], extension = '.pdf') ))
				gd = self.runList[ri].eyeOp.gazeDataDuringExpt
			else:
				#import pickle
				self.runList[ri].eyeOp = EyelinkOperator( inputObject = self.runList[ri].eyeLinkFile, date_format = 'obj_c_experiment' )
				self.runList[ri].eyeOp.loadData()
				self.runList[ri].eyeOp.findELEvents()
				self.runList[ri].eyeOp.findRecordingParameters()
				f = open(self.runFile(stage = 'processed/eye', run = self.runList[ri], extension = '.pickle'), 'w')
				pickle.dump([self.runList[ri].eyeOp.gazeData, self.runList[ri].eyeOp.msgData], f)
				f.close()
				self.secondaryEyeMovementAnalysis(run_index = ri)
				gd = self.gazeDataDuringExpt
			print gd
			np.save(self.runFile(stage = 'processed/eye', run = self.runList[ri], extension = '.npy'), gd)
			print self.runFile(stage = 'processed/eye', run = self.runList[ri], extension = '.npy')
	
	def secondaryEyeMovementAnalysis(self, run_index = 0, wanted_sample_frequency = 60):
		# for ri in self.scanTypeDict['epi_bold']:
		f = open(self.runFile(stage = 'processed/eye', run = self.runList[run_index], extension = '.pickle'), 'r')
		[gazeData, msgData] = pickle.load(f)
		f.close()
		
		startTime = re.findall(re.compile('MSG\t([\d\.]+)\tTrial Phase: stimPresentationPhase'), msgData)[0]
		sampleRate = int(re.findall(re.compile('MSG\t[\d\.]+\t!MODE RECORD CR (\d+) \d+ \d+ (\S+)'), msgData)[0][0])
		blinks = np.array(re.findall(re.compile('EBLINK\t[RL]{1}\t([\d\.]+)\t([\d\.]+)'), msgData), dtype = int)
		
		blinks_binary_indices = np.array([(gazeData[:,0] > b[0]) * (gazeData[:,0] < b[1]) for b in blinks]).sum(axis = 0, dtype = bool)
		
		vel = np.diff(gazeData, axis = 0)
		
		# shell()
		
		startPoint = np.arange(gazeData.shape[0])[gazeData[:,0] == float(startTime)][0] 
		startPoint += 8 * int(sampleRate)
		endPoint = startPoint + int(sampleRate) * 256
		
		subsampling = 10
		
		xData = gazeData[startPoint:endPoint:subsampling, 1].reshape([128,sampleRate*2 / subsampling])
		
		# convert to degrees, and degrees per second.
		export_gaze_data = np.array([14.0 * (gazeData[startPoint:endPoint, 1] + 480.0 ) / 960.0, 20.0 * (gazeData[startPoint:endPoint, 2] - 600.0 ) / 1200.0 , blinks_binary_indices[startPoint:endPoint], np.diff(14.0 * (gazeData[:, 1] + 480.0 ) / 960.0)[startPoint:endPoint] * 1000.0, np.diff(20.0 * (gazeData[:, 2] - 600.0 ) / 1200.0)[startPoint:endPoint]]).T
		
		from scipy.signal import resample
		self.gazeDataDuringExpt = resample(export_gaze_data, 256*wanted_sample_frequency, axis = 0)
		
		pl.figure(figsize = (8,3))
		pl.plot(xData.T, c = 'k', alpha = 0.2, linewidth = 0.5)
		pl.fill([0.0,(sampleRate*2 / subsampling) / 8.0,(sampleRate*2 / subsampling) / 8.0,0.0], [-1000,-1000,1000,1000], 'r', alpha=0.2, edgecolor='r')
		pl.fill([(sampleRate*2 / subsampling) / 2.0, 5*(sampleRate*2 / subsampling) / 8.0, 5*(sampleRate*2 / subsampling) / 8.0,(sampleRate*2 / subsampling) / 2.0], [-1000,-1000,1000,1000], 'r', alpha=0.2, edgecolor='r')
		pl.axis([0,(sampleRate*2 / subsampling),-1000,1000])
		pl.savefig(self.runFile(stage = 'processed/eye', run = self.runList[run_index], extension = '.pdf'))
		pl.draw()
			
	def moreEyeMovementAnalysis(self, threshold = 100, moving_mean_width_on_velocity = 2, stim_offset = 0.1, sample_frequency = 60.0):
		# grab data for all conditions
		run_data = {}
		for condition in self.conditionDict.keys():
			if condition in ('remap', 'sacc_map'):
				run_data.update({condition : []})
				for c in self.conditionDict[condition]:
					this_run_data = np.load(self.runFile(stage = 'processed/eye', run = self.runList[c], extension = '.npy')).squeeze()
				
					# v = sqrt(this_run_data[:,3]**2 + this_run_data[:,4]**2)
					vp = pd.Series(np.abs(this_run_data[:,3]))
					# std_vp = pd.Series(np.roll(np.array(pd.rolling_mean(vp / np.std(vp), moving_mean_width_on_velocity)), -moving_mean_width_on_velocity/2))
					threshold_crossings = np.array(vp > threshold, dtype = int)
					blinks = pd.rolling_mean(np.array(this_run_data[1:,2], dtype = bool), 15) > 0
					# sacc_starts = (np.diff(threshold_crossings) == 1)[blinks]
					# sacc_ends = (np.diff(threshold_crossings) == -1)[blinks]
					sacc_starts = np.diff(threshold_crossings) == 1
					sacc_ends = np.diff(threshold_crossings) == -1
				
					# shell()
					sample_times = np.arange(this_run_data.shape[0]) / sample_frequency
					landing_positions = np.array([this_run_data[:,3][sacc_ends - blinks], this_run_data[:,4][sacc_ends - blinks]]).T
						
					stim_onsets = np.arange(0, float(this_run_data.shape[0] - (48 * sample_frequency))/sample_frequency, 1.0) + stim_offset
					sacc_start_times = sample_times[sacc_starts] # (sacc_starts - blinks)
					sacc_end_times = sample_times[sacc_ends]
					sacc_end_times = sacc_end_times[np.concatenate((np.diff(sacc_end_times) > 0.5, [True]))]
					sacc_start_times = sacc_start_times[np.concatenate((np.diff(sacc_end_times) > 0.5, [True]))]
	
					saccade_latencies = np.array([sacc_start_times[sacc_start_times < this_stim_onset][-1] - this_stim_onset for this_stim_onset in stim_onsets[8:-8]])
					saccade_offsets_per_trial = np.array([this_run_data[np.roll(sample_times == sacc_end_times[sacc_end_times > sacc_start_times[sacc_start_times < this_stim_onset][-1]][0],4), [0,1]] for this_stim_onset in stim_onsets[8:-8]])
					saccade_offsets_per_trial_zero_mean = saccade_offsets_per_trial.copy()
					saccade_offsets_per_trial_zero_mean[saccade_offsets_per_trial[:,0] > 7] = saccade_offsets_per_trial[saccade_offsets_per_trial[:,0] > 7] - saccade_offsets_per_trial[saccade_offsets_per_trial[:,0] > 7].mean(axis = 0)
					saccade_offsets_per_trial_zero_mean[saccade_offsets_per_trial[:,0] < 7] = saccade_offsets_per_trial[saccade_offsets_per_trial[:,0] < 7] - saccade_offsets_per_trial[saccade_offsets_per_trial[:,0] < 7].mean(axis = 0)
					saccade_offsets_per_trial_zero_mean[:,1] = saccade_offsets_per_trial[:,1] - saccade_offsets_per_trial[:,1].mean(axis = 0)
				
					stim_after_TR_delay = stim_offset * sample_frequency
					stim_duration = 0.25 * sample_frequency
					# saccade every second
					stim_indices = np.arange(0, this_run_data.shape[0], sample_frequency) + stim_after_TR_delay
					left_gaze = np.array([[this_run_data[ind:ind+stim_duration, 0], this_run_data[ind:ind+stim_duration, 1]] for ind in stim_indices[::2]]).squeeze()
					right_gaze = np.array([[this_run_data[ind:ind+stim_duration, 0], this_run_data[ind:ind+stim_duration, 1]] for ind in stim_indices[1::2]]).squeeze()
		
					left_gaze_zero_mean = left_gaze.mean(axis = -1) - left_gaze.mean(axis = -1).mean(axis = 0)
					right_gaze_zero_mean = right_gaze.mean(axis = -1) - right_gaze.mean(axis = -1).mean(axis = 0)
				
					left_gaze_zero_mean_dev = np.array([np.linalg.norm(a) for a in left_gaze_zero_mean])
					right_gaze_zero_mean_dev = np.array([np.linalg.norm(a) for a in right_gaze_zero_mean])
				
					saccade_offsets_per_trial_dev = np.array([np.linalg.norm(a) for a in saccade_offsets_per_trial_zero_mean])
					fix_deviations_dev = np.array([[aa, bb] for aa, bb in zip(left_gaze_zero_mean_dev,right_gaze_zero_mean_dev)]).ravel()[8:saccade_offsets_per_trial.shape[0]+8]
				
					with open(self.runFile(stage = 'processed/eye', run = self.runList[c], extension = '.pickle'), 'w') as f:
						pickle.dump([			fix_deviations_dev, saccade_offsets_per_trial_dev, saccade_latencies,
						 						np.median(fix_deviations_dev),np.median(saccade_offsets_per_trial_dev),np.median(saccade_latencies)
												], f)
				
					run_data[condition].append([
												fix_deviations_dev, saccade_offsets_per_trial_dev, saccade_latencies,
						 						np.median(fix_deviations_dev),np.median(saccade_offsets_per_trial_dev),np.median(saccade_latencies)
												])
				
		# shell()
	
	def ASL_file_analysis(self, filename, threshold = 0.4, moving_mean_width_on_velocity = 4, stim_offset = 0.1):
		from scipy.io import loadmat
		import pandas as pd
		
		nrVolumes = 112
		nrSecsBeforeCountingForSelfReqAvgs = 8
		TR = 2.0
		
		fileData = loadmat(filename)['dataEYD'][0,0]
		sampleFrequency = fileData['freq'][0,0]
		sample_times = np.arange(0, float(fileData['time'].ravel().shape[0]) / sampleFrequency, 1.0/sampleFrequency)
		
		TRInfo = np.arange(np.array(fileData['XDAT'], dtype = float).ravel().shape[0])[np.diff(np.array(fileData['XDAT'], dtype = float).ravel()) == -1][int(nrSecsBeforeCountingForSelfReqAvgs/TR):]
		blinks = (fileData['pupil_recogn'].ravel() == 0)
		
		x, y = fileData['horz_gaze_coord'],fileData['vert_gaze_coord']
		vx, vy = np.diff(x.ravel()), np.diff(y.ravel())
		v = sqrt(vx**2 + vy**2)
		vp = pd.Series(v)
		std_vp = pd.Series(np.roll(np.array(pd.rolling_mean(vp / np.std(vp), moving_mean_width_on_velocity)), -moving_mean_width_on_velocity/2))
		threshold_crossings = np.array(std_vp > threshold, dtype = int)
		sacc_starts = np.diff(threshold_crossings) == 1
		sacc_ends = np.diff(threshold_crossings) == -1
		
		# adjust length of arrays to the experiment
		interesting_range = np.arange(TRInfo[0],TRInfo[0]+sampleFrequency*nrVolumes*TR, dtype = int)
		blinks = blinks[interesting_range]
		sacc_starts = sacc_starts[interesting_range]
		sacc_ends = sacc_ends[interesting_range]
		x = x[interesting_range]
		y = y[interesting_range]
		sample_times = sample_times[interesting_range]
		
		landing_positions = np.array([x[sacc_ends - blinks], y[sacc_ends - blinks]]).T
		
		stim_onsets = np.arange(0, nrVolumes * TR, 1.0) + sample_times[0] + stim_offset * sampleFrequency
		sacc_start_times = sample_times[(sacc_starts - blinks)]
		sacc_end_times = sample_times[sacc_ends - blinks]
		
		saccade_latencies = [sacc_start_times[sacc_start_times > this_stim_onset][0] - this_stim_onset for this_stim_onset in stim_onsets if this_stim_onset < sacc_start_times[-1]]
		saccade_offsets_per_trial = [sacc_ends[sacc_ends > sacc_start_times[sacc_start_times > this_stim_onset][0]] for this_stim_onset in stim_onsets if this_stim_onset < sacc_start_times[-1]]
		
	
	def createFunctionalMask(self, exclusionThreshold = 2.0, maskFrame = 0, inclusion_threshold = 4.0):
		"""
		Take the eccen F-values, use as a mask, and take out the F-value mask of the peripheral fixation condition
		results in creation of a mask file which can be accessed later
		"""
		# F-value mask from eccen experiment
#		eccenFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen.nii.gz')
		# fixPeripheryFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['fix_periphery'][0]]), 'polar.nii.gz')
		# imO = ImageMaskingOperator(inputObject = eccenFile, maskObject = fixPeripheryFile, thresholds = [exclusionThreshold])
		# # change the first frame of the mask and input data (-log p-value) to its absolute value
		# imO.maskData[0] = np.abs(imO.maskData[0])
		# imO.inputData[0] = np.abs(imO.inputData[0])
		# maskedDataArray = imO.applySingleMask(whichMask = maskFrame, maskThreshold = exclusionThreshold, nrVoxels = False, maskFunction = '__lt__', flat = False)
#		maskImage = NiftiImage(maskedDataArray)
#		maskImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen_mask-' + str(exclusionThreshold) + '.nii.gz')
#		maskImage.save()
		# F-value mask from polar - fix map experiment
		polarFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'polar.nii.gz')
		fixPeripheryFile = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['fix_periphery'][0]]), 'polar.nii.gz')
		imO = ImageMaskingOperator(inputObject = polarFile, maskObject = fixPeripheryFile, thresholds = [exclusionThreshold])
		# change the first frame of the mask and input data (-log p-value) to its absolute value
		imO.maskData[0] = np.abs(imO.maskData[0])
		imO.inputData[0] = np.abs(imO.inputData[0])
		
		print self.subject.initials + ' ' + str(((imO.maskData[0] > exclusionThreshold) * (imO.inputData[0] > inclusion_threshold)).sum()) + ' of ' + str((imO.maskData[0] > exclusionThreshold).sum()) + ' and ' + str((imO.inputData[0] > inclusion_threshold).sum())
		
		maskedDataArray = imO.applySingleMask(whichMask = maskFrame, maskThreshold = exclusionThreshold, nrVoxels = False, maskFunction = '__lt__', flat = False)
		maskImage = NiftiImage(maskedDataArray)
		maskImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'polar_mask-' + str(exclusionThreshold) + '.nii.gz')
		maskImage.save()
		
		overlap_array = ((imO.maskData[0] > exclusionThreshold) * (imO.inputData[0] > inclusion_threshold))
		excl_mask_array = (imO.maskData[0] > exclusionThreshold)
		incl_mask_data = (imO.inputData[0] > inclusion_threshold)
		
		# this final part is a separate analysis per region of interest
		roiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/visual_areas/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('masking functional data from files %s', str([os.path.split(f)[1] for f in roiFileNames]))
		areas = ['V1','V2','V3','V4','V3AB']
		rois = [[] for a in areas]
		for i in range(len(roiFileNames)):
			for a in areas:
				if roiFileNames[i].split('.')[1] == a:
					rois[areas.index(a)].append(roiFileNames[i])
		ratios = []
		for rs in rois:
			thisArea = np.array(np.array([np.array(NiftiImage(roi).data, dtype = bool) for roi in rs]).sum(axis = 0), dtype = bool)
			print areas[rois.index(rs)], (thisArea * overlap_array).sum(), (thisArea * excl_mask_array).sum(), (thisArea * incl_mask_data).sum(), float((thisArea * overlap_array).sum())/float((thisArea * incl_mask_data).sum())
			ratios.append([(thisArea * overlap_array).sum(), (thisArea * excl_mask_array).sum(), (thisArea * incl_mask_data).sum(), float((thisArea * overlap_array).sum())/float((thisArea * incl_mask_data).sum()), thisArea.sum()])
		ratios = np.array(ratios)
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'voxel_overlap_ratios.npy'), ratios)
		# for rn, roi in enumerate(roiFileNames):
		# 	roi = NiftiImage(roi).data
		# 	print os.path.split(roiFileNames[rn])[-1] + ' ' + str((overlap_array * (roi > 0)).sum()) + ' ' + str((excl_mask_array * (roi > 0)).sum()) + ' ' + str((incl_mask_data * (roi > 0)).sum())
			
	
	def collectConditionFiles(self, add_eccen = True):
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
				thisRunFile = os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[c]), 'polar.nii.gz')
				images[-1].append(NiftiImage(thisRunFile))
		return images
		
	def rescaleEccenFile(self, phase_offset = 0.0, phase_to_degree_ratio = 1.0):
		"""docstring for rescaleEccenFile"""
		self.eccenImage = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'eccen.nii.gz'))
		
	
	def maskFiles(self, dataFiles, maskFile = None, maskThreshold = 5.0, maskFrame = 0, nrVoxels = False, flat = False):
		# anatomical or statistical mask?
		if not maskFile.__class__.__name__ == 'ndarray':
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
	
	def conditionDataForRegions(self, regions = ['V1', 'V2', 'V3', 'V3AB', 'V4', ['V1', 'V2', 'V3']], maskFile = 'polar_mask-2.0.nii.gz', nrVoxels = False, maskThreshold = 4.0, add_eccen = False ):
		"""
		Produce phase-phase correlation plots across conditions.
		['rh.V1', 'lh.V1', 'rh.V2', 'lh.V2', 'rh.V3', 'lh.V3', 'rh.V3AB', 'lh.V3AB', 'rh.V4', 'lh.V4']
		['V1', 'V2', 'V3', 'V3AB', 'V4']
		['V1','V2','V3'],['V3AB','V4'],['lateraloccipital','lingual','fusiform'],['cuneus','precuneus','inferiorparietal','superiorparietal']
		['V1'],['V2'],['V3'],['V3AB'],['V4'],['fusiform'],['superiorparietal']
		['pericalcarine','lateraloccipital','lingual','fusiform','cuneus','precuneus','inferiorparietal', 'superiorparietal']
		"""
		self.rois = regions
		maskedFiles = self.maskFiles(dataFiles = self.collectConditionFiles(add_eccen = add_eccen), maskFile = np.array([np.abs(NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), maskFile )).data[0])]), maskThreshold = maskThreshold, maskFrame = 0)
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
	
	def runDataForRegions(self, regions = [['V1'], ['V2'], ['V3'], ['V3AB'], ['V4']], maskFile = 'polar_mask-1.5.nii.gz', maskThreshold = 4.0, nrVoxels = False ):
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
			print condition, [runFile.filename for runFile in runFiles[i]]
			for roi in regions:
				# print roi[]
				thisRoiData = self.maskFiles(dataFiles = maskedFiles, maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat/'), roi[0] + '.nii.gz' ), maskThreshold = 0.0, maskFrame = 0, nrVoxels = nrVoxels, flat = True)
				maskedRunData[-1].append(thisRoiData)
		self.maskedRunData = maskedRunData
		self.logger.debug('masked roi data shape is ' + str(len(self.maskedRunData)) + ' ' + str(len(self.maskedRunData[0])) + ' ' + str(len(self.maskedRunData[1][0]))+ ' ' + str(len(self.maskedRunData[0][0][0])))
	
	
	def eyeDataForRuns(self):
		"""
		Get eye data from runs in the same sequence as the runDataForRegions function
		"""
		eyes = []
		for condition in ('remap', 'sacc_map'):
			eyes.append([])
			for c in self.conditionDict[condition]:
				thisConditionFile = self.runFile(stage = 'processed/eye', run = self.runList[c], extension = '.pickle')
				with open(thisConditionFile) as f:
					eyes[-1].append(pickle.load(f))
		return eyes
	
	def eyeContingentRemappingPerRun(self):
		"""analyze the remapping signal per run, and correlate this with the saccade and fixation errors."""
		eye_data = self.eyeDataForRuns() # remap, sacc_map
		roi_data_per_run = self.runDataForRegions()
		roi_data_per_condition = self.conditionDataForRegions()
		results = []
		f = pl.figure(figsize = (8,8))
		for area in range(len(self.maskedConditionData)):
			mask = self.maskedConditionData[area][self.conditionDict.keys().index('fix_map')][9] != 0
			fix_map_phases = self.maskedConditionData[area][self.conditionDict.keys().index('fix_map')][9][mask]
			sacc_map_phases = self.maskedConditionData[area][self.conditionDict.keys().index('sacc_map')][9][mask]
			fix_periphery_phases = self.maskedConditionData[area][self.conditionDict.keys().index('fix_periphery')][9][mask]
			remap_phases_per_run = np.array([self.maskedRunData[self.conditionDict.keys().index('remap')][area][run][9][mask] for run in range(len(self.maskedRunData[self.conditionDict.keys().index('remap')][area]))])
			
			# remap is the first of the saccade conditions, so it's eye_data[0]
			eye_parameters_remap = np.array([[eye_data[0][run][i] for i in [3,4]] for run in range(len(eye_data[0]))])
			
			# calculate phase differences and such
			map_diffs = (pi - np.abs(circularDifference(fix_map_phases, sacc_map_phases))) / pi
			periphery_diffs = (pi - np.abs(circularDifference(fix_periphery_phases, sacc_map_phases))) / pi
			remap_diffs = (pi - np.abs(np.array([circularDifference(r, sacc_map_phases) for r in remap_phases_per_run]))) / pi
			
			s = f.add_subplot(len(self.maskedConditionData), 2, (2*area) + 1)
			pl.hist(periphery_diffs, range = [0,1], bins = 50, alpha = 0.3, histtype='step', color = 'b')
			pl.hist(map_diffs, range = [0,1], bins = 50, alpha = 0.3, histtype='stepfilled', color = 'r')
			for i in range(remap_diffs.shape[0]):
				pl.hist(remap_diffs[i], range = [0,1], bins = 50, alpha = 0.3, histtype='step', color = 'g')
			print map_diffs.mean(), periphery_diffs.mean(), remap_diffs.mean(axis = 1)
			
			# shell()
			s = f.add_subplot(len(self.maskedConditionData), 2, (2*area) + 2)
			for i in range(remap_diffs.shape[0]):
				pl.plot( eye_parameters_remap[:,0] - eye_parameters_remap[:,0].mean(), remap_diffs.mean(axis = 1) - remap_diffs.mean(axis = 1).mean(), 'gx', alpha = 0.3 + (0.7 / remap_diffs.shape[0]) )
			for i in range(remap_diffs.shape[0]):
				pl.plot( eye_parameters_remap[:,1] - eye_parameters_remap[:,1].mean(), remap_diffs.mean(axis = 1) - remap_diffs.mean(axis = 1).mean(), 'g+', alpha = 0.3 + (0.7 / remap_diffs.shape[0]) )
			results.append(np.vstack(((eye_parameters_remap - eye_parameters_remap.mean(axis = 0)).T, remap_diffs.mean(axis = 1) - remap_diffs.mean(axis = 1).mean())))
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'eye_remap_run_correlations.pdf' ))
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'eye_remap_run_correlations.npy' ), np.array(results))
	
	def saccadeLatencies(self):
		eye_data = self.eyeDataForRuns() # remap, sacc_map
	
		# saccade latency plots:
		# remap is 0
		saccade_latencies = [np.concatenate([eye_data[j][i][2] for i in range(len(eye_data[j]))]) for j in [0, 1]]
		f = pl.figure(figsize = (6,4))
		s = f.add_subplot(111)
		pl.hist(saccade_latencies[0], normed = True, bins = 200, range = [-0.750, 0], color = 'g', histtype = 'step', alpha = 0.15, cumulative = False)
		pl.hist(saccade_latencies[1], normed = True, bins = 200, range = [-0.750, 0], color = 'r', histtype = 'step', alpha = 0.15, cumulative = False)
		s.set_xlim(-0.75, 0)
		s.set_xlabel('pre-stimulus time [s]')
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'eye_sacc_latencies.pdf' ))
		
		with open(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'eye_sacc_latencies.pickle'), 'w') as f:
			pickle.dump(saccade_latencies, f)
		
		pl.show()
	
	def phasePhasePlots(self, nrBins = 24):
		if not hasattr(self, 'maskedConditionData'):
			self.conditionDataForRegions()
		self.logger.debug('masked roi data shape is ' + str(len(self.maskedConditionData)) + ' ' + str(len(self.maskedConditionData[0])) + ' ' + str(self.maskedConditionData[0][0].shape))
		from itertools import combinations
		f = pl.figure(figsize = (1.5 * len(self.maskedConditionData), 1.5 *len(list(combinations(range(len(self.conditionDict)),2)))))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1
		combs = list(combinations(range(len(self.conditionDict)),2))
		self.logger.debug( str(self.conditionDict) + str(combs) )
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
			self.conditionDataForRegions( regions = ['V1', 'V2', 'V3', 'V3AB', 'V4',['inferiorparietal','superiorparietal'], ['V1', 'V2', 'V3']], maskThreshold = maskThreshold, nrVoxels = nrVoxels )
		
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
				[mu, kappa] =  fitVonMises(-diffs, initial = [pi,0.5] )
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
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitResults.npy' ), self.fitResults)
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phaseDiffs.npy' ), self.allPhaseDiffs)
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitPhaseDifferences.pdf' ))
		
	
		
		
	def fitPhaseDifferencesZeroMean(self, comparisons = [['sacc_map','fix_map'],['sacc_map','remap'],['sacc_map','fix_periphery']], maskThreshold = 4.0, nrVoxels = False, runBootstrap = True, nrBootstrapRepetitions = 1000):
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
				allBootstrapResults = np.load(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitPhaseDifferencesZeroMean.npy' ))
			except OSError:
				runBootstrap = True
				allBootstrapResults = np.zeros((len(self.maskedConditionData), len(comparisons), nrBootstrapRepetitions))
		else:
			allBootstrapResults = np.zeros((len(self.maskedConditionData), len(comparisons), nrBootstrapRepetitions))
		
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
				[kappa] =  fitVonMisesZeroMean(-diffs)
				fitResults[-1].append([kappa])
				allPhaseDiffs[-1].append(diffs)
				phaseHists[-1].append(np.histogram(diffs, bins = 50, range = (-pi, pi), normed = True)[0])
				pl.hist(diffs, range = (-pi,pi), normed = True, bins = 50, color = ['r','g','b'][c], histtype = 'stepfilled', alpha = 0.15)
				pl.plot(np.linspace(-pi,pi,100), scipy.stats.vonmises.pdf(0.0, kappa, np.linspace(pi,-pi,100)), ['r-','g-','b-'][c])
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
			np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitPhaseDifferencesZeroMean.npy' ), allBootstrapResults)
	
		
		self.bootstrapResults = allBootstrapResults
		self.fitResults = np.array(fitResults)
		self.allPhaseDiffs = allPhaseDiffs
		self.phaseHists = np.array(phaseHists)
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitResultsZeroMean.npy' ), self.fitResults)
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phaseDiffsZeroMean.npy' ), self.allPhaseDiffs)
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fitPhaseDifferencesZeroMean.pdf' ))
	
	def collapsePhaseDifferences(self, comparisons = [['sacc_map','fix_map'],['sacc_map','remap'],['sacc_map','fix_periphery'],['sacc_map','remap']], maskThreshold = 4.0, nrVoxels = False):
		""""""
		if not hasattr(self, 'maskedConditionData'):
			self.conditionDataForRegions( maskThreshold = maskThreshold, nrVoxels = nrVoxels )
		collapsedPhaseDiffs = np.zeros((len(self.maskedConditionData),len(comparisons)))
		forHists = []
		f = pl.figure(figsize = (7,12))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		plotNr = 1
		for i in range(len(self.maskedConditionData)):
			forHists.append([])
			sbp = f.add_subplot(len(self.maskedConditionData),1,plotNr)
			for (c,cond) in zip(range(len(comparisons)), comparisons):
				cond1 = self.conditionDict.keys().index(cond[0])
				cond2 = self.conditionDict.keys().index(cond[1])
				summedArray = - ( self.maskedConditionData[i][cond1][0] + self.maskedConditionData[i][cond2][0] == 0.0 )
				diffs = circularDifference(self.maskedConditionData[i][cond1][9][summedArray], self.maskedConditionData[i][cond2][9][summedArray])
				collapsedPhaseDiffs[i,c] = 1.0 - (np.abs(diffs).mean() / pi)
				forHists[-1].append(1.0 - (np.abs(diffs) / pi))
			
			pl.bar(np.arange(0,len(comparisons)), collapsedPhaseDiffs[i])
			sbp.set_title(str(self.rois[i]), fontsize=10)
			sbp.set_ylabel(self.conditionDict.keys()[cond1] + ' - ' + self.conditionDict.keys()[cond2], fontsize=10)
			plotNr += 1
			
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsed.pdf' ))
		self.collapsedPhaseDiffs = collapsedPhaseDiffs
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsed.npy' ), collapsedPhaseDiffs)
		
		f = pl.figure(figsize = (12,12))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		plotNr = 1
		for i in range(len(forHists)):
			sbp = f.add_subplot(len(forHists),2,plotNr)
			for j in range(len(comparisons)):
				pl.hist(forHists[i][j], range = [0,1], bins = 50, alpha = 0.25, normed = True, histtype = 'step', linewidth = 2.5, color = ['r','g','b','c'][j])
			plotNr += 1
			sbp.set_title(str(self.rois[i]), fontsize=10)
			sbp = f.add_subplot(len(forHists),2,plotNr) #sbp.twinx()
			for j in range(len(comparisons)):
				pl.plot(np.linspace(0,1,len(forHists[i][j])), np.sort(np.array(forHists[i][j])), ['r--','g--','b--','c--'][j], alpha = 0.75)
			plotNr += 1
			sbp.set_title(str(self.rois[i]), fontsize=10)
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsed_hist.pdf' ))
		f = open(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsed_hist.pickle' ), 'w')
		pickle.dump(forHists, f)
		f.close()
		
		if False:
			# this is just done for the original 3 comparisons. not adding the third since this didn't show any correlations
			f = pl.figure(figsize = (5,12))
			pl.subplots_adjust(hspace=0.4, wspace=0.4)
			plotNr = 1
			for i in range(len(forHists)):
				sbp = f.add_subplot(len(forHists),1,plotNr)
				sbp.set_title(str(self.rois[i]), fontsize=10)
				print [forHists[i][j].shape[0] for j in range(3)]
				pl.scatter(forHists[i][0][:], forHists[i][1][:forHists[i][0].shape[0]], alpha = 0.25, linewidth = 1.75, color = 'g')
				stt = sp.stats.spearmanr(forHists[i][0], forHists[i][1][:forHists[i][0].shape[0]])
				sbp.annotate('rho: %1.3f' % stt[0] + ' p: %1.3f' % stt[1], (0.525,0.9), va="top", ha="left", size = 9, color = 'g')
				pl.scatter(forHists[i][0], forHists[i][2][:forHists[i][0].shape[0]], alpha = 0.25, linewidth = 1.75, color = 'b')
				stt = sp.stats.spearmanr(forHists[i][0], forHists[i][2][:forHists[i][0].shape[0]])
				sbp.annotate('rho: %1.3f' % stt[0] + ' p: %1.3f' % stt[1], (0.525,0.7), va="top", ha="left", size = 9, color = 'b')
				pl.scatter(forHists[i][1][:forHists[i][0].shape[0]], forHists[i][2][:forHists[i][0].shape[0]], alpha = 0.25, linewidth = 1.75, color = 'k')
				stt = sp.stats.spearmanr(forHists[i][1][:forHists[i][0].shape[0]], forHists[i][2][:forHists[i][0].shape[0]])
				sbp.annotate('rho: %1.3f' % stt[0] + ' p: %1.3f' % stt[1], (0.525,0.5), va="top", ha="left", size = 9, color = 'k')
				plotNr += 1
				sbp.axis([0.5,1,0,1])
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsed_scatter.pdf' ))
	
	def phaseDifferencesPerPhase(self, comparisons = [['fix_map','sacc_map'],['fix_map','remap'],['fix_map','fix_periphery']], baseCondition = 'sacc_map', binSize = 32, maskThreshold = 4.0, smooth = True, smoothSize = 12, stretch = 1.0 ):
		self.conditionDataForRegions(add_eccen = True, maskThreshold = maskThreshold, regions = [['V1','V2','V3'],['V1'],['V2'],['V3'],['V3AB'],['V4'],['inferiorparietal','superiorparietal']] ) 
		if not hasattr(self, 'phasePhaseHistogramDict'):
			self.phasePhaseHistogramDict = {}
		if not hasattr(self, 'phasePhaseTotalDict'):
			self.phasePhaseTotalDict = {}
		
		f = pl.figure(figsize = (10,10))
		pl.subplots_adjust(hspace=0.4)
		pl.subplots_adjust(wspace=0.4)
		plotNr = 1		
		outputData = []
		totalData = []
		for cond in comparisons:
			cond1 = self.conditionDict.keys().index(cond[0])
			cond2 = self.conditionDict.keys().index(cond[1])
			outputData.append([])
			totalData.append([])
			for i in range(len(self.maskedConditionData)):
				sbp = f.add_subplot(len(comparisons),len(self.maskedConditionData),plotNr)
				summedArray = - ( self.maskedConditionData[i][cond1][0] + self.maskedConditionData[i][cond2][0] == 0.0 )
				# base phase data based on eccen which is the last data file in maskedConditionData
				if baseCondition == 'eccen':
					summedArray = summedArray * (-( self.maskedConditionData[i][-1][9] == 0.0 ))
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
				if baseCondition == 'eccen':
					histData = np.histogram2d(positivePhases(baseData) * stretch * np.cos( circDiffData ), positivePhases(baseData) * stretch * np.sin( circDiffData ), [np.linspace(-pi,pi,binSize),np.linspace(-pi,pi,binSize)], normed = True )[0]
				else:
					histData = np.histogram2d(baseData,circDiffData, [np.linspace(-pi,pi,binSize),np.linspace(-pi,pi,binSize)])[0]
				
				if smooth:
					from scipy.signal import convolve2d
					x = np.linspace(-3.0, 3.0, smoothSize)
					y = np.linspace(-3.0, 3.0, smoothSize)
					X,Y = meshgrid(x, y)
					filt = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
					h2 = np.tile(np.tile(histData, 3).T, 3)
					h2f = convolve2d(h2.T, filt, mode= 'same')
					
					histData = h2f[histData.shape[0]:histData.shape[0]*2,histData.shape[1]:histData.shape[1]*2]
					
#					pl.imshow(filt, extent = (-pi,pi,-pi,pi), alpha = 0.5)
				pl.imshow(histData, extent = (-pi,pi,-pi,pi))
				plotNr += 1
				outputData[-1].append(histData)
				totalData[-1].append(np.array([baseData, circDiffData]))
		self.phasePhaseHistogramDict.update( {baseCondition: outputData} )
		self.phasePhaseTotalDict.update( {baseCondition: totalData} )
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phaseDifferencesPerPhase_' + baseCondition + '.pdf' ))
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phaseDifferencesPerPhase_raw_' + baseCondition + '.npy' ), np.array(totalData))
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'phaseDifferencesPerPhase_hist_' + baseCondition + '.npy' ), np.array(outputData))
		return outputData
	
	def collapsePhaseDifferencesPerPhase(self, comparisons = [['sacc_map','fix_map'],['sacc_map','remap'],['sacc_map','fix_periphery']], baseCondition = 'fix_map', binSize = 0.5, nrBins = 100, maskThreshold = 5.0 ):
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
				diffs = (1.0 - (np.abs(circDiffData) / pi) )
				diffs = np.tile(diffs, 3) 
				baseData = np.concatenate((baseData - 2*pi, baseData, baseData + 2*pi))
				res = np.array([diffs[((baseData > (ph - binSize/2.0)) * (baseData <= (ph + binSize/2.0)))].mean() for ph in np.linspace(-pi,pi, nrBins)])
				outputData[i,j] = res
				
				sbp.set_title(str(self.rois[i]), fontsize=8)
				pl.plot(np.linspace(-pi,pi,nrBins), outputData[i,j], ['r-','g-','b-'][j])
			plotNr += 1
		self.collapsedPhaseDiffDict.update( {baseCondition: outputData} )
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsedPhaseDiffPP.pdf' ))
		return outputData
	
	def collapsePhaseDifferencesHorVer(self, comparisons = [['sacc_map','fix_map'],['sacc_map','remap'],['sacc_map','fix_periphery']], baseCondition = 'fix_map', nrBins = 8, maskThreshold = 3.0 ):
		self.conditionDataForRegions(add_eccen = False, regions = [['V1'],['V2'],['V3'],['V3AB'],['V4'],['fusiform'],['superiorparietal']], maskThreshold = maskThreshold ) # [['V1'],['V2'],['V3'],['V3AB'],['V4'],['fusiform'],['superiorparietal']]
		
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
				diffs = (1.0 - (np.abs(circDiffData) / pi) )
				
				res = np.array([diffs[((baseData > ph) * (baseData <= (ph + 1.0/nrBins)))].mean() for ph in np.linspace(0,1, nrBins, endpoint = False)])
				outputData[i,j] = res
				
				sbp.set_title(str(self.rois[i]), fontsize=8)
				pl.plot(np.linspace(-pi,pi,nrBins), outputData[i,j], ['r-','g-','b-'][j])
			plotNr += 1
		self.collapsedPhaseDiffDict.update( {baseCondition + '_HV': outputData} )
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsedPhaseDiffHV.pdf' ))
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsedPhaseDiffHV.npy' ), outputData)
		
		return outputData

	
	def combinationsPhaseDifferences(self, comparisons = [['fix_map','sacc_map'],['fix_map','remap'],['fix_map','fix_periphery'],['sacc_map','remap'],['sacc_map','fix_periphery']], maskThreshold = 3.0, nrVoxels = False):
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
				pl.hist(diffs, range = (-pi,pi), normed = True, bins = 25, color = ['r','g','b','m','c'][j], histtype = 'stepfilled', alpha = 0.15)
				pl.plot(np.linspace(-pi,pi,100), scipy.stats.vonmises.pdf(mu, kappa, np.linspace(pi,-pi,100)), ['r-','g-','b-','m-','c-'][j])
				sbp.set_title(str(self.rois[i]), fontsize=10)
				sbp.set_ylabel(self.conditionDict.keys()[cond1] + ' - ' + self.conditionDict.keys()[cond2] + ' ' + str(diffs.shape[0]), fontsize=10)
			plotNr += 1	
		self.combinationFitResults = np.array(fitResults)
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'combinationsPhaseDifferences.pdf' ))
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'combinationsPhaseDifferences.npy' ), self.combinationFitResults)
	
	def runEyeMovementControlsForRemapping(self, regions = [['V1'],['V2'],['V3'], ['V3AB'],['V4']], maskFile = 'polar_mask-1.5.nii.gz', nrVoxels = False, maskThreshold = 4.0):
		# get original data
		self.conditionDataForRegions( regions = regions, maskFile = maskFile, maskThreshold = maskThreshold )
		
		# get different versions of the remapped data
		remapImages = [NiftiImage(os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['remap'][0]]), fileName + '.nii.gz')) for fileName in ['polar', 'polar_good', 'polar_bad']]
		# mask these files with the basic statistical ROI
		remapImages = self.maskFiles(dataFiles = remapImages, maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), maskFile ), maskThreshold = maskThreshold, maskFrame = 0)
		maskedRemapData = []
		for roi in regions:
			if roi.__class__.__name__ == 'str': 
				thisRoiData = self.maskFiles(dataFiles = remapImages, maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat/'), roi + '.nii.gz' ), maskThreshold = 0.0, maskFrame = 0, nrVoxels = nrVoxels, flat = True)
			elif roi.__class__.__name__ == 'list':
				print roi
				allRoisData = []
				for r in roi:
					allRoisData.append( np.array( self.maskFiles(dataFiles = remapImages, maskFile = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat/'), r + '.nii.gz' ), maskThreshold = 0.0, maskFrame = 0, nrVoxels = nrVoxels, flat = True) ) )
				
				thisRoiData = np.dstack(allRoisData)
			maskedRemapData.append(thisRoiData)
		self.maskedRemapData = maskedRemapData
		
		originalIndex = self.conditionDict.keys().index('sacc_map')
		periIndex = self.conditionDict.keys().index('fix_periphery')
		
		collapsedPhaseDiffs = np.zeros((len(self.maskedRemapData),len(remapImages)))
		collapsedPhaseDiffsPeri = np.zeros((len(self.maskedRemapData),len(remapImages)))
		f = pl.figure(figsize = (7,4))
		colors = ['r','g','b']
		width = 0.05
		sbp = f.add_subplot(1,2,1)
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		plotNr = 1		
		for i in range(collapsedPhaseDiffs.shape[0]):
			for j in range(collapsedPhaseDiffs.shape[1]):
				summedArray = - ( self.maskedConditionData[i][originalIndex][0] + self.maskedRemapData[i][j][0] == 0.0 )
				diffs = circularDifference(self.maskedConditionData[i][originalIndex][9][summedArray], self.maskedRemapData[i][j][9][summedArray])
				diffsPeri = circularDifference(self.maskedConditionData[i][periIndex][9][summedArray], self.maskedRemapData[i][j][9][summedArray])
				collapsedPhaseDiffs[i,j] = 1.0 - (np.abs(diffs).mean() / pi)
				collapsedPhaseDiffsPeri[i,j] = 1.0 - (np.abs(diffsPeri).mean() / pi)
			
				pl.bar(j-i*width, collapsedPhaseDiffs[i,j], color = colors[j])
		sbp = f.add_subplot(1,2,2)
		pl.plot(collapsedPhaseDiffs[:,0],collapsedPhaseDiffs[:,2], 'or')
		pl.plot(collapsedPhaseDiffsPeri[:,0],collapsedPhaseDiffsPeri[:,2], 'ob')
		pl.plot(collapsedPhaseDiffs[:,1]-collapsedPhaseDiffsPeri[:,1],collapsedPhaseDiffs[:,2]-collapsedPhaseDiffsPeri[:,2], 'og')
		sbp.axis([-0.1,0.3,-0.1,0.3])
		pl.plot([0,1],[0,1], 'k--')
			
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsedAllRemap.pdf' ))
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'collapsedAllRemap.npy' ), [collapsedPhaseDiffs,collapsedPhaseDiffsPeri])
		self.collapsedPhaseDiffsRemap = collapsedPhaseDiffs
	
	def wholeBrainComparisons(self, comparisons = [['fix_map','sacc_map'],['fix_map','remap'],['fix_map','fix_periphery'],['sacc_map','remap']] ):
		"""docstring for wholeBrainComparisons"""
		
		allDiffs = []
		
		for comp in comparisons:
			f1 = NiftiImage(os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[comp[0]][0]]), 'polar.nii.gz'))
			f2 = NiftiImage(os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[comp[1]][0]]), 'polar.nii.gz'))
			
			allDiffs.append((1.0 - (np.abs(circularDifference(f1.data[9], f2.data[9])) / pi) ))
			
		allDiffs.append( allDiffs[1]/allDiffs[0] )
		allDiffs.append( allDiffs[2]/allDiffs[0] )
		allDiffs.append( allDiffs[3]/allDiffs[0] )
		allDiffs.append( (allDiffs[1] - allDiffs[2]) )
		allDiffs.append( (allDiffs[1] - allDiffs[2]) / allDiffs[0] )
		
		allDiffs = np.array(allDiffs)
		
		nF = NiftiImage( allDiffs )
		nF.filename = os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'diffs.nii.gz')
		nF.header = f2.header
		nF.save()
		
		imo = ImageMaskingOperator(nF.filename, maskObject = NiftiImage(nF.filename))
		nF = NiftiImage( imo.applySingleMask(whichMask = 0, maskThreshold = 0.75) )
		nF.filename = os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'diffs.nii.gz')
		nF.header = f2.header
		nF.save()
		
		frames = {'full':0, 'remap':1, 'perihery':2, 'remap2':3, 'remap_full':4, 'peripheral_full': 5, 'remap2_full':6, 'remap-peri':7, 'remap-peri_full':8}
		vts = VolToSurfOperator(inputObject = nF)
		vts.configure(frames = frames, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/figs/surf'), 'res_'), surfSmoothingFWHM = 0.5, surfType = 'paint' )
		vts.execute()
		
		# split functional file
		for fr in frames.keys():
			outputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/figs/surf'), 'res_' + fr + '.nii.gz')
			print outputFileName, frames[fr], fr
			tf = NiftiImage(np.array([nF.data[frames[fr]]]))
			tf.header = nF.header
			tf.filename = outputFileName
			tf.save()
		
		# resample to standard subject fsaverage
		for fr in frames.keys():
			for h in ['lh','rh']:
				inputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/figs/surf'), 'res_' + fr + '-' + h + '.w')
				outputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/figs/surf'), 'fsa_' + fr + '-' + h + '.w')
				sts = SurfToSurfOperator(inputObject = inputFileName)
				sts.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'fsaverage', hemi = h, outputFileName = outputFileName, insmooth = 10)
				if not os.path.isfile(outputFileName):
					sts.execute()
					
				# converting the surface to ascii is not necessary when using mri_preprocess and mri_glmfit
#				mscO = MRISConvertOperator(outputFileName)
#				mscO.configure(surfaceFile = os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage', 'surf', h + '.inflated'))
#				mscO.execute()
	
	def smoothRoiDataOverTime(self, data, start, end, width = 1, start_out = 4, end_out = 4):
		timepoints_per_run = end-start
		nr_runs = data.shape[0]/(timepoints_per_run)
		new_nr_timepoints_per_run = timepoints_per_run - end_out - start_out
		out_data = np.zeros((new_nr_timepoints_per_run* nr_runs, data.shape[1]))
		for i in range(nr_runs):
			# smooth over time per voxel
			run_data_s = np.zeros((timepoints_per_run-start_out-end_out,))
			for j in range(data.shape[1]):
				out_data[i*new_nr_timepoints_per_run:(i+1)*new_nr_timepoints_per_run,j] = np.convolve(data[i*timepoints_per_run:(i+1)*timepoints_per_run,j], np.ones((width))/width, 'valid')[start_out:timepoints_per_run-end_out]
		self.logger.info('smoothed roi data from shape ' + str(data.shape) + ' changed to ' + str(out_data.shape))
		return out_data
	
	def phaseDecodingFromBaseConditionRoi(self, roi, base_condition = 'sacc_map', test_conditions = ['fix_map','remap','fix_periphery'], subfigure = None, color = 'k' ):
		start, end = 16, 120
		trainData = np.array(self.gatherRIOData(roi, whichRuns = self.conditionDict[base_condition], whichMask = '_polar', timeSlices = [start,end] ), dtype = np.float64)
		trainData = self.smoothRoiDataOverTime(trainData, start, end)
		trainPhases = np.array(np.mod(np.arange(trainData.shape[0]), 16), dtype = np.float64)
		[nr_train_samples, nr_voxels] = trainData.shape
		
		from ..Operators.ArrayOperator import DecodingOperator
		from scipy.stats import vonmises
		
		if subfigure == None:
			f = pl.figure()
			subfigure = f.add_subplot(111)
		subfigure.set_title(roi)
		for (i, testCondition) in zip(range(3), test_conditions):
			testData = np.array(self.gatherRIOData(roi, whichRuns = self.conditionDict[testCondition], whichMask = '_polar', timeSlices = [start,end] ), dtype = np.float64)
			testData = self.smoothRoiDataOverTime(testData, start, end)
			testPhases = np.array(np.mod(np.arange(testData.shape[0]), 16), dtype = np.float64)
			nr_test_samples = testData.shape[0]
			
			roiData = np.concatenate([trainData, testData])
			dec = DecodingOperator(roiData, decoder = 'multiclass', fullOutput = True)
			
			out = dec.decode(np.arange(trainData.shape[0]), trainPhases, np.arange(trainData.shape[0],roiData.shape[0]), testPhases)[-1]
			pl.hist(circularDifference((testPhases / 16.0 ) * 2.0 * pi, (out / 16.0 ) * 2.0 * pi), alpha = 0.25, range = [-pi,pi], bins = 17, normed = True, histtype = 'stepfilled', linewidth = 2.5, color = ['r','g','b'][i], rwidth = 1.0)
			fts = fitVonMises(circularDifference((testPhases / 16.0 ) * 2.0 * pi, (out / 16.0 ) * 2.0 * pi))
			pl.plot(np.linspace(-pi,pi,100), vonmises.pdf(fts[0] ,fts[1] , np.linspace(-pi,pi,100)), alpha = 0.25, linewidth = 2.5, color = ['r','g','b'][i])
			subfigure.axis([-pi,pi,0,1.5])
			
	
	def phaseDecodingConditionRoi(self, condition, roi, subfigure = None, color = 'k' ):
		
		start, end = 8, 120
		roiData = np.array(self.gatherRIOData(roi, whichRuns = self.conditionDict[condition], whichMask = '_polar', timeSlices = [start,end] ), dtype = np.float64)
		phases = np.array(np.mod(np.arange(roiData.shape[0]), 16), dtype = np.float64)
		
		from ..Operators.ArrayOperator import DecodingOperator
		from scipy.stats import vonmises
		
		# 96 TRs per run.
		# reshape for grouping across runs?
		[nr_samples, nr_voxels] = roiData.shape
#		roiData = roiData.reshape((roiData.shape[0]/(end-start), (end-start), -1))
#		roiData = roiData.transpose((1,0,2)).reshape(nr_samples, -1)
		
		run_width = 64
		dec = DecodingOperator(roiData, decoder = 'multiclass', fullOutput = True)
		print 'nr of samples in ' + condition + ', ' + roi + ': ' + str(nr_samples) + ' whole shape: ' + str(roiData.shape)

		
		if subfigure == None:
			f = pl.figure()
			subfigure = f.add_subplot(111)
		
		all_out = []
		for i in range(0, nr_samples-run_width, 8):
			testThisRun = (np.arange(nr_samples) >= i) * (np.arange(nr_samples) < i+run_width)
#			testThisRun = np.array(np.random.binomial(1, run_width/float(nr_samples), nr_samples), dtype = bool)
			trainingThisRun = -testThisRun
			trainingDataIndices = np.arange(nr_samples)[trainingThisRun]
			testDataIndices = np.arange(nr_samples)[testThisRun]
			trainingsLabels = phases[trainingThisRun]
			testLabels = phases[testThisRun]
			
			out = dec.decode(trainingDataIndices, trainingsLabels, testDataIndices, testLabels)[-1]
#			out = (out / 16.0 ) * 2.0 * pi
#			print out, testLabels, out - testLabels
			all_out.append([testLabels, out, circularDifference((testLabels / 16.0 ) * 2.0 * pi, (out / 16.0 ) * 2.0 * pi)])
		all_out = np.array(all_out)
#		all_out_diffs = np.concatenate(all_out[:,2])
#		pl.hist(all_out[:,0].ravel(), alpha = 0.1, bins = 16, normed = False, histtype = 'step', linewidth = 2.5, color = 'r', rwidth = 1.0)
#		pl.hist(all_out[:,1].ravel(), alpha = 0.1, bins = 16, normed = False, histtype = 'step', linewidth = 2.5, color = 'g', rwidth = 1.0)
		pl.hist(all_out[:,-1].ravel(), alpha = 0.25, range = [-pi,pi], bins = 17, normed = True, histtype = 'stepfilled', linewidth = 2.5, color = color, rwidth = 1.0)
		fts = fitVonMises(all_out[:,-1].ravel())
		pl.plot(np.linspace(-pi,pi,500), vonmises.pdf(fts[0] ,fts[1] , np.linspace(-pi,pi,500)), alpha = 0.25, linewidth = 2.5, color = color)
		subfigure.axis([-pi,pi,0,1.5])
		subfigure.set_title(roi + ' ' + condition)
#		pl.plot(np.linspace(-pi, pi, 16), np.histogram(all_out_diffs, bins = 16, range = [-pi, pi], normed = True)[0], linewidth = 2.5, color = color )
#		im = histogram2d(all_out[:,0].ravel(), all_out[:,1].ravel(), bins = 16, range = [[0,2*pi],[0,2*pi]], normed=True)
#		pl.imshow(im[0])
#		pl.scatter(all_out[:,0].ravel(), all_out[:,1].ravel(), color = color, edgecolor = 'w', s = 10, marker = 'o' )
	
	def phaseDecodingRoi(self, roi, condition_array = ['fix_map', 'sacc_map', 'remap', 'fix_periphery'], colors = ['r', 'm', 'g', 'b'], subfigure = None, figure = None):
		figure = pl.figure(figsize = (12,4))
		for (cond, i) in zip(condition_array, range(len(condition_array))):
			subfigure = figure.add_subplot(1,4,i+1)
			self.phaseDecodingConditionRoi(condition = cond, roi = roi, subfigure = subfigure, color = colors[i])
	
	def phaseDecodingRois(self, roi_array = ['V1', 'V2', 'V3', 'V3AB', 'V4']):
	#	fig = pl.figure(figsize = (6,8))
		for (roi, i) in zip(roi_array, range(len(roi_array))):
#			subfig = fig.add_subplot(len(roi_array), 1, i+1)
			subfig = None
			self.phaseDecodingRoi(roi = roi, subfigure = subfig, figure = None)
#			fig.set_title(roi)
			pl.draw()
		pl.show()
	

	def behavior(self):
		dPrimes = []
		for (i, c) in zip(range(len(self.conditionDict)), self.conditionDict.keys()):
			conditionAnswers = []
			for r in self.conditionDict[c]:
				wrO = WedgeRemappingOperator( self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.dat' )  )
				wrO.segmentOutputData()
				wrO.collectResponsesAfterColorChanges()
				conditionAnswers.append(wrO.answerList)
			# now concatenate all trials for all runs in one condition
			conditionAnswers = np.vstack(conditionAnswers)
			(hits, misses, corr_rej, fa) = ((conditionAnswers[:,-2] == 1.).sum(), (conditionAnswers[:,-2] == 0.).sum(), (conditionAnswers[:,-2] == 2.).sum(), (conditionAnswers[:,-2] == -1.).sum())
			hit_rate, fa_rate = (float(hits) / (hits + misses), float(fa) / (fa + corr_rej))
			if fa_rate == 0.0:
				fa_rate = 0.01
			if hit_rate == 1.0:
				hit_rate = 0.99
			zH, zF = sp.stats.norm.ppf(hit_rate), sp.stats.norm.ppf(fa_rate)
			dPrimes.append(zH-zF)
		print self.conditionDict.keys()
		np.save(os.path.join(self.stageFolder(stage = 'processed/behavior' ), 'dPrime.npy'), np.array(dPrimes))