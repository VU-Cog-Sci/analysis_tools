#!/usr/bin/env python
# encoding: utf-8
"""
EyeOperator.py

Created by Tomas Knapen on 2010-12-19.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess, re
import tempfile, logging
import pickle

import scipy as sp
import scipy.fftpack
import scipy.signal as signal
import numpy as np
import matplotlib.pylab as pl
from math import *
from scipy.io import *

from nifti import *
from Operator import *
from datetime import *

from tables import *

from BehaviorOperator import NewBehaviorOperator

def derivative_normal_pdf( mu, sigma, x ):
	return -( np.exp( - ( (x - mu)**2 / (2.0 * (sigma ** 2))) ) * (x - mu)) / ( sqrt(2.0 * pi) * sigma ** 3)

class EyeOperator( Operator ):
	"""docstring for ImageOperator"""
	def __init__(self, inputObject, **kwargs):
		"""
		EyeOperator operator takes a filename
		"""
		super(EyeOperator, self).__init__(inputObject = inputObject, **kwargs)
		if self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
		self.logger.info('started with ' +os.path.split(self.inputFileName)[-1])
		

class ASLEyeOperator( EyeOperator ):
	"""docstring for ASLEyeOperator"""
	def __init__(self, inputObject, **kwargs):
		super(ASLEyeOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.type = 'ASL'
		self.rawDataFile = loadmat(self.inputFileName)['dataEYD'][0,0]
		
		self.sampleFrequency = self.rawDataFile['freq'][0,0]
	
	def trSignals(self, TR = None):
		self.TRinfo = (np.array(self.rawDataFile['XDAT']-np.min(self.rawDataFile['XDAT']))/(np.max(self.rawDataFile['XDAT'])-np.min(self.rawDataFile['XDAT'])) == 1.0).ravel()
		# take even non-same consecutive TR samples
		self.TRtimeIndices = np.arange(self.TRinfo.shape[0])[self.TRinfo[:-1]!=self.TRinfo[1:]][0::2]
		self.TRtimes = [datetime.strptime(str(t[0]),'%H:%M:%S.%f') for t in self.rawDataFile['time'][self.TRtimeIndices,0]]
		self.firstTR = {'index': self.TRtimeIndices[0], 'time':self.TRtimes[0]}
		if TR == None:
			self.TR = self.TRtimes[1]-self.TRtimes[0]
			self.TR = self.TR.seconds + self.TR.microseconds / 1000.0
		else:
			self.TR = TR
	
	def firstPass(self, nrVolumes, delay, TR = None, makeFigure = False, figureFileName = '' ):
		self.nrVolumes = nrVolumes
		self.delay = delay
		
		# analyze incoming TR signals
		self.trSignals(TR = TR)
		
		self.logger.debug('TR is %f, nr of TRs as per .eyd file is %d, nrVolumes and delay: %i, %i', self.TR, len(self.TRtimes), self.nrVolumes, self.delay)
		if len(self.TRtimes) != self.nrVolumes:
			self.logger.warning('data amount in .eyd file doesn not correspond to the amount of data in the .nii file... Aborting this eye file. \n%s', self.inputFileName)
			self.error = True
			return
		
		self.gazeData = self.rawDataFile['horz_gaze_coord'][self.firstTR['index']:self.firstTR['index'] + self.sampleFrequency * self.TR * self.nrVolumes ]
		self.gazeDataPerTR = self.gazeData.reshape(self.gazeData.shape[0]/(self.sampleFrequency * self.TR), self.sampleFrequency * self.TR).transpose()
		self.pupilRecogn = np.array(self.rawDataFile['pupil_recogn'][self.firstTR['index']: self.firstTR['index'] + self.TR * self.nrVolumes * self.sampleFrequency ], dtype = bool)
		self.pupilRecognPerTR = self.pupilRecogn.reshape(self.gazeData.shape[0]/(self.sampleFrequency * self.TR), self.sampleFrequency * self.TR).transpose()
		
		self.horVelocities = np.concatenate((self.gazeData[:-1]-self.gazeData[1:], [[0]]))
		self.horVelocitiesPerTR = self.horVelocities.reshape(self.horVelocities.shape[0]/(self.sampleFrequency * self.TR), self.sampleFrequency * self.TR).transpose()
		
		self.hVRunningSD = np.concatenate((np.ones((6)) * self.horVelocities.std(), [self.horVelocities[i:i+6].std() for i in range(self.horVelocities.shape[0]-6)]))
		self.hVRunningSDPerTR = self.hVRunningSD.reshape(self.hVRunningSD.shape[0]/(self.sampleFrequency * self.TR), self.sampleFrequency * self.TR).transpose()
		
		if makeFigure:
			if figureFileName == '':
				figureFileName = os.splitext(inputFileName)[0] + '.pdf'
				
			f = pl.figure(figsize = (10,5))
			sbp = f.add_subplot(2,1,1)
			for (g,p,i) in zip(self.gazeDataPerTR.T, self.pupilRecognPerTR.T, range(self.gazeDataPerTR.T.shape[0])):
				if i >= delay:
					pl.plot( np.arange(g.shape[0])[p], g[p], '-', c = 'k', alpha = 0.5, linewidth=0.5 )
			pl.axvspan(0.25 * self.sampleFrequency, 0.5 * self.sampleFrequency, facecolor=(1.0,0.0,0.0), alpha=0.25)
			pl.axvspan(1.25 * self.sampleFrequency, 1.5 * self.sampleFrequency, facecolor=(1.0,0.0,0.0), alpha=0.25)
			
			sbp.annotate(os.path.splitext(os.path.split(figureFileName)[-1])[0], xy=(.5, .5),  xycoords='axes fraction',
			                horizontalalignment='center', verticalalignment='center')
			
			gazeMean = [g[p].mean() for (g,p) in zip(self.gazeDataPerTR, self.pupilRecognPerTR)]
			pl.plot( np.arange(self.TR * self.sampleFrequency), gazeMean, 'o', c = 'k', alpha = 1.0, linewidth = 4.0 )
			
			sbp.axis([0, self.TR * self.sampleFrequency, 50, 210])
			
			sbp = f.add_subplot(2,1,2)
			for (v,p,sd) in zip(self.horVelocitiesPerTR.T, self.pupilRecognPerTR.T, self.hVRunningSDPerTR.T):
				if i >= delay:
					pl.plot( np.arange(v.shape[0])[p], v[p], '-', c = 'k', alpha = 0.5, linewidth=0.5 )
					pl.plot( np.arange(sd.shape[0])[p], sd[p], '+', c = 'b', alpha = 0.75, linewidth=0.5 )
					
			pl.axvspan(0.25 * self.sampleFrequency, 0.5 * self.sampleFrequency, facecolor=(1.0,0.0,0.0), alpha=0.25)
			pl.axvspan(1.25 * self.sampleFrequency, 1.5 * self.sampleFrequency, facecolor=(1.0,0.0,0.0), alpha=0.25)
			sbp.axis([0, self.TR * self.sampleFrequency, -50, 50])
			pl.savefig(figureFileName)
			
	

class EyelinkOperator( EyeOperator ):
	"""docstring for EyelinkOperator"""
	def __init__(self, inputObject, split = True, date_format = 'python_experiment', **kwargs):
		super(EyelinkOperator, self).__init__(inputObject = inputObject, **kwargs)
		
		if os.path.splitext(self.inputObject)[-1] == '.edf':
			self.type = 'eyelink'
			self.inputFileName = self.inputObject
			# in Kwargs there's a variable that we can set to 
			if not split:
				self.messageFile = os.path.splitext(self.inputFileName)[0] + '.msg'
				self.gazeFile = os.path.splitext(self.inputFileName)[0] + '.gaz'
			else:
				from CommandLineOperator import EDF2ASCOperator
				eac = EDF2ASCOperator(self.inputFileName)
				eac.configure()
				eac.execute()
				self.messageFile = eac.messageOutputFileName
				self.gazeFile = eac.gazeOutputFileName
				self.convertGazeData()
				
			if date_format == 'python_experiment':
				# recover time of experimental run from filename
				timeStamp = self.inputFileName.split('_')[-2:]
				[y, m, d] = [int(t) for t in timeStamp[0].split('-')]
				[h, mi, s] = [int(t) for t in timeStamp[1].split('.')[:-1]]
				self.timeStamp = datetime(y, m, d, h, mi, s)
				self.timeStamp_numpy = np.array([y, m, d, h, mi, s], dtype = np.int)
			else:
				timeStamp = self.inputFileName.split('_')[-1].split('.edf')[0]
				print timeStamp
				[d, m] = [int(t) for t in timeStamp.split('|')[1].split('-')]
				[h, mi] = [int(t) for t in timeStamp.split('|')[0].split('.')] 
				self.timeStamp = datetime(2010, m, d, h, mi, 11)
				self.timeStamp_numpy = np.array([2010, m, d, h, mi, 0], dtype = np.int)
				
				
		else:
			self.logger.warning('Input object is not an edf file')
	
	def convertGazeData(self):
		# take out non-readable string elements in order to load numpy array
		f = open(self.gazeFile)
		workingString = f.read()
		f.close()
		
		workingString = re.sub(re.compile('	\.\.\.'), '', workingString)
		
		of = open(self.gazeFile, 'w')
		of.write(workingString)
		of.close()
		
		gd = np.loadtxt(self.gazeFile)
		# make sure the amount of samples is even, so that later filtering is made easier. 
		# deleting the first data point of a session shouldn't matter at all..
		if bool(gd.shape[0] % 2):
			gd = gd[1:]
		np.save( self.gazeFile, gd.astype(np.float32) )
		os.rename(self.gazeFile+'.npy', self.gazeFile)
		
	
	def loadData(self, get_gaze_data = True):
		mF = open(self.messageFile, 'r')
		self.msgData = mF.read()
		mF.close()
		
		if get_gaze_data:
			self.gazeData = np.load(self.gazeFile)
			self.gazeData = self.gazeData.astype(np.float32)
	
	def findAll(self, check_answers = True):
		"""docstring for findAll"""
		if not hasattr(self, 'msgData'):
			self.loadData(get_gaze_data = False)
		
		self.findTrials()
		self.findTrialPhases()
		self.findParameters()
		self.findRecordingParameters()
		self.findKeyEvents()
		self.findELEvents()
		
		logString = 'data parameters:'
		if hasattr(self, 'gazeData'):
			logString += ' samples - ' + str(self.gazeData.shape)
		logString += ' sampleFrequency, eye - ' + str(self.sampleFrequency) + ' ' + self.eye
		logString += ' nrTrials, phases - ' + str(self.nrTrials) + ' ' + str(self.trialStarts.shape)
		self.logger.info(logString)
		
		if check_answers:
			tobedeleted = []
			for r in range(len(self.parameters)):
				if 'answer' not in self.parameters[r].keys():
					self.logger.info( 'no answer in run # ' + self.gazeFile + ' trial # ' + str(r) )
					tobedeleted.append(r + len(tobedeleted))
			for r in tobedeleted:
				self.parameters.pop(r)
				self.phaseStarts.pop(r)
				self.trials = np.delete(self.trials, r)
				
	
	def findOccurences(self, RE = ''):
		return re.findall(re.compile(RE), self.msgData)
	
	def findRecordingParameters(self, sampleRE = 'MSG\t[\d\.]+\t!MODE RECORD CR (\d+) \d+ \d+ (\S+)', screenRE = 'MSG\t[\d\.]+\tGAZE_COORDS (\d+.\d+) (\d+.\d+) (\d+.\d+) (\d+.\d+)', pixelRE = 'MSG\t[\d\.]+\tdegrees per pixel (\d*.\d*)', standardPixelsPerDegree = 84.6):
		self.parameterStrings = self.findOccurences(sampleRE)
		self.sampleFrequency = int(self.parameterStrings[0][0])
		self.eye = self.parameterStrings[0][1]
		
		self.screenStrings = self.findOccurences(screenRE)
		if len(self.screenStrings) > 0:
			self.screenCorners = np.array([float(s) for s in self.screenStrings[0]])
			self.screenSizePixels = [self.screenCorners[2]-self.screenCorners[0], self.screenCorners[3]-self.screenCorners[1]]
		else:
			# just put random stuff in there
			self.screenCorners = [0,0,1280,960]
			self.screenSizePixels = [self.screenCorners[2]-self.screenCorners[0], self.screenCorners[3]-self.screenCorners[1]]
		
		self.pixelStrings = self.findOccurences(pixelRE)
		if len(self.pixelStrings) > 0:
			self.pixelsPerDegree = float(self.pixelStrings[0])
		else:
			# standard is for the 74 cm screen distance on the 24 inch Sony that is running at 1280x960.
			self.pixelsPerDegree = standardPixelsPerDegree
	
	def findELEvents(self,
		saccRE = 'ESACC\t(\S+)[\s\t]+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+.?\d+)', 
		fixRE = 'EFIX\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?', 
		blinkRE = 'EBLINK\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d?.?\d*)?'):
		"""
		searches for the ends of Eyelink events, since they
		contain all the information about the occurrence of the event. Examples:
		ESACC	R	2347313	2347487	174	  621.8	  472.4	  662.0	  479.0	   0.99	 
		EFIX	R	2340362.0	2347312.0	6950	  650.0	  480.4	   5377
		EBLINK	R	2347352	2347423	71
		"""
		saccadeStrings = self.findOccurences(saccRE)
		fixStrings = self.findOccurences(fixRE)
		blinkStrings = self.findOccurences(blinkRE)
		
		self.saccades_from_MSG_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3]),'start_x':float(e[4]),'start_y':float(e[5]),'end_x':float(e[6]),'end_y':float(e[7]), 'peak_velocity':float(e[7])} for e in saccadeStrings]
		self.fixations_from_MSG_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3]),'x':float(e[4]),'y':float(e[5]),'pupil_size':float(e[6])} for e in fixStrings]
		self.blinks_from_MSG_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3])} for e in blinkStrings]
		
		self.saccadesTypeDictionary = np.dtype([(s , np.array(self.saccades_from_MSG_file[0][s]).dtype) for s in self.saccades_from_MSG_file[0].keys()])
		self.fixationsTypeDictionary = np.dtype([(s , np.array(self.fixations_from_MSG_file[0][s]).dtype) for s in self.fixations_from_MSG_file[0].keys()])
		if len(self.blinks_from_MSG_file) > 0:
			self.blinksTypeDictionary = np.dtype([(s , np.array(self.blinks_from_MSG_file[0][s]).dtype) for s in self.blinks_from_MSG_file[0].keys()])
	
	def findTrials(self, startRE = 'MSG\t([\d\.]+)\ttrial (\d+) started at (\d+.\d)', stopRE = 'MSG\t([\d\.]+)\ttrial (\d+) stopped at (\d+.\d)'):
		self.startTrialStrings = self.findOccurences(startRE)
		self.stopTrialStrings = self.findOccurences(stopRE)
		
		if np.unique(np.array(self.startTrialStrings, dtype = np.float64)[:,1]).shape[0] == np.array(self.startTrialStrings, dtype = np.float64).shape[0]:
			self.monotonic = True
			
		else:
			self.monotonic = False
			self.nrRunsInDataFile = np.array(self.startTrialStrings, dtype = np.float64).shape[0] / np.unique(np.array(self.startTrialStrings, dtype = np.float64)[:,1]).shape[0]
			self.logger.info('This edf file contains multiple runs. Analyzing ' + str(self.nrRunsInDataFile) + ' runs.')
		
		self.trialStarts = np.array([[float(s[0]), int(s[1]), float(s[2])] for s in self.startTrialStrings])
		self.trialEnds = np.array([[float(s[0]), int(s[1]), float(s[2])] for s in self.stopTrialStrings])
			
		self.nrTrials = len(self.stopTrialStrings)
		self.trials = np.hstack((self.trialStarts, self.trialEnds))
			
		self.trialTypeDictionary = [('trial_start_EL_timestamp', np.float64), ('trial_start_index',np.int32), ('trial_start_exp_timestamp',np.float64), ('trial_end_EL_timestamp',np.float64), ('trial_end_index',np.int32), ('trial_end_exp_timestamp',np.float64)]
	
	def findTrialPhases(self, RE = 'MSG\t([\d\.]+)\ttrial X phase (\d+) started at (\d+.\d)'):
		phaseStarts = []
		for i in range(self.nrTrials):
			thisRE = RE.replace(' X ', ' ' + str(i) + ' ')
			phaseStrings = self.findOccurences(thisRE)
			phaseStarts.append([[float(s[0]), int(s[1]), float(s[2])] for s in phaseStrings])
		if self.monotonic == False:
			nrPhases = len(phaseStarts[0])/self.nrRunsInDataFile
			newPhases = []
			for j in range(self.nrTrials / self.nrRunsInDataFile):
				for i in range(self.nrRunsInDataFile):
					newPhases.append( phaseStarts[j][i*nrPhases:(i+1)*nrPhases] )
			phaseStarts = newPhases
		self.phaseStarts = phaseStarts
		# sometimes there are not an equal amount of phasestarts in a run.
		self.nrPhaseStarts = np.array([len(ps) for ps in self.phaseStarts]).min()
		self.trialTypeDictionary.append(('trial_phase_timestamps', np.float64, (self.nrPhaseStarts, 3)))
		self.trialTypeDictionary = np.dtype(self.trialTypeDictionary)
	
	def findKeyEvents(self, RE = 'MSG\t([\d\.]+)\ttrial X event \<Event\((\d)-Key(\S*?) {\'scancode\': (\d+), \'key\': (\d+), \'unicode\': u\'(\S*?)\', \'mod\': (\d+)}\)\> at (\d+.\d)'):
		events = []
		for i in range(self.nrTrials):
			thisRE = RE.replace(' X ', ' ' + str(i) + ' ')
			eventStrings = self.findOccurences(thisRE)
			events.append([{'EL_timestamp':float(e[0]),'event_type':int(e[1]),'up_down':e[2],'scancode':int(e[3]),'key':int(e[4]),'unicode':e[5],'modifier':int(e[6]), 'presentation_time':float(e[7])} for e in eventStrings])
		self.events = events
		
	def findParameters(self, RE = 'MSG\t[\d\.]+\ttrial X parameter\t(\S*?) : ([-\d\.]*|[\w]*)'):
		parameters = []
		# if there are no duplicates in the edf file
		trialCounter = 0
		for i in range(self.nrTrials):
			thisRE = RE.replace(' X ', ' ' + str(i) + ' ')
			parameterStrings = self.findOccurences(thisRE)
			if len(parameterStrings) > 0:
				if self.monotonic == False:
					nrParameters = len(parameterStrings)/self.nrRunsInDataFile
					for j in range(self.nrRunsInDataFile):
						thisTrialParameters = dict([[s[0], float(s[1])] for s in parameterStrings[j*nrParameters:(j+1)*nrParameters]])
						thisTrialParameters.update({'trial_nr' : float(trialCounter), 'seen': 0.0})
						parameters.append(thisTrialParameters)
						trialCounter += 1
				else:
					# assuming all these parameters are numeric
					thisTrialParameters = dict([[s[0], float(s[1])] for s in parameterStrings])
					thisTrialParameters.update({'trial_nr' : float(i), 'seen': 0.0})
					parameters.append(thisTrialParameters)
		
		if len(parameters) > 0:		# there were parameters in the edf file
			self.parameters = parameters
		else:		# we have to take the parameters from the output_dict pickle file of the same name as the edf file. 
			self.logger.info('taking parameter data for ' + self.inputFileName + ' from its dictionary neigbor')
			bhO = NewBehaviorOperator(os.path.splitext(self.inputFileName)[0] + '_outputDict.pickle')
			self.parameters = bhO.parameters
			for i in range(len(self.parameters)):
				self.parameters[i].update({'trial_nr' : float(i)})
				if not self.parameters[i].has_key('answer'):
					self.parameters[i].update({'answer' : float(-10000)})
			
		# now create parameters and types for hdf5 file table of trial parameters
		if not self.parameters[0].has_key('answer'):
			self.parameters[0].update({'answer' : float(-10000)})
		if not self.parameters[0].has_key('confidence'):
			self.parameters[0].update({'confidence' : float(-10000)})
		
		self.parameterTypeDictionary = np.dtype([(k, np.float64) for k in self.parameters[0].keys()])
	
	def removeDrift(self, cutoffFrequency = 0.1, cleanup = True):
		"""
		Removes low frequency drift of frequency lower than cutoffFrequency from the eye position signals
		cleanup removes intermediate data formats
		"""
		if not hasattr(self, 'gazeData'):
			self.loadData(get_gaze_data = True)
			
		self.signalNrSamples = self.gazeData.shape[0]
		self.cutoffFrequency = cutoffFrequency
		
		self.representedFrequencies = np.fft.fftfreq(n = int(self.signalNrSamples), d = 1.0/self.sampleFrequency)[:floor(self.signalNrSamples/2.0)]
		self.f = np.ones(self.signalNrSamples)
		self.thres = max(np.arange(self.f.shape[0])[self.representedFrequencies < self.cutoffFrequency])
		# high-pass:
		self.f[:self.thres] = 0.0
		self.f[-self.thres:] = 0.0
		
		# fourier transform all data columns instead of the time column
		self.fourierData = sp.fftpack.fft(self.gazeData[:,1:], axis = 0)
		self.fourierFilteredData = (self.fourierData.T * self.f).T
		self.filteredGazeData = sp.fftpack.ifft(self.fourierFilteredData, axis = 0).astype(np.float64)
		if cleanup:
			del(self.fourierData, self.fourierFilteredData)
			
		self.logger.info('fourier drift correction of data at cutoff of ' + str(cutoffFrequency) + ' finished')
	
	def computeVelocities(self, smoothingFilterWidth = 0.002 ):
		"""
		calculates velocities by multiplying the fourier-transformed raw data and a derivative of gaussian.
		the width of this gaussian determines the extent of temporal smoothing inherent in the calculation.
		Presently works only for one-eye data only - will change this as binocular data comes available.
		"""
		if not hasattr(self, 'gazeData'):
			self.loadData(get_gaze_data = True)
		if not hasattr(self, 'fourierData'):
			self.fourierData = sp.fftpack.fft(self.gazeData[:,1:], axis = 0)
		
		times = np.linspace(-floor(self.gazeData.shape[0]/2) / self.sampleFrequency, floor(self.gazeData.shape[0]/2) / self.sampleFrequency, self.gazeData.shape[0] )
		# gaussian with zero mean scaled to degrees per second, fourier transformed.
		gauss_pdf = sp.stats.norm.pdf( times / smoothingFilterWidth )
		gauss_pdf_kernel = np.roll(gauss_pdf / gauss_pdf.sum(), times.shape[0]/2)
		gauss_pdf_kernel_fft = sp.fftpack.fft( gauss_pdf_kernel )
		
		# difference operator, fourier transformed.
		diff_kernel = np.zeros(times.shape[0])
		diff_kernel[times.shape[0]/2-1] = 1
		diff_kernel[times.shape[0]/2] = -1
		diff_kernel_fft = sp.fftpack.fft( np.roll(diff_kernel, times.shape[0]/2) )
		
		diff_smoothed_data_fft = self.fourierData.T * gauss_pdf_kernel_fft * diff_kernel_fft
		diff_data_fft = self.fourierData.T * diff_kernel_fft
		smoothed_data_fft = self.fourierData.T * gauss_pdf_kernel_fft
		
		self.fourierVelocityData = self.sampleFrequency * sp.fftpack.ifft(( diff_data_fft ).T, axis = 0).astype(np.float32) / self.pixelsPerDegree
		self.normedVelocityData = np.array([np.linalg.norm(xy[0:2]) for xy in self.fourierVelocityData]).reshape((self.fourierVelocityData.shape[0],1))
		
		self.velocityData = np.hstack((self.fourierVelocityData, self.normedVelocityData))
		self.logger.info('velocity calculation of data finished')
		
		self.fourierSmoothedVelocityData = self.sampleFrequency * sp.fftpack.ifft(( diff_smoothed_data_fft ).T, axis = 0).astype(np.float32) / self.pixelsPerDegree
		self.normedSmoothedVelocityData = np.array([np.linalg.norm(xy[0:2]) for xy in self.fourierSmoothedVelocityData]).reshape((self.fourierSmoothedVelocityData.shape[0],1))
		
		self.smoothedVelocityData = np.hstack((self.fourierSmoothedVelocityData, self.normedSmoothedVelocityData))
		self.smoothedGazeData = sp.fftpack.ifft(( smoothed_data_fft ).T, axis = 0).astype(np.float32) / self.pixelsPerDegree
		
		self.logger.info('fourier velocity calculation of data at smoothing width of ' + str(smoothingFilterWidth) + ' s finished')
	
	def processIntoTable(self, tableFile = '', name = 'bla', compute_velocities = True):
		"""
		Take all the existent data from this run's edf file and put it into a standard format hdf5 file using pytables.
		"""
		if tableFile == '':
			self.logger.error('cannot process data into no table')
			return
		
		self.tableFileName = tableFile
		self.runName = name
		if not os.path.isfile(self.tableFileName):
			self.logger.info('starting table file ' + self.tableFileName)
			h5file = openFile(self.tableFileName, mode = "w", title = "Eye file")
		else:
			self.logger.info('opening table file ' + self.tableFileName)
			h5file = openFile(self.tableFileName, mode = "a", title = "Eye file")
		try:
			h5file.getNode(where = '/', name=self.runName, classname='Group')
			self.logger.info('data file ' + self.inputFileName + ' already in ' + self.tableFileName)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + self.runName + ' to this file')
			thisRunGroup = h5file.createGroup("/", self.runName, 'Run ' + str(len(h5file.listNodes(where = '/', classname = 'Group'))) +' imported from ' + self.inputFileName)
			
			# create all the parameters, events and such if they haven't already been created.
			if not hasattr(self, 'parameters'):
				self.findAll()
				
			# create a table for the parameters of this run's trials
			thisRunParameterTable = h5file.createTable(thisRunGroup, 'trial_parameters', self.parameterTypeDictionary, 'Parameters for trials in run ' + self.inputFileName)
			# fill up the table
			trial = thisRunParameterTable.row
			for tr in self.parameters:
				for par in tr.keys():
					trial[par] = tr[par]
				trial.append()
			thisRunParameterTable.flush()
			
			# create a table for the saccades from the eyelink of this run's trials
			thisRunSaccadeTable = h5file.createTable(thisRunGroup, 'saccades_from_EL', self.saccadesTypeDictionary, 'Saccades for trials in run ' + self.inputFileName)
			# fill up the table
			sacc = thisRunSaccadeTable.row
			for tr in self.saccades_from_MSG_file:
				for par in tr.keys():
					sacc[par] = tr[par]
				sacc.append()
			thisRunSaccadeTable.flush()
			
			# create a table for the blinks from the eyelink of this run's trials
			if len(self.blinks_from_MSG_file) > 0:
				thisRunBlinksTable = h5file.createTable(thisRunGroup, 'blinks_from_EL', self.blinksTypeDictionary, 'Blinks for trials in run ' + self.inputFileName)
				# fill up the table
				blink = thisRunBlinksTable.row
				for tr in self.blinks_from_MSG_file:
					for par in tr.keys():
						blink[par] = tr[par]
					blink.append()
				thisRunBlinksTable.flush()
			
			# create a table for the fixations from the eyelink of this run's trials
			thisRunFixationsTable = h5file.createTable(thisRunGroup, 'fixations_from_EL', self.fixationsTypeDictionary, 'Fixations for trials in run ' + self.inputFileName)
			# fill up the table
			fix = thisRunFixationsTable.row
			for tr in self.fixations_from_MSG_file:
				for par in tr.keys():
					fix[par] = tr[par]
				fix.append()
			thisRunFixationsTable.flush()
			
			# create a table for the trial times of this run's trials
			thisRunTimeTable = h5file.createTable(thisRunGroup, 'trial_times', self.trialTypeDictionary, 'Timestamps for trials in run ' + self.inputFileName)
			trial = thisRunTimeTable.row
			for (i, tr) in zip(range(len(self.trials)), self.trials):
				trial['trial_start_EL_timestamp'] = tr[0]
				trial['trial_start_index'] = tr[1]
				trial['trial_start_exp_timestamp'] = tr[2]
				trial['trial_end_EL_timestamp'] = tr[3]
				trial['trial_end_index'] = tr[4]
				trial['trial_end_exp_timestamp'] = tr[5]
				trial['trial_phase_timestamps'] = np.array(self.phaseStarts[i][:self.nrPhaseStarts])
				trial.append()
			thisRunTimeTable.flush()
			
			# create eye arrays for the run's eye movement data
			if not hasattr(self, 'gazeData'):
				self.loadData()
			
			h5file.createArray(thisRunGroup, 'gaze_data', self.gazeData.astype(np.float32), 'Raw gaze data from ' + self.inputFileName)
			
			if not hasattr(self, 'velocityData') and compute_velocities:
				# make the velocities arrays if it hasn't been done yet. 
				self.computeVelocities()
			
			if compute_velocities:
				h5file.createArray(thisRunGroup, 'velocity_data', self.velocityData.astype(np.float32), 'Raw velocity data from ' + self.inputFileName)
				h5file.createArray(thisRunGroup, 'smoothed_gaze_data', self.smoothedGazeData.astype(np.float32), 'Smoothed gaze data from ' + self.inputFileName)
				h5file.createArray(thisRunGroup, 'smoothed_velocity_data', self.smoothedVelocityData.astype(np.float32), 'Smoothed velocity data from ' + self.inputFileName)
			
		h5file.close()
	
	def clean_data(self):
		if hasattr(self, 'velocityData'):	# this is a sign that velocity analysis was run
			del(self.velocityData)
			del(self.smoothedGazeData)
			del(self.smoothedVelocityData)
			del(self.normedVelocityData)
			del(self.fourierSmoothedVelocityData)
			del(self.normedSmoothedVelocityData)
	
	