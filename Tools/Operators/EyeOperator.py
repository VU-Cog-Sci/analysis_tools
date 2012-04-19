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
# import scipy.signal as signal
import numpy as np
import matplotlib.pylab as pl
from math import *
from scipy.io import *

from nifti import *
from Operator import *
from datetime import *

from tables import *

from BehaviorOperator import NewBehaviorOperator
from IPython import embed as shell

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
				self.messageFile = eac.messageOutputFileName
				self.gazeFile = eac.gazeOutputFileName
				if not os.path.isfile(eac.messageOutputFileName):
					eac.execute()
					self.convertGazeData()
				
			if date_format == 'python_experiment':
				# recover time of experimental run from filename
				timeStamp = self.inputFileName.split('_')[-2:]
				[y, m, d] = [int(t) for t in timeStamp[0].split('-')]
				[h, mi, s] = [int(t) for t in timeStamp[1].split('.')[:-1]]
				self.timeStamp = datetime(y, m, d, h, mi, s)
				self.timeStamp_numpy = np.array([y, m, d, h, mi, s], dtype = np.int)
			elif date_format == 'c_experiment':
				timeStamp = self.inputFileName.split('_')[-1].split('.edf')[0]
				[d, m] = [int(t) for t in timeStamp.split('|')[1].split('-')]
				[h, mi] = [int(t) for t in timeStamp.split('|')[0].split('.')] 
				self.timeStamp = datetime(2010, m, d, h, mi, 11)
				self.timeStamp_numpy = np.array([2010, m, d, h, mi, 0], dtype = np.int)
			else:
				self.timeStamp = datetime.now()
				self.timeStamp_numpy = np.array([2011, 0, 0, 0, 0, 0], dtype = np.int)
				
		elif os.path.splitext(self.inputObject)[-1] == '.hdf5':
			self.inputFileName = self.inputObject
			self.hdf5_filename = self.inputObject
				
		else:
			self.logger.warning('Input object is not an edf or hdf5 file')
	
	def convertGazeData(self):
		# take out non-readable string elements in order to load numpy array
		f = open(self.gazeFile)
		self.workingString = f.read()
		f.close()
		
		# optimize this so that it doesn't delete the periods in the float time, for example.
		self.workingString = re.sub(re.compile('\t*\.\.\.+'), '', self.workingString)
		os.system('rm -rf ' + self.gazeFile)
		
		of = open(self.gazeFile, 'w')
		of.write(self.workingString)
		of.close()
		
		gd = np.loadtxt(self.gazeFile)
		# make sure the amount of samples is even, so that later filtering is made easier. 
		# deleting the first data point of a session shouldn't matter at all..
		if bool(gd.shape[0] % 2):
			gd = gd[1:]
		np.save( self.gazeFile, gd.astype(np.float64) )
		os.rename(self.gazeFile+'.npy', self.gazeFile)
		
	
	def loadData(self, get_gaze_data = True):
		mF = open(self.messageFile, 'r')
		self.msgData = mF.read()
		mF.close()
		
		if get_gaze_data:
			self.gazeData = np.load(self.gazeFile)
			self.gazeData = self.gazeData.astype(np.float64)
	
	def findAll(self, check_answers = False):
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
		self.which_trials_actually_exist = np.array([int(i[1]) for i in self.startTrialStrings])
		# print self.trials.shape
			
		self.trialTypeDictionary = [('trial_start_EL_timestamp', np.float64), ('trial_start_index',np.int32), ('trial_start_exp_timestamp',np.float64), ('trial_end_EL_timestamp',np.float64), ('trial_end_index',np.int32), ('trial_end_exp_timestamp',np.float64)]
	
	def findTrialPhases(self, RE = 'MSG\t([\d\.]+)\ttrial X phase (\d+) started at (\d+.\d)'):
		phaseStarts = []
		for i in self.which_trials_actually_exist:
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
		self.nrPhaseStarts = np.array([len(ps) for ps in self.phaseStarts])
		self.trialTypeDictionary.append(('trial_phase_timestamps', np.float64, (self.nrPhaseStarts.max(), 3)))
		self.trialTypeDictionary = np.dtype(self.trialTypeDictionary)
		
#		print self.phaseStarts
	def findKeyEvents(self, RE = 'MSG\t([\d\.]+)\ttrial X event \<Event\((\d)-Key(\S*?) {\'scancode\': (\d+), \'key\': (\d+)(, \'unicode\': u\'\S*?\',|,) \'mod\': (\d+)}\)\> at (\d+.\d)'):
		events = []
		for i in self.which_trials_actually_exist:
			thisRE = RE.replace(' X ', ' ' + str(i) + ' ')
			eventStrings = self.findOccurences(thisRE)
			events.append([{'EL_timestamp':float(e[0]),'event_type':int(e[1]),'up_down':e[2],'scancode':int(e[3]),'key':int(e[4]),'modifier':int(e[6]), 'presentation_time':float(e[7])} for e in eventStrings])
		self.events = events
		#
		# add types to eventTypeDictionary that specify the relevant trial and time in trial for this event - per run.
		#
		self.eventTypeDictionary = np.dtype([('EL_timestamp', np.float64), ('event_type', np.float64), ('up_down', '|S25'), ('scancode', np.float64), ('key', np.float64), ('modifier', np.float64), ('presentation_time', np.float64)])
		
		# print 'self.eventTypeDictionary is ' + str(self.eventTypeDictionary) + '\n' +str(self.events[0])
		
	def findParameters(self, RE = 'MSG\t[\d\.]+\ttrial X parameter\t(\S*?) : ([-\d\.]*|[\w]*)', add_parameters = None):
		parameters = []
		# if there are no duplicates in the edf file
		trialCounter = 0
		for i in self.which_trials_actually_exist:
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
					thisTrialParameters.update({'trial_nr': float(i), 'seen': 0.0})
					parameters.append(thisTrialParameters)
					trialCounter += 1
		
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
		
		ptd = [(k, np.float64) for k in np.unique(np.concatenate([k.keys() for k in self.parameters]))]
		if add_parameters != None:
			for par in add_parameters:
				ptd.append(par, np.float64)
		self.parameterTypeDictionary = np.dtype(ptd)
		import pdb; pdb.set_trace()
		
	
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
		
		self.fourierVelocityData = self.sampleFrequency * sp.fftpack.ifft(( diff_data_fft ).T, axis = 0).astype(np.float64) / self.pixelsPerDegree
		self.normedVelocityData = np.array([np.linalg.norm(xy[0:2]) for xy in self.fourierVelocityData]).reshape((self.fourierVelocityData.shape[0],1))
		
		self.velocityData = np.hstack((self.fourierVelocityData, self.normedVelocityData))
		self.logger.info('velocity calculation of data finished')
		
		self.fourierSmoothedVelocityData = self.sampleFrequency * sp.fftpack.ifft(( diff_smoothed_data_fft ).T, axis = 0).astype(np.float64) / self.pixelsPerDegree
		self.normedSmoothedVelocityData = np.array([np.linalg.norm(xy[0:2]) for xy in self.fourierSmoothedVelocityData]).reshape((self.fourierSmoothedVelocityData.shape[0],1))
		
		self.smoothedVelocityData = np.hstack((self.fourierSmoothedVelocityData, self.normedSmoothedVelocityData))
		self.smoothedGazeData = sp.fftpack.ifft(( smoothed_data_fft ).T, axis = 0).astype(np.float64) / self.pixelsPerDegree
		
		self.logger.info('fourier velocity calculation of data at smoothing width of ' + str(smoothingFilterWidth) + ' s finished')
	
	def processIntoTable(self, hdf5_filename = '', name = 'bla', compute_velocities = False, check_answers = False):
		"""
		Take all the existent data from this run's edf file and put it into a standard format hdf5 file using pytables.
		"""
		if hdf5_filename == '':
			self.logger.error('cannot process data into no table')
			return
		
		self.hdf5_filename = hdf5_filename
		self.runName = name
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('starting table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "w", title = "Eye file")
		else:
			self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "a", title = "Eye file")
		try:
			thisRunGroup = h5file.getNode(where = '/', name=self.runName, classname='Group')
			self.logger.info('data file ' + self.inputFileName + ' already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + self.runName + ' to this file')
			thisRunGroup = h5file.createGroup("/", self.runName, 'Run ' + str(len(h5file.listNodes(where = '/', classname = 'Group'))) +' imported from ' + self.inputFileName)
			
			# create all the parameters, events and such if they haven't already been created.
			if not hasattr(self, 'parameters'):
				self.findAll(check_answers = check_answers)
				
			# create a table for the trial times of this run's trials
			thisRunTimeTable = h5file.createTable(thisRunGroup, 'trial_times', self.trialTypeDictionary, 'Timestamps for trials in run ' + self.inputFileName)
			trial = thisRunTimeTable.row
			for i in range(self.nrTrials):
				trial['trial_start_EL_timestamp'] = self.trials[i][0]
				trial['trial_start_index'] = self.trials[i][1]
				trial['trial_start_exp_timestamp'] = self.trials[i][2]
				trial['trial_end_EL_timestamp'] = self.trials[i][3]
				trial['trial_end_index'] = self.trials[i][4]
				trial['trial_end_exp_timestamp'] = self.trials[i][5]
				# check whether this session ended with a full trial, and only then append
				if len(self.phaseStarts[i]) > 0:
					if self.nrPhaseStarts[i] == np.max(self.nrPhaseStarts):
						trial['trial_phase_timestamps'] = np.array(self.phaseStarts[i])
						# print np.array(self.phaseStarts[i][:self.nrPhaseStarts[i]]), trial['trial_phase_timestamps'][:self.nrPhaseStarts[i]]
					# import pdb; pdb.set_trace()
					trial.append()
					
			thisRunTimeTable.flush()
			
			# create a table for the parameters of this run's trials
			thisRunParameterTable = h5file.createTable(thisRunGroup, 'trial_parameters', self.parameterTypeDictionary, 'Parameters for trials in run ' + self.inputFileName)
			# fill up the table
			trial = thisRunParameterTable.row
			for tr in self.parameters:
				for par in tr.keys():
					trial[par] = tr[par]
				trial.append()
			thisRunParameterTable.flush()
			
			# create a table for the events of this run's trials
			thisRunEventTable = h5file.createTable(thisRunGroup, 'events', self.eventTypeDictionary, 'Events for trials in run ' + self.inputFileName)
			# fill up the table
			trial = thisRunEventTable.row
			for tr in self.events:								# per trial
				if len(tr) > 0:
					for ev in tr:								# per event per trial
						for var in ev.keys():					# per variable in the event.
							trial[var] = ev[var]
						# add timing in trial and trial # here
						# shell()
						trial.append()
			thisRunEventTable.flush()
			
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
			
			# create eye arrays for the run's eye movement data
			if not hasattr(self, 'gazeData'):
				self.loadData()
			
			h5file.createArray(thisRunGroup, 'gaze_data', self.gazeData.astype(np.float64), 'Raw gaze data from ' + self.inputFileName)
			
			if not hasattr(self, 'velocityData') and compute_velocities:
				# make the velocities arrays if it hasn't been done yet. 
				self.computeVelocities()
			
			if compute_velocities:
				h5file.createArray(thisRunGroup, 'velocity_data', self.velocityData.astype(np.float64), 'Raw velocity data from ' + self.inputFileName)
				h5file.createArray(thisRunGroup, 'smoothed_gaze_data', self.smoothedGazeData.astype(np.float64), 'Smoothed gaze data from ' + self.inputFileName)
				h5file.createArray(thisRunGroup, 'smoothed_velocity_data', self.smoothedVelocityData.astype(np.float64), 'Smoothed velocity data from ' + self.inputFileName)
			
		h5file.close()
	
	def clean_data(self):
		if hasattr(self, 'velocityData'):	# this is a sign that velocity analysis was run
			del(self.velocityData)
			del(self.smoothedGazeData)
			del(self.smoothedVelocityData)
			del(self.normedVelocityData)
			del(self.fourierSmoothedVelocityData)
			del(self.normedSmoothedVelocityData)
	
	def import_parameters(self, run_name = 'run_'):
		parameter_data = []
		h5f = openFile(self.hdf5_filename, mode = "r" )
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if run_name in r._v_name:
				# try to take care of the problem that parameter composition of runs may change over time - we choose the common denominator for now.
				# perhaps later a more elegant solution is possible
				this_dtype = np.array(r.trial_parameters.read().dtype.names)
				if len(parameter_data) == 0:	# if the first run, we construct a dtype_array
					dtype_array = this_dtype
				else:	# common denominator by intersection
					dtype_array = np.intersect1d(dtype_array, this_dtype)
				parameter_data.append(np.array(r.trial_parameters.read()))
		parameter_data = [p[:][dtype_array] for p in parameter_data]
		self.timings = r.trial_times.read()
		self.events = r.events.read()
		self.parameter_data = np.concatenate(parameter_data)
		self.logger.info('imported parameter data from ' + str(self.parameter_data.shape[0]) + ' trials')
		h5f.close()
	
	def get_EL_samples_per_trial(self, run_name = 0, trial_ranges = [[0,-1]], trial_phase_range = [0,-1], data_type = 'smoothed_velocity'):
		h5f = openFile(self.hdf5_filename, mode = "r" )
		run = None
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if run_name == r._v_name:
				run = r
				break
		if run == None:
			self.logger.error('No run named ' + run_name + ' in this session\'s hdf5 file ' + self.hdf5_filename )
		self.timings = run.trial_times.read()
		gaze_timestamps = run.gaze_data.read()[:,0]

		# select data_type
		if data_type == 'smoothed_velocity':
			all_data_of_requested_type = run.smoothed_velocity_data.read()[:,-1]
		elif data_type == 'smoothed_velocity_x':
			all_data_of_requested_type = run.smoothed_velocity_data.read()[:,0]
		elif data_type == 'smoothed_velocity_y':
			all_data_of_requested_type = run.smoothed_velocity_data.read()[:,1]
		elif data_type == 'smoothed_velocity_xy':
			all_data_of_requested_type = run.smoothed_velocity_data.read()[:,[0,1]]
		elif data_type == 'velocity':
			all_data_of_requested_type = run.velocity_data.read()[:,-1]
		elif data_type == 'velocity_x':
			all_data_of_requested_type = run.velocity_data.read()[:,0]
		elif data_type == 'velocity_y':
			all_data_of_requested_type = run.velocity_data.read()[:,1]
		elif data_type == 'velocity_xy':
			all_data_of_requested_type = run.velocity_data.read()[:,[0,1]]
		elif data_type == 'gaze_xy':
			all_data_of_requested_type = run.gaze_data.read()[:,[1,2]]
		elif data_type == 'gaze_x':
			all_data_of_requested_type = run.gaze_data.read()[:,1]
		elif data_type == 'gaze_y':
			all_data_of_requested_type = run.gaze_data.read()[:,2]
		elif data_type == 'smoothed_gaze_xy':
			all_data_of_requested_type = run.smoothed_gaze_data.read()[:,[0,1]]
		elif data_type == 'smoothed_gaze_x':
			all_data_of_requested_type = run.smoothed_gaze_data.read()[:,0]
		elif data_type == 'smoothed_gaze_y':
			all_data_of_requested_type = run.smoothed_gaze_data.read()[:,1]
		elif data_type == 'pupil_size':
			all_data_of_requested_type = run.gaze_data.read()[:,3]
		
		# make sure we always take the last of the trials into account, too.
		for tr in trial_ranges:
			if tr[-1] == -1:
				tr[-1] = self.timings.shape[0]
		
		# run for loop for actual data
		export_data = []
		for (i, trial_range) in zip(range(len(trial_ranges)), trial_ranges):
			export_data.append([])
			for t in self.timings[trial_range[0]:trial_range[1]]:
				phase_timestamps = np.concatenate((np.array([t['trial_start_EL_timestamp']]), t['trial_phase_timestamps'][:,0], np.array([t['trial_end_EL_timestamp']])))
				which_samples = (gaze_timestamps >= phase_timestamps[trial_phase_range[0]]) * (gaze_timestamps <= phase_timestamps[trial_phase_range[1]])
				export_data[-1].append(np.vstack((gaze_timestamps[which_samples].T, all_data_of_requested_type[which_samples].T)).T)
		# clean-up
		h5f.close()
		return export_data
	
	def get_EL_events_per_trial(self, run_name = '', trial_ranges = [[0,-1]], trial_phase_range = [0,-1], data_type = 'saccades'):
		h5f = openFile(self.hdf5_filename, mode = "r" )
		run = None
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if run_name == r._v_name:
				run = r
				break
		if run == None:
			self.logger.error('No run named ' + run_name + ' in this session\'s hdf5 file ' + self.hdf5_filename )
		timings = run.trial_times.read()
		
		if data_type == 'saccades':
			table = run.saccades_from_EL
		elif data_type == 'fixations':
			table = run.fixations_from_EL
		elif data_type == 'blinks':
			table = run.blinks_from_EL
			
		# make sure we always take the last of the trials into account, too.
		for tr in trial_ranges:
			if tr[-1] == -1:
				tr[-1] = self.timings.shape[0]
		
		# run for loop for actual data
		export_data = []
		for (i, trial_range) in zip(range(len(trial_ranges)), trial_ranges):
			export_data.append([])
			for t in timings[trial_range[0]:trial_range[1]]:
				phase_timestamps = np.concatenate((np.array([t['trial_start_EL_timestamp']]), t['trial_phase_timestamps'][:,0], np.array([t['trial_end_EL_timestamp']])))
				where_statement = '(start_timestamp >= ' + str(phase_timestamps[trial_phase_range[0]]) + ') & (start_timestamp < ' + str(phase_timestamps[trial_phase_range[1]]) + ')' 
				export_data[-1].append(np.array([s[:] for s in table.where(where_statement) ], dtype = table.dtype))
		# import pdb; pdb.set_trace()
		h5f.close()
		return export_data
	
	def detect_saccade_from_data(self, xy_data = None, xy_velocity_data = None, l = 5, sample_times = None, pixels_per_degree = 26.365, plot = False):
		"""
		detect_saccade_from_data takes a sequence (2 x N) of xy gaze position or velocity data and uses the engbert & mergenthaler algorithm (PNAS 2006) to detect saccades.
		L determines the threshold - standard set at 5 median-based standard deviations from the median
		"""
		minimum_saccade_duration = 12 # in ms, as we assume the sampling to be
		
		if xy_velocity_data == None:
			vel_data = np.zeros(xydata.shape)
			vel_data[1:] = np.diff(xydata, axis = 0)
		else:
			vel_data = xy_velocity_data
			
		if sample_times == None:
			sample_times = np.arange(vel_data.shape[1])
			
		# median-based standard deviation
		med = np.median(vel_data, axis = 0)
		scaled_vel_data = vel_data/np.mean(np.sqrt(((vel_data - med)**2)), axis = 0)
		
		# when are we above the threshold, and when were the crossings
		over_threshold = (np.array([np.linalg.norm(s) for s in scaled_vel_data]) > l)
		# integers instead of bools preserve the sign of threshold transgression
		over_threshold_int = np.array(over_threshold, dtype = np.int16)
		
		# crossings come in pairs
		threshold_crossings_int = np.concatenate([[0], np.diff(over_threshold_int)])
		threshold_crossing_indices = np.arange(threshold_crossings_int.shape[0])[threshold_crossings_int != 0]
		
		# check for shorter saccades and gaps
		tci = []
		sacc_on = False
		for i in range(0, threshold_crossing_indices.shape[0]):
			# last transgression, is an offset of a saccade
			if i == threshold_crossing_indices.shape[0]-1:
				if threshold_crossings_int[threshold_crossing_indices[i]] == -1:
					tci.append(threshold_crossing_indices[i])
					sacc_on = False # be complete
				else: pass
			# first transgression, start of a saccade
			elif i == 0:
				if threshold_crossings_int[threshold_crossing_indices[i]] == 1:
					tci.append(threshold_crossing_indices[i])
					sacc_on = True
				else: pass
			elif threshold_crossings_int[threshold_crossing_indices[i]] == 1 and sacc_on == False: # start of a saccade that occurs without a prior saccade en route
				tci.append(threshold_crossing_indices[i])
				sacc_on = True
			# don't want to add any point that borders on a too-short interval
			elif (threshold_crossing_indices[i+1] - threshold_crossing_indices[i] <= minimum_saccade_duration):
				if threshold_crossings_int[threshold_crossing_indices[i]] == -1: # offset but the next is too short - disregard offset
					pass
				elif threshold_crossings_int[threshold_crossing_indices[i]] == 1: # onset but the next is too short - disregard offset if there is already a previous saccade going on
					if sacc_on: # there already is a saccade going on - no need to include this afterbirth
						pass
					else:	# this should have been caught earlier
						tci.append(threshold_crossing_indices[i])
						sacc_on = True
			elif (threshold_crossing_indices[i] - threshold_crossing_indices[i-1] <= minimum_saccade_duration):
				if threshold_crossings_int[threshold_crossing_indices[i]] == -1: # offset but the previous one is too short - use offset offset
					if sacc_on:
						tci.append(threshold_crossing_indices[i])
						sacc_on = False
			# but add anything else
			else:
				tci.append(threshold_crossing_indices[i])
				if threshold_crossings_int[threshold_crossing_indices[i]] == 1:
					sacc_on = True
				else:
					sacc_on = False
					
		threshold_crossing_indices = np.array(tci)
		
		if threshold_crossing_indices.shape[0] > 0:
			saccades = np.zeros( (floor(sample_times[threshold_crossing_indices].shape[0]/2.0)) , dtype = self.saccade_dtype )
			
			# construct saccades:
			for i in range(0,sample_times[threshold_crossing_indices].shape[0]-1,2):
				j = i/2
				saccades[j]['start_time'] = sample_times[threshold_crossing_indices[i]] - sample_times[0]
				saccades[j]['end_time'] = sample_times[threshold_crossing_indices[i+1]] - sample_times[0]
				saccades[j]['start_point'][:] = xy_data[threshold_crossing_indices[i],:]
				saccades[j]['end_point'][:] = xy_data[threshold_crossing_indices[i+1],:]
				saccades[j]['duration'] = saccades[j]['end_time'] - saccades[j]['start_time']
				saccades[j]['vector'] = saccades[j]['end_point'] - saccades[j]['start_point']
				saccades[j]['amplitude'] = np.linalg.norm(saccades[j]['vector'])
				saccades[j]['direction'] = math.atan(saccades[j]['vector'][0] / (saccades[j]['vector'][1] + 0.00001))
				saccades[j]['peak_velocity'] = vel_data[threshold_crossing_indices[i]:threshold_crossing_indices[i+1]].max()
		else: saccades = np.array([])
		
		if plot:
			fig = pl.figure(figsize = (8,3))
#			pl.plot(sample_times[:vel_data[0].shape[0]], vel_data[0], 'r')
#			pl.plot(sample_times[:vel_data[0].shape[0]], vel_data[1], 'c')
			pl.plot(sample_times[:scaled_vel_data.shape[0]], np.abs(scaled_vel_data), 'k', alpha = 0.5)
			pl.plot(sample_times[:scaled_vel_data.shape[0]], np.array([np.linalg.norm(s) for s in scaled_vel_data]), 'b')
			if saccades.shape[0] > 0:
				pl.scatter(sample_times[threshold_crossing_indices], np.ones((sample_times[threshold_crossing_indices].shape[0]))* 10, s = 25, color = 'k')
			pl.ylim([-20,20])
			
		return saccades
		
	
	
