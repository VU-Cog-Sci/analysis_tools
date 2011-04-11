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
	def __init__(self, inputObject, split = True, **kwargs):
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
		np.save( self.gazeFile, gd )
		os.rename(self.gazeFile+'.npy', self.gazeFile)
		
	
	def loadData(self, get_gaze_data = True):
		mF = open(self.messageFile, 'r')
		self.msgData = mF.read()
		mF.close()
		
		if get_gaze_data:
			self.gazeData = np.load(self.gazeFile)
		else:
			self.gazeData = None
		
	def findAll(self):
		"""docstring for findAll"""
		self.findTrials()
		self.findTrialPhases()
		self.findParameters()
		self.findRecordingParameters()
		
		logString = 'data parameters:'
		if self.gazeData != None:
			logString += ' samples - ' + str(self.gazeData.shape)
		logString += ' sampleFrequency, eye - ' + str(self.sampleFrequency) + ' ' + self.eye
		logString += ' nrTrials, phases - ' + str(self.nrTrials) + ' ' + str(self.trialStarts.shape)
		self.logger.info(logString)
			
	
	def findOccurences(self, RE = ''):
		return re.findall(re.compile(RE), self.msgData)
	
	def findRecordingParameters(self, sampleRE = 'MSG\t[\d\.]+\t!MODE RECORD CR (\d+) \d+ \d+ (\S+)', screenRE = 'MSG\t[\d\.]+\tGAZE_COORDS (\d+.\d+) (\d+.\d+) (\d+.\d+) (\d+.\d+)', pixelRE = 'MSG\t[\d\.]+\tdegrees per pixel (\d*.\d*)', standardPixelsPerDegree = 84.6):
		self.parameterStrings = self.findOccurences(sampleRE)
		self.sampleFrequency = int(self.parameterStrings[0][0])
		self.eye = self.parameterStrings[0][1]
		
		self.screenStrings = self.findOccurences(screenRE)
		self.screenCorners = np.array([float(s) for s in self.screenStrings[0]])
		self.screenSizePixels = [self.screenCorners[2]-self.screenCorners[0], self.screenCorners[3]-self.screenCorners[1]]
		
		self.pixelStrings = self.findOccurences(pixelRE)
		if len(self.pixelStrings) > 0:
			self.pixelsPerDegree = float(self.pixelStrings[0])
		else:
			# standard is for the 74 cm screen distance on the 24 inch Sony that is running at 1280x960.
			self.pixelsPerDegree = standardPixelsPerDegree
	
	def findTrials(self, startRE = 'MSG\t([\d\.]+)\ttrial (\d+) started at (\d+.\d)', stopRE = 'MSG\t([\d\.]+)\ttrial (\d+) stopped at (\d+.\d)'):
		
		self.startTrialStrings = self.findOccurences(startRE)
		self.stopTrialStrings = self.findOccurences(stopRE)
		
		self.nrTrials = len(self.stopTrialStrings)
		self.trialStarts = np.array([[float(s[0]), int(s[1]), float(s[2])] for s in self.startTrialStrings])
		self.trialEnds = np.array([[float(s[0]), int(s[1]), float(s[2])] for s in self.stopTrialStrings])
	
	def findTrialPhases(self, RE = 'MSG\t([\d\.]+)\ttrial X phase (\d+) started at (\d+.\d)'):
		phaseStarts = []
		for i in range(self.nrTrials):
			thisRE = RE.replace(' X ', ' ' + str(i) + ' ')
			phaseStrings = self.findOccurences(thisRE)
			phaseStarts.append([[float(s[0]), int(s[1]), float(s[2])] for s in phaseStrings])
		self.phaseStarts = phaseStarts
		# self.phaseStarts = np.array(phaseStarts)
	
	def findParameters(self, RE = 'MSG\t[\d\.]+\ttrial X parameter\t(\S*?) : ([-\d\.]*|[\w]*)'):
		parameters = []
		# if there are no duplicates in the edf file
		if np.unique(self.trialStarts[:,1]).shape[0] == len(self.startTrialStrings):
			for i in range(self.nrTrials):
				thisRE = RE.replace(' X ', ' ' + str(i) + ' ')
				parameterStrings = self.findOccurences(thisRE)
				# assuming all these parameters are numeric
				parameters.append(dict([[s[0], float(s[1])] for s in parameterStrings]))
		else:	# there are duplicates in the edf file - take care of this by using the stop times.
			for stop_time in self.stopTrialStrings:
				thisRE = RE.replace(' X ', ' ' + stop_time[0] + ' ')
				thisRE = thisRE.replace('\t[\d\.]+', stop_time[1])
				parameterStrings = self.findOccurences(thisRE)
				# assuming all these parameters are numeric
				parameters.append(dict([[s[0], float(s[1])] for s in parameterStrings]))
		self.parameters = parameters	
			
	def removeDrift(self, cutoffFrequency = 0.1, cleanup = True):
		"""
		Removes low frequency drift of frequency lower than cutoffFrequency from the eye position signals
		cleanup removes intermediate data formats
		"""
		if self.gazeData == None:
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
#		self.filteredData = self.backFourierFilteredData.reshape(self.inputObject.data.shape).astype(np.float32)
		if cleanup:
			del(self.fourierData, self.fourierFilteredData)
			
		self.logger.info('fourier drift correction of data at cutoff of ' + str(cutoffFrequency) + ' finished')
	
	def computeVelocities(self, smoothingFilterWidth = 0.0025 ):
		"""
		calculates velocities by multiplying the fourier-transformed raw data and a derivative of gaussian.
		the width of this gaussian determines the extent of temporal smoothing inherent in the calculation
		"""
		if self.gazeData == None:
			self.loadData(get_gaze_data = True)
		if not hasattr(self, 'fourierData'):
			self.fourierData = sp.fftpack.fft(self.gazeData[:,1:], axis = 0)
		
		times = np.linspace(-floor(self.gazeData.shape[0]/2) / self.sampleFrequency, floor(self.gazeData.shape[0]/2) / self.sampleFrequency, self.gazeData.shape[0] )
		# derivative of gaussian with zero mean scaled to degrees per second, fourier transformed.
		dergausspdf = derivative_normal_pdf( mu = 0, sigma = smoothingFilterWidth, x = times)
		c = np.sum(dergausspdf * times)
		dergausspdf = np.roll(dergausspdf, times.shape[0]/2) * c
		
		print np.max(dergausspdf), np.abs(dergausspdf).sum(), c
		diffKernelFFT = sp.fftpack.fft( dergausspdf )
		
		self.fourierVelocityData = sqrt(times.shape[0]) * self.sampleFrequency * sp.fftpack.ifft((self.fourierData.T * diffKernelFFT).T, axis = 0).astype(np.float64) / self.pixelsPerDegree
		self.velocityData = self.sampleFrequency * np.diff(self.gazeData[:,1:], axis = 0) / self.pixelsPerDegree
		self.logger.info('fourier velocity calculation of data at smoothing width of ' + str(smoothingFilterWidth) + ' finished')
		
		
