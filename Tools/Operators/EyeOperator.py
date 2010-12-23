#!/usr/bin/env python
# encoding: utf-8
"""
EyeOperator.py

Created by Tomas Knapen on 2010-12-19.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import tempfile, logging
import pickle

import scipy as sp
import scipy.fftpack
import numpy as np
import matplotlib.pylab as pl
from math import *
from scipy.io import *

from nifti import *
from Operator import *

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
		

	def trSignals(self):
		self.TRinfo = (np.array(self.rawDataFile['XDAT']-np.min(self.rawDataFile['XDAT']))/(np.max(self.rawDataFile['XDAT'])-np.min(self.rawDataFile['XDAT'])) == 1.0).ravel()
		# take even non-same consecutive TR samples
		self.TRtimeIndices = np.arange(self.TRinfo.shape[0])[self.TRinfo[:-1,0]==self.TRinfo[1:,0]][::2]
		self.TRtimes = self.rawDataFile['time'][self.TRtimeIndices]
		self.firstTR = {'index': self.TRtimeIndices[0], 'time':self.TRTimes[0]}
		self.TR = self.TRtimes[1]-self.TRtimes[0]
		
		
	def firstPass(self, nrVolumes, TR, delay ):
		sampleFrequency = self.inputObject['freq'][0,0]
		# analyze incoming TR signals
		self.trSignals()
		
		self.gazeData = self.rawDataFile['horz_gaze_coord'][self.firstTRSample:self.firstTRSample + sampleFrequency * self.TR * self.nrVolumes ]
		self.gazeData = self.gazeData.reshape(self.gazeData.shape[0]/(sampleFrequency * self.TR), sampleFrequency * self.TR).transpose()
		self.blinks = self.rawDataFile['pupil_recogn'][self.firstTRSample: self.firstTRSample + self.TR * self.nrVolumes ]
		
		
class EyelinkOperator( EyeOperator ):
	"""docstring for EyelinkOperator"""
	def __init__(self, inputObject, **kwargs):
		super(EyelinkOperator, self).__init__(inputObject = inputObject, **kwargs)
		
		if os.path.splitext(self.inputObject)[-1] == '.edf':
			self.type = 'eyelink'
			self.inputFileName = self.inputObject
			# in Kwargs there's a variable that we can set to 
			if hasattr(self, is_split):
				self.messageFile = os.path.splitext(self.inputFileName)[0] + '.msg'
				self.gazeFile = os.path.splitext(self.inputFileName)[0] + '.gaz'
			else:
				eac = EDF2ASCOperator(self.inputFileName)
				eac.configure()
				eac.execute()
		else:
			self.logger.warning('Input object is not an edf file')
	
	def loadData(self):
		self.gazeData = np.loadtxt(self.gazeFile)
		mF = open(self.messagefile, 'r')
		self.msgData = mF.readlines()
		mF.close()
	
	def splitToTrials