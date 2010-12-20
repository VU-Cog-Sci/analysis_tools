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
		# convert inputObject to NiftiImage while generating inputFileName from it, if necessary.
		if self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
			dataFile = loadmat(self.inputObject)['dataEYD'][0,0]
			self.inputObject = dataFile
		self.logger.info('started with ' +os.path.split(self.inputFileName)[-1])
		
	def trSignals(self):
		self.TRinfo = (np.array(self.inputObject['XDAT']-np.min(self.inputObject['XDAT']))/(np.max(self.inputObject['XDAT'])-np.min(self.inputObject['XDAT'])) == 1.0).ravel()
		# take even non-same consecutive TR samples
		self.TRtimeIndices = np.arange(self.TRinfo.shape[0])[self.TRinfo[:-1,0]==self.TRinfo[1:,0]][::2]
		self.TRtimes = self.inputObject['time'][self.TRtimeIndices]
		self.firstTR = {'index': self.TRtimeIndices[0], 'time':self.TRTimes[0]}
		self.TR = self.TRtimes[1]-self.TRtimes[0]
		
		
	def firstPass(self, nrVolumes, TR, delay ):
		sampleFrequency = self.inputObject['freq'][0,0]
		# analyze incoming TR signals
		self.trSignals()
		
		self.gazeData = self.inputObject['horz_gaze_coord'][self.firstTRSample:self.firstTRSample + sampleFrequency * self.TR * self.nrVolumes ]
		self.gazeData = self.gazeData.reshape(self.gazeData.shape[0]/(sampleFrequency * self.TR), sampleFrequency * self.TR).transpose()
		self.blinks = self.inputObject['pupil_recogn'][self.firstTRSample: self.firstTRSample + self.TR * self.nrVolumes ]
		
		
