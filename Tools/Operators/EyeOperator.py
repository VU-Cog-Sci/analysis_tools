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
from datetime import *

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
		
		
	def firstPass(self, nrVolumes, delay, TR = None, makeFigure = False ):
		self.nrVolumes = nrVolumes
		self.delay = delay
		
		# analyze incoming TR signals
		self.trSignals(TR = TR)
		
		self.logger.debug('TR is %f, nr of TRs is %d, nrVolumes and delay: %i, %i', self.TR, len(self.TRtimes), self.nrVolumes, self.delay)
		
		self.gazeData = self.rawDataFile['horz_gaze_coord'][self.firstTR['index']:self.firstTR['index'] + self.sampleFrequency * self.TR * self.nrVolumes ]
		self.gazeDataPerTR = self.gazeData.reshape(self.gazeData.shape[0]/(self.sampleFrequency * self.TR), self.sampleFrequency * self.TR).transpose()
		self.pupilRecogn = np.array(self.rawDataFile['pupil_recogn'][self.firstTR['index']: self.firstTR['index'] + self.TR * self.nrVolumes * self.sampleFrequency ], dtype = bool)
		self.pupilRecognPerTR = self.pupilRecogn.reshape(self.gazeData.shape[0]/(self.sampleFrequency * self.TR), self.sampleFrequency * self.TR).transpose()
		
		if makeFigure:
			f = pl.figure(figsize = (10,5))
			sbp = f.add_subplot(1,1,1)
			for (g,p,i) in zip(self.gazeDataPerTR.T, self.pupilRecognPerTR.T, range(self.gazeDataPerTR.T.shape[0])):
				if i >= delay:
					nrStimDesignatorElements = 30
					desSign = '|'
					pl.plot( np.arange(g.shape[0])[p], g[p], c = 'k', alpha = 1.0, linewidth=0.15 )
					pl.plot([0.25 * self.sampleFrequency for l in range(nrStimDesignatorElements)],np.linspace(0,250,nrStimDesignatorElements), desSign, c = 'r')
					pl.plot([0.5 * self.sampleFrequency for l in range(nrStimDesignatorElements)],np.linspace(0,250,nrStimDesignatorElements), desSign, c = 'r') 
					pl.plot([1.25 * self.sampleFrequency for l in range(nrStimDesignatorElements)],np.linspace(0,250,nrStimDesignatorElements), desSign, c = 'r') 
					pl.plot([1.5 * self.sampleFrequency for l in range(nrStimDesignatorElements)],np.linspace(0,250,nrStimDesignatorElements), desSign, c = 'r') 
			sbp.axis([0, self.TR * self.sampleFrequency, 0, 250])
		
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
	
