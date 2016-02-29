#!/usr/bin/env python
# encoding: utf-8
"""
BehaviorOperator.py

Created by Tomas Knapen on 2010-11-06.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import tempfile, logging, pickle

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from nifti import *
from Operator import *
from ..log import *
import re
from IPython import embed as shell


def removeRepetitions(array, position = 0):
	a = array
	return a[np.concatenate(([True], a[:-1,position] != a[1:,position]))]


class BehaviorOperator(Operator):
	def __init__(self, inputObject, **kwargs):
		"""
		BehaviorOperator will receive a filename/wildcard combo as inputobject. 
		"""
		super(BehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		
		if self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
			self.logger.info(self.__repr__() + ' initialized with file ' + os.path.split(self.inputFileName)[-1])
			if not os.path.isfile(self.inputFileName):
				self.logger.warning('inputFileName is not a file at initialization')
			
		# first thing is to change the wildcard to an actual filename that contains the date and timestamp of the run
		fileNameList = subprocess.Popen('ls ' + self.inputFileName, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		if len(fileNameList) > 1:
			self.logger.debug('more than one behavioral file (%s) corresponds to the wildcard %s suggested. Will be using the last in the list ', str(fileNameList), self.inputFileName)
		self.inputFileName = fileNameList[-1]
	


			
