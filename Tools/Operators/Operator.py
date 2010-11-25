#!/usr/bin/env python
# encoding: utf-8
"""
Operator.py

Created by Tomas Knapen on 2010-09-17.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import tempfile, logging

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from nifti import *
from ..log import *

# in os.path, extensions contain leading periods
standardMRIExtension = '.nii.gz'

class Operator( object ):
	def __init__(self, inputObject, **kwargs):
		self.inputObject = inputObject
		for k,v in kwargs.items():
			setattr(self, k, v)
			
		# setup logging for this operator.
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(logging.DEBUG)
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		
		if self.inputObject.__class__.__name__ == 'NiftiImage':
			self.inputFileName = self.inputObject.filename
			self.logger.info(self.__repr__() + ' initialized with ' + os.path.split(self.inputFileName)[-1])
		elif self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
			self.logger.info(self.__repr__() + ' initialized with file ' + os.path.split(self.inputFileName)[-1])
			if not os.path.isfile(self.inputFileName):
				self.logger.warning('inputFileName is not a file at initialization')
		elif self.inputObject.__class__.__name__ == 'list':
			# if there's a list of input objects, this 
			self.inputList = self.inputObject
			self.logger.info(self.__repr__() + ' initialized with list of files')
	
	def configure(self):
		"""
		placeholder for configure
		to be filled in by subclasses
		"""
		pass

	def execute(self):
		"""
		placeholder for execute
		to be filled in by subclasses
		"""


# leave the mp part for after understanding how to use processing queues in pp.
class MultiOperator( Operator ):
	def __init__(self):
		"""docstring for __init__"""
		pass
		
	def prepare(self):
		"""docstring for prepare"""
		pass
		
	def runAll(self):
		pass
		
