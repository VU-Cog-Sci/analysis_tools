#!/usr/bin/env python
# encoding: utf-8
"""
PhysioOperator.py

Created by Tomas Knapen on 2010-12-19.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

# the following I consider standard.
import os, sys, subprocess, re
import pickle

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from math import *

from Operator import Operator

from IPython import embed as shell

class TemplateOperator( Operator ):
	"""PhysioOperator is an operator that takes a log file from the scanphyslog system and preprocesses these data.
	this includes temporal filtering, kernel convolution and regressor creation.
	"""
	def __init__(self, inputObject, **kwargs):
		"""
		PhysioOperator operator takes a filename for a log file
		"""
		super(TemplateOperator, self).__init__(inputObject = inputObject, **kwargs)
		if self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
		self.logger.info('started with ' +os.path.split(self.inputFileName)[-1])
		
	def method_placeholder(self, argument1, argument2 = None):
		pass