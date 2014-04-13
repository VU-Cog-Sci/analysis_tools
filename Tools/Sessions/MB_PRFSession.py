#!/usr/bin/env python
# encoding: utf-8
"""
MB_PRFSession.py

Created by Tomas HJ Knapen on 2014-04-13.
Copyright (c) 2014 TK. All rights reserved.
"""

import os, sys, pickle, math
from subprocess import *
import datetime

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
from nifti import *

import pp
import logging, logging.handlers, logging.config

from ..log import *
from ..Run import *
from ..Subjects.Subject import *
from ..Operators.Operator import *
from ..Operators.CommandLineOperator import *
from ..Operators.ImageOperator import *
from ..Operators.BehaviorOperator import *
# from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from IPython import embed as shell


class MB_PRFSession(PopulationReceptiveFieldMappingSession):
	"""
	Class for population receptive field mapping sessions analysis.
	"""
	def transplant_headers_to_MB_niftis(self):
		"""docstring for transplant_headers_to_MB_niftis"""
		MB_conditions = [c for c in self.conditionDict.keys() if 'MB' in c]
		first_non_MB_condition = [c for c in self.conditionDict.keys() if 'MB' not in c][0]
		standard_file = nifti.NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[first_non_MB_condition][0]], postFix = ['mcf']) ).header
		for condition in MB_conditions:
			for r in [self.runList[i] for i in self.conditionDict[condition]]:
				this_MB_file = NiftiImage(nifti.NiftiImage(self.runFile(stage = 'processed/mri', run = r) ))
				os.system('cp %s %s'%(self.runFile(stage = 'processed/mri', run = r), self.runFile(stage = 'processed/mri', run = r, postFix = ['noheader']) ))
				this_MB_file.header = standard_file.header
				this_MB_file.rtime = standard_file.rtime / r.MB_factor
				this_MB_file.save_to_file(self.runFile(stage = 'processed/mri', run = r))
		
		