#!/usr/bin/env python
# encoding: utf-8
"""
project.py

Created by Tomas HJ Knapen on 2009-12-08.
Copyright (c) 2009 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from nifti import *
from math import *
from itertools import *

class Project(object):
	"""docstring for project"""
	def __init__(self, projectName, subject, base_dir = '', **kwargs):
		self.projectName = projectName
		self.subject = subject
		self.base_dir = base_dir
		for k,v in kwargs.items():
			setattr(self, k, v)
			
		