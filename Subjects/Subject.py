#!/usr/bin/env python
# encoding: utf-8
"""
subject.py

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

class Subject(object):
	def __init__(self, initials, firstName, birthdate, standardFSID, labelFolderOfPreference ):
		self.initials = initials
		self.firstName = firstName
		self.birthdate = birthdate
		self.standardFSID = standardFSID
		self.labelFolderOfPreference = labelFolderOfPreference
