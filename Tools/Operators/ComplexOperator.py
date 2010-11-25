#!/usr/bin/env python
# encoding: utf-8
"""
ComplexOperator.py

Created by Tomas Knapen on 2010-10-11.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import tempfile, logging

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from nifti import *
from Operator import *
from CommandLineOperator import *
from ImageOperator import *
from ..log import *

class ComplexOperator( Operator ):
	"""docstring for ComplexOperator"""
	def __init__(self, inputObject, **kwargs):
		super(ComplexOperator, self).__init__(inputObject, **kwargs)


		
		
		