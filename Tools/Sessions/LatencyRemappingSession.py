#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import * 
from RetinotopicMappingSession import *
from ..circularTools import *
import matplotlib.cm as cm
from pylab import *

class LatencyRemappingSession(Session):
	def b