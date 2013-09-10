#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Sessions import * 
from pylab import *
from IPython import embed as shell


class TemplateSession(Session):
	"""
	Template Class for fMRI sessions analysis.
	"""
	def __init__(self, ID, date, project, subject, parallelize = True, loggingLevel = logging.DEBUG):
		self.session_label = session_label
		super(TemplateSession, self).__init__(ID, date, project, subject, parallelize = parallelize, loggingLevel = loggingLevel)
	
		
	
	def method_placeholder(self, argument1, argument2 = None):
		pass
		
		
		



	
