#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Tomas Knapen on 2010-09-15.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, datetime
from subprocess import *
#from volumesAndSurfaces import *

from Tools.Sessions import *
from Operators.BehaviorOperator import *

class Run(object):
	def __init__(self, **kwargs ): #ID, condition, dataType, 
		"""
		run takes an ID, condition, dataType, rawDataFilePath
		"""
		# integer that will tell the run what number it is in the session
		self.indexInSession = None
		
		self.behaviorFile = None
		self.eyeLinkFile = None
		
		self.trialList = []
		
		# datetime of this run is the creation time of the raw data file
		for k,v in kwargs.items():
			setattr(self, k, v)
			
		if not hasattr(self, 'condition'):
			self.condition = ''
		
		if hasattr(self, 'rawDataFilePath'):
			if os.path.isfile(self.rawDataFilePath)			:
				self.dateTime = os.path.getctime(self.rawDataFilePath)
			else:
				print 'rawDataFilePath %s is not file.' % self.rawDataFilePath
		elif hasattr(self, 'behaviorFile'):
			#			self.dateTime = os.path.getctime(self.behaviorFile)
			self.dateTime = datetime.date.today()
		elif hasattr(self, 'eyeFile'):
			self.dateTime = os.path.getctime(self.eyeFile)
	
	def addTrial(self, trial):
		"""docstring for addTrial"""
		trial.indexInRun = trialList.len()
		self.trialList.append(trial)
	
class RivalryLearningRun(Run):
	def behavior(self):
		"""docstring for behavior"""
		# First we'll have to set up the basic data extraction and the like
		# self.behaviorFile is a wildcard, the operator knows what to do with it.
		if self.condition == 'rivalry':
			self.bO = RivalryTrackingBehaviorOperator(self.behaviorFile)
			self.bO.joinButtonDownAndUps()
		elif self.condition == 'disparity':
			self.bO = DisparityLocalizerBehaviorOperator(self.behaviorFile)
			self.bO.separateConditions()
			self.bO.analyzePsychophysics()
			
class RivalryReplayRun(Run):
	def behavior(self):
		self.bO = RivalryReplayBehaviorOperator(self.behaviorFile)
