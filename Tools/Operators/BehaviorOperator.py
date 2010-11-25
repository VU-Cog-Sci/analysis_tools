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
from pypsignifit import *
from Operator import *
from ..log import *

def removeRepetitions(array, position = 0):
	a = array
	return a[np.concatenate(([True], a[:-1,position] != a[1:,position]))]


class BehaviorOperator(Operator):
	def __init__(self, inputObject, **kwargs):
		"""
		BehaviorOperator will receive a filenam/wildcard combo as inputobject. 
		"""
		super(BehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		
		# first thing is to change the wildcard to an actual filename that contains the date and timestamp of the run
		fileNameList = subprocess.Popen('ls ' + self.inputFileName, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		if len(fileNameList) > 1:
			self.logger.debug('more than one behavioral file (%s) corresponds to the wildcard %s suggested. Will be using the last in the list ', str(fileNameList), self.inputFileName)
		self.inputFileName = fileNameList[-1]
	

class RivalryLearningBehaviorOperator(BehaviorOperator):
	def openData(self):
		"""
		Opens data file in pickle format, containing events and stimulus settings
		"""
		file = open(self.inputFileName, 'r')
		self.pickledData = pickle.load(file)
		
		self.parameters = self.pickledData[0]
		self.rawEventData = self.pickledData[1]
		if len(self.pickledData) > 2:
			self.rawScreenData = self.pickledData[2]
		else: self.rawScreenData = None
		
	def separateEventsFromData(self):
		# do this removeRepetitions again later on
		self.nonDuplicateEventData = removeRepetitions(self.rawEventData)
		# TR key up events are -116. no need to keep these as they are not informative
		self.nonDuplicateEventData = self.nonDuplicateEventData[self.nonDuplicateEventData[:,0] != -116.0]
		
		#convert times to scanner times
		firstTRDelay = self.parameters['presentation_start_time'] - self.parameters['start_time']
		self.nonDuplicateEventData = self.nonDuplicateEventData - np.array([0,firstTRDelay])
		
		# separate button events and TR events and times.
		self.repetitionTimeEvents = removeRepetitions(self.nonDuplicateEventData[self.nonDuplicateEventData[:,0] == 0.0])
		self.buttonEvents = removeRepetitions(self.nonDuplicateEventData[self.nonDuplicateEventData[:,0] != 0.0])
	

class DisparityLocalizerBehaviorOperator(RivalryLearningBehaviorOperator):
	def __init__(self, inputObject, **kwargs):
		super(DisparityLocalizerBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.openData()
		self.separateEventsFromData()
		
	def separateConditions(self):
		"""
		Separate conditions and timings in the m-seq design. 
		Correlate responses to stimulus presentations
		"""
		# time coded after first TR, as in event data -> self.buttonEvents
		self.disparityEvents = removeRepetitions(self.rawScreenData[:,[-2, 0]])
		self.disparities = np.unique(self.disparityEvents[:,0])
		
		# button down events 
		self.buttonDownEvents = self.buttonEvents[self.buttonEvents[:,0] > 0]
		self.inputEventTimes = [self.disparityEvents[:,0] == d for d in self.disparities]	# don't make an np.array of this 'cos it may be irregularly shaped.
		
	def joinResponsesAndStimPresentations(self, disparityEvents, buttonDownEvents, responseInterval = [0.5,2.0]):
		"""docstring for joinResponsesAndStimPresentations"""
		stimResponsePairs = []
		for dE in disparityEvents:
			# select button presses that occurred after stimulus presentation
			#  but within response interval
			responseTimeIndices = ((buttonDownEvents[:,1] - dE[1]) > responseInterval[0]) * ((buttonDownEvents[:,1] - dE[1]) <= responseInterval[1])
			responseTimes = buttonDownEvents[responseTimeIndices]
			if responseTimes.shape[0] == 0:		# no answer means 0.5
#				noResponseTrials.append(dE)
				answer = 0.5
			else:	# one or many answers - take the last answer
				answer = buttonDownEvents[responseTimeIndices][-1,0] - 1.0
				
			stimResponsePairs.append([dE[0],answer,dE[1]])	# stim value, answer, time
		return np.array(stimResponsePairs)
	
	def analyzePsychophysics(self, noAnswerBehavior = 'disregard'):
		"""
		analyze this run's psychopysics.
		takes together the input event times and searches button presses that are likely to belong to it
		then creates raw trial data and average trial data, and then fits the thing
		noAnswerBehavior argument can be 'disregard' or 'asChanceAnswer'
		"""
		self.stimResponsePairs = self.joinResponsesAndStimPresentations(self.disparityEvents, self.buttonDownEvents)
		if noAnswerBehavior == 'disregard':
			self.stimResponsePairsAllAnswers = self.stimResponsePairs[self.stimResponsePairs[:,1] != 0.5]
		elif noAnswerBehavior == 'asChanceAnswer':
			self.stimResponsePairsAllAnswers = self.stimResponsePairs
		self.answersPerStimulusValue = [self.stimResponsePairsAllAnswers[self.stimResponsePairsAllAnswers[:, 0] == d,1] for d in self.disparities]
		self.meanAnswersPerStimulusValue = [np.array(self.stimResponsePairsAllAnswers[self.stimResponsePairsAllAnswers[:, 0] == d,1]).mean() for d in self.disparities]
		
		self.fitPC()
		
	def fitPC(self, plot = True):
		"""invoke pypsignifit for fitting psychometric curves"""
		data = zip(self.disparities, [f.sum() for f in self.answersPerStimulusValue], [f.shape[0] for f in self.answersPerStimulusValue])
		# standard: no constraints
		pf = BootstrapInference(data, sigmoid = 'gauss', core = 'ab', nafc = 1, cuts = [0.25,0.5,0.75])
		pf.sample()
		GoodnessOfFit(pf)
		self.fit = pf
	

class RivalryPercept(object):
	"""docstring for RivalryPercept"""
	def __init__(self, startReport, endReport):
		super(RivalryPercept, self).__init__()
		self.startReport = startReport
		self.endReport = endReport
		self.duration = self.endReport[1] - self.startReport[1]
		self.startTime = self.startReport[1]
		self.endTime = self.endReport[1]
		self.identity = self.startReport[0]
		

class RivalryTrackingBehaviorOperator(RivalryLearningBehaviorOperator):
	"""docstring for ClassName"""
	def __init__(self, inputObject, **kwargs):
		super(RivalryTrackingBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.openData()
		self.separateEventsFromData()
		
	def joinButtonDownAndUps(self):
		"""
		convert button ups to the right format because first sessions didn't do that
		then, create events that may be percepts or transitions. these are stored in lists
		"""
		# -1 is -98., -2 is -121.
		# correct this and move on.
		# once the presentation software is corrected for this, these statements will just pass
		self.buttonEvents[self.buttonEvents[:,0] == -98.,0] = -1.
		self.buttonEvents[self.buttonEvents[:,0] == -121.,0] = -2.
		
		# report types
		# every report chimes in a period of a certain type - a transition, a definite percept, a double-percept (both at the same time)
		lastReport = self.buttonEvents[0]
		beforeLastReport = None
		percepts = []
		transitions = []
		for bE in self.buttonEvents[1:]:
			if (bE[0] < 0) and (lastReport[0] == -bE[0]) :	# the present buttonpress ends a normal definite percept
				thisPercept = RivalryPercept(lastReport, bE)
				thisPercept.type = 0
				percepts.append(thisPercept)
			elif (lastReport[0] == -1 and bE[0] == 2) or (lastReport[0] == -2 and bE[0] == 1):	# the present buttonpress ends a (prolonged) transition
				thisTransition = RivalryPercept(lastReport, bE)
				thisTransition.type = 0
				transitions.append(thisTransition)
			elif (bE[0] > 0) and (lastReport[0] == -bE[0]):	# the present buttonpress ends a return transition
				thisTransition = RivalryPercept(lastReport, bE)
				thisTransition.type = 1
				transitions.append(thisTransition)
			elif (bE[0] > 0) and (lastReport[0] > 0):	# this is a percept starting before the previous percept ends.
				thisPercept = RivalryPercept(lastReport, bE)
				thisPercept.type = 1
				percepts.append(thisPercept)
			elif (bE[0] < 0) and (lastReport[0] != -bE[0]):	# this buttonpress ends a double-pressed period. This could be described as a negative-duration transition. We do not add thes as such to the transitions array but log it as info
				thisTransition = RivalryPercept(lastReport, bE)
				thisTransition.type = 2
				thisTransition.duration = -thisTransition.duration
				self.logger.info('Negative duration transition at %f, lasting %f', thisTransition.startTime, thisTransition.duration)
			# that should be all. 
			lastReport = bE
			beforeLastReport = lastReport
			
		perceptsNoTransitions = removeRepetitions(self.buttonEvents[self.buttonEvents[:,0]>0])
		self.perceptsNoTransitions = [perceptsNoTransitions[:-1,1], perceptsNoTransitions[1:,1] - perceptsNoTransitions[:-1,1], perceptsNoTransitions[:-1,0]]
		self.perceptsNoTransitionsAsArray = np.array(self.perceptsNoTransitions).T
		
		self.transitions = transitions
		self.percepts = percepts
		
		# information for fsl event files and further analyses
		self.transitionEventsAsArray = np.array([[tr.startTime, tr.duration, tr.identity] for tr in self.transitions])
		self.perceptEventsAsArray = np.array([[p.startTime, p.duration, p.identity] for p in self.percepts])
		
		self.meanPerceptDuration = self.perceptEventsAsArray.mean(axis = 0)[1]
		self.meanTransitionDuration = self.transitionEventsAsArray.mean(axis = 0)[1]
		self.meanPerceptsNoTransitionsDuration = self.perceptsNoTransitionsAsArray.mean(axis = 0)[1]
	

