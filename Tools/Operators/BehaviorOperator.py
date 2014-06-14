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
import re
from IPython import embed as shell


def removeRepetitions(array, position = 0):
	a = array
	return a[np.concatenate(([True], a[:-1,position] != a[1:,position]))]


class BehaviorOperator(Operator):
	def __init__(self, inputObject, **kwargs):
		"""
		BehaviorOperator will receive a filename/wildcard combo as inputobject. 
		"""
		super(BehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		
		if self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
			self.logger.info(self.__repr__() + ' initialized with file ' + os.path.split(self.inputFileName)[-1])
			if not os.path.isfile(self.inputFileName):
				self.logger.warning('inputFileName is not a file at initialization')
			
		# first thing is to change the wildcard to an actual filename that contains the date and timestamp of the run
		fileNameList = subprocess.Popen('ls ' + self.inputFileName, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		if len(fileNameList) > 1:
			self.logger.debug('more than one behavioral file (%s) corresponds to the wildcard %s suggested. Will be using the last in the list ', str(fileNameList), self.inputFileName)
		self.inputFileName = fileNameList[-1]
	

class NewBehaviorOperator(BehaviorOperator):
	"""
	A class that takes a now standard inputDict file and takes its parameterArray and events
	"""
	def __init__(self, inputObject, **kwargs):
		super(NewBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
	
		f = open(self.inputFileName, 'r')
		self.pickledData = pickle.load(f)
		f.close()
		
		self.parameters = self.pickledData['parameterArray']
		self.rawEventData = self.pickledData['eventArray']

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
		
		# self.buttonDownEvents depends on the types of buttons pushed. we may correct for this.
		minmax = np.array([self.buttonDownEvents[:,0].min(),self.buttonDownEvents[:,0].max()])
		minmaxPositions = np.array([self.buttonDownEvents[:,0] == minmax[0], self.buttonDownEvents[:,0] == minmax[1]])
		minmaxNormalized = (minmax - minmax[0])/(minmax[1] - minmax[0])
		self.buttonDownEvents[minmaxPositions[0],0] = minmaxNormalized[0]
		self.buttonDownEvents[minmaxPositions[1],0] = minmaxNormalized[1]
		
#		self.logger.debug('self.buttonDownEvents = %s, self.buttonEvents = %s', str(self.buttonDownEvents), str(self.buttonEvents))
		
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
				answer = buttonDownEvents[responseTimeIndices][-1,0]
				
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
		
	def joinButtonDownAndUps(self):	# buttons when the thumb button is broken: [2,3]
		"""
		convert button ups to the right format because first sessions didn't do that
		then, create events that may be percepts or transitions. these are stored in lists
		"""
		# -1 is -98., -2 is -121.
		# correct this and move on.
		# once the presentation software is corrected for this, these statements will just pass
		self.buttonEvents[self.buttonEvents[:,0] == -98.,0] = -1.
		self.buttonEvents[self.buttonEvents[:,0] == -121.,0] = -2.
		
		# 1 and 2 for normal sessions, 2 and 3 (or different things, equally shitty) for fucked up button sessions
		mm = [np.min(abs(self.buttonEvents[:,0])),np.max(abs(self.buttonEvents[:,0]))]
		if mm != [1.,2.]:
			self.buttonEvents[self.buttonEvents[:,0] == mm[0],0] = 1.
			self.buttonEvents[self.buttonEvents[:,0] == -mm[0],0] = -1.
			self.buttonEvents[self.buttonEvents[:,0] == mm[1],0] = 2.
			self.buttonEvents[self.buttonEvents[:,0] == -mm[1],0] = -2.
			
		# from now on we can be sure of this
		answers = [1,2]
		
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
			elif (lastReport[0] == -answers[0] and bE[0] == answers[1]) or (lastReport[0] == -answers[1] and bE[0] == answers[0]):	# the present buttonpress ends a (prolonged) transition
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
		
		self.yokedPeriods = []
	


class RivalryReplayBehaviorOperator(BehaviorOperator):
	"""
	This behavior operator parses the outputs of Jan's behavioral output format in order to take out event timings.
	"""
	def __init__(self, inputObject, **kwargs):
		super(RivalryReplayBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.openData()
		self.separateEventsFromData()
	
	def openData(self):
		"""docstring for openData"""
		self.rawData = np.loadtxt(self.inputFileName)
		self.parameters = self.rawData[self.rawData[:,0] == 4.0]
		
		# rescale and remean time column
		self.rawData[4:,1] = (self.rawData[4:,1] - self.rawData[0,1]) * 10000
	
	def separateEventsFromData(self, reactionTime = 0.4, timeRange = [10,130]):
		"""docstring for separateEventsFromData"""
		self.yokedRawEvents = self.rawData[self.rawData[:,0] == 5.0]
		self.buttonRawEvents = self.rawData[self.rawData[:,0] == 2.0]
		
		# take all button events, take their time and identity at each row. then remove repetitions from this new list
		# order for button presses is = [0,0,0] -> [green, red, transition]
		# this is corrected in order to correspond to the yoked conditions by shuffling the order of columns to [2,4,3], see below
		self.allButtonEvents = np.array([self.buttonRawEvents[4:,1], (self.buttonRawEvents[4:,[2,3,4]] * [1,2,3]).sum(axis = 1)]).T
		self.buttonEvents = removeRepetitions(self.allButtonEvents, position = 1)
		# we're interested in the period the subject was reporting actual rivalry
		self.rivalryButtonEvents = self.buttonEvents[(self.buttonEvents[:,0] < timeRange[1]) * (self.buttonEvents[:,0] > timeRange[0])]
		# and from this period we want to know when and for how long events of what type were occurring
		self.rivalryButtonPeriods = np.array([[self.rivalryButtonEvents[ev][0] - reactionTime, self.rivalryButtonEvents[ev+1][0] - self.rivalryButtonEvents[ev][0], self.rivalryButtonEvents[ev][1]]  for ev in range(self.rivalryButtonEvents.shape[0]-1)])
		# throw out percepts and transitions too short to be bothered with
		self.rivalryPeriods = self.rivalryButtonPeriods[self.rivalryButtonPeriods[:,1] > reactionTime]
		# take actual percepts
		self.percepts = self.rivalryButtonPeriods[(self.rivalryButtonPeriods[:,2] == 1.) + (self.rivalryButtonPeriods[:,2] == 3.)]
		self.transitions = self.rivalryButtonPeriods[(self.rivalryButtonPeriods[:,2] == 2.)]
		
		# yoked events - coded as [green, transition, red] - don't ask me why this is different from the earlier one
		instantYokedStartEvents = np.arange(self.yokedRawEvents.shape[0])[self.yokedRawEvents[:,2] == 4.]
		instantYokedEndEvents = instantYokedStartEvents + 1
		if instantYokedStartEvents.shape[0] > 0:
			self.startInstantYokedEventOnsets = self.yokedRawEvents[instantYokedStartEvents]
			self.startInstantYokedEventOffsets = self.yokedRawEvents[instantYokedEndEvents]
			self.yokedEvents = np.vstack((self.startInstantYokedEventOnsets[:,1],self.startInstantYokedEventOffsets[:,1]-self.startInstantYokedEventOnsets[:,1],self.startInstantYokedEventOffsets[:,2])).T
			self.yokedPeriods = [[self.yokedRawEvents[ev][1],self.yokedRawEvents[ev+1][1]-self.yokedRawEvents[ev][1], self.yokedRawEvents[ev][2]] for ev in range(1, self.yokedRawEvents.shape[0]-1)]
		else:
			self.startInstantYokedEventOnsets = np.zeros((1,3))
			self.startInstantYokedEventOffsets = np.zeros((1,3))
			self.yokedEvents = np.zeros((1,3))
			self.yokedPeriods = np.zeros((1,3))
		
		# information for fsl event files and further analyses
		self.transitionEventsAsArray = np.array(self.transitions)
		self.perceptEventsAsArray = np.array(removeRepetitions(self.percepts, position = 2))
		self.perceptsNoTransitionsAsArray = np.array(self.percepts)
		
		
		self.halfwayTransitionsAsArray = np.array([[e[0] + e[1]/2.0, e[1], e[2]] for e in self.transitionEventsAsArray])
		
		self.meanPerceptDuration = self.perceptEventsAsArray.mean(axis = 0)[1]
		self.meanTransitionDuration = self.transitionEventsAsArray.mean(axis = 0)[1]
		self.meanPerceptsNoTransitionsDuration = self.meanPerceptDuration
	

class ApparentMotionBehaviorOperator(BehaviorOperator):
	"""
	This behavior operator parses the outputs of by own AM behavioral output format in order to take out event timings.
	"""
	def __init__(self, inputObject, **kwargs):
		super(ApparentMotionBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.openData()
		self.separateEventsFromData()
	
	def openData(self):
		"""docstring for openData"""
		self.rawData = np.loadtxt(self.inputFileName)
		self.parameters = self.rawData[:6]
	
	def separateEventsFromData(self, reactionTime = 0.4, timeRange = [10,130]):
		"""docstring for separateEventsFromData"""
		
		self.rawEvents = self.rawData[6:]
		self.rawEvents = self.rawEvents.reshape((self.rawEvents.shape[0]/2, 2))
		
		self.buttonEvents = self.rawEvents[(self.rawEvents[:,0] > 0) * (self.rawEvents[:,0] < 999)]
		self.TREvents = self.rawEvents[self.rawEvents[:,0] == 0]
		# we're interested in the period the subject was reporting actual rivalry
		self.rivalryButtonEvents = self.buttonEvents[(self.buttonEvents[:,1] < timeRange[1]) * (self.buttonEvents[:,1] > timeRange[0])]
		# and from this period we want to know when and for how long events of what type were occurring
		self.rivalryButtonPeriods = np.array([[self.rivalryButtonEvents[ev][1] - reactionTime, self.rivalryButtonEvents[ev+1][1] - self.rivalryButtonEvents[ev][1], self.rivalryButtonEvents[ev][0]]  for ev in range(self.rivalryButtonEvents.shape[0]-1)])
		# throw out percepts and transitions too short to be bothered with
		self.rivalryPeriods = self.rivalryButtonPeriods[self.rivalryButtonPeriods[:,1] > reactionTime]
		# take actual percepts
		self.percepts = self.rivalryButtonPeriods[(self.rivalryButtonPeriods[:,2] == 1.) + (self.rivalryButtonPeriods[:,2] == 3.)]
		
		self.yokedPeriods = []
	
class SphereBehaviorOperator(BehaviorOperator):
	"""
	This behavior operator parses the outputs of by own AM behavioral output format in order to take out event timings.
	"""
	def __init__(self, inputObject, **kwargs):
		super(SphereBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.openData()
		self.separateEventsFromData()
	
	def openData(self):
		"""docstring for openData"""
		self.rawData = np.loadtxt(self.inputFileName)
		self.parameters = None
		
	def separateEventsFromData(self, reactionTime = 0.4, timeRange = [0,1000], startEndPeriods = [16, -16], time_resolution = 0.0001):
		self.startTime = self.rawData[0,2]
		self.timedData = self.rawData
		self.timedData[:,2] = self.timedData[:,2] - self.startTime
		self.timedData[:,2] = self.timedData[:,2] * time_resolution
		
		self.TREvents = self.timedData[self.timedData[:,1] == 49][:,[1,2]]
		self.buttonEvents = self.timedData[self.timedData[:,1] > 60][:,[1,2]]
		self.TR = round(np.mean(self.TREvents[:,0] * 1000)) / 1000
		
		lastStimulusTime = self.TREvents[-1,1] + startEndPeriods[1]
		
		self.transitionEvents = removeRepetitions(self.buttonEvents)
		
		self.percepts = np.vstack((self.transitionEvents[:,0], self.transitionEvents[:,1], np.concatenate((self.transitionEvents[1:,1], [lastStimulusTime])))).T
		
		# np.hstack(, np.concatenate((self.transitionEvents[-1], lastStimulusTime)) )
	
standardNrVolumes = 124
standardTR = 2
standardNrSecsBeforeCountingForSelfReqAvgs = 8.0
standardPeriodDuration = 32.0
standardPermittedResponseLag = 1.5

class WedgeRemappingOperator(BehaviorOperator):
	def __init__(self, inputObject, **kwargs):
		super(WedgeRemappingOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.importFile()
	
	def importFile(self):
		f = open(self.inputFileName)
		stringArray = f.readlines()
		f.close()
		
		inputs = []
		outputs = []
		for i in range(1,len(stringArray),4):
			inputs.append( np.array(stringArray[i].split('\t')[:-1],dtype = float) )
			outputData = np.array(stringArray[i+2].split('\t')[:-1],dtype = float)
			outputs.append(outputData.reshape(len(outputData)/4,4))
		
		if i == 1:	# only one line of data - one trial per run
			self.inputParams = inputs[0]
			self.outputData = outputs[0]
		else: 		# more trials per run
			self.inputParams = np.array(inputs, dtype = float)	# these are all the same size (128), so convertable to array
			self.outputData = outputs							# these likely differ in size, so don't convert to numpy
	
	def segmentOutputData(self):
		self.stimColorEvents = self.outputData[(self.outputData[:,2]==5003)+(self.outputData[:,2]==5004)]
		self.colorResponseEvents = self.outputData[self.outputData[:,2]==49]
		self.scannerTREvents = self.outputData[self.outputData[:,2]==96]
		if len(self.scannerTREvents) > 0:	# this occurs when there were no TRs
			startTime = self.scannerTREvents[0][1]
		else:
			startTime = self.outputData[self.outputData[:,2] == 0.1,1][0]
		# correct for start time offset
		self.stimColorEvents[:,1] =  self.stimColorEvents[:,1] - startTime
		self.colorResponseEvents[:,1] =  self.colorResponseEvents[:,1] - startTime
		self.scannerTREvents[:,1] =  self.scannerTREvents[:,1] - startTime
	
	def collectResponsesAfterColorChanges(self, permittedResponseLag = standardPermittedResponseLag):
		answers = []
		for colorEvent in self.stimColorEvents:
			laterColorResponses = self.colorResponseEvents[self.colorResponseEvents[:,1]>colorEvent[1]]
			if len(laterColorResponses) > 0:
				latestResponse = laterColorResponses[0]
				responseLag = latestResponse[1] - colorEvent[1]
				if responseLag < permittedResponseLag:	# correct answer
					answers.append([(colorEvent[1]-standardNrSecsBeforeCountingForSelfReqAvgs)/standardPeriodDuration, colorEvent[1], colorEvent[2], 1, responseLag])
				else: # answer too late
					answers.append([(colorEvent[1]-standardNrSecsBeforeCountingForSelfReqAvgs)/standardPeriodDuration, colorEvent[1], colorEvent[2], 0, responseLag])
			else: # no answer
				answers.append([(colorEvent[1]-standardNrSecsBeforeCountingForSelfReqAvgs)/standardPeriodDuration, colorEvent[1], colorEvent[2], 0, 0])
		noColorEventIndices = np.array([((self.stimColorEvents[:,1] < i[1]) * (self.stimColorEvents[:,1] > i[0])).sum() == 0 for i in np.vstack((np.arange(0,263), np.arange(1,264))).T])
		faEvents = np.ones((263,3))
		faEvents[:,1] = np.arange(1,264) * faEvents[:,1]
		faEvents[:,2] = -1 * faEvents[:,2]
		self.noColorEvents = faEvents[noColorEventIndices]
		for colorEvent in self.noColorEvents:
			laterColorResponses = self.colorResponseEvents[self.colorResponseEvents[:,1]>colorEvent[1]]
			if len(laterColorResponses) > 0:
				latestResponse = laterColorResponses[0]
				responseLag = latestResponse[1] - colorEvent[1]
				if responseLag < permittedResponseLag:	# false alarm
					answers.append([(colorEvent[1]-standardNrSecsBeforeCountingForSelfReqAvgs)/standardPeriodDuration, colorEvent[1], colorEvent[2], -1, responseLag])
				else: # answer too late - no false alarm
					answers.append([(colorEvent[1]-standardNrSecsBeforeCountingForSelfReqAvgs)/standardPeriodDuration, colorEvent[1], colorEvent[2], 2, responseLag])
			else: # no answer	- correct rejection
				answers.append([(colorEvent[1]-standardNrSecsBeforeCountingForSelfReqAvgs)/standardPeriodDuration, colorEvent[1], colorEvent[2], 2, 0])
		self.answerList = np.array(answers, dtype = float)


class PopulationReceptiveFieldResponse(object):
	"""docstring for PopulationReceptiveFieldResponse"""
	def __init__(self, phase, ):
		super(PopulationReceptiveFieldResponse, self).__init__()
		self.arg = arg
		


class TrialEventSequence(object):
	def __init__(self, parameters, events, index = 0, run_start_time = 0.0):
		self.parameters = parameters
		self.events = events
		self.run_start_time = run_start_time
		self.index = index
	
	def convert_events(self):
		"""convert the string-based event-array from the pickle file 
		to a set of lists that we can work with 
		and are easily interpretable."""
		rec_button = re.compile('trial %i key: (\S+) at time: (-?\d+\.?\d*) for task [\S+]' % self.index)
		self.button_events = [re.findall(rec_button, e)[0] for e in self.events if 'trial %i key:' % self.index in e]
		self.button_events = [[b[0], float(b[1]) - self.run_start_time] for b in self.button_events]
		
		rec_phase = re.compile('trial %d phase (\d+) started at (-?\d+\.?\d*)' % self.index)
		self.phase_events = [re.findall(rec_phase, e)[0] for e in self.events if 'trial %d phase'% self.index in e]
		self.phase_events = [[p[0], float(p[1]) - self.run_start_time] for p in self.phase_events]
		
		self.find_task()	# need to know the task this trial to be able to extract.
		rec_signal = re.compile('signal in task (\S+) at (-?\d+\.?\d*) value 1.0')
		self.signal_events = [re.findall(rec_signal, e)[0] for e in self.events if 'signal' in e]
		self.signal_events = [[s[0], float(s[1]) - self.run_start_time] for s in self.signal_events]
		self.task_signal_events = np.array([s[0] == self.task for s in self.signal_events])
		# shell()
		
		self.task_signal_times = np.array(np.array(self.signal_events)[self.task_signal_events,1], dtype = float)
		
	
	def find_task(self):
		"""find the task from parameters, 
		if the 'y' button has been pressed, 
		this means the fixation task."""
		self.task = self.parameters['task']
		
		if np.array([b[0] == 'y' for b in self.button_events]).sum() > 0:
			self.task = 'fix'
			self.task_button_event_times = np.array([b[1] for b in self.button_events if b[0] == 'y'])
		else:
			self.task_button_event_times = np.array([b[1] for b in self.button_events if b[0] == 'b'])
	
	def check_answers(self, maximal_reaction_time = 1.0):
		"""take the task signal events and the button presses
		and evaluate which signals were responded to and which were not.
		maximal_reaction_time defines the window in which button presses are counted."""
		
		all_response_delays = np.array([self.task_button_event_times - s for s in self.task_signal_times])
		# response_delays = [r[r>0][0] if (r>0).shape[0] > 0 else -1 for r in all_response_delays]
		
		self.signal_times_response_times = np.array([[self.task_signal_times[i], r[r>0][0]] if (r>0).shape[0] > 0 else -1 for i, r in enumerate(all_response_delays)])
		
		# maak hier een lijst van responses voor deze trial

class PopulationReceptiveFieldBehaviorOperator(NewBehaviorOperator):
	def __init__(self, inputObject, **kwargs):
		"""docstring for __init__"""
		super(PopulationReceptiveFieldBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		with open( self.inputFileName ) as f:
			file_data = pickle.load(f)
		self.events = file_data['eventArray']
		self.parameters = file_data['parameterArray']
		
		run_start_time_string = [e for e in self.events[0] if e[:len('trial 0 phase 1')] == 'trial 0 phase 1']
		self.run_start_time = float(run_start_time_string[0].split(' ')[-1])
	
	def convert_events(self):
		self.trials = []
		for p, e, i in zip(self.parameters, self.events, range(len(self.parameters))):
			tes = TrialEventSequence(p, e, i, self.run_start_time)
			tes.convert_events()
			self.trials.append(tes)
			
	def analyze_SDT(self):
		"""analyze_SDT analyzes trial data to find sdt category responses and further behavioral analysis."""
		tasks = [t.task for t in self.trials]
		u_tasks = np.unique(np.array(tasks))
		all_responses_per_task = []
		for u in u_tasks:
			all_responses_per_task.append([]) # [[e, aiadjk, ask, akddk], [jdfkl, akkdsjk, adksj, jadkjfks, ]]
			for i, t in enumerate(tasks): # over trials loop
				if t == u:
					all_responses_per_task[-1].append(self.trials[i].responses) # responses is een lijst van Response objecten.
		
	def trial_times(self, stim_offsets = [1.0, 0.0]):
		if not hasattr(self, 'trials'):
			self.convert_events()
		self.trial_times = []
		self.all_button_times = []
		for i, t in enumerate(self.trials):
			stim_on_time = [p[1] for p in t.phase_events if p[0] == '2']
			stim_off_time = [p[1] for p in t.phase_events if p[0] == '3' ]
			
			if len(stim_off_time) != 0 and len(stim_on_time) != 0:
				self.trial_times.append([t.task, stim_on_time[0] + stim_offsets[0], stim_off_time[0] + stim_offsets[1]])
				self.all_button_times.append([[t.task, bt[1]] for bt in t.button_events])
		self.all_button_times = np.concatenate(self.all_button_times)
	
			
