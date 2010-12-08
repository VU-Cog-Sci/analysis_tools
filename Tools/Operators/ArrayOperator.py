#!/usr/bin/env python
# encoding: utf-8
"""
DataOperator.py

Created by Tomas Knapen on 2010-11-26.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

from Operator import *


class ArrayOperator(Operator):
	"""
	DataOperator takes in numpy arrays or 
	numpy data files and runs analysis on these. 
	Examples are time-course and event-based data analysis like deconvolution and event-related averages.
	"""
	def __init__(self, inputObject, **kwargs):
		super(ArrayOperator, self).__init__(inputObject, **kwargs)
		
		if self.inputObject.__class__.__name__ == 'ndarray':
			self.dataArray = self.inputObject
			# or the input object could be a numpy file
		elif self.inputObject.__class__.__name__ == 'str':
			self.dataArray = np.load(self.inputObject)
			
		# if input data is 4D, reshape to do voxel-by time analysis. 
		# revert back at end, perhaps
		self.inputShape = self.dataArray.shape
		if len(self.inputShape) > 2:
			self.dataArray = self.dataArray.reshape((-1, self.inputShape[-1]))
	
	
class EventDataOperator(ArrayOperator):
	"""docstring for EventDataOperator"""
	def __init__(self, inputObject, eventObject, TR = 2.0, **kwargs):
		super(EventDataOperator, self).__init__(inputObject, **kwargs)
		
		self.eventObject = eventObject
		
		# eventArray is a 1D array of timepoints
		if self.eventObject.__class__.__name__ == 'ndarray':
			self.eventArray = self.eventObject
			# or the input object could be a numpy file
		elif self.eventObject.__class__.__name__ == 'str':
			self.eventArray = np.load(self.eventObject)
			# or a regular list
		elif self.eventObject.__class__.__name__ == 'list':
			self.eventArray = self.eventObject
			
			
		self.TR = TR
	

class DeconvolutionOperator(EventDataOperator):
	"""docstring for DeconvolutionOperator"""
	def __init__(self, inputObject, eventObject, TR = 2.0, deconvolutionSampleDuration = 0.5, deconvolutionInterval = 12.0, **kwargs):
		super(DeconvolutionOperator, self).__init__(inputObject, eventObject, TR, **kwargs)
		
		self.deconvolutionSampleDuration = deconvolutionSampleDuration
		self.deconvolutionInterval = deconvolutionInterval
		
		self.ratio = TR / deconvolutionSampleDuration
		self.nrSamplesInInterval = self.deconvolutionInterval / deconvolutionSampleDuration
		
		self.upsampleDataTimeSeries()
		self.createDesignMatrix()
		self.rawDeconvolvedTimeCourse = self.h()
		self.deconvolvedTimeCoursesPerEventType = self.rawDeconvolvedTimeCourse.reshape((self.rawDeconvolvedTimeCourse.shape[0]/self.nrSamplesInInterval,self.nrSamplesInInterval))
		
	def upsampleDataTimeSeries(self):
		"""upsampleDataTimeSeries takes a timeseries of data points
		 and upsamples them according to 
		the ratio between TR and deconvolutionSampleDuration."""
		self.workingDataArray = (np.ones((int(self.ratio),self.dataArray.shape[0])) * self.dataArray).T.ravel()
#		self.workingDataArray = np.array([self.dataArray for i in range(int(self.ratio))]).transpose().ravel()
		self.logger.debug('upsampled from %s to %s according to ratio %s', str(self.dataArray.shape), str(self.workingDataArray.shape), str(self.ratio))
	
	def designMatrixFromVector(self, eventTimesVector):
		"""designMatrixFromVector creates a deconvolution design matrix from 
		an event vector. To do this, it rounds the event times to the nearest 
		deconvolutionSampleDuration interval and thus discretizes them. 
		Afterwards, it creates a matrix by shifting the discretized array
		nrSamplesInInterval times. """
		startArray = np.zeros(self.workingDataArray.shape[0])
		startArray[np.array((eventTimesVector * self.deconvolutionSampleDuration).round(),dtype = int)] = 1.0
		allArray = np.zeros((self.nrSamplesInInterval,self.workingDataArray.shape[0]))
		allArray[0] = startArray
		for i in range(1,int(self.nrSamplesInInterval)):
			allArray[i,i:] = startArray[:-i]
		return np.mat(allArray)
	
	def createDesignMatrix(self):
		"""createDesignMatrix takes the event times array 
		and creates a design timecourse matrix
		by discretizing the events per deconvolutionSampleDuration period."""
		m = []
		for i in range(len(self.eventArray)):
			m.append(self.designMatrixFromVector(self.eventArray[i]))
		self.designMatrix = np.mat(np.concatenate(m,axis = 0)).T
	
	def h(self):
		"""
		run the actual deconvolution least-squares approximation
		"""
		return ((self.designMatrix.T * self.designMatrix).I * self.designMatrix.T) * np.mat(self.workingDataArray).T
	

class EventRelatedAverageOperator(EventDataOperator):
	"""Event related averages for all voxel/ROI elements in the input data array.
	In most cases, one can save time by averaging across voxels before construction."""
	def __init__(self, inputObject, eventObject, TR = 2.0, interval = [-5.0,21.0], **kwargs):
		super(EventRelatedAverageOperator, self).__init__(inputObject, eventObject, TR, **kwargs)
		
		self.interval = np.array(interval)
		self.intervalInTRs = np.array(self.interval)/self.TR
		self.intervalRange = np.arange(self.interval[0] + self.TR / 2.0, self.interval[1], self.TR)
		self.TRTimes = np.arange(self.TR / 2.0, self.dataArray.shape[-1] * self.TR + self.TR / 2.0, self.TR)
		
		# throw out events that happen too near the beginning and end of the run to fit in the averaging interval -- this has already been done in gatherBehavioralData in session
		self.selectedEventArray = self.eventArray # [( self.eventArray > self.interval[0] ) * ( self.eventArray < (self.dataArray.shape[-1] - self.interval[-1]) )]
		
		self.prepareEventData()
	
	def prepareEventData(self):
		self.eventSampleTimes = np.zeros(np.concatenate([[self.selectedEventArray.shape[0]],self.intervalRange.shape]))
		self.eventData = np.zeros(np.concatenate([[self.dataArray.shape[0],self.selectedEventArray.shape[0]],self.intervalRange.shape]))
		self.logger.debug('eventSampleTimes array shape: %s, eventData array shape: %s, dataArray shape: %s',self.eventSampleTimes.shape, self.eventData.shape, self.dataArray.shape )
		for i in range(self.selectedEventArray.shape[0]):
			# set back the times of the recorded TRs
			zeroTime = np.fmod(np.fmod(self.selectedEventArray[i], self.TR) + self.TR, self.TR) - self.TR
			self.eventSampleTimes[i] = self.intervalRange - zeroTime
#			print zeroTime, self.eventSampleTimes[i], self.intervalRange, self.selectedEventArray[i]
#			print self.TRTimes[( self.TRTimes > self.interval[0] + self.selectedEventArray[i] ) * ( self.TRTimes < self.interval[1] + self.selectedEventArray[i] )]
#			print self.dataArray[:,( self.TRTimes > self.interval[0] + self.selectedEventArray[i] ) * ( self.TRTimes < self.interval[-1] + self.selectedEventArray[i] )]
			self.eventData[:,i] = self.dataArray[:,( self.TRTimes > self.interval[0] + self.selectedEventArray[i] ) * ( self.TRTimes < self.interval[-1] + self.selectedEventArray[i] )]
	
	def averageEventsInTimeInterval(self, averagingInterval):
		theseData = self.eventData[:,( self.eventSampleTimes > averagingInterval[0] ) * ( self.eventSampleTimes <= averagingInterval[1] )].ravel()
		return [averagingInterval[0] + (averagingInterval[1] - averagingInterval[0]) / 2.0, theseData.mean(), theseData.std(), theseData.shape[0]]
	
	def run(self, binWidth = 2.0, stepSize = 0.5):
		self.averagingIntervals = np.array([[t, t + binWidth] for t in np.arange(self.interval[0], self.interval[1] - binWidth, stepSize)])
		self.output = np.array([self.averageEventsInTimeInterval(i) for i in self.averagingIntervals])
		return self.output
	
