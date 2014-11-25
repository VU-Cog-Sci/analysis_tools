#!/usr/bin/env python
# encoding: utf-8
"""
DataOperator.py

Created by Tomas Knapen on 2010-11-26.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

from Operator import *
from IPython import embed as shell

# weird workaround for scipy stats import bug. try and except: do again!
try:
	import scipy.stats as stats
except ValueError:
	import scipy.stats as stats

from math import *


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
	def __init__(self, inputObject, eventObject, TR = 2.0, deconvolutionSampleDuration = 0.5, deconvolutionInterval = 12.0, run = True, **kwargs):
		super(DeconvolutionOperator, self).__init__(inputObject, eventObject, TR, **kwargs)
		
		self.deconvolutionSampleDuration = deconvolutionSampleDuration
		self.deconvolutionInterval = deconvolutionInterval
		
		self.ratio = TR / deconvolutionSampleDuration
		self.nrSamplesInInterval = floor(self.deconvolutionInterval / deconvolutionSampleDuration)
		
		self.upsampleDataTimeSeries()
		self.createDesignMatrix()
		if run:
			self.run()
			# self.rawDeconvolvedTimeCourse = self.h()
			# self.deconvolvedTimeCoursesPerEventType = np.array(self.rawDeconvolvedTimeCourse).reshape((int(self.rawDeconvolvedTimeCourse.shape[0]/self.nrSamplesInInterval),int(self.nrSamplesInInterval),-1))
			# shell()
	
	def run(self):
		self.rawDeconvolvedTimeCourse = self.h()
		self.deconvolvedTimeCoursesPerEventType = np.array(self.rawDeconvolvedTimeCourse).reshape((int(self.rawDeconvolvedTimeCourse.shape[0]/self.nrSamplesInInterval),int(self.nrSamplesInInterval),-1))


	def upsampleDataTimeSeries(self):
		"""upsampleDataTimeSeries takes a timeseries of data points
		 and upsamples them according to 
		the ratio between TR and deconvolutionSampleDuration."""
		# self.workingDataArray = np.tile(self.dataArray,int(self.ratio)).reshape(int(self.ratio),self.dataArray.shape[0]).T.ravel()
		
		# new version for whole-array analyses
		new_size = list(self.dataArray.shape)
		new_size[0] *= int(self.ratio)
		new_array = np.zeros(new_size)
		for i in np.arange(self.dataArray.shape[0]) * int(self.ratio):
			new_array[i:i+int(self.ratio)] = self.dataArray[i/int(self.ratio)]
		self.workingDataArray = new_array
		self.logger.info('upsampled from %s to %s according to ratio %s', str(self.dataArray.shape), str(self.workingDataArray.shape), str(self.ratio))
		
	def designMatrixFromVector(self, eventTimesVector):
		"""designMatrixFromVector creates a deconvolution design matrix from 
		an event vector. To do this, it rounds the event times to the nearest 
		deconvolutionSampleDuration interval and thus discretizes them. 
		Afterwards, it creates a matrix by shifting the discretized array
		nrSamplesInInterval times. """
		allArray = np.zeros((int(self.nrSamplesInInterval),self.workingDataArray.shape[0]))
		for i in range(0,int(self.nrSamplesInInterval)):
			which_further_time_points = ((eventTimesVector / self.deconvolutionSampleDuration) + i).round()
			which_further_time_points = np.array(which_further_time_points[which_further_time_points < self.workingDataArray.shape[0]], dtype = int)
			allArray[i,which_further_time_points] = 1
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
		return ((self.designMatrix.T * self.designMatrix).I * self.designMatrix.T) * np.mat(self.workingDataArray.T).T
	
	def runWithConvolvedNuisanceVectors(self, nuisanceVectors):
		designShape = self.designMatrix.shape
		if nuisanceVectors.shape[0] != self.designMatrix.shape[0]:
			self.logger.error('nuisance dimensions does not correspond to the designmatrix, shapes %s, %s' % (nuisanceVectors.shape, designShape))
		else:
			newNuisanceVectors = nuisanceVectors
			newNuisanceVectors = nuisanceVectors / nuisanceVectors.max(axis=0)
			# newNuisanceVectors = newNuisanceVectors - newNuisanceVectors.mean(axis = 0)
			self.newDesignMatrix = np.mat(np.hstack((self.designMatrix, newNuisanceVectors)))
			#run and segment
			self.deconvolvedTimeCoursesNuisanceAll = ((self.newDesignMatrix.T * self.newDesignMatrix).I * self.newDesignMatrix.T) * np.mat(self.workingDataArray.T).T
			self.deconvolvedTimeCoursesPerEventTypeNuisance = np.zeros((int(designShape[1]/self.nrSamplesInInterval),int(self.nrSamplesInInterval),self.deconvolvedTimeCoursesNuisanceAll.shape[1]))
			for i in range(int(round(designShape[1]/self.nrSamplesInInterval))):
				self.deconvolvedTimeCoursesPerEventTypeNuisance[i] = self.deconvolvedTimeCoursesNuisanceAll[i*self.nrSamplesInInterval:(i+1)*self.nrSamplesInInterval]
			self.deconvolvedNuisanceBetas = self.deconvolvedTimeCoursesNuisanceAll[(i+1)*self.nrSamplesInInterval:]
	
	def residuals(self):
		if hasattr(self, 'newDesignMatrix'):	# means we've run this with nuisances
			betas = self.deconvolvedTimeCoursesNuisanceAll
			design_matrix = self.newDesignMatrix
		elif hasattr(self, 'designMatrix'):
			betas = self.rawDeconvolvedTimeCourse
			design_matrix = self.designMatrix
		else:
			self.logger.error("To compute residuals, we need to calculate betas. Use runWithConvolvedNuisanceVectors or re-initialize with argument run = True")
		self.residuals = self.workingDataArray - np.squeeze(np.dot(design_matrix, betas))
		return np.array(self.residuals)
	
	def sse(self):
		"""use the residuals to create a sum of squared error for each of the event types."""
		res_sq = np.squeeze(self.residuals() ** 2)
		# res_sq = res_sq - res_sq.mean(axis = -1)
		if hasattr(self, 'newDesignMatrix'):	# means we've run this with nuisances
			design_matrix = self.newDesignMatrix
		elif hasattr(self, 'designMatrix'):
			design_matrix = self.designMatrix
		self.logger.info('mean squared sse over time is %f' % res_sq.mean())
		self.sse = np.squeeze(np.array(((design_matrix.T * design_matrix).I * design_matrix.T) * np.mat(res_sq.T).T))
		return self.sse
		
	
	

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
		self.selectedEventArray = self.eventArray[( self.eventArray > self.interval[0] ) * ( self.eventArray < ((self.dataArray.shape[-1] * self.TR) - self.interval[-1]) )]
		
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
			thisData =  self.dataArray[:,( self.TRTimes > (self.interval[0] + self.selectedEventArray[i]) ) * ( self.TRTimes < (self.interval[-1] + self.selectedEventArray[i]) )]
			if thisData.shape[1] == self.intervalRange.shape[0]:
				self.eventData[:,i] = thisData
	
	def averageEventsInTimeInterval(self, averagingInterval, output_raw_data = False):
		theseData = self.eventData[:,( self.eventSampleTimes > averagingInterval[0] ) * ( self.eventSampleTimes <= averagingInterval[1] )].ravel()
		if not output_raw_data:
			return [averagingInterval[0] + (averagingInterval[1] - averagingInterval[0]) / 2.0, theseData.mean(), theseData.std(), theseData.shape[0]]
		else:
			return [averagingInterval[0] + (averagingInterval[1] - averagingInterval[0]) / 2.0, theseData.mean(), theseData.std(), theseData.shape[0], theseData]
	
	def run(self, binWidth = 2.0, stepSize = 0.5, output_raw_data = False):
		self.averagingIntervals = np.array([[t, t + binWidth] for t in np.arange(self.interval[0], self.interval[1] - binWidth, stepSize)])
		self.output = [self.averageEventsInTimeInterval(i, output_raw_data) for i in self.averagingIntervals]
		if output_raw_data:
			return self.output
		else:
			self.output = np.array(self.output)
			return self.output

### took out decoding from shogun, to be replaced by skikits-sklearn and mdp. 
### libsvm is also installed to subserve svm and svr.