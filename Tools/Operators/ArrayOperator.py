#!/usr/bin/env python
# encoding: utf-8
"""
DataOperator.py

Created by Tomas Knapen on 2010-11-26.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

from Operator import *

# weird workaround for scipy stats import bug. try and except: do again!
try:
	import scipy.stats as stats
except ValueError:
	import scipy.stats as stats



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
		self.selectedEventArray = self.eventArray[( self.eventArray > self.interval[0] ) * ( self.eventArray < (self.dataArray.shape[-1] - self.interval[-1]) )]
		
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
	
	def averageEventsInTimeInterval(self, averagingInterval):
		theseData = self.eventData[:,( self.eventSampleTimes > averagingInterval[0] ) * ( self.eventSampleTimes <= averagingInterval[1] )].ravel()
		return [averagingInterval[0] + (averagingInterval[1] - averagingInterval[0]) / 2.0, theseData.mean(), theseData.std(), theseData.shape[0]]
	
	def run(self, binWidth = 2.0, stepSize = 0.5):
		self.averagingIntervals = np.array([[t, t + binWidth] for t in np.arange(self.interval[0], self.interval[1] - binWidth, stepSize)])
		self.output = np.array([self.averageEventsInTimeInterval(i) for i in self.averagingIntervals])
		return self.output
	

		# for decoding we need the following:
		from shogun.Features import SparseRealFeatures, RealFeatures, Labels
		from shogun.Kernel import GaussianKernel
		from shogun.Classifier import LibSVM
		from shogun.Classifier import SVMLin
		
		
from shogun.Features import SparseRealFeatures, RealFeatures, Labels
from shogun.Kernel import GaussianKernel
from shogun.Classifier import LibSVM
from shogun.Classifier import SVMLin
	
def svmLinDecoder(train, test, trainLabels, testLabels, fullOutput = False, C = 0.9, epsilon = 1e-5, nThreads = 4):
	realfeat=RealFeatures(train)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(test)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)
	
	num_threads=nThreads
	labels=Labels(trainLabels)
	
	svm=SVMLin(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(True)
	svm.train()
	
	svm.set_features(feats_test)
#	svm.get_bias()
#	svm.get_w()
	out = svm.classify().get_labels()
	
	predictions = np.sign(out)
	accuracy = ( predictions == testLabels ).sum() / float(testLabels.shape[0])
	
	if fullOutput:
		return [accuracy, predictions, out]
	else:
		return accuracy


def libSVMDecoder(train, test, trainLabels, testLabels, fullOutput = False, width = 2.1, C = 1, epsilon = 1e-5 ):
	pl.figure()
	pl.imshow(train)
	pl.show()
	feats_train=RealFeatures(train)
	feats_test=RealFeatures(test)
	
	kernel=GaussianKernel(feats_train, feats_train, width)
	labels=Labels(trainLabels)
	
	svm=LibSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	
	svm.train()
	
	kernel.init(feats_train, feats_test)
	
	out = svm.classify().get_labels()
	
	predictions = np.sign(out)
	accuracy = ( predictions == testLabels ).sum() / float(testLabels.shape[0])
	
	if fullOutput:
		return [accuracy, predictions, out]
	else:
		return accuracy

def classifier_libsvm_modular(fm_train_real, fm_test_real, label_train_multiclass, width = 1.7, C = 0.4, epsilon= 1e-5):
	
	from shogun.Features import RealFeatures, Labels
	from shogun.Kernel import GaussianKernel
	from shogun.Classifier import LibSVMMultiClass
	
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	kernel=GaussianKernel(feats_train, feats_train, width)
	
	labels=Labels(label_train_multiclass)
	
	svm=LibSVMMultiClass(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.train()
	
	kernel.init(feats_train, feats_test)
	out = svm.classify().get_labels()
	predictions = svm.classify()
	
	return predictions, svm, predictions.get_labels()

def regression_libsvr_modular(fm_train, fm_test,label_train, width=2.1,C=1,epsilon=1e-5,tube_epsilon=1e-2):

	from shogun.Features import Labels, RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Regression import LibSVR

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)

	kernel=GaussianKernel(feats_train, feats_train, width)
	labels=Labels(label_train)

	svr=LibSVR(C, epsilon, kernel, labels)
	svr.set_tube_epsilon(tube_epsilon)
	svr.train()

	kernel.init(feats_train, feats_test)
	out1=svr.classify().get_labels()
	out2=svr.classify(feats_test).get_labels()

	return out1,out2,kernel

class DecodingOperator(ArrayOperator):
	def __init__(self, inputObject, decoder = 'libSVM', fullOutput = False, **kwargs ):
		"""
		inputObject is an array of voxels x Time. 
		It contains all the feature data. configure function will define the trainings/test partitions on the data and the labels associated with those.
		"""
		super(DecodingOperator, self).__init__(inputObject, **kwargs)
		self.decoder = decoder
		self.fullOutput = fullOutput
		self.dataArray = np.array(self.dataArray, dtype = np.float64)
		
	def decode(self, trainingDataIndices, trainingLabels, testDataIndices, testLabels):
		
		train = self.dataArray[trainingDataIndices].T
		test = self.dataArray[testDataIndices].T
				
		if self.decoder == 'libSVM':
			return libSVMDecoder(train, test, np.asarray(trainingLabels, dtype = np.float64).T, np.asarray(testLabels, dtype = np.float64).T, fullOutput = self.fullOutput)
		elif self.decoder == 'svmLin':
			return svmLinDecoder(train, test, np.asarray(trainingLabels, dtype = np.float64).T, np.asarray(testLabels, dtype = np.float64).T, fullOutput = self.fullOutput)
		elif self.decoder == 'multiclass':
			return classifier_libsvm_modular( train, test, trainingLabels )
		elif self.decoder == 'regression':
			return regression_libsvr_modular( train, test, trainingLabels )
		