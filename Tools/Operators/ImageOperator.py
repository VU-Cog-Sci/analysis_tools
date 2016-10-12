#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Tomas Knapen on 2010-09-23.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import tempfile, logging
import pickle

import scipy as sp
import scipy.fftpack
import numpy as np
import matplotlib.pylab as pl
from math import *
from IPython import embed as shell

from nifti import *
from Operator import *
# import nipy.labs.glm
from Tools.other_scripts.savitzky_golay import *
from scipy.signal import fftconvolve, resample

import bottleneck as bn


class ImageOperator( Operator ):
	"""docstring for ImageOperator"""
	def __init__(self, inputObject, **kwargs):
		"""
		image operator takes a filename, an ndarray, or a niftiImage file object
		and creates internal variable inputObject to always be a nifti image object.
		its data accessible through inputObject.data, and filename inputFileName 
		"""
		super(ImageOperator, self).__init__(inputObject = inputObject, **kwargs)
		# convert inputObject to NiftiImage while generating inputFileName from it, if necessary.
		if self.inputObject.__class__.__name__ == 'str':
			dataFile = NiftiImage(self.inputObject)
			self.inputObject = dataFile
			self.inputFileName = self.inputObject.filename
			self.logger.info('started with ' +os.path.split(self.inputFileName)[-1])
		if self.inputObject.__class__.__name__ == 'ndarray':
		# don't care about file name. will be added for saving, later
		# we will use an in-memory nifti file object as a data container here.
			self.inputArray = self.inputObject
			self.inputObject = NiftiImage(self.inputArray)
			self.logger.info('started with ndarray of shape ' + str(self.inputArray.shape))
			# inputFileName has to be set in the **kwargs now, or it will be the empty string
			if not hasattr(self, 'inputFileName'):
				self.inputFileName = ''
		
		

class ImageMaskingOperator( ImageOperator ):
	"""
	ImageMaskingOperator is used to extract data from those voxels in a nifti file, designated by inputObject,
	that are identified by the mask file, designated by maskObject.
	"""
	def __init__(self, inputObject, maskObject = None, thresholds = [0.0], nrVoxels = [False], outputFileName = False, **kwargs):
		super(ImageMaskingOperator, self).__init__(inputObject = inputObject, **kwargs)
		if maskObject == None:
			raise IOError('mask file not given - without a mask no masking operation')
			self.logger.error('mask file not given - without a mask no masking operation')
			pass
		self.maskObject = maskObject
		
		if self.maskObject.__class__.__name__ == 'str':
			self.maskObject = NiftiImage(self.maskObject)
		if self.maskObject.__class__.__name__ == 'NiftiImage':
			# not typically the filename I'd want to be using
			self.maskFileName = self.maskObject.filename
		if self.maskObject.__class__.__name__ == 'ndarray':
			self.maskObject = NiftiImage(self.maskObject)
		
		# if all thresholds for all mask volumes will be equal:
		if len(thresholds) < self.maskObject.data.shape[0]:
			self.thresholds = [thresholds[0] for t in range(self.maskObject.data.shape[0])]
		elif len(thresholds) == self.maskObject.data.shape[0]:
			self.thresholds = thresholds
		else:
			self.logger.error('dimensions of thresholds argument do not fit with nr of masks in maskobject.')
			
		# if all nrVoxels for all mask volumes will be equal:
		if len(nrVoxels) < self.maskObject.data.shape[0]:
			self.nrVoxels = [nrVoxels[0] for t in range(self.maskObject.data.shape[0])]
		elif len(nrVoxels) == self.maskObject.data.shape[0]:
			self.nrVoxels = nrVoxels
		else:
			self.logger.error('dimensions of nrVoxels argument do not fit with nr of masks in maskobject.')
			
		if not outputFileName:
			self.outputFileName = os.path.splitext(self.inputObject.filename)[0] + '_' + os.path.split(self.maskObject.filename)[-1]
		else:
			self.outputFileName = outputFileName
		
		self.buildData()
	
	def buildData(self):
		"""
		buildData creates the necessary arrays of data in 
		such a way that even single-volume masks will have 4D shapes.
		This makes all masking functions transparent.
		self.maskObject and self.inputObject already exist before calling 
		this method, but afterwards the relevant data are in self.maskData
		and self.inputData.
		"""
		self.maskFrameNames = self.maskObject.description.split('|')
		if self.maskObject.data.shape[0] != len(self.maskFrameNames):
			# the mask has more than one volume of 3D binary masks but doesn't fit with the naming
			self.logger.warning('mask Frame names do not match number of volumes in maskfile, %s != %s. \n This doesn\'t matter if the file wasn\'t made by me.', str(self.maskObject.data.shape[0]), str(len(self.maskFrameNames)) )
		
		# may need separate headers for reference
		self.maskHeader = self.maskObject.header
		self.inputHeader = self.inputObject.header
		print self.maskObject.data.shape
		
		if len(self.maskObject.data.shape) == 3:
			# only one volume of mask data
			self.maskData = self.maskObject.data.reshape(np.concatenate(([1],self.maskObject.data.shape)))
			self.logger.debug('too small mask - reshaped.')
		elif len(self.maskObject.data.shape) == 5:
			self.maskData = self.maskObject.data.reshape(list(self.maskObject.data.shape)[1:])
		elif len(self.maskObject.data.shape) == 4:
			self.maskData = self.maskObject.data
			
		if len(self.inputObject.data.shape) == 3:
			# only one volume of to-be-masked-data
			self.inputData = self.inputObject.data.reshape(np.concatenate(([1],self.inputObject.data.shape)))
		else:
			self.inputData = self.inputObject.data
			
		# test for data shape here
		if self.inputData.shape[-3:] != self.maskData.shape[-3:]:
			# the mask has more than one volume of 3D binary masks but doesn't fit with the naming
			self.logger.warning('mask and input data dimensions do not match')
		
		self.logger.debug('input and mask data dimensions: %s, %s', str(self.inputData.shape), str(self.maskData.shape) )
	
	def applySingleMask(self, whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = False):
		"""
		applySingleMask extracts data from self.inputData using a mask in self.maskData. argument 'flat' affects the shape of the
		array that is returned.
		"""
		
		if nrVoxels:	# if nrVoxels we'll need to set the threshold to reach that nr of Voxels.
			sortedData = np.sort(self.maskData[whichMask].ravel())
			if maskFunction == '__gt__':
				maskThreshold = sortedData[-(nrVoxels+1)]
			elif maskFunction == '__lt__':
				maskThreshold = sortedData[(nrVoxels+1)]
				
		# this piece must be done regardless of nrVoxels or manual threshold setting
		mask = eval('self.maskData[whichMask].' + maskFunction + '(' + str(maskThreshold) + ')')
		self.logger.debug('mask dimensions: %s', str(mask.shape) )
		if flat:
			maskedData = self.inputData[:,mask]
		else:
			maskedData = np.zeros(self.inputData.shape)
			for i in range(self.inputData.shape[0]):
				maskedData[i] = mask * self.inputData[i]
				
		self.logger.debug('data masked, mask # %s, threshold %s resultant array is shaped: %s', str(whichMask), str(maskThreshold), str(maskedData.shape))
		
		return maskedData
	
	def applyAllMasks(self, save = True, maskFunction = '__gt__', flat = False):
		"""docstring for applyAllMasks"""
		if not flat:	# output is in nii file format and shape
			allMaskedData = np.zeros(np.concatenate(([self.maskData.shape[0]],self.inputData.shape)))
			for i in range(self.maskData.shape[0]):
				allMaskedData[i] = self.applySingleMask(i, self.thresholds[i], self.nrVoxels[i], maskFunction, flat = flat)
		
			if save:
				fileName = self.outputFileName + standardMRIExtension
				maskedDataFile = NiftiImage(allMaskedData, self.inputObject.header)
				maskedDataFile.save(fileName)
		
		elif flat:	# flatten output arrays into voxels by time
			allMaskedData = []
			for i in range(self.maskData.shape[0]):
				allMaskedData.append(self.applySingleMask(i, self.thresholds[i], self.nrVoxels[i], maskFunction, flat = flat))
			
			if save:
#				fileName = self.outputFileName + '.pickle'
#				maskedDataFile = open(fileName, 'w')
#				pickle.dump(allMaskedData, maskedDataFile)
				np.save(self.outputFileName + '.npy', allMaskedData)
#				maskedDataFile.close()
				oldDataFile = self.outputFileName + '.pickle'
				try:
					os.system('rm ' + oldDataFile )
				except OSError:
					pass
			
		return allMaskedData
		
	
	def execute(self):
		super(ImageMaskingOperator, self).execute()
		# this will return the allMaskedData into thin air while saving the data to the standard file.
		# if you want the masked data array returned run applySingleMask or applyAllMasks
		self.applyAllMasks()
	
class PercentSignalChangeOperator(ImageOperator):
	"""
	PercentSignalChangeOperator
	does exactly what its name implies
	"""
	def __init__(self, inputObject, outputFileName = None, **kwargs):
		super(PercentSignalChangeOperator, self).__init__(inputObject = inputObject, **kwargs)
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = self.inputObject.filename[:-7] + '_psc.nii.gz'
	
	def execute(self):
		meanImage = self.inputObject.data.mean(axis = 0)
		pscData = (100.0 * (self.inputObject.data / meanImage)) - 100.0
		
		# make sure to remove intensity mapping slope...
		header = self.inputObject.header
		header['scl_inter'] = 0
		header['scl_slope'] = 1 
		
		outputFile = NiftiImage(pscData.astype(np.float32), header)
		outputFile.save(self.outputFileName)
		
class MeanBrainSubtractionOperator(ImageOperator):
	"""
	MeanBrainSubtractionOperator
	subtracts the mean brain value at each timepoint from the timeseries.
	This alleviates temporal oscillations due to for example multiband sequences.
	"""
	def __init__(self, inputObject, outputFileName = None, **kwargs):
		super(MeanBrainSubtractionOperator, self).__init__(inputObject = inputObject, **kwargs)
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = self.inputObject.filename[:-7] + '_mbs.nii.gz'
	
	def execute(self):
		meanBrainTimeseries = self.inputObject.data.reshape((self.inputObject.data.shape[0], -1)).mean(axis = 1)
		outputFile = NiftiImage((self.inputObject.data.reshape((self.inputObject.data.shape[0], -1)).T - meanBrainTimeseries).astype(np.float32).T.reshape(self.inputObject.data.shape), self.inputObject.header)
		outputFile.save(self.outputFileName)
		
	
class ZScoreOperator(ImageOperator):
	"""
	PercentSignalChangeOperator
	does exactly what its name implies
	"""
	def __init__(self, inputObject, outputFileName = None, **kwargs):
		super(ZScoreOperator, self).__init__(inputObject = inputObject, **kwargs)
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = self.inputObject.filename[:-7] + '_Z.nii.gz'

	def execute(self):
		meanImage = bn.nanmean(self.inputObject.data, axis = 0)
		stdImage = bn.nanstd(self.inputObject.data, axis = 0)
		outputFile = NiftiImage(((self.inputObject.data - meanImage) / stdImage).astype(np.float32), self.inputObject.header)
		outputFile.save(self.outputFileName)
		

class SavitzkyGolayHighpassFilterOperator(ImageOperator):
	"""
	SavitzkyGolayHighpassFilterOperator performs a type of highpass filter.
	It uses the local regression smoothing savitzky-golay algorithm to smooth the data,
	which it then subtracts from the data to create a high-pass filtered signal.
	"""
	
	def __init__(self, inputObject, outputFileName = None, **kwargs):
		super(SavitzkyGolayHighpassFilterOperator, self).__init__(inputObject = inputObject, **kwargs)
		
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = self.inputObject.filename[:-7] + '_sgtf.nii.gz'
	
	def configure(self, mask_file = None, TR = 1.5, width = 240, order = 3):
		"""mask_file is a BET mask, width is in seconds, as is TR, and the order is the polynomial order of the sg procedure."""
		self.mask_file = mask_file
		
		self.TR = TR
		self.width = width
		self.order = order
		
		self.logger.info('configured: inputfile %s with mask %s, TR %s'%(self.inputObject, self.mask_file, self.TR))
		self.logger.info('SG parameters: width %s order %s' %(str(self.width), str(self.order)))
	
	def execute(self):
		"""execute the filtering operation."""
		# sg filter needs odd sized window size
		window_size = int(self.width / self.TR)
		if window_size % 2 == 0:
			window_size = window_size + 1
		
		# mask or reshape incoming data
		input_data = self.inputObject.data
		if self.mask_file != None:
			mask = np.array(NiftiImage(self.mask_file).data, dtype = bool)
			input_data = input_data[:,mask]
		else:
			input_data = input_data.reshape((input_data.shape[0], -1))
		
		# per-voxel estimation of sg low-pass and subtract it from the input data
		output_data = np.zeros(input_data.shape)
		percentage_done = 0
		for vox in range(input_data.shape[-1]):
			output_data[:,vox] = input_data[:,vox] - savitzky_golay(input_data[:,vox], window_size = window_size, order = self.order) + input_data[:,vox].mean()
			if percentage_done < ceil(100 * vox / input_data.shape[-1]):
				percentage_done = percentage_done + 1
				self.logger.info( 'voxel # ' + str(vox) + ' done ' + str(percentage_done) + ' percent' )
		
		# reshape filtered data back to original format
		if self.mask_file != None:
			new_output_data = np.zeros(self.inputObject.data.shape)
			new_output_data[:,mask] = output_data
			output_data = new_output_data
		else:
			output_data = output_data.reshape(self.inputObject.data.shape)
		
		# save output file, as np.float32
		output_image = NiftiImage(np.array(output_data, dtype = np.float32))
		output_image.header = self.inputObject.header
		output_image.save(self.outputFileName)
		

# GLM type code..

def doubleGamma(x, a1 = 6, a2 = 12, b1 = 0.9, b2 = 0.9, c = 0.35):
	d1 = a1 * b1
	d2 = a2 * b2
	return np.array([(t/(d1))**a1 * np.exp(-(t-d1)/b1) - c*(t/(d2))**a2 * np.exp(-(t-d2)/b2) for t in x])	

def doubleGamma_with_d(x, a1 = 6, a2 = 12, b1 = 0.9, b2 = 0.9, c = 0.35,d1=5.4,d2=10.8):
	return np.array([(t/(d1))**a1 * np.exp(-(t-d1)/b1) - c*(t/(d2))**a2 * np.exp(-(t-d2)/b2) for t in x])	


def double_gamma(x, a1, sh1, sc1, a2, sh2, sc2 ): 
	return a1 * sp.stats.gamma.pdf(x, sh1, loc=0.0, scale = sc1) + a2 * sp.stats.gamma.pdf(x, sh2, loc=0.0, scale = sc2)


def singleGamma(timepoints, a = 6, b = 0.9):
	d = a * b
	return np.array([(t/(d))**a * exp(-(t-d)/b) for t in timepoints])

class Design(object):
	"""Design represents the design matrix of a given run"""
	def __init__(self, nrTimePoints, rtime, subSamplingRatio = 5):
		self.nrTimePoints = nrTimePoints
		self.rtime = rtime
		self.subSamplingRatio = subSamplingRatio
		
		self.rawDesignMatrix = []
		self.timeValuesForConvolution = np.arange(0,nrTimePoints * self.rtime, 1.0/self.subSamplingRatio)
	
	def addRegressor(self, regressor):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressorValues = np.zeros(self.timeValuesForConvolution.shape[0])
		for event in regressor:
			startTime = event[0]
			endTime = event[0]+event[1]
			regressorValues[(self.timeValuesForConvolution > startTime) * (self.timeValuesForConvolution < endTime)] = event[2]
		self.rawDesignMatrix.append(regressorValues)
		
		return regressorValues
	
	def convolveWithHRF(self, hrfType = 'singleGamma', hrfParameters = {'a': 6, 'b': 0.9}): # hrfType = 'doubleGamma', hrfParameters = {'a1': 6, 'a2': 12, 'b1': 0.9, 'b2': 0.9, 'c': 0.35}
		# hrfType = 'singleGamma', hrfParameters = {'a': 6, 'b': 0.9} OR hrfType = 'doubleGamma', hrfParameters = {a1, sh1, sc1, a2, sh2, sc2} OR 
		"""convolveWithHRF convolves the designMatrix with the specified HRF and build final regressors by resampling to TR times"""
		self.hrfType = hrfType
		if self.rtime > 1000:
			self.rtime /= 1000
		self.hrfKernel = eval(self.hrfType + '(np.arange(0,32,self.rtime/(self.subSamplingRatio)), **hrfParameters)')
		if self.hrfKernel.shape[0] % 2 == 1:
			self.hrfKernel = np.r_[self.hrfKernel, 0]
		# self.hrfKernel/=self.hrfKernel.sum()
		self.designMatrix = np.zeros([len(self.rawDesignMatrix), self.nrTimePoints])
		for i, ds in enumerate(self.rawDesignMatrix):
			kl = self.hrfKernel.shape[0]
			self.dpadded = np.zeros(ds.shape[0] + kl*2)
			self.dpadded[kl:-kl] = ds
			self.intermediate_signal = fftconvolve( self.dpadded, self.hrfKernel, 'full' )[:(-kl+1)]
			self.designMatrix[i,:] = self.intermediate_signal[np.array(np.linspace(kl, self.intermediate_signal.shape[0] - kl, self.nrTimePoints), dtype = int)]
			# self.designMatrix[i,:] = self.intermediate_signal[np.array(np.linspace(0, self.intermediate_signal.shape[0]-1, self.nrTimePoints), dtype = int)]
			if self.intermediate_signal.sum() > 0.0:
				self.example_intermediate_signal = self.intermediate_signal
				self.example_raw_design_signal = self.dpadded
#		self.designMatrix = np.array([sp.convolve(ds, self.hrfKernel, 'full')[:-(self.hrfKernel.shape[0]-1)][::round(self.subSamplingRatio * self.rtime)] for ds in self.rawDesignMatrix]).T
			
	def configure(self, regressors, hrfType = 'singleGamma', hrfParameters = {'a': 6, 'b': 0.9} ):
		"""
		configure takes the design matrix in FSL EV format and sets up the design matrix
		"""
		for reg in regressors:
			self.addRegressor(reg)
		# standard HRF for now - double gamma
		self.convolveWithHRF(hrfType = hrfType, hrfParameters = hrfParameters)
		
class NewDesign(object):
	"""Design represents the design matrix of a given run"""
	def __init__(self, nr_time_points, rtime, sample_duration = 0.01):
		self.nr_time_points = nr_time_points
		self.rtime = rtime
		self.sample_duration = sample_duration
		
		self.raw_design_matrix = []
		self.time_values_for_convolution = np.arange(0,nr_time_points * self.rtime, self.sample_duration)
	
	def add_regressor(self, regressor):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.time_values_for_convolution.shape[0])
		for event in regressor:
			start_time = event[0]
			end_time = event[0]+event[1]
			regressor_values[(self.time_values_for_convolution > start_time) * (self.time_values_for_convolution < end_time)] = event[2]
		self.raw_design_matrix.append(regressor_values)
		
		return regressor_values
	
	def convolve_with_HRF(self, hrf_type = 'singleGamma', hrf_parameters = {'a': 6, 'b': 0.9}): # hrfType = 'doubleGamma', hrfParameters = {'a1': 6, 'a2': 12, 'b1': 0.9, 'b2': 0.9, 'c': 0.35}
		# hrfType = 'singleGamma', hrfParameters = {'a': 6, 'b': 0.9} OR hrfType = 'doubleGamma', hrfParameters = {a1, sh1, sc1, a2, sh2, sc2} OR 
		"""convolveWithHRF convolves the designMatrix with the specified HRF and build final regressors by resampling to TR times"""
		self.hrf_type = hrf_type
		self.hrf_kernel = eval(self.hrf_type + '(np.arange(0,32,self.sample_duration), **hrf_parameters)')
		if self.hrf_kernel.shape[0] % 2 == 1:
			self.hrf_kernel = np.r_[self.hrf_kernel, 0]
		self.hrf_kernel /= self.hrf_kernel.sum()
		self.convolved_design_matrix = np.zeros_like(self.raw_design_matrix)
		for i, ds in enumerate(self.raw_design_matrix):
			self.convolved_design_matrix[i] = fftconvolve( ds, self.hrf_kernel, 'full' )[:self.convolved_design_matrix[i].shape[0]]
		self.design_matrix = resample(self.convolved_design_matrix, self.nr_time_points, axis = 1)
			
	def configure(self, regressors, hrf_type = 'singleGamma', hrf_parameters = {'a': 6, 'b': 0.9} ):
		"""
		configure takes the design matrix in FSL EV format and sets up the design matrix
		"""
		for reg in regressors:
			self.add_regressor(reg)
		# standard HRF for now - double gamma
		self.convolve_with_HRF(hrf_type = hrf_type, hrf_parameters = hrf_parameters)


# class ImageRegressOperator(ImageOperator):
# 	"""
# 	class for running glms on functional data
# 	takes a functional data file and creates a design matrix for it
# 	calculates glm and returns results
# 	"""
# 	def __init__(self, inputObject, regressors, **kwargs):
# 		"""docstring for __init__"""
# 		super(ImageRegressOperator, self).__init__(inputObject = inputObject, **kwargs)
# 		self.design = Design(nrTimePoints = self.inputObject.timepoints, rtime = self.inputObject.rtime)
# 		self.design.configure(regressors)
	
# 	def execute(self, outputFormat = ['betas','sse','rank','sing']):
# 		"""docstring for execute"""
# 		super(ImageRegressOperator, self).execute()
# 		origShape = self.inputObject.data.shape
# 		designShape = self.design.designMatrix.shape
# 		fitData = self.inputObject.data.reshape(self.inputObject.timepoints,-1).astype(np.float64)
# 		design = self.design.designMatrix.astype(np.float64)
# 		# self.betas, self.sse, self.rank, self.sing = sp.linalg.lstsq( design, fitData, overwrite_a = True, overwrite_b = True )
# 		# self.logger.info('regress operator betas & sse shape ' + str(self.betas.shape) + ' ' + str(self.sse.shape) + ' rank ' + str(self.rank) + ' from design shaped ' + str(designShape) + ' and data shaped ' + str(origShape))
# 		# returnDict = {}
# 		# if 'betas' in outputFormat: 
# 		# 	returnDict['betas'] = self.betas.reshape(np.concatenate(([designShape[1]], origShape[1:])))
# 		# if 'sse' in outputFormat:
# 		# 	returnDict['sse'] = self.sse.reshape(origShape[1:])
# 		# if 'rank' in outputFormat:
# 		# 	returnDict['rank'] = self.rank
# 		# if 'sing' in outputFormat:
# 		# 	returnDict['sing'] = self.sing
# 		# return returnDict
		
# 		# using nipy this might become:
# 		model = "ar1"
# 		method = "kalman"
# 		my_glm = nipy.labs.glm.glm.glm()
# 		glm = my_glm.fit(fitData.T, design, method="kalman", model="ar1")
		
	

class Filter1D(object):
	"""
	Filter
	constructs filters in order to pass in fourier space
	band-pass filters should be unimportant for now - will add them later
	"""
	def __init__(self, sampleInterval, signalNrSamples, cutoff):
		self.sampleInterval = sampleInterval
		self.signalNrSamples = signalNrSamples
		self.cutoff = cutoff
		
		self.representedFrequencies = np.fft.fftfreq(n = int(self.signalNrSamples), d = int(self.sampleInterval))[:floor(self.signalNrSamples/2.0)]
		self.f = np.ones(self.signalNrSamples)
		self.thres = max(np.arange(self.f.shape[0])[self.representedFrequencies < self.cutoff])
	

class HighPassFilter1D(Filter1D):
	def __init__(self, sampleInterval, signalNrSamples, cutoff):
		super(HighPassFilter1D, self).__init__(sampleInterval, signalNrSamples, cutoff)
		self.f[:self.thres] = 0.0
		self.f[-self.thres:] = 0.0
		
#		self.f = self.f / self.f.sum()
	

class LowPassFilter1D(Filter1D):
	def __init__(self, sampleInterval, signalNrSamples, cutoff):
		super(LowPassFilter1D, self).__init__(sampleInterval, signalNrSamples, cutoff)
		self.f[self.thres:-self.thres] = 0.0
		
#		self.f = self.f / self.f.sum()
	

class ImageTimeFilterOperator(ImageOperator):
	"""
	class for filtering functional data
	takes a functional data file and creates a filter for it
	execute returns the filtered functional data 
	Uses fft for filtering
	"""
	def __init__(self, inputObject, filterType = 'highpass', **kwargs):
		"""docstring for __init__"""
		super(ImageTimeFilterOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.filterType = filterType
	
	def configure(self, frequency, outputFileName = None):
		"""docstring for configure"""
		self.frequency = frequency
		if self.filterType == 'highpass':
			self.f = HighPassFilter1D(self.inputObject.rtime, self.inputObject.timepoints, self.frequency).f
		elif self.filterType == 'lowpass':
			self.f = LowPassFilter1D(self.inputObject.rtime, self.inputObject.timepoints, self.frequency).f
		self.outputFileName = outputFileName
	
	def execute(self):
		"""docstring for execute"""
		super(ImageTimeFilterOperator, self).execute()
		self.fourierData = sp.fftpack.fft(self.inputObject.data.reshape((self.inputObject.timepoints, -1)), axis = 0)
		self.fourierFilteredData = (self.fourierData.T * self.f).T
		self.backFourierFilteredData = sp.fftpack.ifft(self.fourierFilteredData, axis = 0)
		self.filteredData = self.backFourierFilteredData.reshape(self.inputObject.data.shape).astype(np.float32)
		if self.outputFileName == None:
			outputFile = NiftiImage(self.filteredData, self.inputObject.header)	# data type will be according to the datatype of the input array
			self.outputFileName = self.inputObject.filename[:-7] + '_' + self.filterType[0] + 'p.nii.gz'
			outputFile.setDescription('filtered with ImageTimeFilterOperator: cutoff = ' + str(self.frequency)[:5])
			outputFile.save(self.outputFileName)
		else:
			outputFile = NiftiImage(self.filteredData, self.inputObject.header)
			outputFile.setDescription('filtered with ImageTimeFilterOperator: cutoff = ' + str(self.frequency)[:5])
			outputFile.save(self.outputFileName)
	

