#!/usr/bin/env python
# encoding: utf-8
"""
PRFModel.py

Created by Tomas Knapen on 2013-08-23.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess, re
import pickle

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from math import *
import scipy.stats as stats

from IPython import embed as shell
from Operators.ImageOperator import *

class PRFModelTrial(object):
	"""docstring for PRFModelTrial"""
	def __init__(self, orientation, n_elements, duration, sample_freq, bar_width = 0.1):
		super(PRFModelTrial, self).__init__()
		self.orientation = orientation
		self.n_elements = n_elements
		self.duration = duration
		self.sample_freq = sample_freq
		self.bar_width = bar_width
		self.n_timepoints = int(round(duration / sample_freq) + 1)
		
		self.rotation_matrix = np.matrix([[cos(self.orientation), -sin(self.orientation)],[sin(self.orientation), cos(self.orientation)]])
		
		x, y = np.meshgrid(np.linspace(-1,1,self.n_elements), np.linspace(-1,1,self.n_elements))
		self.xy = np.matrix([x.ravel(), y.ravel()]).T  
		self.rotated_xy = np.array(self.xy * self.rotation_matrix)
		self.ecc_test = (np.array(self.xy) ** 2).sum(axis = 1) < 1.0
	
	def in_bar(self, time = 0):
		"""in_bar"""
		# a bar of self.bar_width width
		position = 2.0 * (((time * (1.0 + self.bar_width)) / self.n_timepoints) - (0.5 + self.bar_width/2.0))
		extent = [-self.bar_width/2.0 + position, self.bar_width/2.0 + position] 
		
		# rotating the xy matrix itself allows us to test only the x component 
		return ((self.rotated_xy[:,0] > extent[0]) * (self.rotated_xy[:,0] < extent[1]) * self.ecc_test).reshape((self.n_elements, self.n_elements))
	
	def pass_through(self, padding = [1.5,3.0]):
		"""pass_through models a single pass-through of the bar, 
		with padding as in the padding list for start and end."""
		self.pass_matrix = np.zeros((self.n_timepoints + int(round(((padding[0] + padding[1]) / self.sample_freq))), self.n_elements, self.n_elements))
		for i in range(self.n_timepoints):
			self.pass_matrix[i + int(round((padding[0])/self.sample_freq))] = self.in_bar(i)
	
class PRFModelRun(object):
	"""docstring for PRFModelRun"""
	def __init__(self, orientation_list, n_elements, duration, sample_freq, bar_width = 0.1, padding = [1.5,3.0]):
		super(PRFModelRun, self).__init__()
		self.orientation_list = orientation_list
		self.n_elements = n_elements
		self.duration = duration
		self.sample_freq = sample_freq
		self.bar_width = bar_width
		self.padding = padding
	
	def simulate_run(self):
		"""docstring for simulate_run"""
		self.run_matrix = np.zeros((len(self.orientation_list), 1 + int(round(((self.duration + self.padding[0] + self.padding[1]) / self.sample_freq))), self.n_elements, self.n_elements))
		for i in range(len(self.orientation_list)):
			pt = PRFModelTrial(orientation = self.orientation_list[i], n_elements = self.n_elements, duration = self.duration, sample_freq = self.sample_freq, bar_width = self.bar_width)
			pt.pass_through(padding = self.padding)
			self.run_matrix[i] = pt.pass_matrix
		self.run_matrix = self.run_matrix.reshape((len(self.orientation_list) * (1 + int(round(((self.duration + self.padding[0] + self.padding[1]) / self.sample_freq)))), self.n_elements, self.n_elements))

class PRFModelDesign(object):
	"""docstring for PRFModelFit"""
	def __init__(self, orientation_lists, n_elements, period, TR, stim_refresh, padding, bar_width):
		super(PRFModelDesign, self).__init__()
		self.orientation_lists = orientation_lists
		self.n_elements = n_elements
		self.period = period
		self.TR = TR
		self.stim_refresh = stim_refresh
		self.period = period
		self.padding = padding
		self.bar_width = bar_width
	
	def design_matrix(self, method = 'gamma', gamma_hrfType = 'singleGamma', gamma_hrfParameters = {'a': 6, 'b': 0.9}, fir_ratio = 6):
		"""design_matrix creates a design matrix for the runs
			method can be gamma or fir. when gamma, we can specify the parameters of gamma and double-gamma, etc.
		"""
		self.stim_matrix_list = []
		self.design_matrix_list = []
		for ol in self.orientation_lists:
			mr = PRFModelRun(ol, self.n_elements, self.period, self.stim_refresh, self.bar_width, self.padding)
			mr.simulate_run()
			self.stim_matrix_list.append(mr.run_matrix)
			
			if method == 'gamma':
				run_design = Design(mr.run_matrix.shape[0], self.stim_refresh)
				run_design.rawDesignMatrix = mr.run_matrix.reshape((mr.run_matrix.shape[0], -1))
				run_design.convolveWithHRF(hrfType = gamma_hrfType, hrfParameters = gamma_hrfParameters)
				workingDesignMatrix = run_design.designMatrix
			elif method == 'fir':
				new_size = list(mr.run_matrix.shape)
				new_size[0] *= int(fir_ratio)
				new_array = np.zeros(new_size)
				for i in np.arange(mr.run_matrix.shape[0]) * int(fir_ratio):
					new_array[i:i+int(fir_ratio)] = mr.run_matrix[i/int(fir_ratio)]
				workingDesignMatrix = new_array
			
			self.design_matrix_list.append(workingDesignMatrix)
		self.full_design_matrix = np.hstack(design_matrix_list)
		
		
		