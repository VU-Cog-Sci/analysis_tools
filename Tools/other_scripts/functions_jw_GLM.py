from __future__ import division

import numpy as np
import scipy as sp
import sympy
import matplotlib.pyplot as plt
import statsmodels.api as sm

import math

from IPython import embed as shell

class GeneralLinearModel(object):
	"""Design represents the design matrix of a given run"""
	def __init__(self, input_object, event_object, sample_dur=2, new_sample_dur=1):
		
		# variables:
		self.input_object = input_object
		self.event_object = event_object
		self.sample_dur = sample_dur
		self.new_sample_dur = new_sample_dur
		self.resample_ratio = sample_dur / new_sample_dur
		self.timepoints = np.arange(0, input_object.shape[0]*sample_dur, new_sample_dur)
		self.raw_design_matrix = []
		
	def configure(self, IRF='pupil', IRF_params=None, regressor_types='stick', IRF_dt=False, subsample=False):
		
		# resample input_object:
		if subsample:
			self.resample_input_object()
		else:
			self.working_data_array = self.input_object
		
		# create raw regressors, and add them to self.raw_design_matrix:
		for i, reg in enumerate(self.event_object):
			if regressor_types[i] == 'stick':
				self.add_stick_regressor(np.atleast_2d(reg))
			if regressor_types[i] == 'box':
				self.add_box_regressor(np.atleast_2d(reg))
			if regressor_types[i] == 'upramp':
				self.add_upramp_regressor(np.atleast_2d(reg))
			if regressor_types[i] == 'downramp':
				self.add_downramp_regressor(np.atleast_2d(reg))
		
		self.IRF_dt = IRF_dt
		
		# create IRF:
		if IRF == 'pupil':
			self.IRF = self.IRF_pupil(dur=IRF_params['dur'], s=IRF_params['s'], n=IRF_params['n'], tmax=IRF_params['tmax'])
		if IRF == 'BOLD':
			self.IRF = self.HRF(dur=IRF_params['dur'])
		else:
			self.IRF = IRF
		
		# convolve raw regressors with IRF to obtain the full design matrix:
		self.convolve_with_IRF()
		self.z_score()
		
	def resample_input_object(self):
		"""resample_input_object takes a timeseries of data points and resamples them according to the ratio between sample duration and the new sample duration."""
		
		self.working_data_array = sp.signal.resample(self.input_object, self.timepoints.shape[0])
		
	def convolve_with_IRF(self):
		"""convolve_wit_IRF convolves the designMatrix with the specified IRF (sampled according to resample_ratio)"""
		
		print
		print len(self.IRF)
		
		self.design_matrix = np.zeros([len(self.raw_design_matrix)*len(self.IRF), self.timepoints.shape[0]])
		i = 0
		for reg in self.raw_design_matrix:
			for IRF in self.IRF:
				self.design_matrix[i,:] = (sp.signal.fftconvolve(reg, IRF, 'full'))[:-(IRF.shape[0]-1)]
				i += 1
	
	def z_score(self):
		"""z scores design matrix"""
		
		for i in range(self.design_matrix.shape[0]):
			self.design_matrix[i,:] = (self.design_matrix[i,:] - self.design_matrix[i,:].mean()) / self.design_matrix[i,:].std()
	
	def execute(self):
		
		# GLM:
		GLM = sm.GLM(self.working_data_array,self.design_matrix.T)
		GLM_results = GLM.fit()
		GLM_results.summary()
		
		# betas:
		self.betas = GLM_results.params
		
		# predicted signal:
		self.predicted = GLM_results.predict()
		
		# residuals:
		self.residuals = self.working_data_array - self.predicted
		
	# --------------------------------------
	# A variety of regressor shapes:       -
	# --------------------------------------
	
	def add_stick_regressor(self, regressor):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.timepoints.shape[0])
		for event in regressor:
			start_time = np.floor((event[0]+event[1])/self.new_sample_dur)
			regressor_values[start_time] = event[2]
		self.raw_design_matrix.append(regressor_values)
	
	def add_box_regressor(self, regressor):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.timepoints.shape[0])
		for event in regressor:
			start_time = np.floor(event[0]/self.new_sample_dur)
			end_time = np.floor((event[0]+event[1])/self.new_sample_dur)
			dur = sum((self.timepoints > start_time) * (self.timepoints < end_time))
			height = np.linspace(event[2]/dur, event[2]/dur, dur)
			regressor_values[(self.timepoints > start_time) * (self.timepoints < end_time)] = height
		self.raw_design_matrix.append(regressor_values)
	
	def add_upramp_regressor(self, regressor):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.timepoints.shape[0])
		for event in regressor:
			start_time = event[0]
			end_time = event[0]+event[1]
			dur = sum((self.timepoints > start_time) * (self.timepoints < end_time))
			height = np.linspace(0, (event[2]*2/dur), dur)
			regressor_values[(self.timepoints > start_time) * (self.timepoints < end_time)] = height
		self.raw_design_matrix.append(regressor_values)
		
	def add_downramp_regressor(self, regressor):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.timepoints.shape[0])
		for event in regressor:
			start_time = event[0]
			end_time = event[0]+event[1]
			dur = sum((self.timepoints > start_time) * (self.timepoints < end_time))
			height = np.linspace((event[2]*2/dur), 0, dur)
			regressor_values[(self.timepoints > start_time) * (self.timepoints < end_time)] = height
		self.raw_design_matrix.append(regressor_values)
	
	# --------------------------------------
	# Impulse Response Functions (IRF)     -
	# --------------------------------------
	
	def HRF(self, dur=25, a1=6.0, a2=12.0, b1=0.9, b2=0.9, c=0.35):
		
		# parameters:
		timepoints = np.arange(0, dur, self.new_sample_dur)
		
		# sympy variable:
		t = sympy.Symbol('t')
		
		# function:
		d1 = a1 * b1
		d2 = a2 * b2
		y = ( (t/(d1))**a1 * sympy.exp(-(t-d1)/b1) - c*(t/(d2))**a2 * sympy.exp(-(t-d2)/b2) )
		
		# derivative:
		y_dt = y.diff(t)
		
		# lambdify:
		y = sympy.lambdify(t, y, "numpy")
		y_dt = sympy.lambdify(t, y_dt, "numpy")
		
		# evaluate:
		y = y(timepoints)
		y_dt = y_dt(timepoints)
		
		if self.IRF_dt:
			return [y/np.std(y), y_dt/np.std(y_dt)]
		else:
			return [y/np.std(y)]

	def IRF_pupil(self, dur=3, s=1.0/(10**26), n=10.1, tmax=.930):
		"""
		Canocial pupil impulse fucntion [ref]: 
		"""
		
		# parameters:
		timepoints = np.arange(0, dur, self.new_sample_dur)
		
		# sympy variable:
		t = sympy.Symbol('t')
		
		# function:
		y = ( (s) * (t**n) * (math.e**((-n*t)/tmax)) )
		
		# derivative:
		y_dt = y.diff(t)
		
		# lambdify:
		y = sympy.lambdify(t, y, "numpy")
		y_dt = sympy.lambdify(t, y_dt, "numpy")
		
		# evaluate:
		y = y(timepoints)
		y_dt = y_dt(timepoints)
		
		if self.IRF_dt:
			return [y/np.std(y), y_dt/np.std(y_dt)]
		else:
			return [y/np.std(y)]