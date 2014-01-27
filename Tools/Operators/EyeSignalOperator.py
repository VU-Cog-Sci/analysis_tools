#!/usr/bin/env python
# encoding: utf-8
"""
EyeSignalOperator.py

Created by Tomas Knapen on 2010-12-19.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess, re
import pickle

import scipy as sp
import numpy as np
import pandas as pd
from datetime import *
from tables import *
from ..other_scripts.savitzky_golay import *
import matplotlib.pylab as pl
from math import *
from scipy.io import *
from scipy.signal import butter, lfilter, filtfilt, fftconvolve, resample
import scipy.stats as stats
import mne

from Operator import Operator

from IPython import embed as shell


class EyeSignalOperator(Operator):
	"""EyeSignalOperator operates on eye signals, preferably sampled at 1000 Hz. 
	This operator is just created by feeding it timepoints, eye signals and pupil size signals in separate arrays, on a per-eye basis.
	"""
	def __init__(self, inputObject, **kwargs):
		"""inputObject is a dictionary with timepoints, gazeXY and pupil keys and timeseries as values"""
		super(EyeSignalOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.timepoints = self.inputObject['timepoints']
		self.raw_gazeXY = self.inputObject['gazeXY']
		self.raw_pupil = self.inputObject['pupil']
		
		if not hasattr(self, 'sample_rate'): # this should have been set as a kwarg, but if it hasn't we just assume a standard 1000 Hz
			self.sample_rate = 1000.0
	
	def blink_detection_pupil(self, coalesce_period = 0.25, threshold_level = 0.01):
		"""blink_detection_pupil detects blinks in the pupil signal depending on when signals go below threshold_level, dilates these intervals by period coalesce_period"""
		self.blinks_indices = pd.rolling_mean(np.array(self.raw_pupil < threshold_level, dtype = float), int(coalesce_period * self.sample_rate)) > 0
		self.blink_starts = self.timepoints[:-1][np.diff(self.blinks_indices) == 1]
		self.blink_ends = self.timepoints[:-1][np.diff(self.blinks_indices) == -1]
		
		# now make sure we're only looking at the blnks that fall fully inside the data stream
		if self.blink_starts[0] > self.blink_ends[0]:
			self.blink_ends = self.blink_ends[1:]
		if self.blink_starts[-1] > self.blink_ends[-1]:
			self.blink_starts = self.blink_starts[:-1]
	
	def interpolate_blinks(self, method = 'spline', spline_interpolation_points = [[-0.15, -0.075],[0.075, 0.15]]):
		"""interpolate_blinks interpolates blink periods with method, which can be spline or linear. 
		Use after blink_detection_pupil.
		spline_interpolation_points is an 2 by X list detailing the data points around the blinks (in s offset from blink start and end) that should be used for fitting the interpolation spline.
		"""
		
		self.interpolated_pupil = self.raw_pupil
		
		if method == 'spline':
			points_for_interpolation = np.array(np.array(spline_interpolation_points) * self.sample_rate, dtype = int)
			
			for bs, be in zip(self.blink_starts, self.blink_ends):
				# interpolate
				samples = np.ravel(np.array([bs + points_for_interpolation[0], be + points_for_interpolation[1]]))
				sample_indices = np.arange(self.raw_pupil.shape)[np.sum(np.array([self.time_points == s for s in samples]), axis = 0)]
				spline = interpolate.InterpolatedUnivariateSpline(itp,self.raw_pupil[sample_indices])
				# replace with interpolated data, from the inside points of the interpolation lists. 
				self.interpolated_pupil[sample_indices[0]:sample_indices[-1]] = spline[sample_indices[1]:sample_indices[-2]]
		
		elif method == 'linear':
			for bs, be in zip(self.blink_starts, self.blink_ends):
				samples = [bs, be]
				sample_indices = np.arange(self.raw_pupil.shape)[np.sum(np.array([self.time_points == s for s in samples]), axis = 0)]
				step = self.raw_pupil[sample_indices[1]] - self.raw_pupil[sample_indices[0]]
				self.interpolated_pupil[sample_indices[0]:sample_indices[1]] = self.raw_pupil[sample_indices[0]] + np.arange(sample_indices[1] - sample_indices[0]) * step
	
	def band_pass_filter_pupil(self, hp = 5.0, lp = 0.05):
		"""band_pass_filter_pupil band pass filters the pupil signal using a butterworth filter of order 3. after interpolation."""
		# band-pass filtering of signal, high pass first and then low-pass
		hp_cof_sample = hp / (self.raw_pupil.shape[0] / (self.sample_rate / 2))
		bhp, ahp = butter(3, hp_cof_sample, btype = 'high')
		
		hp_c_pupil_size = filtfilt(bhp, ahp, self.interpolated_pupil)
		
		lp_cof_sample = lp / (self.raw_pupil.shape[0] / (self.sample_rate / 2))
		blp, alp = butter(3, lp_cof_sample, btype = 'high')
		
		self.bp_filt_pupil = filtfilt(blp, alp, hp_c_pupil_size)
	
	def zscore_pupil(self):
		"""zscore_pupil: simple zscoring of pupil sizes."""
		self.pupil_zscore = (self.bp_filt_pupil - self.bp_filt_pupil.mean()) / self.bp_filt_pupil.std() 
	
	def time_frequency_decomposition_pupil(self, min_freq = 0.01, max_freq = 3.0, freq_stepsize = 0.25, n_cycles = 7):
		"""time_frequency_decomposition_pupil uses the mne package to perform a time frequency decomposition on the pupil data after interpolation"""
		
		frequencies = np.arange(min_freq, max_freq, freq_stepsize)  # define frequencies of interest
		n_cycles = frequencies / float(n_cycles)  # different number of cycle per frequency
		
		self.pupil_tf, pl = mne.time_frequency.induced_power(np.array([[self.interpolated_pupil]]), self.sample_rate, frequencies, use_fft=True, n_cycles=n_cycles, decim=3, n_jobs=1, zero_mean=True)
	