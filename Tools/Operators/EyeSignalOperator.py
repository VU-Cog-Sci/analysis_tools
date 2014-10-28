#!/usr/bin/env python
# encoding: utf-8

"""@package Operators
This module offers various methods to process eye movement data

Created by Tomas Knapen on 2010-12-19.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.

More details.
"""

import os, sys, subprocess, re
import pickle

import scipy as sp
import numpy as np
import pandas as pd
import numpy.linalg as LA
from Tools.other_scripts.savitzky_golay import *
import matplotlib.pylab as pl
from math import *
from scipy.signal import butter, lfilter, filtfilt, fftconvolve, resample
import scipy.interpolate as interpolate
import scipy.stats as stats
import mne

from Operator import Operator

from IPython import embed as shell

def detect_saccade_from_data(xy_data = None, vel_data = None, l = 5, sample_rate = 1000.0):
	"""Uses the engbert & mergenthaler algorithm (PNAS 2006) to detect saccades.
	
	This function expects a sequence (2 x N) of xy gaze position or velocity data. 
	
	Arguments:
		xy_data (numpy.ndarray, optional): a sequence (2 x N) of xy gaze (float/integer) positions. Defaults to None
		vel_data (numpy.ndarray, optional): a sequence (2 x N) of velocity data (float/integer). Defaults to None.
		l (float, optional):determines the threshold. Defaults to 5 median-based standard deviations from the median
		sample_rate (float, optional) - the rate at which eye movements were measured per second). Defaults to 1000.0
	
	Returns:
		list of dictionaries, which each correspond to a saccade.
		
		The dictionary contains the following items:
			
	Raises:
		ValueError: If neither xy_data and vel_data were passed to the function.
	
	"""
	
	# If xy_data and vel_data are both None, function can't continue
	if xy_data is None and vel_data is None:
		raise ValueError("Supply either xy_data or vel_data")	
	
	minimum_saccade_duration = 0.012 # in s
	
	#If xy_data is given, process it
	if not xy_data is None:
		xy_data = np.array(xy_data)
		# when are both eyes zeros?
		xy_data_zeros = (xy_data == 0.0001).sum(axis = 1)
	
	# Calculate velocity data if it has not been given to function
	if vel_data is None:
		# Check for shape of xy_data. If x and y are ordered in columns, transpose array.
		# Should be 2 x N array to use np.diff namely (not Nx2)
		rows, cols = xy_data.shape
		if rows == 2:
			vel_data = np.diff(xy_data)
		if cols == 2:
			vel_data = np.diff(xy_data.T)
		
	# median-based standard deviation, for x and y separately
	med = np.median(vel_data, axis = 0)
	
	scaled_vel_data = vel_data/np.mean(np.array(np.sqrt((vel_data - med)**2)), axis = 0)
	# normalize and to acceleration and its sign
	if (np.__version__.split('.')[0] == 1) and (np.__version__.split('.')[1] > 6):
		normed_scaled_vel_data = LA.norm(scaled_vel_data, axis = 1)
		normed_vel_data = LA.norm(vel_data, axis = 1)
	else:
		normed_scaled_vel_data = np.array([LA.norm(svd) for svd in np.array(scaled_vel_data)])
		normed_vel_data = np.array([LA.norm(vd) for vd in np.array(vel_data)])
	normed_acc_data = np.r_[0,np.diff(normed_scaled_vel_data)]
	signed_acc_data = np.sign(normed_acc_data)
	
	# when are we above the threshold, and when were the crossings
	over_threshold = (normed_scaled_vel_data > l)
	# integers instead of bools preserve the sign of threshold transgression
	over_threshold_int = np.array(over_threshold, dtype = np.int16)
	
	# crossings come in pairs
	threshold_crossings_int = np.concatenate([[0], np.diff(over_threshold_int)])
	threshold_crossing_indices = np.arange(threshold_crossings_int.shape[0])[threshold_crossings_int != 0]
	
	valid_threshold_crossing_indices = []
	
	# if no saccades were found, then we'll just go on and record an empty saccade
	if threshold_crossing_indices.shape[0] > 1:
		# the first saccade cannot already have started now
		if threshold_crossings_int[threshold_crossing_indices[0]] == -1:
			threshold_crossings_int[threshold_crossing_indices[0]] = 0
			threshold_crossing_indices = threshold_crossing_indices[1:]
	
		# the last saccade cannot be in flight at the end of this data
		if threshold_crossings_int[threshold_crossing_indices[-1]] == 1:
			threshold_crossings_int[threshold_crossing_indices[-1]] = 0
			threshold_crossing_indices = threshold_crossing_indices[:-1]
		
#		if threshold_crossing_indices.shape == 0:
#			break
		# check the durations of the saccades
		threshold_crossing_indices_2x2 = threshold_crossing_indices.reshape((-1,2))
		raw_saccade_durations = np.diff(threshold_crossing_indices_2x2, axis = 1).squeeze()
	
		# and check whether these saccades were also blinks...
		blinks_during_saccades = np.ones(threshold_crossing_indices_2x2.shape[0], dtype = bool)
		for i in range(blinks_during_saccades.shape[0]):
			if np.sum(xy_data_zeros[threshold_crossing_indices_2x2[i,0]-20:threshold_crossing_indices_2x2[i,1]+20]) > 0:
				blinks_during_saccades[i] = False
	
		# and are they too close to the end of the interval?
		right_times = threshold_crossing_indices_2x2[:,1] < xy_data.shape[0]-30
	
		valid_saccades_bool = (((raw_saccade_durations / sample_rate) > minimum_saccade_duration) * blinks_during_saccades ) * right_times
		if type(valid_saccades_bool) != np.ndarray:
			valid_threshold_crossing_indices = threshold_crossing_indices_2x2
		else:
			valid_threshold_crossing_indices = threshold_crossing_indices_2x2[valid_saccades_bool]
	
		# print threshold_crossing_indices_2x2, valid_threshold_crossing_indices, blinks_during_saccades, ((raw_saccade_durations / sample_rate) > minimum_saccade_duration), right_times, valid_saccades_bool
		# print raw_saccade_durations, sample_rate, minimum_saccade_duration
	
		
	
	saccades = []
	for i, cis in enumerate(valid_threshold_crossing_indices):
		# find the real start and end of the saccade by looking at when the acceleleration reverses sign before the start and after the end of the saccade:
		# sometimes the saccade has already started?
		expanded_saccade_start = np.arange(cis[0])[np.r_[0,np.diff(signed_acc_data[:cis[0]] != 1)] != 0]
		if expanded_saccade_start.shape[0] > 0:
			expanded_saccade_start = expanded_saccade_start[-1]
		else:
			expanded_saccade_start = 0
			
		expanded_saccade_end = np.arange(cis[1],np.min([cis[1]+50, xy_data.shape[0]]))[np.r_[0,np.diff(signed_acc_data[cis[1]:np.min([cis[1]+50, xy_data.shape[0]])] != -1)] != 0]
		# sometimes the deceleration continues crazily, we'll just have to cut it off then. 
		if expanded_saccade_end.shape[0] > 0:
			expanded_saccade_end = expanded_saccade_end[0]
		else:
			expanded_saccade_end = np.min([cis[1]+50, xy_data.shape[0]])
		
		try:
			this_saccade = {
				'expanded_start_time': expanded_saccade_start,
				'expanded_end_time': expanded_saccade_end,
				'expanded_duration': expanded_saccade_end - expanded_saccade_start,
				'expanded_start_point': xy_data[expanded_saccade_start],
				'expanded_end_point': xy_data[expanded_saccade_end],
				'expanded_vector': xy_data[expanded_saccade_end] - xy_data[expanded_saccade_start],
				'expanded_amplitude': np.sum(normed_vel_data[expanded_saccade_start:expanded_saccade_end]) / sample_rate,
				'peak_velocity': np.max(normed_vel_data[expanded_saccade_start:expanded_saccade_end]),

				'raw_start_time': cis[0],
				'raw_end_time': cis[1],
				'raw_duration': cis[1] - cis[0],
				'raw_start_point': xy_data[cis[1]],
				'raw_end_point': xy_data[cis[0]],
				'raw_vector': xy_data[cis[1]] - xy_data[cis[0]],
				'raw_amplitude': np.sum(normed_vel_data[cis[0]:cis[1]]) / sample_rate,
			}
			saccades.append(this_saccade)
		except IndexError:
			pass
		
	
	# if this fucker was empty
	if len(valid_threshold_crossing_indices) == 0:
		this_saccade = {
			'expanded_start_time': 0,
			'expanded_end_time': 0,
			'expanded_duration': 0.0,
			'expanded_start_point': [0.0,0.0],
			'expanded_end_point': [0.0,0.0],
			'expanded_vector': [0.0,0.0],
			'expanded_amplitude': 0.0,
			'peak_velocity': 0.0,

			'raw_start_time': 0,
			'raw_end_time': 0,
			'raw_duration': 0.0,
			'raw_start_point': [0.0,0.0],
			'raw_end_point': [0.0,0.0],
			'raw_vector': [0.0,0.0],
			'raw_amplitude': 0.0,
		}
		saccades.append(this_saccade)
	
	return saccades

class EyeSignalOperator(Operator):
	"""EyeSignalOperator operates on eye signals, preferably sampled at 1000 Hz. 
	This operator is just created by feeding it timepoints, eye signals and pupil size signals in separate arrays, on a per-eye basis.
	"""
	def __init__(self, inputObject, **kwargs):
		"""inputObject is a dictionary with timepoints, gazeXY and pupil keys and timeseries as values"""
		super(EyeSignalOperator, self).__init__(inputObject = inputObject, **kwargs)
		self.timepoints = np.array(self.inputObject['timepoints']).squeeze()
		self.raw_gazeXY = np.array(self.inputObject['gazeXY']).squeeze()
		self.raw_pupil = np.array(self.inputObject['pupil']).squeeze()
		
		if hasattr(self, 'eyelink_blink_data'):
			self.blink_dur = np.array(self.eyelink_blink_data['duration']) 
			self.blink_starts = np.array(self.eyelink_blink_data['start_timestamp'])[self.blink_dur<4000]
			self.blink_ends = np.array(self.eyelink_blink_data['end_timestamp'])[self.blink_dur<4000]
			
		if not hasattr(self, 'sample_rate'): # this should have been set as a kwarg, but if it hasn't we just assume a standard 1000 Hz
			self.sample_rate = 1000.0
	
	def blink_detection_pupil(self, coalesce_period = 250, threshold_level = 0.01):
		"""blink_detection_pupil detects blinks in the pupil signal depending on when signals go below threshold_level, dilates these intervals by period coalesce_period"""
		
		if hasattr(self, 'eyelink_blink_data'):
			
			# set all blinks to 0:
			for i in range(len(self.blink_starts)):
				self.raw_pupil[(self.timepoints>self.blink_starts[i])*(self.timepoints<self.blink_ends[i])] = 0
		
			# set all missing data to 0:
			self.raw_pupil[self.raw_pupil<threshold_level] = 0
		
			# we do not want to start or end with a 0:
			pupil_median_start = np.median(self.raw_pupil[:int(self.sample_rate*10)][self.raw_pupil[:int(self.sample_rate*10)]!=0])
			pupil_median_end = np.median(self.raw_pupil[int(self.sample_rate*10):][self.raw_pupil[int(self.sample_rate*10):]!=0])
			self.raw_pupil[:coalesce_period+1] = pupil_median_start
			self.raw_pupil[-coalesce_period+1:] = pupil_median_end
		
			# detect zero edges (we just created from blinks, plus missing data):
			zero_edges = np.arange(self.raw_pupil.shape[0])[np.diff(( self.raw_pupil < threshold_level ))]
			if zero_edges.shape[0] == 0:
				pass
			else:
				zero_edges = zero_edges[:int(2 * np.floor(zero_edges.shape[0]/2.0))].reshape(-1,2)
		
			# check for neighbouring blinks (coalesce_period, default is 250ms), and string them together:
			start_indices = np.ones(zero_edges.shape[0], dtype=bool)
			end_indices = np.ones(zero_edges.shape[0], dtype=bool)
			for i in range(zero_edges.shape[0]):               
				try:
					if zero_edges[i+1,0] - zero_edges[i,1] <= coalesce_period:
						start_indices[i+1] = False
						end_indices[i] = False
				except IndexError:
					pass
		
			# these are the blink start and end samples to work with:
			
			
			
			if sum(start_indices) > 0:
				self.blink_starts = zero_edges[start_indices,0]
				self.blink_ends = zero_edges[end_indices,1]
			else:
				self.blink_starts = None
				self.blink_ends = None
		
		else:
			self.blinks_indices = pd.rolling_mean(np.array(self.raw_pupil < threshold_level, dtype = float), int(coalesce_period)) > 0
			self.blinks_indices = np.array(self.blinks_indices, dtype=int)
			self.blink_starts = self.timepoints[:-1][np.diff(self.blinks_indices) == 1]
			self.blink_ends = self.timepoints[:-1][np.diff(self.blinks_indices) == -1]
		
			# now make sure we're only looking at the blnks that fall fully inside the data stream
			if self.blink_starts[0] > self.blink_ends[0]:
				self.blink_ends = self.blink_ends[1:]
			if self.blink_starts[-1] > self.blink_ends[-1]:
				self.blink_starts = self.blink_starts[:-1]
		
	def interpolate_blinks(self, method = 'linear', lin_interpolation_points = [[-100],[100]], spline_interpolation_points = [[-0.15, -0.075],[0.075, 0.15]]):
		"""interpolate_blinks interpolates blink periods with method, which can be spline or linear. 
		Use after blink_detection_pupil.
		spline_interpolation_points is an 2 by X list detailing the data points around the blinks (in s offset from blink start and end) that should be used for fitting the interpolation spline.
		"""
		import copy
		
		self.interpolated_pupil = copy.copy(self.raw_pupil[:])
		self.interpolated_x = copy.copy(self.raw_gazeXY[:,0])
		self.interpolated_y = copy.copy(self.raw_gazeXY[:,1])
		
		if method == 'spline':
			points_for_interpolation = np.array(np.array(spline_interpolation_points) * self.sample_rate, dtype = int)
			for bs, be in zip(self.blink_starts, self.blink_ends):
				# interpolate
				samples = np.ravel(np.array([bs + points_for_interpolation[0], be + points_for_interpolation[1]]))
				sample_indices = np.arange(self.raw_pupil.shape[0])[np.sum(np.array([self.timepoints == s for s in samples]), axis = 0)]
				spline = interpolate.InterpolatedUnivariateSpline(sample_indices,self.raw_pupil[sample_indices])
				# replace with interpolated data, from the inside points of the interpolation lists. 
				self.interpolated_pupil[sample_indices[0]:sample_indices[-1]] = spline(np.arange(sample_indices[1],sample_indices[-2]))
				spline = interpolate.InterpolatedUnivariateSpline(sample_indices,self.raw_gazeXY[sample_indices,0])
				self.interpolated_x[sample_indices[0]:sample_indices[-1]] = spline(np.arange(sample_indices[1],sample_indices[-2]))
				spline = interpolate.InterpolatedUnivariateSpline(sample_indices,self.raw_gazeXY[sample_indices,1])
				self.interpolated_y[sample_indices[0]:sample_indices[-1]] = spline(np.arange(sample_indices[1],sample_indices[-2]))

		
		elif method == 'linear':
			
			if self.blink_starts != None:
				points_for_interpolation = np.array([self.blink_starts, self.blink_ends], dtype=int).T + np.array(lin_interpolation_points).T
				for itp in points_for_interpolation:
					self.interpolated_pupil[itp[0]:itp[-1]] = np.linspace(self.interpolated_pupil[itp[0]], self.interpolated_pupil[itp[-1]], itp[-1]-itp[0])
					self.interpolated_x[itp[0]:itp[-1]] = np.linspace(self.interpolated_x[itp[0]], self.interpolated_x[itp[-1]], itp[-1]-itp[0])
					self.interpolated_y[itp[0]:itp[-1]] = np.linspace(self.interpolated_y[itp[0]], self.interpolated_y[itp[-1]], itp[-1]-itp[0])
		
	def filter_pupil(self, hp = 0.01, lp = 4.0):
		"""band_pass_filter_pupil band pass filters the pupil signal using a butterworth filter of order 3. after interpolation."""
		# band-pass filtering of signal, high pass first and then low-pass
		# High pass:
		hp_cof_sample = hp / (self.interpolated_pupil.shape[0] / self.sample_rate / 2)
		bhp, ahp = sp.signal.butter(3, hp_cof_sample, btype='high')
		self.hp_filt_pupil = sp.signal.filtfilt(bhp, ahp, self.interpolated_pupil)
		# Low pass:
		lp_cof_sample = lp / (self.interpolated_pupil.shape[0] / self.sample_rate / 2)
		blp, alp = sp.signal.butter(3, lp_cof_sample)
		self.lp_filt_pupil = sp.signal.filtfilt(blp, alp, self.interpolated_pupil)
		# Band pass:
		self.bp_filt_pupil = sp.signal.filtfilt(blp, alp, self.hp_filt_pupil)
	
	def zscore_pupil(self):
		"""zscore_pupil: simple zscoring of pupil sizes."""
		self.bp_pupil_zscore = (self.bp_filt_pupil - self.bp_filt_pupil.mean()) / self.bp_filt_pupil.std() 
		self.lp_pupil_zscore = (self.lp_filt_pupil - self.lp_filt_pupil.mean()) / self.lp_filt_pupil.std() 
	
	def time_frequency_decomposition_pupil(self, min_freq = 0.01, max_freq = 3.0, freq_stepsize = 0.25, n_cycles = 7):
		"""time_frequency_decomposition_pupil uses the mne package to perform a time frequency decomposition on the pupil data after interpolation"""
		
		frequencies = np.arange(min_freq, max_freq, freq_stepsize)  # define frequencies of interest
		n_cycles = frequencies / float(n_cycles)  # different number of cycle per frequency
		
		self.pupil_tf, pl = mne.time_frequency.induced_power(np.array([[self.interpolated_pupil]]), self.sample_rate, frequencies, use_fft=True, n_cycles=n_cycles, decim=3, n_jobs=1, zero_mean=True)
	
