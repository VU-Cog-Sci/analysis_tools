#!/usr/bin/env python
# encoding: utf-8
"""
EyeOperator.py

Created by Tomas Knapen on 2010-12-19.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess, re
import pickle

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from math import *
from scipy.io import *
from scipy.signal import butter, lfilter, filtfilt, fftconvolve, resample
import scipy.stats as stats

from nifti import *
from Operator import Operator
from datetime import *
from tables import *
from ..savitzky_golay import *

from BehaviorOperator import NewBehaviorOperator
from IPython import embed as shell

class PhysioOperator( Operator ):
	"""docstring for ImageOperator"""
	def __init__(self, inputObject, **kwargs):
		"""
		PhysioOperator operator takes a filename
		"""
		super(PhysioOperator, self).__init__(inputObject = inputObject, **kwargs)
		if self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
		self.logger.info('started with ' +os.path.split(self.inputFileName)[-1])
		self.read_file()
		
	def read_file(self, sample_rate = 500.0):
		"""read_file(self) reads the physio log file and separates the data, finding its beginning and end."""
		# columns = ['v1raw', 'v2raw',  'v1', 'v2',  'ppu', 'resp',  'gx', 'gy', 'gz', 'mark']
		self.logger.info('reading log file for physio data')
		self.columns = ['v1raw', 'v2raw',  'v1', 'v2',  'ppu', 'resp',  'gx', 'gy', 'gz', 'mark']
		self.sample_rate = sample_rate
		
		self.log_data = np.loadtxt(self.inputFileName)
		self.start_index = np.arange(self.log_data.shape[0])[self.log_data[:,-1] == 10][-1]
		self.end_index = np.arange(self.log_data.shape[0])[self.log_data[:,-1] == 20][-1]
	
	def filter_resp(self, hp_frequency = 0.05, lp_frequency = 0.5):
		"""docstring for filter_resp"""
		# band-pass filtering of resp signal, high pass first and then low-pass parameters
		self.logger.info('band-pass filtering resp signals and normalizing them')
		
		hp_cof_sample = hp_frequency / (self.log_data.shape[0] / (self.sample_rate / 2))
		bhp, ahp = butter(3, hp_cof_sample, btype = 'high')
		
		lp_cof_sample = lp_frequency / (self.log_data.shape[0] / (self.sample_rate / 2))
		blp, alp = butter(3, lp_cof_sample)
		
		# filter operations and normalization after filtering
		hp_c_resp = filtfilt(bhp, ahp, self.log_data[:,self.columns.index('resp')])
		self.bp_resp = filtfilt(blp, alp, hp_c_resp)
		self.continuous_bp_resp_norm = (self.bp_resp - self.bp_resp.min()) / (self.bp_resp.max() - self.bp_resp.min())
		self.continuous_bp_resp_norm = self.continuous_bp_resp_norm - self.continuous_bp_resp_norm.mean()
	
	def filter_ppu(self, filter_width = 5.0, filter_sample_width = 10000):
		"""docstring for filter_ppu"""
		self.logger.info('filtering ppu signals')
		# filter setup
		smooth_width = filter_width * self.sample_rate
		kern = stats.norm.pdf( np.arange(filter_sample_width) - filter_sample_width/2.0, loc = 0, scale = smooth_width )
		kern = kern / kern.sum()
		
		# data setup - we take the scanner's detected heartbeats
		heart_beat_diracs = np.array(self.log_data[:,self.columns.index('mark')] == 2.0, dtype = float)
		self.continuous_ppu_signal = fftconvolve( heart_beat_diracs, kern, 'full' )[kern.shape[0]/2:-kern.shape[0]/2]
	
	def convolve_hrf(self, data, which_rf = 'gamma', filter_length = 20.0):
		"""docstring for convolve_hrf"""
		from ImageOperator import singleGamma
		
		padded_data = np.zeros(data.shape[0] + filter_length * self.sample_rate -1 )
		padded_data[filter_length * self.sample_rate / 2:filter_length * self.sample_rate / 2 + data.shape[0]] = data
		
		t = np.linspace(0, filter_length, filter_length * self.sample_rate, endpoint = False)
		if which_rf == 'gamma':
			hrf = singleGamma(t)
		elif which_rf == 'rrf':	
			# respiration response function from 
			# Birn, R., Smith, M., Jones, T. & Bandettini, P. Neuroimage (2008).
			hrf = 0.6 * t ** 2.1 * np.exp(-t/1.6) - 0.0023 * t ** 3.54 * np.exp(-t/4.25)
		elif which_rf == 'crf':
			# cardiac response function from 
			# Chang, C., Cunningham, J. P. & Glover, G. H. Neuroimage 44, 857–869 (2009).
			hrf = (0.6 * t ** 2.7) * np.exp(-t/1.6) - 16.0 * (1.0/np.sqrt(2*pi*9)) * np.exp(-0.5*((t-12.0)*(t-12.0))/9)
			
		return fftconvolve(padded_data, hrf, 'full')[filter_length * self.sample_rate / 2:filter_length * self.sample_rate / 2 + data.shape[0]]
	
	def find_scan_interval_by_gradients(self, which_gradients = ['x','y','z'], TR = 1.5, nr_TRs = 835):
		"""find_scan_interval_by_gradients finds the interval of scanning by using the gradient channels. 
		which_gradient_channels sums different gradient channels. """
		self.logger.info('detecting signal period from gradient samples')
		channel_dict = {'x': -4, 'y': -3, 'z': -2}
		gradient_signal = np.abs(self.log_data[:,[channel_dict[g] for g in which_gradients]]).sum(axis = 1)
		# what was the last sample in which there were no gradient pulses? 
		# That is our last TR end.
		self.end_index = np.arange(gradient_signal.shape[0])[np.array(np.diff(gradient_signal != 0), dtype = bool)][-1]
		self.start_index = self.end_index - float(TR * nr_TRs * self.sample_rate)
	
	def preprocess_to_continuous_signals(self, TR = 1.5, nr_TRs = 834, hp_frequency = 0.01, lp_frequency = 4.0, filter_width = 3.0, filter_sample_width = 10000, sg_width = 241, sg_order = 3):
		
		self.filter_resp(hp_frequency = hp_frequency, lp_frequency = lp_frequency)
		self.filter_ppu(filter_width = filter_width, filter_sample_width = filter_sample_width)
		
		self.logger.info('convolving ppu and resp data with hrf')
		self.continuous_bp_resp_norm_hrf = self.convolve_hrf(self.continuous_bp_resp_norm - self.continuous_bp_resp_norm.mean(), which_rf = 'rrf', filter_length = 40.0)
		self.continuous_ppu_signal_hrf = self.convolve_hrf(self.continuous_ppu_signal - self.continuous_ppu_signal.mean(), which_rf = 'crf', filter_length = 30.0)
		
		self.find_scan_interval_by_gradients()
		
		self.logger.info('resampling physio data to TRs')
		# resampling is okay like this, picking single indices, because it's all very low-pass filtered already
		self.resamples = np.array(np.round(np.linspace(self.start_index, self.end_index, nr_TRs, endpoint = False)), dtype = int)
		self.res_continuous_bp_resp_signal_hrf = self.continuous_bp_resp_norm_hrf[self.resamples]
		self.res_continuous_ppu_signal_hrf = self.continuous_ppu_signal_hrf[self.resamples]
		
		self.logger.info('detrend physio signal with savitzky-golay filter as a high-pass filter')
		self.detrended_res_continuous_bp_resp_signal_hrf = self.res_continuous_bp_resp_signal_hrf - savitzky_golay(self.res_continuous_bp_resp_signal_hrf, sg_width, sg_order)
		self.detrended_res_continuous_ppu_signal_hrf = self.res_continuous_ppu_signal_hrf - savitzky_golay(self.res_continuous_ppu_signal_hrf, sg_width, sg_order)
		
		self.detrended_res_continuous_bp_resp_signal_hrf = self.detrended_res_continuous_bp_resp_signal_hrf / self.detrended_res_continuous_bp_resp_signal_hrf.std()
		self.detrended_res_continuous_ppu_signal_hrf = self.detrended_res_continuous_ppu_signal_hrf / self.detrended_res_continuous_ppu_signal_hrf.std()
	
		f = pl.figure(figsize = (15,3))
		s = f.add_subplot(111)
		pl.plot(self.detrended_res_continuous_bp_resp_signal_hrf, label = 'resp')
		pl.plot(self.detrended_res_continuous_ppu_signal_hrf, label = 'hr')
		s.set_title('resp and hr signals from file ' + os.path.split(self.inputFileName)[-1])
		s.set_xlabel('time [TRs]')
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (j, l) in enumerate(leg.get_lines()):
				l.set_linewidth(3.5)  # the legend line width
		pl.savefig(os.path.splitext(self.inputFileName)[0] + '.pdf')
		np.savetxt(os.path.splitext(self.inputFileName)[0] + '_resp.txt', self.detrended_res_continuous_bp_resp_signal_hrf.T, delimiter = '\t', fmt = '%3.2f')
		np.savetxt(os.path.splitext(self.inputFileName)[0] + '_ppu.txt', self.detrended_res_continuous_ppu_signal_hrf.T, delimiter = '\t', fmt = '%3.2f')




