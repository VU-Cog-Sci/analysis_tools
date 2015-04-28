#!/usr/bin/env python
# encoding: utf-8

"""@package Operators
This module offers various methods to process eye movement data

Created by Tomas Knapen on 2010-12-19.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.

More details.
"""
from __future__ import division
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


class StarStimOperator(Operator):
	"""
	StarStimOperator operates on starstim EEG signals, sampled at 500 Hz.
	Its input is a .easy file, which in the session's folders should be a .log file.


	"""
	def __init__(self, inputObject, **kwargs):
		"""inputObject is a filename, channel_names is a list of channel names, 
		referring to electrode positions and acceleromter X, Y or Z only,
		i.e. without mention of time and trigger channel."""
		super(StarStimOperator, self).__init__(inputObject = inputObject, **kwargs)
		if not hasattr(self, 'channel_names'):
			self.logger.error('channel names not defined')

	def easy_to_fif(self, fif_file_name = None, htps_file = None, n_eeg_channels = 8):
		"""read the .easy file into a numpy array, and add this data to a fif file.
		This method strips off the time column and converts the trigger column
		of the easy file to mne compatible events."""
		if fif_file_name == None:
			fif_file_name = os.path.splitext(self.inputObject)[0] + '_raw.fif.gz'
		self.raw_data = np.loadtxt(self.inputObject)

		# rescale eeg channels by z-scoring
		self.logger.info('converting %i channels from nanovolts to volts'%(n_eeg_channels))
		self.raw_data[:,:n_eeg_channels] = self.raw_data[:,:n_eeg_channels] / 1e9
		# self.logger.info('new range for %i channels from %f, %f'%(n_eeg_channels, self.raw_data[:,:n_eeg_channels].min(), self.raw_data[:,:n_eeg_channels].max() ))

		# drop the time column from the data
		raw_data = np.copy(self.raw_data)[:,:-1]
		# and empty the trigger column
		raw_data[:,-1] = 0
		self.channel_names.append('STI 014')

		# prepare info, such as channel types
		ch_types = ['eeg' for a in range(len(self.channel_names))]
		for i in range(len(self.channel_names)):	# accelerator samples are not eeg, but misc type
			if self.channel_names[i] in ['X','Y','Z','STI 014']:
				ch_types[i] = 'misc'
		# prepare info, such as sampling frequency distilled from the time column
		sfreq = 1000.0 / np.median(np.diff(self.raw_data[:,-1]))

		# create mne fif info structure
		info = mne.create_info(ch_names = self.channel_names, sfreq = sfreq, ch_types = ch_types)
		# input the data and info structure in raw array.
		raw = mne.io.RawArray(raw_data.T, info)

		# now create valid-type events for the events channel
		event_samples = np.arange(raw_data.shape[0])[self.raw_data[:,-2] != 0] 
		event_samples = np.concatenate((event_samples, event_samples+1, event_samples+2, event_samples+3, event_samples+4, event_samples+5)) # blowing up the event samples.
		event_values = np.mod(self.raw_data[:,-2][self.raw_data[:,-2] != 0], 100)
		event_values = np.tile(event_values, 6) # to accommodate the event_samples that have been blown up.
		events = np.array([event_samples, np.zeros(event_samples.shape[0]), event_values]).T
		raw.add_events(events)

		# shell()
		# raw.plot(n_channels = 1, block = True, scalings= {'eeg':20e-9})


		if htps_file != None:
			htps_dtype = np.dtype([('name','S10'),('x','f'),('y','f8'),('z','f8')])
			positions = np.loadtxt(htps_file, dtype = htps_dtype)
			all_names = list(positions['name'])
			all_positions = np.array([[positions[positions['name'] == an][pos][0] for pos in ['x','y','z']] for an in all_names])

			raw.set_channel_positions(all_positions, all_names)

		# shell()

		self.logger.info('saving data and events from %s to fif file %s'%(self.inputObject, fif_file_name))
		# save all data to file name
		raw.save(fif_file_name)

	def line_filter_fif(self, fif_input_file_name = None, fif_output_file_name = None, notch_freq = 50):
		if fif_input_file_name == None:
			fif_input_file_name = os.path.splitext(self.inputObject)[0] + '_raw.fif.gz'
		if fif_output_file_name == None:
			fif_output_file_name = os.path.splitext(self.inputObject)[0] + '_notch_raw.fif.gz'
		raw = mne.io.RawFIF(fif_input_file_name, preload=True)
		raw.notch_filter(freqs = np.arange(notch_freq, notch_freq * 3 + 1, notch_freq), n_jobs=8)
		raw.save(fif_output_file_name)

	def filter_fif(self, fif_input_file_name = None, fif_output_file_name = None, l_freq = 0.2, h_freq = 50, l_trans_bandwidth = 0.1, h_trans_bandwidth = 2.5 ):
		if fif_input_file_name == None:
			fif_input_file_name = os.path.splitext(self.inputObject)[0] + '_notch_raw.fif.gz'
		if fif_output_file_name == None:
			fif_output_file_name = os.path.splitext(self.inputObject)[0] + '_bp_notch_raw.fif.gz'
		raw = mne.io.RawFIF(fif_input_file_name, preload=True)
		raw.filter(l_freq = l_freq, h_freq = h_freq, l_trans_bandwidth = l_trans_bandwidth, h_trans_bandwidth = h_trans_bandwidth, n_jobs=8)
		raw.save(fif_output_file_name)

	def psd_quality_control(self, fif_input_file_name = None, l_freq = 0.2, h_freq = 50 ):
		if fif_input_file_name == None:
			fif_input_file_name = os.path.splitext(self.inputObject)[0] + '_bp_notch_raw.fif.gz'
		raw = mne.io.RawFIF(fif_input_file_name, preload=True)
		raw.plot_psds(tmin=0.0, tmax=raw.index_as_time(raw.n_times - 4), fmin=l_freq, fmax=h_freq)
		pl.savefig(os.path.splitext(fif_input_file_name)[0] + '.pdf')





