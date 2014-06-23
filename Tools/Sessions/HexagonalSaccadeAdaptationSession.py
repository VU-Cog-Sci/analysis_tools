#!/usr/bin/env python
# encoding: utf-8
"""
EyeLinkSession.py

Created by Tomas Knapen on 2011-04-27.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import os, math

import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as pl
import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
from scipy.optimize import curve_fit
from scipy import stats, polyval, polyfit
from lmfit import minimize, Parameters, Parameter, report_fit

from joblib import Parallel, delayed
import itertools
from itertools import chain


import logging, logging.handlers, logging.config

from ..log import *
from ..Operators import EDFOperator, HDFEyeOperator, EyeSignalOperator
from ..Operators.EyeSignalOperator import detect_saccade_from_data
from ..Operators.CommandLineOperator import ExecCommandLine
from ..other_scripts.plotting_tools import *
from ..other_scripts.circularTools import *

from IPython import embed as shell

class HexagonalSaccadeAdaptationSession(object):
	"""HexagonalSaccadeAdaptationSession"""
	def __init__(self, subject, experiment_name, project_directory, conditions, loggingLevel = logging.DEBUG):
		self.subject = subject
		self.experiment_name = experiment_name
		self.conditions = conditions
		try:
			os.mkdir(os.path.join( project_directory, experiment_name ))
			os.mkdir(os.path.join( project_directory, experiment_name, self.subject.initials ))
		except OSError:
			pass
		self.project_directory = project_directory
		self.base_directory = os.path.join( self.project_directory, self.experiment_name, self.subject.initials )
		
		self.create_folder_hierarchy()
		self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
		self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
		
		self.nr_trials_per_block = 150
		self.nr_blocks = 8
		self.block_trial_indices = np.array([np.arange(0,self.nr_trials_per_block) + (i*self.nr_trials_per_block) for i in range(self.nr_blocks)])
		
		self.all_block_colors = ['k', 'r', 'g', 'r', 'g', 'r', 'g', 'k']
		
		self.velocity_profile_duration = self.signal_profile_duration = 100
		# add logging for this session
		# sessions create their own logging file handler
		self.loggingLevel = loggingLevel
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(self.loggingLevel)
		addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.base_directory, 'log', 'sessionLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		self.logger.info('starting analysis in ' + self.base_directory)
	
	def create_folder_hierarchy(self):
		"""createFolderHierarchy does... guess what."""
		this_dir = self.project_directory
		for d in [self.experiment_name, self.subject.initials]:
			try:
				this_dir = os.path.join(this_dir, d)
				os.mkdir(this_dir)
			except OSError:
				pass
		for p in ['raw','processed','figs','log']:
			try:
				os.mkdir(os.path.join(self.base_directory, p))
			except OSError:
				pass
	
	def import_raw_data(self, edf_files, aliases):
		"""import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""
		for edf_file, alias in zip(edf_files, aliases):
			self.logger.info('importing file ' + edf_file + ' as ' + alias)
			ExecCommandLine('cp "' + edf_file + '" "' + os.path.join(self.base_directory, 'raw', alias + '.edf"') )

	def import_all_data(self, aliases):
		"""import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
		for alias in aliases:
			self.ho.add_edf_file(os.path.join(self.base_directory, 'raw', alias + '.edf'))
			self.ho.edf_message_data_to_hdf(alias = alias)
			self.ho.edf_gaze_data_to_hdf(alias = alias)
	
	def detect_all_saccades(self, alias, n_jobs = -1, threshold = 5.0):
		"""docstring for detect_all_saccades"""
		self.logger.info('starting saccade detection of ' + alias)
		all_saccades = []
		for bi, tb in enumerate(self.block_trial_indices):
			this_block_eye = self.ho.eye_during_trial(trial_nr = tb[0], alias = alias)
			this_block_sr = self.ho.sample_rate_during_trial(trial_nr = tb[0], alias = alias)
			this_block_res = []
			for tbe in this_block_eye:
				xy_data = [self.ho.signal_from_trial_phases(trial_nr = tr, trial_phases = [1,3], alias = alias, signal = 'gaze', requested_eye = tbe, time_extensions = [0,200]) for tr in tb]
				vel_data = [self.ho.signal_from_trial_phases(trial_nr = tr, trial_phases = [1,3], alias = alias, signal = 'vel', requested_eye = tbe, time_extensions = [0,200]) for tr in tb]
				res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(detect_saccade_from_data)(xy, vel, l = threshold, sample_rate = this_block_sr ) for xy, vel in zip(xy_data, vel_data))
				# res = [detect_saccade_from_data(xy, vel, l = threshold, sample_rate = this_block_sr ) for xy, vel in zip(xy_data, vel_data)]
				for (tr, r) in zip(tb,res):
					r[0].update({'trial': tr, 'eye': tbe, 'block': bi})
				this_block_res.append([r[0] for r in res])
			all_saccades.append(this_block_res)
		all_saccades = list(chain.from_iterable(all_saccades))
		all_saccades_pd = pd.DataFrame(list(chain.from_iterable(all_saccades)))
		self.ho.data_frame_to_hdf(alias = alias, name = 'saccades_per_trial', data_frame = all_saccades_pd)
		
	def trial_selection(self, alias, no_std_vel_cutoff = 3, amp_range = [5,15], degree_cutoff = 2):
		"""
		select_trials returns a boolean array with 1's for trials to use. Trials are deselected when:
		1. there was a blink in trialphase 2 of that trial (the phase where the script polls for a saccade)
		2. the amplitude of the saccade was below or above the 'amp_range' thresholds
		3. the peak velocity is more than 'no_std_vel_cutoff' std from mean
		4. the starting gaze position is more than 'degree_cutoff' distance off
		"""
		
		self.logger.info('starting bad trial detection of ' + alias + '...')
		
		# import required datafiles
		with pd.get_store(self.ho.inputObject) as h5_file:
			saccade_table = h5_file['%s/saccades_per_trial'%alias]
			trial_phases = h5_file['%s/trial_phases'%alias]
			blinks = h5_file['%s/blinks_from_message_file'%alias]
			params = h5_file['%s/parameters'%alias]
				
		# initialize outcome array
		trials2use = np.ones(saccade_table.shape[0])	
		
		# 1. deselect trial when the blink onset was within trial phase 2 of that trial:
		phase_2_start_times = np.array(trial_phases.trial_phase_EL_timestamp[trial_phases.trial_phase_index==2])
		phase_2_end_times = np.array(trial_phases.trial_phase_EL_timestamp[trial_phases.trial_phase_index==3])
				
		for blink_start in blinks.start_timestamp:
 			if blink_start < phase_2_end_times[np.argmax(phase_2_start_times > blink_start)]:
 				trials2use[np.argmax(phase_2_start_times > blink_start)] = 0
		c_blink = int(saccade_table.shape[0] - sum(trials2use))
				
		# 2. deselect trial when: peak velocity more than no_std_vel_cutoff std off the mean
		for bi in range(self.nr_blocks):
			trials_this_block = np.arange(bi*self.nr_trials_per_block, bi*self.nr_trials_per_block+self.nr_trials_per_block)
			mean_vel = np.median(saccade_table.peak_velocity[trials_this_block])
			std_vel = no_std_vel_cutoff*np.std(saccade_table.peak_velocity[trials_this_block]) 
			vel_range = [int(mean_vel-std_vel), int(mean_vel+std_vel)]
			for ti, vel in enumerate(saccade_table.peak_velocity[trials_this_block]):
				if np.any([vel < vel_range[0], vel > vel_range[1]]):
					trials2use[ti] = 0
		c_vel = int(saccade_table.shape[0] - sum(trials2use) - c_blink)
		
		# 3. deselect trial when amplitude falls within the 'amp_range'
		for i, amp in enumerate(saccade_table.expanded_amplitude):
			if np.any([amp < amp_range[0], amp > amp_range[1]]):
				trials2use[i] = 0
		c_amp = int(saccade_table.shape[0] - sum(trials2use) - c_blink - c_vel)
		
		# 4. deselect trials when the gaze starting position is more than 'degree_cutoff' degrees off
		# because the original dot locations were not saved to the parameters file, I recreate them here:
		physical_screen_size = (40, 30)
		physical_screen_distance = 50
		screen_resolution = (800,600)
		amplitude = 10
		n_directions = 6
		screen_height_degrees = 2.0 * 180.0/np.pi * math.atan((physical_screen_size[1]/2.0)/physical_screen_distance)
		pixels_per_degree = (screen_resolution[1]) / screen_height_degrees
		dot_directions = (np.arange(n_directions) * 2.0 * np.pi) / float(n_directions)
		dot_xy = np.array([[-np.sin(d), -np.cos(d)] for d in dot_directions])
		dot_xy_screen = amplitude * pixels_per_degree * dot_xy
		
		# shell()
		
		# now that we know the dot positions, we can compute the actual positions by subtracting the offsets and deselect unwanted trials
		gaze_offset = np.zeros(saccade_table.shape[0])
		for ti in range(np.size(saccade_table,0)):
			actual_dot_position = [dot_xy_screen[params.dot_index[ti]][0] + params.stim_offset_pre_sacc_x[ti], dot_xy_screen[params.dot_index[ti]][1] + params.stim_offset_pre_sacc_y[ti]]
			gaze_offset[ti] = LA.norm([actual_dot_position[0]-(np.array(saccade_table.expanded_start_point[ti])[0]-400), actual_dot_position[1]-(np.array(saccade_table.expanded_start_point[ti])[1]-300)]) /  params.pixels_per_degree[ti]
		trials2use[gaze_offset > degree_cutoff] = 0
		c_gaze = int(saccade_table.shape[0] - sum(trials2use) - c_blink - c_vel - c_amp)
		
		# log amount of trials rejected. NOTE: the counters do not include trials that were already rejected by the former step, so reflect only ADDITIONAL AMOUNT OF TRIALS rejected by ... procedure
		self.logger.info('Trials rejected: total (' + str(c_blink+c_amp+c_vel+c_gaze) + ') blinks(' + str(c_blink) + ') amplitude(' + str(c_amp) + ', range: ' + str(amp_range) + ') velocity(' + str(c_vel) + ', range: ' + str(vel_range) + ') gaze(' + str(c_gaze) + ', offset: ' + str(degree_cutoff) + ' degrees) in  ' + alias)
		
		# also return vel_range for plotting purposes, as this is different for every participant (because of std +/- mean computation)
		return trials2use.astype(bool), vel_range, amp_range
		
	def velocity_for_saccades(self, alias):
		"""
		velocity_profiles_for_saccades takes the velocity profiles for all saccades that have been detected earlier
		"""
		with pd.get_store(self.ho.inputObject) as h5_file:
			saccade_table = h5_file['%s/saccades_per_trial'%alias]
		signal_profiles = np.ones((saccade_table.shape[0], self.velocity_profile_duration)) * np.nan
		for s in saccade_table.iterrows():
			index = s[0]
			s = s[1] # lose the index
			if s['raw_start_time'] != 0.0:
				int_signal = self.ho.signal_from_trial_phases(trial_nr = s['trial'], trial_phases = [1,3], alias = alias, signal = 'vel', requested_eye = s['eye'], time_extensions = [s['raw_start_time']-10,350])
				if (np.__version__.split('.')[0] == 1) and (np.__version__.split('.')[1] > 6):
					signal_profiles[index,:np.min((self.velocity_profile_duration,int(s['raw_duration'])+20))] = LA.norm(int_signal, axis = 1)[:np.min((self.velocity_profile_duration,int(s['raw_duration'])+20))]
				else:
					signal_profiles[index,:np.min((self.velocity_profile_duration,int(s['raw_duration'])+20))] = np.array([LA.norm(ints) for ints in np.array(int_signal)])[:np.min((self.velocity_profile_duration,int(s['raw_duration'])+20))]
			else:
				pass
		s_name = 'velocity'
		self.ho.data_frame_to_hdf(alias = alias, name = 'saccade_%s_profiles'%s_name, data_frame = pd.DataFrame(np.array(signal_profiles)))
	
	def gaze_for_saccades(self, alias):
		"""
		velocity_profiles_for_saccades takes the velocity profiles for all saccades that have been detected earlier
		"""
		s_name = 'gaze'
		with pd.get_store(self.ho.inputObject) as h5_file:
			saccade_table = h5_file['%s/saccades_per_trial'%alias]
			signal_profiles = np.ones((saccade_table.shape[0], self.signal_profile_duration, 2)) * np.nan
		for s in saccade_table.iterrows():
			index = s[0]
			s = s[1] # lose the index
			if s['raw_start_time'] != 0.0:
				int_signal = self.ho.signal_from_trial_phases(trial_nr = s['trial'], trial_phases = [1,3], alias = alias, signal = s_name, requested_eye = s['eye'], time_extensions = [s['raw_start_time']-10,350])
				signal_profiles[index,:np.min((self.signal_profile_duration,int(s['raw_duration'])+20))] = int_signal[:np.min((self.signal_profile_duration,int(s['raw_duration'])+20))]
			else:
				pass
		self.ho.data_frame_to_hdf(alias = alias, name = 'saccade_%s_profiles'%s_name, data_frame = pd.Panel(signal_profiles))
	
	def plot_velocity_profiles_for_blocks(self, bins_per_block = 5):
		""""""
		bin_size = self.nr_trials_per_block / bins_per_block
		panel_list = self.signals_per_block()
		f = pl.figure(figsize = (20,8))
		sub_plot_counter = 1
		for i, alias in enumerate(self.conditions.keys()):
			labels = panel_list[i][0]
			wp = panel_list[i][1]
			for j in range(self.nr_blocks):
				ss = f.add_subplot(2, self.nr_blocks, sub_plot_counter)
				for e in 'LR':
					if e == 'L':
						ls = '-'
						c = 'b'
					elif e == 'R':
						ls = '--'
						c = 'r'
					if str(j)+'_'+e in wp.items:
						this_block_data = np.array(wp[str(j)+'_'+e])
						for bin_nr in range(bins_per_block):
							bin_data = this_block_data[bin_nr*bin_size:(bin_nr+1)*bin_size]
							ss.plot(bn.nanmean(bin_data, axis = 0), c = c, linewidth = 1.75, alpha = 0.2 + (0.8 * bin_nr * 1.0/bins_per_block), linestyle = ls)
					else:
						print 'no %s in block %i' % (e, j)
				ss.axis([0,60,0,450])
				simpleaxis(ss)
				spine_shift(ss)
				ss.axhline(0, linewidth = 0.5)
				sub_plot_counter += 1
		pl.savefig(os.path.join(self.base_directory, 'figs', 'blocked_velocity_profiles.pdf'))
		
	def signals_per_block(self, signal_type = 'velocity'):
		"""docstring for velocity_signals_per_block"""
		panel_list = []
		for alias in self.conditions.keys():
			eyes = [self.ho.eye_during_trial(tbi[0], alias = alias) for tbi in self.block_trial_indices]
			with pd.get_store(self.ho.inputObject) as h5_file:
				saccade_signal_profiles_table = h5_file['%s/saccade_%s_profiles'%(alias, signal_type)]
				saccades_per_trial_table = h5_file['%s/saccades_per_trial'%alias]
			nr_eye_blocks = len("".join(eyes))
			alias_signal_profiles = np.ones((nr_eye_blocks,self.nr_trials_per_block,self.signal_profile_duration)) * np.nan
			eye_block_counter = 0
			labels = []
			for i, be in enumerate(eyes): # loop across blocks, for the eyes
				for e in be: # loop across the eyes in the string "be"
					which_trials_this_block_eye = (saccades_per_trial_table['block'] == i) * (saccades_per_trial_table['eye'] == e)
					alias_signal_profiles[eye_block_counter] = np.array(saccade_signal_profiles_table[which_trials_this_block_eye])
					labels.append({'block' : i, 'eye' : e})
					eye_block_counter += 1
			wp = pd.Panel(alias_signal_profiles, items = [str(l['block'])+'_'+l['eye'] for l in labels], major_axis=np.arange(self.nr_trials_per_block))
			panel_list.append([labels, wp])
		return panel_list
	
	def amplitudes_first_last_bino_block(self, which_amplitude = 'raw_amplitude', acceptance_amplitude_range = [4.0, 14.0]):
		""""""
		block_amps = []
		block_diffs = []
		for alias in self.conditions.keys():
			with pd.get_store(self.ho.inputObject) as h5_file:
				saccade_table = h5_file['%s/saccades_per_trial'%alias]
			nr_saccs_in_session = saccade_table.shape[0]
			
			first_block_amplitudes = np.array([saccade_table[which_amplitude][i*self.nr_trials_per_block:(i+1) * self.nr_trials_per_block] for i in [0,1]])
			# take out non-relevant trials, such as zero amplitude or such
			first_block_mask = ((first_block_amplitudes > acceptance_amplitude_range[0]) * (first_block_amplitudes < acceptance_amplitude_range[1])).prod(axis = 0, dtype = bool)
			first_block_amplitudes = first_block_amplitudes[:,first_block_mask]
			first_block_eyes = np.array([saccade_table['eye'][i*self.nr_trials_per_block] for i in [0,1]])
			
			last_block_amplitudes = np.array([saccade_table[which_amplitude][nr_saccs_in_session - self.nr_trials_per_block * (i+1):nr_saccs_in_session - self.nr_trials_per_block * i] for i in [1,0]])
			# take out non-relevant trials, such as zero amplitude or such
			last_block_mask = ((last_block_amplitudes > acceptance_amplitude_range[0]) * (last_block_amplitudes < acceptance_amplitude_range[1])).prod(axis = 0, dtype = bool)
			last_block_amplitudes = last_block_amplitudes[:,last_block_mask]
			last_block_eyes = np.array([saccade_table['eye'][nr_saccs_in_session - self.nr_trials_per_block * (i+1)] for i in [1,0]])
			
			print last_block_mask.sum(), first_block_mask.sum(), last_block_amplitudes.shape, first_block_amplitudes.shape
			
			# determine first-adapted eye and order accordingly
			first_adapted_eye = saccade_table['eye'][2*self.nr_trials_per_block + 1]
			if last_block_eyes[0] == first_adapted_eye:
				eye_order = [0,1]	# is out-adapted eye, in-adapted eye
			else:
				eye_order = [1,0]
				
			# take ordered differences on a per-trial basis
			first_diffs = np.diff(first_block_amplitudes[eye_order], axis = 0)[0]
			last_diffs = np.diff(last_block_amplitudes[eye_order], axis = 0)[0]
			first_amps = first_block_amplitudes[eye_order]
			last_amps = last_block_amplitudes[eye_order]
			
			block_diffs.append([first_diffs, last_diffs])
			block_amps.append([first_amps, last_amps])
		
		f = pl.figure(figsize = (4,8))
		for (sess, alias) in enumerate(self.conditions.keys()):
			s = f.add_subplot(2,1,sess+1)
			simpleaxis(s)
			spine_shift(s)
			for j in range(2):
				s.plot(block_amps[sess][j][0], 'b' + [':', '--'][j], linewidth = 1.0, alpha = 0.5 )
				s.plot(block_amps[sess][j][1], 'k' + [':', '--'][j], linewidth = 1.0, alpha = 0.5 )
				# s.set_ylim([5,13])
			s.set_title(alias)
		pl.savefig(os.path.join(self.base_directory, 'figs', 'first_last_bino_block_%s_amps.pdf'%which_amplitude))
		
		f = pl.figure(figsize = (4,8))
		for (sess, alias) in enumerate(self.conditions.keys()):
			s = f.add_subplot(2,1,sess+1)
			simpleaxis(s)
			spine_shift(s)
			pl.hist(block_diffs[sess][0], color = 'g', alpha = 0.5, histtype = 'step', bins = 100, range = [-3,3.1], label = 'Pre', normed = True, cumulative = True, linewidth = 3.0)
			pl.hist(block_diffs[sess][1], color = 'r', alpha = 0.5, histtype = 'step', bins = 100, range = [-3,3.1], label = 'Post', normed = True, cumulative = True, linewidth = 3.0)
			s.axhline(0.5, linewidth = 0.5)
			s.axvline(0, linewidth = 0.5)
			s.set_title(alias)
			leg = s.legend(loc = 0, fancybox = True)
			leg.get_frame().set_alpha(0.5)
			s.axis([-3,3,0,1])
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for (j, l) in enumerate(leg.get_lines()):
					l.set_linewidth(3.5)  # the legend line width
		pl.savefig(os.path.join(self.base_directory, 'figs', 'first_last_bino_block_%s_diffs_dist.pdf'%which_amplitude))
		
		f = pl.figure(figsize = (4,8))
		for (sess, alias) in enumerate(self.conditions.keys()):
			s = f.add_subplot(2,1,sess+1)
			simpleaxis(s)
			spine_shift(s)
			pl.plot(block_diffs[sess][0], color = 'g', label = 'Pre', alpha = 0.5, linewidth = 3.0)
			pl.plot(block_diffs[sess][1], color = 'r', label = 'Post', alpha = 0.5, linewidth = 3.0)
			s.axhline(0, linewidth = 0.5)
			s.axvline(0, linewidth = 0.5)
			s.set_title(alias)
			leg = s.legend(loc = 0, fancybox = True)
			leg.get_frame().set_alpha(0.5)
			# s.axis([-3,3,0,1])
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for (j, l) in enumerate(leg.get_lines()):
					l.set_linewidth(3.5)  # the legend line width
		pl.savefig(os.path.join(self.base_directory, 'figs', 'first_last_bino_block_%s_diffs.pdf'%which_amplitude))
		
		diff_fr = np.zeros((2,2,2))
		for (sess, alias) in enumerate(self.conditions.keys()):
			for j in [0,1]:
				diff_fr[sess,j] = (np.mean(block_diffs[sess][j]), np.std(block_diffs[sess][j]))
		
		diff_fr_P = pd.Panel(diff_fr)
		# diff_fr_P = pd.Panel(diff_fr, major_axis = self.conditions.keys(), minor_axis = ['pre','post'])
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/first_last_bino_block", diff_fr_P)
		
		# diff_zs = diff_fr[...,0] / diff_fr[...,1]
		diff_zs_pp = diff_fr[:,0,0] - diff_fr[:,1,0]
		
		print diff_fr
		print diff_zs_pp
		print diff_zs_pp[0] - diff_zs_pp[1]
	
	def fit_adaptation_timecourse_one_block_powerlaw(self, data, trials_okay):
		
		(ar,br)=polyfit(np.log10(np.arange(1,self.nr_trials_per_block+1))[trials_okay],np.log10(data)[trials_okay],1)
		xr=polyval([ar,br],np.log10(np.arange(1,self.nr_trials_per_block+1)))
		
		return (ar, br, xr, 10**ar, 10**br, 10**xr)
	
	def fit_adaptation_timecourse_one_block_exponential(self, data, trials_okay):
		
		(ar,br)=polyfit(np.log10(np.arange(1,self.nr_trials_per_block+1)[trials_okay]),data[trials_okay],1)
		xr=polyval([ar,br],np.log10(np.arange(1,self.nr_trials_per_block+1)))
		
		return (ar, br, xr, 10**ar, br, xr)
	
	def fit_adaptation_timecourse_one_block_exponential_mts(self, data, trials_okay):
		
		(ar,br)=polyfit(np.log10(np.arange(1,self.nr_trials_per_block+1))[trials_okay],np.log10(data)[trials_okay],1)
		xr=polyval([ar,br],np.log10(np.arange(1,self.nr_trials_per_block+1)))
		
		return (ar, br, xr, 10**ar, 10**br, 10**xr)
	
	def amplitudes_all_adaptation_blocks(self, which_amplitude = 'raw_amplitude', acceptance_amplitude_range = [5.0, 13.0]):
		""""""
		alias_amps = []
		alias_fitted_amps = []
		for alias in self.conditions.keys():
			with pd.get_store(self.ho.inputObject) as h5_file:
				saccade_table = h5_file['%s/saccades_per_trial'%alias]
				# pixels per degree
				parameters = h5_file['%s/parameters'%alias]
				pixels_per_degree = parameters['pixels_per_degree'][0]
				
			nr_saccs_in_session = saccade_table.shape[0]
			
			ad_bl_tr = []
			f_ad_bl_tr = []
			ad_for_trial_sel =[]
			for i in range(self.nr_blocks):
				if which_amplitude == 'peak_velocity':
					if (saccade_table['block'] == i).sum() > self.nr_trials_per_block: # average the two eyes together for these trials
						ad_bl_tr.append(((np.array(saccade_table[which_amplitude][(saccade_table['block'] == i) * (saccade_table['eye'] == 'R')]) + np.array(saccade_table[which_amplitude][(saccade_table['block'] == i) * (saccade_table['eye'] == 'L')]))/2.0))
					else:
						ad_bl_tr.append(np.array(saccade_table[which_amplitude][(saccade_table['block'] == i)]))
						ad_for_trial_sel.append(np.array(saccade_table['raw_amplitude'][(saccade_table['block'] == i)]))
						which_trials_okay = (ad_for_trial_sel[i] > acceptance_amplitude_range[0]) * (ad_for_trial_sel[i] < acceptance_amplitude_range[1])
				else:
					if (saccade_table['block'] == i).sum() > self.nr_trials_per_block: # average the two eyes together for these trials
						ad_bl_tr.append(((np.array(saccade_table[which_amplitude][(saccade_table['block'] == i) * (saccade_table['eye'] == 'R')]) + np.array(saccade_table[which_amplitude][(saccade_table['block'] == i) * (saccade_table['eye'] == 'L')]))/2.0))
					else:
						ad_bl_tr.append(np.array(saccade_table[which_amplitude][(saccade_table['block'] == i)]))
						which_trials_okay = (ad_bl_tr[i] > acceptance_amplitude_range[0]) * (ad_bl_tr[i] < acceptance_amplitude_range[1])
				# do some fitting of power-law
				f_ad_bl_tr.append( [self.fit_adaptation_timecourse_one_block_powerlaw(ad_bl_tr[i], which_trials_okay), self.fit_adaptation_timecourse_one_block_exponential(ad_bl_tr[i], which_trials_okay)] )
				
			alias_amps.append(ad_bl_tr)
			alias_fitted_amps.append(f_ad_bl_tr)
			
			f = pl.figure(figsize = (12,4))
			s1 = f.add_subplot(111)
			for i in range(len(ad_bl_tr)):
				s1.plot(self.nr_trials_per_block * i + np.arange(self.nr_trials_per_block)[which_trials_okay], ad_bl_tr[i][which_trials_okay], self.all_block_colors[i] + 'o', mew = 1.75, alpha = 0.875, mec = 'w', ms = 6  )
				s1.plot(self.nr_trials_per_block * i + np.arange(self.nr_trials_per_block), f_ad_bl_tr[i][0][-1], self.all_block_colors[i] + '--', alpha = 0.75, linewidth = 4.75 )
				s1.plot(self.nr_trials_per_block * i + np.arange(self.nr_trials_per_block), f_ad_bl_tr[i][1][-1], self.all_block_colors[i] + ':', alpha = 0.75, linewidth = 4.75 )
			simpleaxis(s1)
			spine_shift(s1)
	
			s1.set_xticks(np.arange(0,self.nr_trials_per_block * self.nr_blocks,self.nr_trials_per_block))
			s1.grid(axis = 'x', linestyle = '--', linewidth = 0.25)
			if which_amplitude != 'peak_velocity':
				s1.axhline(10.0, linewidth = 0.25)
				s1.axis([-20,self.nr_trials_per_block * self.nr_blocks + 20,acceptance_amplitude_range[0],acceptance_amplitude_range[1]])
				s1.set_ylabel('saccade gain')
			else:
				s1.set_ylabel('peak velocity')
			s1.set_xlabel('trials within blocks, ' + alias)
			s1.set_title(alias + '\nTrials + fit')
			
			pl.savefig(os.path.join(self.base_directory, 'figs', 'adap_time_course_%s_%s.pdf'%(alias, which_amplitude)))
			
		return alias_amps, alias_fitted_amps
		
		
		
	