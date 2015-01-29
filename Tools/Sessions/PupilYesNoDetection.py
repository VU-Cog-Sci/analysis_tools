#!/usr/bin/env python
# encoding: utf-8
"""
Created by Jan Willem de Gee on 2011-02-16.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
import os
import sys
import datetime
import pickle
import math
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
import glob
from joblib import Parallel, delayed
import itertools
from itertools import chain
import logging
import logging.handlers
import logging.config

import hddm

from IPython import embed as shell

sys.path.append(os.environ['ANALYSIS_HOME'])
from Tools.log import *
from Tools.Operators import ArrayOperator, EDFOperator, HDFEyeOperator, EyeSignalOperator
from Tools.Operators.EyeSignalOperator import detect_saccade_from_data
from Tools.Operators.CommandLineOperator import ExecCommandLine
from Tools.other_scripts.plotting_tools import *
from Tools.other_scripts.circularTools import *
from Tools.other_scripts import functions_jw as myfuncs
from Tools.other_scripts import functions_jw_GLM as GLM


class pupilPreprocessSession(object):
	"""pupilPreprocessing"""
	def __init__(self, subject, experiment_name, experiment_nr, version, project_directory, loggingLevel = logging.DEBUG, sample_rate_new = 50):
		self.subject = subject
		self.experiment_name = experiment_name
		self.experiment = experiment_nr
		self.version = version
		try:
			os.mkdir(os.path.join(project_directory, experiment_name))
			os.mkdir(os.path.join(project_directory, experiment_name, self.subject.initials))
		except OSError:
			pass
		self.project_directory = project_directory
		self.base_directory = os.path.join(self.project_directory, self.experiment_name, self.subject.initials)
		self.create_folder_hierarchy()
		self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
		self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
		self.velocity_profile_duration = self.signal_profile_duration = 100
		self.loggingLevel = loggingLevel
		self.logger = logging.getLogger(self.__class__.__name__)
		self.logger.setLevel(self.loggingLevel)
		addLoggingHandler(logging.handlers.TimedRotatingFileHandler(os.path.join(self.base_directory, 'log', 'sessionLogFile.log'), when='H', delay=2, backupCount=10), loggingLevel=self.loggingLevel)
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		
		self.logger.info('starting analysis in ' + self.base_directory)
		self.sample_rate_new = int(sample_rate_new)
		self.downsample_rate = int(1000 / sample_rate_new)
		
	def create_folder_hierarchy(self):
		"""createFolderHierarchy does... guess what."""
		this_dir = self.project_directory
		for d in [self.experiment_name, self.subject.initials]:
			try:
				this_dir = os.path.join(this_dir, d)
				os.mkdir(this_dir)
			except OSError:
				pass

		for p in ['raw',
		 'processed',
		 'figs',
		 'log']:
			try:
				os.mkdir(os.path.join(self.base_directory, p))
			except OSError:
				pass
	
	def delete_hdf5(self):
		os.system('rm {}'.format(os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')))
	
	def import_raw_data(self, edf_files, aliases):
		"""import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""
		for (edf_file, alias,) in zip(edf_files, aliases):
			self.logger.info('importing file ' + edf_file + ' as ' + alias)
			ExecCommandLine('cp "' + edf_file + '" "' + os.path.join(self.base_directory, 'raw', alias + '.edf"'))
	
	def import_all_data(self, aliases):
		"""import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
		for alias in aliases:
			self.ho.add_edf_file(os.path.join(self.base_directory, 'raw', alias + '.edf'))
			self.ho.edf_message_data_to_hdf(alias=alias)
			self.ho.edf_gaze_data_to_hdf(alias=alias)




	def compute_omission_indices(self):
		"""
		Here we're going to determine which trials should be counted as omissions due to
		(i) fixation errors (in decision interval):
				->gaze of 150px or more away from fixation
				->10 percent of time gaze of 75px or more away from fixation.
		(ii) blinks (in window 0.5s before decision interval till 0.5s after decision interval)
		(iii) too long (>3000ms, >2500ms) or too short (<250ms) RT
		(iv) first two trials
		(v) indicated by subject (only in exp 1)
		"""
		
		self.omission_indices_answer = (self.rt > 4000.0)
		# self.omission_indices_answer = (self.rt > 4000.0) * (np.array(self.parameters.answer == 0))
		
		self.omission_indices_sac = np.zeros(self.nr_trials, dtype=bool)
		middle_x = 0
		middle_y = 0
		cut_off = 75
		x_matrix = []
		y_matrix = []
		
		for t in range(self.nr_trials):
			
			try:
				indices = (self.time > self.cue_times[t]) * (self.time < self.choice_times[t])
			except:
				shell()
			
			x = self.gaze_x[indices]
			x = x - bn.nanmean(x)
			y = self.gaze_y[indices]
			y = y - bn.nanmean(y)
			if (x < -175).sum() > 0 or (x > 175).sum() > 0:
				self.omission_indices_sac[t] = True
			if (y < -175).sum() > 0 or (y > 175).sum() > 0:
				self.omission_indices_sac[t] = True
			if ((x > middle_x + cut_off).sum() + (x < middle_x - cut_off).sum()) / float(self.rt[t]) * 100 > 10:
				self.omission_indices_sac[t] = True
			if ((y > middle_y + cut_off).sum() + (y < middle_y - cut_off).sum()) / float(self.rt[t]) * 100 > 10:
				self.omission_indices_sac[t] = True

		self.omission_indices_blinks = np.zeros(self.nr_trials, dtype=bool)
		for t in range(self.nr_trials):
			if sum((self.blink_start_times > self.cue_times[t] - 500) * (self.blink_end_times < self.choice_times[t] + 500)) > 0:
				self.omission_indices_blinks[t] = True
		
		self.omission_indices_rt = np.zeros(self.nr_trials, dtype=bool)
		for t in range(self.nr_trials):
			if self.rt[t] < 250:
				self.omission_indices_rt[t] = True
		
		self.omission_indices_first = np.zeros(self.nr_trials, dtype=bool)
		self.omission_indices_first[0:2] = True
		if self.experiment == 1:
			self.omission_indices_subject = np.array(self.parameters['confidence'] == -1)
			self.omission_indices_subject = np.array(self.parameters['correct'] == -1)
		else:
			self.omission_indices_subject = np.zeros(self.nr_trials, dtype=bool)
		self.omission_indices = self.omission_indices_answer + self.omission_indices_sac + self.omission_indices_blinks + self.omission_indices_rt + self.omission_indices_first + self.omission_indices_subject
	
	def trial_params(self):
		blinks_nr = np.zeros(self.nr_trials)
		number_blinks = np.zeros(self.nr_trials)
		for t in range(self.nr_trials):
			blinks_nr[t] = sum((self.blink_start_times > self.cue_times[t] - 5500) * (self.blink_start_times < self.cue_times[t] - 500))

		sacs_nr = np.zeros(self.nr_trials)
		sacs_dur = np.zeros(self.nr_trials)
		sacs_vel = np.zeros(self.nr_trials)
		for t in range(self.nr_trials):
			saccades_in_trial_indices = (self.saccade_start_times > self.cue_times[t] - 500) * (self.saccade_start_times < self.choice_times[t] + 1500)
			sacs_nr[t] = sum(saccades_in_trial_indices)
			sacs_dur[t] = sum(self.saccade_durs[saccades_in_trial_indices])
			if sacs_nr[t] != 0:
				sacs_vel[t] = max(self.saccade_peak_velocities[saccades_in_trial_indices])
		if self.version == 3:
			present = np.array(self.parameters['signal_present'] == 1)
		else:
			present = np.array(self.parameters['target_present_in_stimulus'] == 1)
		correct = np.array(self.parameters['correct'] == 1)
		yes = present * correct + -present * -correct
		hit = present * yes
		fa = -present * yes
		miss = present * -yes
		cr = -present * -yes
		self.parameters['omissions'] = self.omission_indices
		self.parameters['omissions_answer'] = self.omission_indices_answer
		self.parameters['omissions_sac'] = self.omission_indices_sac
		self.parameters['omissions_blinks'] = self.omission_indices_blinks
		self.parameters['omissions_rt'] = self.omission_indices_rt
		self.parameters['rt'] = self.rt
		self.parameters['yes'] = yes
		self.parameters['present'] = present
		self.parameters['hit'] = hit
		self.parameters['fa'] = fa
		self.parameters['miss'] = miss
		self.parameters['cr'] = cr
		self.parameters['blinks_nr'] = blinks_nr
		self.parameters['sacs_nr'] = sacs_nr
		self.parameters['sacs_dur'] = sacs_dur
		self.parameters['sacs_vel'] = sacs_vel
		self.ho.data_frame_to_hdf(self.alias, 'parameters2', self.parameters)
		print '{} total trials'.format(self.nr_trials)
		print '{} hits'.format(sum(hit))
		print '{} false alarms'.format(sum(fa))
		print '{} misses'.format(sum(miss))
		print '{} correct rejects'.format(sum(cr))
		print '{} omissions'.format(sum(self.omission_indices))
		(d_prime, criterion,) = myfuncs.SDT_measures(present, hit, fa)
		print "d' = {}\nc = {}".format(round(d_prime, 4), round(criterion, 4))
		print '{} mean RT (including omissions)'.format(round(np.mean(self.rt), 4))
		print ''
	
	def pupil_zscore(self):
		start = self.cue_times[(-self.omission_indices)] - 500
		end = self.choice_times[(-self.omission_indices)] + 1500
		include_indices = np.zeros(len(self.time), dtype=bool)
		for i in range(len(start)):
			include_indices[(self.time > start[i]) * (self.time < end[i])] = True

		mean = self.pupil_lp[include_indices].mean()
		std = self.pupil_lp[include_indices].std()
		self.pupil_lp_z = (self.pupil_lp - mean) / std
		mean = self.pupil_bp[include_indices].mean()
		std = self.pupil_bp[include_indices].std()
		self.pupil_bp_z = (self.pupil_bp - mean) / std
	
	def create_timelocked_arrays(self):
		cue_locked_array_lp = np.empty((self.nr_trials, 7500))
		cue_locked_array_lp[:, :] = np.NaN
		cue_locked_array_bp = np.empty((self.nr_trials, 7500))
		cue_locked_array_bp[:, :] = np.NaN
		for i in range(self.nr_trials):
			indices = (self.time > self.cue_times[i] - 1000) * (self.time < self.cue_times[i] + 6500)
			cue_locked_array_lp[i,:self.pupil_lp_z[indices].shape[0]] = self.pupil_lp_z[indices]
			cue_locked_array_bp[i,:self.pupil_bp_z[indices].shape[0]] = self.pupil_bp_z[indices]

		choice_locked_array_lp = np.empty((self.nr_trials, 7500))
		choice_locked_array_lp[:, :] = np.NaN
		choice_locked_array_bp = np.empty((self.nr_trials, 7500))
		choice_locked_array_bp[:, :] = np.NaN
		for i in range(self.nr_trials):
			indices = (self.time > self.choice_times[i] - 4000) * (self.time < self.choice_times[i] + 3500)
			choice_locked_array_lp[i,:self.pupil_lp_z[indices].shape[0]] = self.pupil_lp_z[indices]
			choice_locked_array_bp[i,:self.pupil_bp_z[indices].shape[0]] = self.pupil_bp_z[indices]
		
		if self.experiment == 1:
			feedback_locked_array_lp = np.empty((self.nr_trials, 5000))
			feedback_locked_array_lp[:, :] = np.NaN
			feedback_locked_array_bp = np.empty((self.nr_trials, 5000))
			feedback_locked_array_bp[:, :] = np.NaN
			for i in range(self.nr_trials):
				indices = (self.time > self.feedback_times[i] - 1000) * (self.time < self.feedback_times[i] + 4000)
				feedback_locked_array_lp[i,:self.pupil_lp_z[indices].shape[0]] = self.pupil_lp_z[indices]
				feedback_locked_array_bp[i,:self.pupil_bp_z[indices].shape[0]] = self.pupil_bp_z[indices]
		
		# to hdf5:
		self.ho.data_frame_to_hdf(self.alias, 'time_locked_cue_lp', pd.DataFrame(cue_locked_array_lp))
		self.ho.data_frame_to_hdf(self.alias, 'time_locked_cue_bp', pd.DataFrame(cue_locked_array_bp))
		self.ho.data_frame_to_hdf(self.alias, 'time_locked_choice_lp', pd.DataFrame(choice_locked_array_lp))
		self.ho.data_frame_to_hdf(self.alias, 'time_locked_choice_bp', pd.DataFrame(choice_locked_array_bp))
		if self.experiment == 1:
			self.ho.data_frame_to_hdf(self.alias, 'time_locked_feedback_lp', pd.DataFrame(feedback_locked_array_lp))
			self.ho.data_frame_to_hdf(self.alias, 'time_locked_feedback_bp', pd.DataFrame(feedback_locked_array_bp))
	
	
	
	def problem_trials(self):
		
		# problem trials:
		answers = np.array(self.parameters.answer)
		if self.version == 3:
			problem_trials = np.where(answers == -1)[0]
		else:
			problem_trials = np.where(answers == 0)[0]
		
		print
		print answers
		print
		
		# fix:
		for i in problem_trials:
			
			events = self.events[(self.events.EL_timestamp > self.trial_starts[i]) * (self.events.EL_timestamp < self.trial_ends[i])]
			response_times = np.array(events[((events.key == 275) + (events.key == 276)) * (events.up_down == 'Down')].EL_timestamp)
			
			if len(response_times) != 0:
				
				try:
					ind = np.where(response_times>=self.choice_times[i])[0][0]
					response_time = response_times[ind]
					response_key = np.array(events[((events.key == 275) + (events.key == 276)) * (events.up_down == 'Down')].key)[ind]
					
					# choice time:
					self.choice_times[i] = response_time
			
					# answer & correct:
					if self.version == 1:
						if response_key == 275:
							self.parameters.answer[i] = -1
							if self.parameters.target_present_in_stimulus[i] == 0:
								self.parameters.correct[i] = 1
							else:
								self.parameters.correct[i] = 0
						if response_key == 276:
							self.parameters.answer[i] = 1
							if self.parameters.target_present_in_stimulus[i] == 0:
								self.parameters.correct[i] = 0
							else:
								self.parameters.correct[i] = 1
					if self.version == 2:
						if response_key == 275:
							self.parameters.answer[i] = 1
							if self.parameters.target_present_in_stimulus[i] == 0:
								self.parameters.correct[i] = 0
							else:
								self.parameters.correct[i] = 1
						if response_key == 276:
							self.parameters.answer[i] = -1
							if self.parameters.target_present_in_stimulus[i] == 0:
								self.parameters.correct[i] = 1
							else:
								self.parameters.correct[i] = 0
					
				except:
					pass
			
			else:
				pass
			
	
	def create_pupil_BOLD_regressor(self):
		
		# load number of trs in nifti (unequal to above because we aborted runs manually, and not all trs are recorded by experiment script):
		if self.subject.initials == 'AV':
			trs = [259, 264, 254, 257, 251, 249, 256, 247, 249, 250, 246, 247,]
		if self.subject.initials == 'BL':
			trs = [273, 267, 255, 259, 254, 266, 252, 273, 261, 259, 256, 250,] 
		if self.subject.initials == 'DL':
			trs = [214, 299, 183, 259, 260, 266, 259, 253, 261, 251, 258, 249, 258, 256, 258, 259,]
		if self.subject.initials == 'EP':
			trs = [252, 255, 258, 264, 255, 257, 255, 258, 257, 254, 247,]
		if self.subject.initials == 'JG':
			trs = [252, 269, 274, 253, 257, 260, 256, 250, 254, 260, 260, 251,]
		if self.subject.initials == 'LH':
			trs = [264, 300, 300, 294, 251, 260, 263, 253, 256, 260,]
		if self.subject.initials == 'LP':
			trs = [249, 252, 249, 259, 260, 261, 262, 256, 256, 253, 255, 257,]
		if self.subject.initials == 'MG':
			trs = [265, 265, 252, 255, 304, 256, 251, 250, 256, 258, 253, 248,]
		nr_trs_nifti = trs[int(self.alias.split('_')[-1]) - 1]
		
		# load trigger timings:
		trigger_timings = np.array(self.events['EL_timestamp'][(self.events.up_down == 'Down') & (self.events.key == 116)])
		tr = min(np.diff(trigger_timings))
		print 'tr = {} ms'.format(tr)
		print 'len trigger exp = {}'.format(len(trigger_timings))
		print 'len trigger nifti = {}'.format(nr_trs_nifti)
		print
		
		if nr_trs_nifti - len(trigger_timings) == -1: # this is the case for the last run of subjects EP... I don't understand this
			trigger_timings = trigger_timings[:-1]
		
		# create pupil timeseries in fMRI tr resolution:
		pupil = np.zeros((6, nr_trs_nifti))
		pupil[:,:] = np.NaN
		try:
			pupil[0,:len(trigger_timings)] = np.array([np.mean(self.pupil_lp_z[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
			pupil[1,:len(trigger_timings)] = np.array([np.max(self.pupil_lp_z[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
			pupil[2,:len(trigger_timings)] = np.array([np.mean(self.pupil_bp_z[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
			pupil[3,:len(trigger_timings)] = np.array([np.max(self.pupil_bp_z[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
			pupil[4,:len(trigger_timings)] = np.array([np.mean(np.array(self.pupil_data[(self.eye + '_pupil_lp_diff')])[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
			pupil[5,:len(trigger_timings)] = np.array([np.max(np.array(self.pupil_data[(self.eye + '_pupil_lp_diff')])[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
		except ValueError:
			shell()
		
		# save:
		self.ho.data_frame_to_hdf(self.alias, 'pupil_BOLD_regressors', pd.DataFrame(pupil))
		#
	def process_runs(self, alias, create_pupil_BOLD_regressor=False):
		print 'subject {}; {}'.format(self.subject.initials, alias)
		print '##############################'
		
		# load data:
		self.alias = alias
		self.events = self.ho.read_session_data(alias, 'events')
		self.parameters = self.ho.read_session_data(alias, 'parameters')
		self.nr_trials = len(self.parameters['trial_nr'])
		self.trial_times = self.ho.read_session_data(alias, 'trials')
		self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
		self.trial_starts = np.array(self.trial_times['trial_start_EL_timestamp'])
		self.trial_ends = np.array(self.trial_times['trial_end_EL_timestamp'])
		self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
		self.cue_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)])
		self.choice_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)])
		self.problem_trials()
		self.rt = self.choice_times - self.cue_times
		if self.experiment == 1:
			self.confidence_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 5)])
			self.feedback_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 6)])
		self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
		self.saccade_data = self.ho.read_session_data(alias, 'saccades_from_message_file')
		self.blink_start_times = np.array(self.blink_data['start_timestamp'])
		self.blink_end_times = np.array(self.blink_data['end_timestamp'])
		self.saccade_start_times = np.array(self.saccade_data['start_timestamp'])
		self.saccade_end_times = np.array(self.saccade_data['end_timestamp'])
		self.saccade_durs = np.array(self.saccade_data['duration'])
		self.saccade_peak_velocities = np.array(self.saccade_data['peak_velocity'])
		self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
		self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
		self.time = np.array(self.pupil_data['time'])
		self.pupil_raw = np.array(self.pupil_data[(self.eye + '_pupil')])
		self.pupil_lp = np.array(self.pupil_data[(self.eye + '_pupil_lp')])
		self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
		self.gaze_x = np.array(self.pupil_data[(self.eye + '_gaze_x')])
		self.gaze_y = np.array(self.pupil_data[(self.eye + '_gaze_y')])
		
		self.compute_omission_indices()
		self.trial_params()
		self.pupil_zscore()
		self.create_timelocked_arrays()
		
		if create_pupil_BOLD_regressor:
			self.create_pupil_BOLD_regressor()
		
	
	def process_across_runs(self, aliases, create_pupil_BOLD_regressor=False):
		
		# load data:
		parameters = []
		cue_locked_array_lp = []
		cue_locked_array_bp = []
		choice_locked_array_lp = []
		choice_locked_array_bp = []
		feedback_locked_array_lp = []
		feedback_locked_array_bp = []
		pupil_BOLD_regressors = []
		for alias in aliases:
			parameters.append(self.ho.read_session_data(alias, 'parameters2'))
			cue_locked_array_lp.append(self.ho.read_session_data(alias, 'time_locked_cue_lp'))
			cue_locked_array_bp.append(self.ho.read_session_data(alias, 'time_locked_cue_bp'))
			choice_locked_array_lp.append(self.ho.read_session_data(alias, 'time_locked_choice_lp'))
			choice_locked_array_bp.append(self.ho.read_session_data(alias, 'time_locked_choice_bp'))
			if self.experiment == 1:
				feedback_locked_array_lp.append(self.ho.read_session_data(alias, 'time_locked_feedback_lp'))
				feedback_locked_array_bp.append(self.ho.read_session_data(alias, 'time_locked_feedback_bp'))
			if create_pupil_BOLD_regressor:
				pupil_BOLD_regressors.append(np.array(self.ho.read_session_data(alias, 'pupil_BOLD_regressors')))
		
		# join over runs:
		parameters_joined = pd.concat(parameters)
		cue_locked_array_lp_joined = np.array(pd.concat(cue_locked_array_lp))
		cue_locked_array_bp_joined = np.array(pd.concat(cue_locked_array_bp))
		choice_locked_array_lp_joined = np.array(pd.concat(choice_locked_array_lp))
		choice_locked_array_bp_joined = np.array(pd.concat(choice_locked_array_bp))
		if self.experiment == 1:
			feedback_locked_array_lp_joined = np.array(pd.concat(feedback_locked_array_lp))
			feedback_locked_array_bp_joined = np.array(pd.concat(feedback_locked_array_bp))
		if create_pupil_BOLD_regressor:
			pupil_BOLD_regressors_joined = np.hstack(pupil_BOLD_regressors)
			np.save(os.path.join(self.project_directory, 'data', self.subject.initials, 'pupil_BOLD_regressors'), pupil_BOLD_regressors_joined)
		
		# baseline pupil measures:
		start = 500
		end = 1000
		bpd_lp = myfuncs.pupil_scalar_mean(cue_locked_array_lp_joined, start, end)
		bpd_bp = myfuncs.pupil_scalar_mean(cue_locked_array_bp_joined, start, end)
		if self.experiment == 1:
			bpd_lp_feed = myfuncs.pupil_scalar_mean(feedback_locked_array_lp_joined, start, end)
			bpd_bp_feed = myfuncs.pupil_scalar_mean(feedback_locked_array_bp_joined, start, end)
		
		# baseline time locked arrays:
		cue_locked_array_lp_joined_b = np.zeros(cue_locked_array_lp_joined.shape)
		cue_locked_array_bp_joined_b = np.zeros(cue_locked_array_bp_joined.shape)
		choice_locked_array_lp_joined_b = np.zeros(choice_locked_array_lp_joined.shape)
		choice_locked_array_bp_joined_b = np.zeros(choice_locked_array_bp_joined.shape)
		for i in range(len(bpd_lp)):
			cue_locked_array_lp_joined_b[i, :] = cue_locked_array_lp_joined[i, :] - bpd_lp[i]
			cue_locked_array_bp_joined_b[i, :] = cue_locked_array_bp_joined[i, :] - bpd_bp[i]
			choice_locked_array_lp_joined_b[i, :] = choice_locked_array_lp_joined[i, :] - bpd_lp[i]
			choice_locked_array_bp_joined_b[i, :] = choice_locked_array_bp_joined[i, :] - bpd_bp[i]
		if self.experiment == 1:
			feedback_locked_array_lp_joined_b = np.zeros(feedback_locked_array_lp_joined.shape)
			feedback_locked_array_bp_joined_b = np.zeros(feedback_locked_array_bp_joined.shape)
			for i in range(len(bpd_lp_feed)):
				feedback_locked_array_lp_joined_b[i, :] = feedback_locked_array_lp_joined[i, :] - bpd_lp_feed[i]
				feedback_locked_array_bp_joined_b[i, :] = feedback_locked_array_bp_joined[i, :] - bpd_bp_feed[i]
		
		# phasic pupil responses 
		start = 3000
		end = 5500
		ppr_peak_lp = myfuncs.pupil_scalar_peak(choice_locked_array_lp_joined_b, start, end)
		ppr_mean_lp = myfuncs.pupil_scalar_mean(choice_locked_array_lp_joined_b, start, end)
		template = np.mean(choice_locked_array_lp_joined_b[-np.array(parameters_joined['omissions']),start:end], axis=0)
		sign_template_lp = np.sign(sum(template))
		ppr_proj_lp = myfuncs.pupil_scalar_lin_projection(choice_locked_array_lp_joined_b, start, end, template)
		ppr_peak_bp = myfuncs.pupil_scalar_peak(choice_locked_array_bp_joined_b, start, end)
		ppr_mean_bp = myfuncs.pupil_scalar_mean(choice_locked_array_bp_joined_b, start, end)
		template = np.mean(choice_locked_array_bp_joined_b[-np.array(parameters_joined['omissions']),start:end], axis=0)
		sign_template_bp = np.sign(sum(template))
		ppr_proj_bp = myfuncs.pupil_scalar_lin_projection(choice_locked_array_bp_joined_b, start, end, template)
		if self.experiment == 1:
			start = 1000
			end = 3000
			ppr_peak_feed_lp = myfuncs.pupil_scalar_peak(feedback_locked_array_lp_joined_b, start, end)
			ppr_mean_feed_lp = myfuncs.pupil_scalar_mean(feedback_locked_array_lp_joined_b, start, end)
			template = np.mean(feedback_locked_array_lp_joined_b[-np.array(parameters_joined['omissions']),start:end], axis=0)
			ppr_proj_feed_lp = myfuncs.pupil_scalar_lin_projection(feedback_locked_array_lp_joined_b, start, end, template)
			ppr_peak_feed_bp = myfuncs.pupil_scalar_peak(feedback_locked_array_bp_joined_b, start, end)
			ppr_mean_feed_bp = myfuncs.pupil_scalar_mean(feedback_locked_array_bp_joined_b, start, end)
			template = np.mean(feedback_locked_array_bp_joined_b[-np.array(parameters_joined['omissions']),start:end], axis=0)
			ppr_proj_feed_bp = myfuncs.pupil_scalar_lin_projection(feedback_locked_array_bp_joined_b, start, end, template)
		target_joined = parameters_joined['present'][(-parameters_joined['omissions'])]
		hit_joined = parameters_joined['hit'][(-parameters_joined['omissions'])]
		fa_joined = parameters_joined['fa'][(-parameters_joined['omissions'])]
		(d, c,) = myfuncs.SDT_measures(target_joined, hit_joined, fa_joined)
		target = [ param['present'][(-param['omissions'])] for param in parameters ]
		hit = [ param['hit'][(-param['omissions'])] for param in parameters ]
		fa = [ param['fa'][(-param['omissions'])] for param in parameters ]
		d_run = []
		c_run = []
		for i in range(len(aliases)):
			d_run.append(myfuncs.SDT_measures(target[i], hit[i], fa[i])[0])
			c_run.append(myfuncs.SDT_measures(target[i], hit[i], fa[i])[1])
		
		# add to dataframe and save to hdf5:
		parameters_joined['bpd_lp'] = bpd_lp
		parameters_joined['bpd_bp'] = bpd_bp
		parameters_joined['ppr_peak_lp'] = ppr_peak_lp
		parameters_joined['ppr_peak_bp'] = ppr_peak_bp
		parameters_joined['ppr_mean_lp'] = ppr_mean_lp
		parameters_joined['ppr_mean_bp'] = ppr_mean_bp
		parameters_joined['ppr_proj_lp'] = ppr_proj_lp
		parameters_joined['ppr_proj_bp'] = ppr_proj_bp
		parameters_joined['sign_template_lp'] = sign_template_lp
		parameters_joined['sign_template_bp'] = sign_template_bp
		parameters_joined['d_prime'] = d
		parameters_joined['criterion'] = c
		for i in range(len(aliases)):
			parameters_joined['d_prime_' + str(i)] = d_run[i]
			parameters_joined['criterion_' + str(i)] = c_run[i]
		if self.experiment == 1:
			parameters_joined['bpd_feed_lp'] = bpd_lp_feed
			parameters_joined['bpd_feed_bp'] = bpd_bp_feed
			parameters_joined['ppr_peak_feed_lp'] = ppr_peak_feed_lp
			parameters_joined['ppr_peak_feed_bp'] = ppr_peak_feed_bp
			parameters_joined['ppr_mean_feed_lp'] = ppr_mean_feed_lp
			parameters_joined['ppr_mean_feed_bp'] = ppr_mean_feed_bp
			parameters_joined['ppr_proj_feed_lp'] = ppr_proj_feed_lp
			parameters_joined['ppr_proj_feed_bp'] = ppr_proj_feed_bp
		parameters_joined['subject'] = self.subject.initials
		self.ho.data_frame_to_hdf('', 'parameters_joined', parameters_joined)
		
		
	# def process_across_runs2(self, aliases):
	#
	# 	downsample_rate = 50 # 50
	# 	new_sample_rate = 1000 / downsample_rate
	#
	# 	# load data:
	# 	parameters = []
	# 	pupil = []
	# 	time = []
	# 	cue_times = []
	# 	choice_times = []
	# 	blink_times = []
	# 	time_to_add = 0
	# 	for alias in aliases:
	# 		parameters.append(self.ho.read_session_data(alias, 'parameters2'))
	#
	# 		self.alias = alias
	# 		self.trial_times = self.ho.read_session_data(alias, 'trials')
	#
	# 		# load pupil:
	# 		self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
	# 		self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
	# 		self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
	# 		pupil.append(self.pupil_bp / np.std(self.pupil_bp))
	# 		pupil_diff.append( np.array(self.pupil_data[(self.eye + '_pupil_lp_diff')]) )
	#
	# 		# load times:
	# 		self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
	# 		self.time = np.array(self.pupil_data['time']) - self.session_start
	# 		time.append( self.time + time_to_add)
	# 		self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
	# 		cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
	# 		choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
	# 		# load blinks:
	# 		self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
	# 		blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
	#
	# 		time_to_add += self.time[-1]
	#
	# 	# join over runs:
	# 	parameters_joined = pd.concat(parameters)
	# 	pupil = np.concatenate(pupil)
	# 	pupil_diff = np.concatenate(pupil_diff)
	# 	time = np.concatenate(time)
	# 	cue_times = np.concatenate(cue_times) / 1000.0
	# 	choice_times = np.concatenate(choice_times) / 1000.0
	# 	blink_times = np.concatenate(blink_times) / 1000.0
	# 	omissions = np.array(parameters_joined.omissions, dtype=bool)
	# 	correct = np.array(parameters_joined.correct, dtype=bool)*-omissions
	# 	error = -np.array(parameters_joined.correct, dtype=bool)*-omissions
	# 	hit = np.array(parameters_joined.hit, dtype=bool)*-omissions
	# 	fa = np.array(parameters_joined.fa, dtype=bool)*-omissions
	# 	miss = np.array(parameters_joined.miss, dtype=bool)*-omissions
	# 	cr = np.array(parameters_joined.cr, dtype=bool)*-omissions
	#
	# 	# event related averages:
	# 	interval = 5
	#
	# 	# stimulus locked:
	# 	# ---------------
	#
	# 	# output:
	# 	kernel_cue_hit = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (cue_times[correct*drug]-0.5)*1000]), axis=0)
	# 	kernel_cue_fa = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (cue_times[correct*-drug]-0.5)*1000]), axis=0)
	# 	kernel_cue_error_A = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (cue_times[error*drug]-0.5)*1000]), axis=0)
	# 	kernel_cue_error_B = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (cue_times[error*-drug]-0.5)*1000]), axis=0)
	#
	# 	# choice locked:
	# 	# -------------
	#
	# 	# output:
	# 	kernel_choice_correct_A = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (choice_times[correct*drug]-2)*1000]), axis=0)
	# 	kernel_choice_correct_B = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (choice_times[correct*-drug]-2)*1000]), axis=0)
	# 	kernel_choice_error_A = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (choice_times[error*drug]-2)*1000]), axis=0)
	# 	kernel_choice_error_B = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (choice_times[error*-drug]-2)*1000]), axis=0)
	#
	#
	# 	# save:
	# 	np.save(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_{}.npy'.format(self.subject.initials)), np.vstack((kernel_cue_correct_A, kernel_cue_correct_B, kernel_cue_error_A, kernel_cue_error_B)))
	# 	np.save(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_{}.npy'.format(self.subject.initials)), np.vstack((kernel_choice_correct_A, kernel_choice_correct_B, kernel_choice_error_A, kernel_choice_error_B)))
	#
	# 	# plot:
	# 	# -----
	# 	fig = plt.figure(figsize=(10,10))
	#
	# 	ax = fig.add_subplot(221)
	# 	ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A, 'g', label='correct, A')
	# 	ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_B, 'g', ls='--', label='correct, B')
	# 	ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_A, 'r', label='error, A')
	# 	ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_B, 'r', ls='--', label='error, B')
	# 	plt.axvline(0, color='k', ls='--', lw=0.5)
	# 	plt.axvline(np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
	# 	ax.set_xlim(-0.5, interval-0.5)
	# 	ax.set_ylim(-1.5, 1.5)
	# 	ax.set_xlabel('time from stimulus onset (s)')
	# 	ax.set_ylabel('pupil size (Z)')
	# 	ax.legend()
	#
	# 	ax = fig.add_subplot(222)
	# 	ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A-kernel_cue_error_A, 'b', label='difference wave, A')
	# 	ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_B-kernel_cue_error_B, 'b', ls='--', label='difference wave, B')
	# 	plt.axvline(0, color='k', ls='--', lw=0.5)
	# 	plt.axvline(np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
	# 	plt.axhline(0, lw=0.5)
	# 	ax.set_xlim(-0.5, interval-0.5)
	# 	ax.set_ylim(-1.5, 1.5)
	# 	ax.set_xlabel('time from stimulus onset (s)')
	# 	ax.set_ylabel('correct - error')
	# 	ax.legend()
	#
	# 	ax = fig.add_subplot(223)
	# 	ax.plot(np.linspace(-2, interval-2, kernel_choice_correct_A.shape[0]), kernel_choice_correct_A, 'g', label='correct, A')
	# 	ax.plot(np.linspace(-2, interval-2, kernel_choice_correct_A.shape[0]), kernel_choice_correct_B, 'g', ls='--', label='correct, B')
	# 	ax.plot(np.linspace(-2, interval-2, kernel_choice_error_A.shape[0]), kernel_choice_error_A, 'r', label='error, A')
	# 	ax.plot(np.linspace(-2, interval-2, kernel_choice_error_A.shape[0]), kernel_choice_error_B, 'r', ls='--', label='error, B')
	# 	plt.axvline(0, color='k', ls='--', lw=0.5)
	# 	plt.axvline(-np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
	# 	ax.set_xlim(-2, interval-2)
	# 	ax.set_ylim(-1.5, 1.5)
	# 	ax.set_xlabel('time from choice (s)')
	# 	ax.set_ylabel('pupil size (Z)')
	# 	ax.legend()
	#
	# 	ax = fig.add_subplot(224)
	# 	ax.plot(np.linspace(-2, interval-2, kernel_choice_error_A.shape[0]), kernel_choice_correct_A-kernel_choice_error_A, 'b', label='difference wave, A')
	# 	ax.plot(np.linspace(-2, interval-2, kernel_choice_error_A.shape[0]), kernel_choice_correct_B-kernel_choice_error_B, 'b', ls='--', label='difference wave, B')
	# 	plt.axvline(0, color='k', ls='--', lw=0.5)
	# 	plt.axvline(-np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
	# 	plt.axhline(0, lw=0.5)
	# 	ax.set_xlim(-2, interval-2)
	# 	ax.set_ylim(-1.5, 1.5)
	# 	ax.set_xlabel('time from choice (s)')
	# 	ax.set_ylabel('correct - error')
	# 	ax.legend()
	#
	# 	plt.tight_layout()
	# 	fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_responses_avg_{}.pdf'.format(self.subject.initials)))
		
		
		
		
class pupilAnalyses(object):
	"""pupilAnalyses"""
	def __init__(self, subject, experiment_name, experiment_nr, project_directory, aliases, sample_rate_new=50):
		self.subject = subject
		self.experiment_name = experiment_name
		self.experiment = experiment_nr
		self.project_directory = project_directory
		self.base_directory = os.path.join(self.project_directory, self.experiment_name, self.subject.initials)
		self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
		self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
		self.sample_rate_new = int(sample_rate_new)
		self.downsample_rate = int(1000 / sample_rate_new)
		
		# load data:
		self.cue_locked_array_lp = []
		self.cue_locked_array_bp = []
		self.choice_locked_array_lp = []
		self.choice_locked_array_bp = []
		self.feedback_locked_array_lp = []
		self.feedback_locked_array_bp = []
		for alias in aliases:
			self.cue_locked_array_lp.append(self.ho.read_session_data(alias, 'time_locked_cue_lp'))
			self.cue_locked_array_bp.append(self.ho.read_session_data(alias, 'time_locked_cue_bp'))
			self.choice_locked_array_lp.append(self.ho.read_session_data(alias, 'time_locked_choice_lp'))
			self.choice_locked_array_bp.append(self.ho.read_session_data(alias, 'time_locked_choice_bp'))
			if self.experiment == 1:
				self.feedback_locked_array_lp.append(self.ho.read_session_data(alias, 'time_locked_feedback_lp'))
				self.feedback_locked_array_bp.append(self.ho.read_session_data(alias, 'time_locked_feedback_bp'))
		self.cue_locked_array_joined = np.vstack(self.cue_locked_array_lp)
		self.choice_locked_array_joined = np.vstack(self.choice_locked_array_lp)
		if self.experiment == 1:
			self.feedback_locked_array_joined = np.vstack(self.feedback_locked_array_lp)
			
		self.parameters_joined = self.ho.read_session_data('', 'parameters_joined')
		
		self.rt = self.parameters_joined['rt']
		
		self.omissions = np.array(self.parameters_joined['omissions'])
		self.hit = np.array(self.parameters_joined['hit']) * -self.omissions
		self.fa = np.array(self.parameters_joined['fa']) * -self.omissions
		self.miss = np.array(self.parameters_joined['miss']) * -self.omissions
		self.cr = np.array(self.parameters_joined['cr']) * -self.omissions
		
		self.bpd = np.array(self.parameters_joined['bpd_lp'])
		
		self.ppr = np.array(self.parameters_joined['ppr_proj_lp'])
		
		self.criterion = np.array(self.parameters_joined['criterion'])[0]
		
		if self.experiment == 1:
			self.ppr_feed = np.array(self.parameters_joined['ppr_peak_feed_lp'])
			self.bpd_feed = np.array(self.parameters_joined['bpd_feed_lp'])
	
	def trial_wise_pupil(self):
		
		nr_runs = (pd.Series(np.array(self.parameters_joined.trial_nr))==0).sum()
		start_run = np.where(pd.Series(np.array(self.parameters_joined.trial_nr))==0)[0]
		run_nr = np.ones(len(np.array(self.parameters_joined.trial_nr)))
		for i in range(nr_runs):
			if i != (nr_runs-1):
				run_nr[start_run[i]:start_run[i+1]] = i
			if i == (nr_runs-1): 
				run_nr[start_run[i]:] = i
		
		# data:
		d = {
		'trial_nr' : pd.Series(np.array(self.parameters_joined.trial_nr)),
		'run_nr' : pd.Series(run_nr),
		'omissions' : pd.Series(np.array(self.omissions)),
		'pupil_mean' : pd.Series(np.array(self.parameters_joined.ppr_mean_lp)),
		'pupil_proj' : pd.Series(np.array(self.ppr)),
		'pupil_b' : pd.Series(np.array(self.bpd)),
		'hit' : pd.Series(np.array(self.hit)),
		'fa' : pd.Series(np.array(self.fa)),
		'miss' : pd.Series(np.array(self.miss)),
		'cr' : pd.Series(np.array(self.cr)),
		'rt' : pd.Series(np.array(self.rt)/1000.0),
		}
		data = pd.DataFrame(d)
		data.to_csv(os.path.join(self.project_directory, 'data', self.subject.initials, 'pupil_data.csv'))
		
	def timelocked_plots(self):
		
		# baseline:
		cue_timings = [-499, 4000]
		choice_timings = [-2499, 2000]
		cue_data = self.cue_locked_array_joined[:, cue_timings[0]+1000:cue_timings[1]+1000]
		choice_data = self.choice_locked_array_joined[:, choice_timings[0]+4000:choice_timings[1]+4000]
		for i in range(len(self.bpd)):
			cue_data[i,:] = cue_data[i,:] - self.bpd[i] 
			choice_data[i,:] = choice_data[i,:] - self.bpd[i]
		
		if self.experiment == 1:
			
			# baseline:
			feed_timings = [-499, 2000]
			feed_data = self.feedback_locked_array_joined[:,feed_timings[0]+1000:feed_timings[1]+1000]
			for i in range(len(self.bpd_feed)):
				feed_data[i,:] = feed_data[i,:] - self.bpd_feed[i]		
			
			# indices:
			if self.subject.initials == 'dh':
				conf1 = np.array(self.parameters_joined['confidence'] == 1)
			else:
				conf1 = np.array(self.parameters_joined['confidence'] == 0)
			conf2 = np.array(self.parameters_joined['confidence'] == 1)
			conf3 = np.array(self.parameters_joined['confidence'] == 2)
			conf4 = np.array(self.parameters_joined['confidence'] == 3)
			correct_conf1 = conf1 * (self.hit + self.cr)
			correct_conf2 = conf2 * (self.hit + self.cr)
			correct_conf3 = conf3 * (self.hit + self.cr)
			correct_conf4 = conf4 * (self.hit + self.cr)
			error_conf1 = conf1 * (self.fa + self.miss)
			error_conf2 = conf2 * (self.fa + self.miss)
			error_conf3 = conf3 * (self.fa + self.miss)
			error_conf4 = conf4 * (self.fa + self.miss)
			yes_conf1 = conf1 * (self.hit + self.fa)
			yes_conf2 = conf2 * (self.hit + self.fa)
			yes_conf3 = conf3 * (self.hit + self.fa)
			yes_conf4 = conf4 * (self.hit + self.fa)
			no_conf1 = conf1 * (self.cr + self.miss)
			no_conf2 = conf2 * (self.cr + self.miss)
			no_conf3 = conf3 * (self.cr + self.miss)
			no_conf4 = conf4 * (self.cr + self.miss)
		
		for aaaaaaaaa in range(4):
			
			if self.experiment == 1:
				if aaaaaaaaa == 0:
			
					condition = [self.hit, self.fa, self.miss, self.cr]
					colors = ['r', 'r', 'b', 'b']
					alphas = [1,0.5,0.5,1]
					labels = ['H', 'FA', 'M', 'CR']
					filename = ''
			
				if aaaaaaaaa == 1:
			
					condition = [conf1, conf2, conf3, conf4]
					colors = ['b', 'b', 'r', 'r']
					alphas = [1,0.5,0.5,1]
					labels = ['--', '-', '+', '++']
					filename = 'confidence_'
			
				if aaaaaaaaa == 2:
			
					condition = [correct_conf1, correct_conf2, correct_conf3, correct_conf4, error_conf1, error_conf2, error_conf3, error_conf4]
					colors = ['g', 'g', 'g', 'g', 'r', 'r', 'r', 'r']
					alphas = [0.25,0.5,0.75,1,0.25,0.5,0.75,1]
					labels = ['correct --', 'correct -', 'correct +', 'correct ++', 'error --', 'error -', 'error +', 'error ++']
					filename = 'confidence_correct_'
			
				if aaaaaaaaa == 3:
			
					condition = [yes_conf1, yes_conf2, yes_conf3, yes_conf4, no_conf1, no_conf2, no_conf3, no_conf4]
					colors = ['r', 'r', 'r', 'r', 'b', 'b', 'b', 'b']
					alphas = [0.25,0.5,0.75,1,0.25,0.5,0.75,1]
					labels = ['yes --', 'yes -', 'yes +', 'yes ++', 'no --', 'no -', 'no +', 'no ++']
					filename = 'confidence_yes_'
			
			if self.experiment == 2:
				condition = [self.hit, self.fa, self.miss, self.cr]
				colors = ['r', 'r', 'b', 'b']
				alphas = [1,0.5,0.5,1]
				labels = ['H', 'FA', 'M', 'CR']
				filename = ''
			
			print filename
			
			# ----------------------
			# do some plotting:    -
			# ----------------------
	
			# create downsampled means and sems:
			cue_means = []
			cue_sems = []
			choice_means = []
			choice_sems = []
			for i in range(len(condition)):
				cue_means.append(sp.signal.decimate(bn.nanmean(cue_data[condition[i]], axis=0), self.downsample_rate, 1))
				cue_sems.append(sp.signal.decimate(bn.nanstd(cue_data[condition[i]], axis=0), self.downsample_rate, 1) / sp.sqrt(condition[i].sum()))
				choice_means.append(sp.signal.decimate(bn.nanmean(choice_data[condition[i]], axis=0), self.downsample_rate, 1))
				choice_sems.append(sp.signal.decimate(bn.nanstd(choice_data[condition[i]], axis=0), self.downsample_rate, 1) / sp.sqrt(condition[i].sum()))
	
			# stuff for ylim:
			max_y_cue = max(np.concatenate(np.vstack(cue_means) + np.vstack(cue_sems)))
			min_y_cue = min(np.concatenate(np.vstack(cue_means) - np.vstack(cue_sems)))
			diff_cue = max_y_cue - min_y_cue
			max_y_choice = max(np.concatenate(np.vstack(choice_means) + np.vstack(choice_sems)))
			min_y_choice = min(np.concatenate(np.vstack(choice_means) - np.vstack(choice_sems)))
			diff_choice = max_y_choice - min_y_choice
			max_y = max((max_y_cue, max_y_choice))
			min_y = max((min_y_cue, min_y_choice))
			diff = abs(max_y - min_y)
	
			# cue locked plot:
			fig = plt.figure(figsize=(4, 3))
			a = plt.subplot(111)
			x = np.linspace(cue_timings[0], cue_timings[1], len(cue_means[0]))
			for i in range(len(condition)):
				a.plot(x, cue_means[i], linewidth=2, color=colors[i], alpha=alphas[i], label=labels[i] + ' ' + str(condition[i].sum()) + ' trials')
				a.fill_between(x, cue_means[i] + cue_sems[i], cue_means[i] - cue_sems[i], color=colors[i], alpha=0.1)
			# a.set_xlim((-500, 4000))
			leg = plt.legend(loc=2, fancybox=True)
			leg.get_frame().set_alpha(0.9)
			if leg:
				for t in leg.get_texts():
					t.set_fontsize(7)

				for l in leg.get_lines():
					l.set_linewidth(2)
			a.axes.tick_params(axis='both', which='major', labelsize=8)
			a.set_xticks((0, 1000, 2000, 3000, 4000))
			a.set_xticklabels((0, 1, 2, 3, 4))
			if diff < 0.5:
				a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
			elif diff < 1:
				a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
			else:
				a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
			a.set_ylim(ymin=min_y - diff / 20.0, ymax=max_y + diff / 20.0)
			a.set_title('{}; c = {}'.format(self.subject.initials, round(self.criterion, 3)), size=12)
			a.set_ylabel('Pupil diameter (s.d.)', size=10)
			a.set_xlabel('Time from cue (s)', size=10)
			a.vlines(np.mean(self.rt[(condition[0] + condition[1])]), plt.axis()[2], plt.axis()[3], color='r', linestyle='--', alpha=0.5)
			a.vlines(np.mean(self.rt[(condition[2] + condition[3])]), plt.axis()[2], plt.axis()[3], color='b', linestyle='--', alpha=0.5)
			a.vlines(0, plt.axis()[2], plt.axis()[3], linewidth=1)
			plt.tight_layout()
			fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_response_cue_locked_' + filename + self.subject.initials + '.pdf'))
	
			# choice locked plot:
			fig = plt.figure(figsize=(4, 3))
			a = plt.subplot(111)
			x = np.linspace(choice_timings[0], choice_timings[1], len(choice_means[0]))
			for i in range(len(condition)):
				a.plot(x, choice_means[i], linewidth=2, color=colors[i], alpha=alphas[i], label=labels[i] + ' ' + str(condition[i].sum()) + ' trials')
				a.fill_between(x, choice_means[i] + choice_sems[i], choice_means[i] - choice_sems[i], color=colors[i], alpha=0.1)
			a.set_xlim((-2500, 2000))
			leg = plt.legend(loc=2, fancybox=True)
			leg.get_frame().set_alpha(0.9)
			if leg:
				for t in leg.get_texts():
					t.set_fontsize(7)
				for l in leg.get_lines():
					l.set_linewidth(2)
			a.axes.tick_params(axis='both', which='major', labelsize=8)
			a.set_xticks((-2000, -1000, 0, 1000, 2000))
			a.set_xticklabels((-2, -1, 0, 1, 2))
			if diff < 0.5:
				a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
			elif diff < 1:
				a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
			else:
				a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
			a.set_ylim(ymin=min_y - diff / 20.0, ymax=max_y + diff / 20.0)
			a.set_title('{}; c = {}'.format(self.subject.initials, round(self.criterion, 3)), size=12)
			a.set_ylabel('Pupil diameter (s.d.)', size=10)
			a.set_xlabel('Time from choice (s)', size=10)
			a.vlines(0 - np.mean(self.rt[(condition[0] + condition[1])]), plt.axis()[2], plt.axis()[3], color='r', linestyle='--', alpha=0.5)
			a.vlines(0 - np.mean(self.rt[(condition[2] + condition[3])]), plt.axis()[2], plt.axis()[3], color='b', linestyle='--', alpha=0.5)
			a.vlines(0, plt.axis()[2], plt.axis()[3], linewidth=1)
			plt.tight_layout()
			fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_response_choice_locked_' + filename + self.subject.initials + '.pdf'))
	
			if self.experiment == 1:
		
				# create downsampled means and sems:
				feed_means = []
				feed_sems = []
				for i in range(len(condition)):
					feed_means.append(sp.signal.decimate(bn.nanmean(feed_data[condition[i]], axis=0), self.downsample_rate, 1))
					feed_sems.append(sp.signal.decimate(bn.nanstd(feed_data[condition[i]], axis=0), self.downsample_rate, 1) / sp.sqrt(condition[i].sum()))
		
				# stuff for ylim:
				max_y_feed = max(np.concatenate( np.vstack(feed_means)+np.vstack(feed_sems) ))
				min_y_feed = min(np.concatenate( np.vstack(feed_means)-np.vstack(feed_sems) ))
				diff_feed = max_y_feed - min_y_feed
		
				# feedback locked plot:
				fig = plt.figure(figsize=(4, 3))
				a = plt.subplot(111)
				x = np.linspace(feed_timings[0], feed_timings[1], len(feed_means[0]))
				for i in range(len(condition)):
					a.plot(x, feed_means[i], linewidth=2, color=colors[i], alpha=alphas[i], label=labels[i] + ' ' + str(condition[i].sum()) + ' trials')
					a.fill_between(x, feed_means[i] + feed_sems[i], feed_means[i] - feed_sems[i], color=colors[i], alpha=0.1)
				a.set_xlim((-500, 2000))
				leg = plt.legend(loc=2, fancybox=True)
				leg.get_frame().set_alpha(0.9)
				if leg:
					for t in leg.get_texts():
						t.set_fontsize(7)
					for l in leg.get_lines():
						l.set_linewidth(2)
				a.axes.tick_params(axis='both', which='major', labelsize=8)
				a.set_xticks([-500,-0,500,1000,1500,2000])
				a.set_xticklabels([-.5,0,.5,1,1.5,2.0])
				a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
				a.set_ylim(ymin=min_y_feed-(diff_feed/20.0), ymax=max_y_feed+(diff_feed/20.0))
				a.set_title('{}; c = {}'.format(self.subject.initials, round(self.criterion, 3)), size=12)
				a.set_ylabel('Pupil diameter (s.d.)', size=10)
				a.set_xlabel('Time from feedback (s)', size=10)
				a.vlines(0, plt.axis()[2], plt.axis()[3], linewidth=1)
				plt.tight_layout()
				fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_response_feedback_locked_' + filename + self.subject.initials + '.pdf'))
				
				if aaaaaaaaa == 2:
					shell()
					
					means = np.vstack(feed_means)
					np.savetxt(os.path.join(self.base_directory, 'figs', 'means.csv'), means, delimiter=",")
					sems = np.vstack(feed_sems)
					np.savetxt(os.path.join(self.base_directory, 'figs', 'sems.csv'), sems, delimiter=",")
				
				
	def behavior_confidence(self):
		
		conf1 = np.array(self.parameters_joined['confidence'] == 0)
		conf2 = np.array(self.parameters_joined['confidence'] == 1)
		conf3 = np.array(self.parameters_joined['confidence'] == 2)
		conf4 = np.array(self.parameters_joined['confidence'] == 3)
		
		# conditions = [conf1+conf2, conf3+conf4]
		conditions = [conf1, conf2, conf3, conf4]
		
		d_prime = []
		criterion = []
		for cond in conditions:
			d, c = myfuncs.SDT_measures((self.hit+self.miss)[cond], self.hit[cond], self.fa[cond],)
			d_prime.append(d)
			criterion.append(c)
		
		my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

		N = 4
		ind = np.linspace(0,2,N)  # the x locations for the groups
		bar_width = 0.6   # the width of the bars
		spacing = [0, 0, 0, 0]
		
		# FIGURE 1
		MEANS = np.array(d_prime)
		MEANS2 = np.array(criterion)
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		ax.plot(ind, MEANS, color = 'g', label="d'")
		ax2 = ax.twinx()
		ax2.plot(ind, MEANS2, color = 'k', label='criterion')
		ax.set_xticklabels( ('--','-','+','++') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		leg = plt.legend(loc=2, fancybox=True)
		leg.get_frame().set_alpha(0.9)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize(7)
			for l in leg.get_lines():
				l.set_linewidth(2)
		ax.set_ylabel("d'")
		ax2.set_ylabel('criterion')
		ax.set_xlabel('confidence')
		ax.set_ylim(ymin=0, ymax=2)
		ax2.set_ylim(ymin=-0.7, ymax=0.7)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
		plt.tight_layout()
		fig.savefig(os.path.join(self.base_directory, 'figs', "behavior_confidence_" + self.subject.initials + '.pdf'))
	
	def pupil_bars(self):
		
		conf1 = np.array(self.parameters_joined['confidence'] == 0)
		conf2 = np.array(self.parameters_joined['confidence'] == 1)
		conf3 = np.array(self.parameters_joined['confidence'] == 2)
		conf4 = np.array(self.parameters_joined['confidence'] == 3)
		
		conditions = [conf1, conf2, conf3, conf4]
		
		my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

		N = 4
		ind = np.linspace(0,2,N)  # the x locations for the groups
		bar_width = 0.6   # the width of the bars
		spacing = [0, 0, 0, 0]
		
		# FIGURE 1
		MEANS_yes = np.array([np.mean(self.ppr[(self.hit+self.fa)*cond]) for cond in conditions])
		SEMS_yes = np.array([sp.stats.sem(self.ppr[(self.hit+self.fa)*cond]) for cond in conditions])
		MEANS_no = np.array([np.mean(self.ppr[(self.miss+self.cr)*cond]) for cond in conditions])
		SEMS_no = np.array([sp.stats.sem(self.ppr[(self.miss+self.cr)*cond]) for cond in conditions])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		ax.errorbar(ind, MEANS_yes, yerr=SEMS_yes, color = 'r', capsize = 0)
		ax.errorbar(ind, MEANS_no, yerr=SEMS_no, color = 'b', capsize = 0)
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('--','-','+','++') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		ax.set_ylim(ymin=0.2, ymax=1.6)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.xlabel('confidence')
		plt.ylabel('pupil response (a.u.)')
		plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
		plt.tight_layout()
		fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_choice_2_' + self.subject.initials + '.pdf'))
		
		# FIGURE 1
		MEANS = np.array([np.mean(self.ppr[(self.hit+self.fa)*cond])-np.mean(self.ppr[(self.miss+self.cr)*cond]) for cond in conditions])
		SEMS = np.array([(sp.stats.sem(self.ppr[(self.hit+self.fa)*cond])+sp.stats.sem(self.ppr[(self.miss+self.cr)*cond]))/2 for cond in conditions])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		ax.errorbar(ind, MEANS, yerr=SEMS, color = 'k', capsize = 0)
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('--','-','+','++') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		ax.set_ylim(ymin=-0.4, ymax=1.0)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.xlabel('confidence')
		plt.ylabel('pupil response (a.u.)')
		plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
		plt.tight_layout()
		fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_choice_' + self.subject.initials + '.pdf'))
		
		# FIGURE 2
		MEANS = np.array([np.mean(self.ppr[(self.hit+self.cr)*cond])-np.mean(self.ppr[(self.fa+self.miss)*cond]) for cond in conditions])
		SEMS = np.array([(sp.stats.sem(self.ppr[(self.hit+self.cr)*cond])+sp.stats.sem(self.ppr[(self.fa+self.miss)*cond]))/2 for cond in conditions])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = 'b', alpha = [.25,0.5,0.75,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('--','-','+','++') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.ylabel('pupil correctness effect (a.u.)')
		plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
		plt.tight_layout()
		fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_correct_' + self.subject.initials + '.pdf'))
		
		# FIGURE 3
		MEANS = np.array([np.mean(self.ppr[(self.hit+self.cr+self.fa+self.miss)*cond]) for cond in conditions])
		SEMS = np.array([sp.stats.sem(self.ppr[(self.hit+self.cr+self.fa+self.miss)*cond]) for cond in conditions])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = 'b', alpha = [.25,0.5,0.75,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('--','-','+','++') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.ylabel('pupil response (a.u.)')
		plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
		plt.tight_layout()
		fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_overall_' + self.subject.initials + '.pdf'))
		
		if self.experiment == 1:
			# FIGURE 4
			MEANS_yes = np.array([np.mean(self.ppr_feed[(self.hit+self.cr)*cond]) for cond in conditions])
			SEMS_yes = np.array([sp.stats.sem(self.ppr_feed[(self.hit+self.cr)*cond]) for cond in conditions])
			MEANS_no = np.array([np.mean(self.ppr_feed[(self.fa+self.miss)*cond]) for cond in conditions])
			SEMS_no = np.array([sp.stats.sem(self.ppr_feed[(self.fa+self.miss)*cond]) for cond in conditions])
			fig = plt.figure(figsize=(4,3))
			ax = fig.add_subplot(111)
			ax.errorbar(ind, MEANS_no-MEANS_yes, yerr=(SEMS_yes+SEMS_no)/2.0, color = 'k', capsize = 0)
			simpleaxis(ax)
			spine_shift(ax)
			ax.set_xticklabels( ('--','-','+','++') )
			ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
			ax.tick_params(axis='x', which='major', labelsize=10)
			ax.tick_params(axis='y', which='major', labelsize=10)
			# ax.set_ylim(ymin=0.2, ymax=1.6)
			plt.gca().spines["bottom"].set_linewidth(.5)
			plt.gca().spines["left"].set_linewidth(.5)
			plt.xlabel('confidence')
			plt.ylabel('prediction error (a.u.)')
			plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
			plt.tight_layout()
			fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_prediction_error_' + self.subject.initials + '.pdf'))
			
	
	def sequential_effects(self):
		
		high_feedback_n = np.concatenate((np.array([False]), np.array(self.parameters_joined['ppr_peak_feed_lp'] > np.median(self.parameters_joined['ppr_peak_feed_lp']))))[:-1]
		
		d, c = myfuncs.SDT_measures(np.array(self.parameters_joined['present'])[high_feedback_n], np.array(self.parameters_joined['hit'])[high_feedback_n], np.array(self.parameters_joined['fa'])[high_feedback_n])
		d1, c1 = myfuncs.SDT_measures(np.array(self.parameters_joined['present'])[-high_feedback_n], np.array(self.parameters_joined['hit'])[-high_feedback_n], np.array(self.parameters_joined['fa'])[-high_feedback_n])
		
		print
		print
		print self.subject.initials
		print '---------------------'
		print 'd prime: {} vs {}'.format(d, d1)
		print 'criterion: {} vs {}'.format(c, c1)
		print
		print
		
	def rescorla_wagner(self):
		
		# rescorla wagner model:
		def simulate_run(reward_history, conf_history, learning_rate=.2):
			global_reward = np.zeros(len(reward_history))
			global_error = np.zeros(len(reward_history))
			local_error = np.zeros(len(reward_history))
			for i in xrange(0, len(reward_history)):
				local_error[i] = reward_history[i] - conf_history[i]
			for i in xrange(0, len(reward_history)): 
				try:
					global_error[i] = global_reward[i] - reward_history[i]
					global_reward[i+1] = global_reward[i] + (learning_rate * global_error[i])
				except IndexError:
					pass
			return global_reward, global_error, local_error
		
		# variables:
		correct = np.array(parameters_joined['correct'] == 1, dtype=int)[-np.array(parameters_joined['omissions'])]
		conf = np.array(parameters_joined['confidence'])[-np.array(parameters_joined['omissions'])]
		conf[conf == 0] = .20
		conf[conf == 1] = .40
		conf[conf == 2] = .60
		conf[conf == 3] = .80
		ppd = np.array(parameters_joined['ppr_peak_feed_lp'])[-np.array(parameters_joined['omissions'])]
		global_reward, global_error, local_error = simulate_run(reward_history=correct, conf_history=conf, learning_rate=1.0)
		
		# boxplot:
		import matplotlib.collections as collections
		errors = np.unique(local_error)
		data = [ppd[local_error == error] for error in errors] 
		fig = plt.figure(figsize=(3,5))
		ax = plt.subplot(111)
		ax.boxplot(data)
		ax.set_xticklabels(errors)
		c1 = collections.BrokenBarHCollection(xranges=[(0.0,4.5)], yrange=ax.get_ylim(), facecolor='red', alpha=0.25)
		c2 = collections.BrokenBarHCollection(xranges=[(4.5,9.0)], yrange=ax.get_ylim(), facecolor='green', alpha=0.25)
		ax.add_collection(c1)
		ax.add_collection(c2)
		ax.set_xlabel('prediction error')
		ax.set_ylabel('pupil response (a.u.)')
		fig.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_suprise.pdf'))
		
	def GLM(self):
		
		# create pupil timeseries:
		choice_timings = [-2999, 3000]
		pupil_time_series = self.choice_locked_array_joined[:, choice_timings[0]+4000:choice_timings[1]+4000]
		for i in range(pupil_time_series.shape[0]):
			pupil_time_series[i,:] = pupil_time_series[i,:] - self.bpd[i]
		pupil_time_series = pupil_time_series[-self.omissions,:]
		pupil_time_series = np.concatenate(pupil_time_series)
		
		# create events:
		nr_trials = sum(-self.omissions)
		
		event_a = np.zeros((nr_trials, 3))
		event_a[:,0] = np.cumsum(np.repeat(6, nr_trials)) - 3
		event_a[:,1] = 0
		event_a[:,2] = 1
		
		events = [event_a]
		
		linear_model = GLM.GeneralLinearModel(input_object=pupil_time_series, event_object=events, sample_dur=0.001, new_sample_dur=0.05)
		linear_model.configure(IRF_type='pupil', IRF_params={'dur':3, 's':1.0/(10**26), 'n':10.1, 'tmax':0.93}, regressor_types=['stick'])
		linear_model.execute()
		
		
class pupilAnalysesAcross(object):
	def __init__(self, subjects, experiment_name, project_directory, sample_rate_new=50):
		
		self.subjects = subjects
		self.nr_subjects = len(self.subjects)
		self.experiment_name = experiment_name
		self.project_directory = project_directory
		self.sample_rate_new = int(sample_rate_new)
		self.downsample_rate = int(1000 / sample_rate_new)
		
		parameters = []
		for s in self.subjects:
			self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
			self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
			self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
			
			try:
				parameters.append(self.ho.read_session_data('', 'parameters_joined'))
			except:
				shell()
		self.parameters_joined = pd.concat(parameters)
		self.omissions = np.array(self.parameters_joined['omissions'])
		self.parameters_joined = self.parameters_joined[-self.omissions]
		
		self.rt = self.parameters_joined['rt']
		
		self.hit = np.array(self.parameters_joined['hit'], dtype=bool)
		self.fa = np.array(self.parameters_joined['fa'], dtype=bool)
		self.miss = np.array(self.parameters_joined['miss'], dtype=bool)
		self.cr = np.array(self.parameters_joined['cr'], dtype=bool)
		
		self.yes = np.array(self.parameters_joined['yes'], dtype=bool)
		self.no = -np.array(self.parameters_joined['yes'], dtype=bool)
		
		try:
			self.present = np.array(self.parameters_joined['signal_present'], dtype=bool)
		except:
			self.present = np.array(self.parameters_joined['target_present_in_stimulus'], dtype=bool)
		self.absent = -self.present
		self.correct = np.array(self.parameters_joined['correct'], dtype=bool)
		self.error = -np.array(self.parameters_joined['correct'], dtype=bool)
		
		self.bpd = np.array(self.parameters_joined['bpd_lp'])
		self.ppr = np.array(self.parameters_joined['ppr_proj_lp'])
		
		self.subj_idx = np.concatenate(np.array([np.repeat(i, sum(self.parameters_joined['subject'] == self.subjects[i])) for i in range(len(self.subjects))]))
		
		self.criterion = np.array([np.array(self.parameters_joined[self.parameters_joined['subject']==subj]['criterion'])[0] for subj in self.subjects])
		
		self.pupil_l_ind = np.concatenate([self.parameters_joined.ppr_mean_lp[self.parameters_joined.subject == subj_idx] <= np.percentile(self.parameters_joined.ppr_mean_lp[self.parameters_joined.subject == subj_idx], 33) for subj_idx in self.subjects])
		self.pupil_h_ind = np.concatenate([self.parameters_joined.ppr_mean_lp[self.parameters_joined.subject == subj_idx] >= np.percentile(self.parameters_joined.ppr_mean_lp[self.parameters_joined.subject == subj_idx], 66) for subj_idx in self.subjects])
		self.pupil_rest_ind = -(self.pupil_h_ind + self.pupil_l_ind)
		
	
	def behavior(self):
		
		sns.set_style("ticks")
		
		# initialize behavior operator:
		d = {
		'subj_idx' : pd.Series(self.subj_idx),
		'correct' : pd.Series(np.array(self.parameters_joined['correct'], dtype=int)),
		'choice' : pd.Series(np.array(self.parameters_joined['yes'], dtype=int)),
		'stimulus' : pd.Series(np.array(self.parameters_joined['present'], dtype=int)),
		'rt' : pd.Series(np.array(self.parameters_joined['rt'])) / 1000.0,
		'pupil' : pd.Series(np.array(self.ppr)),
		'pupil_b' : pd.Series(np.array(self.bpd)),
		'pupil_high' : pd.Series(self.pupil_h_ind)
		}
		df = pd.DataFrame(d)[-self.pupil_rest_ind]
		behavior = myfuncs.behavior(df)
		
		# SDT fractions:
		yes_f, no_f, cr_f, miss_f, hit_f, fa_f = behavior.choice_fractions()
		MEANS_correct = (np.mean(hit_f), np.mean(cr_f))
		SEMS_correct = (sp.stats.sem(hit_f), sp.stats.sem(cr_f))
		MEANS_error = (np.mean(miss_f), np.mean(fa_f))
		SEMS_error = (sp.stats.sem(miss_f), sp.stats.sem(fa_f))
		N = 2
		ind = np.linspace(0,N/2,N)
		bar_width = 0.50
		fig = plt.figure(figsize=(2,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(ind[i], height=MEANS_correct[i], width = bar_width, yerr = SEMS_correct[i], color = ['r', 'b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		for i in range(N):
			ax.bar(ind[i], height=MEANS_error[i], bottom = MEANS_correct[i], width = bar_width, color = ['b', 'r'][i], alpha = 0.5, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		ax.set_ylabel('fraction of trials', size = 10)
		ax.set_title('SDT fractions', size = 12)
		ax.set_xticks( (ind) )
		ax.set_xticklabels( ('signal+\nnoise', 'noise') )
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'SDT_fractions.pdf'))
		
		# RT histograms:
		# --------------
		
		x_grid = [0.5, 3.5, 100]
		c0_pdf, c1_pdf, c0_correct_pdf, c0_error_pdf, c1_correct_pdf, c1_error_pdf = behavior.rt_kernel_densities(x_grid=x_grid, bandwidth=0.07)
		
		yes = np.vstack(c0_pdf)
		no = np.vstack(c1_pdf)
		cr = np.vstack(c0_correct_pdf)
		miss = np.vstack(c0_error_pdf)
		hit = np.vstack(c1_correct_pdf)
		fa = np.vstack(c1_error_pdf)
		
		step = pd.Series(np.linspace(x_grid[0], x_grid[1], x_grid[2]), name='rt (s)')
		
		# Make the plt.plot
		fig = plt.figure(figsize=(4, 5))
		ax = plt.subplot(211)
		conditions = pd.Series(['hits'], name='trial type')
		sns.tsplot(hit, time=step, condition=conditions, value='kde', color='red', ci=66, lw=1, ls='-', ax=ax)
		conditions = pd.Series(['miss'], name='trial type')
		sns.tsplot(miss, time=step, condition=conditions, value='kde', color='blue', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
		# ax.axvline(np.mean([np.median(df.rt[(df.choice==0) & (df.correct==1) & (df.subj_idx==i])) for i in range(self.nr_subjects)]), color='r', linestyle='--')
		# ax.axvline(np.mean([np.median(df.rt[(df.choice==0) & (df.correct==1) & (df.subj_idx==i])) for i in range(self.nr_subjects)]), color='b', linestyle='--')
		ax = plt.subplot(212)
		conditions = pd.Series(['cr'], name='trial type')
		sns.tsplot(cr, time=step, condition=conditions, value='kde', color='blue', ci=66, lw=1, ls='-', ax=ax)
		conditions = pd.Series(['fa'], name='trial type')
		sns.tsplot(fa, time=step, condition=conditions, value='kde', color='red', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
		# ax.axvline(np.mean([np.median(df.rt[(df.choice==0) & (df.correct==1) & (df.subj_idx==i])) for i in range(self.nr_subjects)]), color='b', linestyle='--')
		# ax.axvline(np.mean([np.median(df.rt[(df.choice==0) & (df.correct==1) & (df.subj_idx==i])) for i in range(self.nr_subjects)]), color='r', linestyle='--')
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'rt_hists.pdf'))

		
	def behavior_two_conditions(self):
		
		sns.set(style="ticks")
		
		for j in range(2):
				
			if j == 0:
				ind = self.pupil_l_ind
			if j == 1:
				ind = self.pupil_h_ind
			
			# SDT fractions:
			# --------------
			hit_fraction = np.zeros(len(self.subjects))
			fa_fraction = np.zeros(len(self.subjects))
			miss_fraction = np.zeros(len(self.subjects))
			cr_fraction = np.zeros(len(self.subjects))
			for i in range(len(self.subjects)):
				hit_fraction[i] = sum(self.hit[(self.subj_idx==i)*ind]) / float(sum(self.present[(self.subj_idx==i)*ind]))
				fa_fraction[i] = sum(self.fa[(self.subj_idx==i)*ind]) / float(sum(self.absent[(self.subj_idx==i)*ind]))
				miss_fraction[i] = sum(self.miss[(self.subj_idx==i)*ind]) / float(sum(self.present[(self.subj_idx==i)*ind]))
				cr_fraction[i] = sum(self.cr[(self.subj_idx==i)*ind]) / float(sum(self.absent[(self.subj_idx==i)*ind]))
			MEANS_correct = (np.mean(hit_fraction), np.mean(cr_fraction))
			SEMS_correct = (sp.stats.sem(hit_fraction), sp.stats.sem(cr_fraction))
			MEANS_error = (np.mean(miss_fraction), np.mean(fa_fraction))
			SEMS_error = (sp.stats.sem(miss_fraction), sp.stats.sem(fa_fraction))
			
			N = 2
			locs = np.linspace(0,N/2,N)  # the x locations for the groups
			bar_width = 0.50	   # the width of the bars
			fig = plt.figure(figsize=(2,3))
			ax = fig.add_subplot(111)
			for i in range(N):
				ax.bar(locs[i], height=MEANS_correct[i], width = bar_width, yerr = SEMS_correct[i], color = ['r', 'b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
			for i in range(N):
				ax.bar(locs[i], height=MEANS_error[i], bottom = MEANS_correct[i], width = bar_width, color = ['b', 'r'][i], alpha = 0.5, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
			ax.axhline(np.mean(MEANS_correct), color='g', ls='--')
			print np.mean(MEANS_correct)
			ax.set_ylabel('fraction of trials', size = 10)
			ax.set_title('SDT fractions', size = 12)
			ax.set_xticks( (locs) )
			ax.set_xticklabels( ('signal+\nnoise', 'noise') )
			ax.set_ylim((0,1))
			plt.gca().spines["bottom"].set_linewidth(.5)
			plt.gca().spines["left"].set_linewidth(.5)
			plt.tight_layout()
			sns.despine(offset=10, trim=True)
			fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_fractions_{}.pdf'.format(j)))
			
			# RT histograms:
			# --------------
			nbins = 20
			
			RESPONSE_TIME = np.array(self.rt)/1000.0
			# max_y = max( max(plt.hist(RESPONSE_TIME[self.hit*ind], bins=nbins)[0]), max(plt.hist(RESPONSE_TIME[self.cr*ind], bins=nbins)[0]) )
			max_y = 80
			y1,binEdges1 = np.histogram(RESPONSE_TIME[self.hit*ind],bins=nbins)
			bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
			y2,binEdges2 = np.histogram(RESPONSE_TIME[self.fa*ind],bins=nbins)
			bincenters2 = 0.5*(binEdges2[1:]+binEdges2[:-1])
			y3,binEdges3 = np.histogram(RESPONSE_TIME[self.miss*ind],bins=nbins)
			bincenters3 = 0.5*(binEdges3[1:]+binEdges3[:-1])
			y4,binEdges4 = np.histogram(RESPONSE_TIME[self.cr*ind],bins=nbins)
			bincenters4 = 0.5*(binEdges4[1:]+binEdges4[:-1])
		
			fig = plt.figure(figsize=(4, 5))
			# present:
			a = plt.subplot(211)
			a.plot(bincenters1,y1,'-', color='r', label='hit')
			a.plot(bincenters3,y3,'-', color='b', alpha=0.5, label='miss')
			a.legend()
			a.set_ylim(ymax=max_y)
			a.set_xlim(xmin=0.25, xmax=3.5)
			simpleaxis(a)
			spine_shift(a)
			a.axes.tick_params(axis='both', which='major', labelsize=8)
			a.set_title('RT histograms', size = 12)
			a.set_ylabel('# trials')
			a.axvline(np.median(RESPONSE_TIME[self.hit*ind]), color='r', linestyle='--')
			a.axvline(np.median(RESPONSE_TIME[self.miss*ind]), color='b', linestyle='--')
			plt.gca().spines["bottom"].set_linewidth(.5)
			plt.gca().spines["left"].set_linewidth(.5)
			# absent:
			b = plt.subplot(212)
			b.plot(bincenters2,y2,'-', color='r', alpha=0.5, label='fa')
			b.plot(bincenters4,y4,'-', color='b', label='cr')
			b.legend()
			b.set_ylim(ymax=max_y)
			b.set_xlim(xmin=0.25, xmax=3.5)
			simpleaxis(b)
			spine_shift(b)
			b.axes.tick_params(axis='both', which='major', labelsize=8)
			b.set_xlabel('RT (s)')
			b.set_ylabel('# trials')
			b.axvline(np.median(RESPONSE_TIME[self.fa*ind]), color='r', linestyle='--')
			b.axvline(np.median(RESPONSE_TIME[self.cr*ind]), color='b', linestyle='--')
			plt.gca().spines["bottom"].set_linewidth(.5)
			plt.gca().spines["left"].set_linewidth(.5)
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_{}.pdf'.format(j)))
			
			
			

		# SDT fractions:
		# --------------
		hit_fraction = np.zeros(len(self.subjects))
		fa_fraction = np.zeros(len(self.subjects))
		miss_fraction = np.zeros(len(self.subjects))
		cr_fraction = np.zeros(len(self.subjects))
		for i in range(len(self.subjects)):
			hit_fraction[i] = sum(self.yes[(self.subj_idx==i)*self.pupil_l_ind]) / float(sum(self.pupil_l_ind[self.subj_idx==i]))
			fa_fraction[i] = sum(self.no[(self.subj_idx==i)*self.pupil_h_ind]) / float(sum(self.pupil_h_ind[self.subj_idx==i]))
			miss_fraction[i] = sum(self.no[(self.subj_idx==i)*self.pupil_l_ind]) / float(sum(self.pupil_l_ind[self.subj_idx==i]))
			cr_fraction[i] = sum(self.yes[(self.subj_idx==i)*self.pupil_h_ind]) / float(sum(self.pupil_h_ind[self.subj_idx==i]))
		MEANS_correct = (np.mean(hit_fraction), np.mean(cr_fraction))
		SEMS_correct = (sp.stats.sem(hit_fraction), sp.stats.sem(cr_fraction))
		MEANS_error = (np.mean(miss_fraction), np.mean(fa_fraction))
		SEMS_error = (sp.stats.sem(miss_fraction), sp.stats.sem(fa_fraction))
		
		N = 2
		locs = np.linspace(0,N/2,N)  # the x locations for the groups
		bar_width = 0.50	   # the width of the bars
		fig = plt.figure(figsize=(2,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(locs[i], height=MEANS_correct[i], width = bar_width, yerr = SEMS_correct[i], color = ['r', 'r'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		for i in range(N):
			ax.bar(locs[i], height=MEANS_error[i], bottom = MEANS_correct[i], width = bar_width, color = ['b', 'b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		ax.axhline(np.mean(MEANS_correct), color='g', ls='--')
		print np.mean(MEANS_correct)
		ax.set_ylabel('fraction of yes trials', size = 10)
		ax.set_xticks( (locs) )
		ax.set_xticklabels( ('low', 'high') )
		ax.set_ylim((0,0.6))
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.tight_layout()
		sns.despine(offset=10, trim=True)
		fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_fractions_2_{}.pdf'.format(j)))
	
	
	
	def rt_distributions(self, bins=25):
		
		quantiles = [0.5, 10, 30, 50, 70, 90, 99.5]
		
		data = self.parameters_joined
		pupil = 'ppr_proj_lp'
		
		data.rt = data.rt / 1000.0
		
		
		plot_width = self.nr_subjects * 4
		
		# plot 1 -- rt combined
		plot_nr = 1
		fig = plt.figure(figsize=(plot_width,4))
		for i in xrange(self.nr_subjects):
			
			ax1 = fig.add_subplot(1,self.nr_subjects,plot_nr)
			data_subj = data[data.subject==self.subjects[i]]
			rt = np.array(data_subj.rt)
			myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax1, xlim=(0,4))
			ax1.set_xlabel('rt')
			plot_nr += 1
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'rt.pdf'))

		# plot 2 -- rt split by pupil
		plot_nr = 1
		fig = plt.figure(figsize=(plot_width,8))
		for i in xrange(self.nr_subjects):

			data_subj = data[data.subject==self.subjects[i]]
			max_ylim = max(max(np.histogram(np.array(data_subj.rt[data_subj[pupil] < np.median(data_subj[pupil])]), bins=bins)[0]), max(np.histogram(np.array(data_subj.rt[data_subj[pupil] > np.median(data_subj[pupil])]), bins=bins)[0]))

			ax1 = plt.subplot(2,self.nr_subjects,plot_nr)
			rt = np.array(data_subj.rt[data_subj[pupil] < np.median(data_subj[pupil])])
			myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax1, quantiles_color='k', alpha=0.75, xlim=(0,4), ylim=(0,max_ylim))
			ax2 = plt.subplot(2,self.nr_subjects,plot_nr+self.nr_subjects)
			rt = np.array(data_subj.rt[data_subj[pupil] > np.median(data_subj[pupil])])
			myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax2, quantiles_color='k', xlim=(0,4), ylim=(0,max_ylim))
			ax1.set_xlabel('rt')
			ax2.set_xlabel('rt')
			plot_nr += 1
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'rt_split_' + pupil + '.pdf'))

		# plot 3 -- rt split by SDT trial types
		plot_nr = 1
		fig = plt.figure(figsize=(plot_width,16))
		for i in xrange(self.nr_subjects):

			data_subj = data[data.subject==self.subjects[i]]
			max_ylim = max(max(np.histogram(np.array(data_subj.rt[data_subj.hit]), bins=bins)[0]), max(np.histogram(np.array(data_subj.rt[data_subj.cr]), bins=bins)[0]))

			ax1 = plt.subplot(4, self.nr_subjects,plot_nr)
			rt = np.array(data_subj.rt[data_subj.hit])
			myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax1, quantiles_color='r', xlim=(0,4), ylim=(0,max_ylim))
			ax2 = plt.subplot(4, self.nr_subjects,plot_nr+self.nr_subjects)
			rt = np.array(data_subj.rt[data_subj.fa])
			myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax2, quantiles_color='r', alpha=0.5, xlim=(0,4), ylim=(0,max_ylim))
			ax3 = plt.subplot(4, self.nr_subjects,plot_nr+(2*self.nr_subjects))
			rt = np.array(data_subj.rt[data_subj.miss])
			myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax3, quantiles_color='b', alpha=0.5, xlim=(0,4), ylim=(0,max_ylim))
			ax4 = plt.subplot(4, self.nr_subjects,plot_nr+(3*self.nr_subjects))
			rt = np.array(data_subj.rt[data_subj.cr])
			myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax4, quantiles_color='b', xlim=(0,4), ylim=(0,max_ylim))
			ax1.set_xlabel('rt')
			ax2.set_xlabel('rt')
			ax3.set_xlabel('rt')
			ax4.set_xlabel('rt')
			plot_nr += 1
		plt.tight_layout()
	
		fig.savefig(os.path.join(self.project_directory, 'figures', 'rt_split_answer.pdf'))
	
	def pupil_bars(self):
		
		ppr_hit = [np.mean(self.ppr[self.subj_idx == i][self.hit[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		ppr_fa = [np.mean(self.ppr[self.subj_idx == i][self.fa[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		ppr_miss = [np.mean(self.ppr[self.subj_idx == i][self.miss[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		ppr_cr = [np.mean(self.ppr[self.subj_idx == i][self.cr[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		
		ppr_yes = [np.mean(self.ppr[self.subj_idx == i][(self.hit+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		ppr_no = [np.mean(self.ppr[self.subj_idx == i][(self.miss+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		ppr_correct = [np.mean(self.ppr[self.subj_idx == i][(self.hit+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		ppr_error = [np.mean(self.ppr[self.subj_idx == i][(self.miss+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		
		bpd_hit = [np.mean(self.bpd[self.subj_idx == i][self.hit[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		bpd_fa = [np.mean(self.bpd[self.subj_idx == i][self.fa[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		bpd_miss = [np.mean(self.bpd[self.subj_idx == i][self.miss[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		bpd_cr = [np.mean(self.bpd[self.subj_idx == i][self.cr[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		
		bpd_yes = [np.mean(self.bpd[self.subj_idx == i][(self.hit+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		bpd_no = [np.mean(self.bpd[self.subj_idx == i][(self.miss+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		bpd_correct = [np.mean(self.bpd[self.subj_idx == i][(self.hit+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		bpd_error = [np.mean(self.bpd[self.subj_idx == i][(self.miss+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
		
		my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

		N = 4
		ind = np.linspace(0,2,N)  # the x locations for the groups
		bar_width = 0.6   # the width of the bars
		spacing = [0, 0, 0, 0]
		
		sns.set_style("ticks")
		
		# FIGURE 1
		p_values = np.array([myfuncs.permutationTest(ppr_hit, ppr_miss)[1], myfuncs.permutationTest(ppr_fa, ppr_cr)[1]])
		ppr = [ppr_hit, ppr_miss, ppr_fa, ppr_cr]
		MEANS = np.array([np.mean(values) for values in ppr])
		SEMS = np.array([sp.stats.sem(values) for values in ppr])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','r','b'][i], alpha=[1,0.5,0.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('H','M','FA','CR') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.title('phasic pupil responses')
		plt.ylabel('pupil response (a.u.)')
		plt.text(x=np.mean((ind[0],ind[1])), y=1, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
		plt.text(x=np.mean((ind[2],ind[3])), y=1, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'STD1_ppr.pdf'))
		
		# FIGURE 2
		p_values = np.array([myfuncs.permutationTest(ppr_yes, ppr_no)[1], myfuncs.permutationTest(ppr_correct, ppr_error)[1]])
		ppr = [ppr_yes, ppr_no, ppr_correct, ppr_error]
		MEANS = np.array([np.mean(values) for values in ppr])
		SEMS = np.array([sp.stats.sem(values) for values in ppr])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','k','k'][i], alpha=[1,1,1,0.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('YES','NO','CORRECT','ERROR') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.title('phasic pupil responses')
		plt.ylabel('pupil response (a.u.)')
		plt.text(x=np.mean((ind[0],ind[1])), y=1, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
		plt.text(x=np.mean((ind[2],ind[3])), y=1, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'STD2_ppr.pdf'))
		
		# FIGURE 3
		p_values = np.array([myfuncs.permutationTest(bpd_hit, bpd_miss)[1], myfuncs.permutationTest(bpd_fa, bpd_cr)[1]])
		bpd = [bpd_hit, bpd_miss, bpd_fa, bpd_cr]
		MEANS = np.array([np.mean(values) for values in bpd])
		SEMS = np.array([sp.stats.sem(values) for values in bpd])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','r','b'][i], alpha=[1,0.5,0.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('H','M','FA','CR') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.title('baseline pupil responses')
		plt.ylabel('pupil response (z)')
		plt.text(x=np.mean((ind[0],ind[1])), y=0.02, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
		plt.text(x=np.mean((ind[2],ind[3])), y=0.02, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'STD1_bpd.pdf'))
		
		# FIGURE 4
		p_values = np.array([myfuncs.permutationTest(bpd_yes, bpd_no)[1], myfuncs.permutationTest(bpd_correct, bpd_error)[1]])
		bpd = [bpd_yes, bpd_no, bpd_correct, bpd_error]
		MEANS = np.array([np.mean(values) for values in bpd])
		SEMS = np.array([sp.stats.sem(values) for values in bpd])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','k','k'][i], alpha=[1,1,1,0.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('YES','NO','CORRECT','ERROR') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.title('baseline pupil responses')
		plt.ylabel('pupil response (a.u.)')
		plt.text(x=np.mean((ind[0],ind[1])), y=0.02, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
		plt.text(x=np.mean((ind[2],ind[3])), y=0.02, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'STD2_bpd.pdf'))
	
	
	def pupil_criterion(self):
		
		# shell()
		
		ppr_yes = np.array([np.mean(self.ppr[self.subj_idx == i][(self.hit+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_no = np.array([np.mean(self.ppr[self.subj_idx == i][(self.miss+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		
		sns.set_style("darkgrid")
		
		fig = myfuncs.correlation_plot(self.criterion, ppr_yes-ppr_no)
		plt.xlabel('criterion')
		plt.ylabel('choice effect (yes-no)')
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'criterion.pdf'))
		
	def pupil_prediction_error(self):
		
		self.confidence = np.array(self.parameters_joined['confidence'])
		
		conf1 = np.array(self.parameters_joined['confidence'] == 0)
		conf1[self.subj_idx ==np.where(np.array(self.subjects) == 'dh')[0][0]] = np.array(self.parameters_joined['confidence'][self.subj_idx ==np.where(np.array(self.subjects) == 'dh')[0][0]] == 1)
		conf2 = np.array(self.parameters_joined['confidence'] == 1)
		conf3 = np.array(self.parameters_joined['confidence'] == 2)
		conf4 = np.array(self.parameters_joined['confidence'] == 3)
		
		
		self.ppr_feed = np.array(self.parameters_joined['ppr_mean_feed_lp'])
		# self.ppr_feed = np.array(self.parameters_joined['ppr_proj_feed_lp'])
		
		ppr_correct_0 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.hit+self.cr)*conf1)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_error_0 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.miss+self.fa)*conf1)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_correct_1 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.hit+self.cr)*conf2)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_error_1 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.miss+self.fa)*conf2)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_correct_2 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.hit+self.cr)*conf3)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_error_2 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.miss+self.fa)*conf3)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_correct_3 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.hit+self.cr)*conf4)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_error_3 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.miss+self.fa)*conf4)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		
		performance_0 = np.array([sum(self.correct[self.subj_idx == i][conf1[self.subj_idx == i]]) / float(len(self.correct[self.subj_idx == i][conf1[self.subj_idx == i]])) for i in range(self.nr_subjects)]) * 100.0
		performance_1 = np.array([sum(self.correct[self.subj_idx == i][conf2[self.subj_idx == i]]) / float(len(self.correct[self.subj_idx == i][conf2[self.subj_idx == i]])) for i in range(self.nr_subjects)]) * 100.0 
		performance_2 = np.array([sum(self.correct[self.subj_idx == i][conf3[self.subj_idx == i]]) / float(len(self.correct[self.subj_idx == i][conf3[self.subj_idx == i]])) for i in range(self.nr_subjects)]) * 100.0
		performance_3 = np.array([sum(self.correct[self.subj_idx == i][conf4[self.subj_idx == i]]) / float(len(self.correct[self.subj_idx == i][conf4[self.subj_idx == i]])) for i in range(self.nr_subjects)]) * 100.0
		
		my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

		N = 4
		ind = np.linspace(0,2,N)  # the x locations for the groups
		bar_width = 0.6   # the width of the bars
		spacing = [0, 0, 0, 0]
		
		# FIGURE 1
		ppr = [ppr_error_0-ppr_correct_0, ppr_error_1-ppr_correct_1, ppr_error_2-ppr_correct_2, ppr_error_3-ppr_correct_3]
		MEANS = np.array([np.mean(values) for values in ppr])
		SEMS = np.array([sp.stats.sem(values) for values in ppr])
		
		performance = [performance_0, performance_1, performance_2, performance_3]
		MEANS2 = np.array([np.mean(values) for values in performance])
		SEMS2 = np.array([sp.stats.sem(values) for values in performance])
		
				
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		ax.plot(ind, MEANS, ls='--', color='k', alpha=0.75)
		ax.errorbar(ind, MEANS, yerr=SEMS, color = 'k', fmt="o", ms=10, capsize = 1, label='pupil')
		ax.set_xticklabels( ('--','-','+','++') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.set_ylabel('pupil response amplitude\n(error - correct) (Z)')
		ax2 = plt.twinx(ax)
		ax2.plot(ind, MEANS2, ls='--', color='g', alpha=0.75)
		ax2.errorbar(ind, MEANS2, yerr=SEMS2, color = 'g', fmt="o", ms=10, capsize = 1, label='accuracy')
		ax2.set_ylabel('accuracy (% correct)')
		plt.xlim(ind[0]-0.5, ind[-1]+0.5)
		ax.legend(loc=2)
		ax2.legend()
		plt.title('N=6')
		plt.xlabel('confidence')
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'prediction_error_ppr.pdf'))
		
		# FIGURE 2
		ppr = np.concatenate([ppr_error_0-ppr_correct_0, ppr_error_1-ppr_correct_1, ppr_error_2-ppr_correct_2, ppr_error_3-ppr_correct_3])
		conf = np.concatenate((np.ones(self.nr_subjects), np.ones(self.nr_subjects)*2, np.ones(self.nr_subjects)*3, np.ones(self.nr_subjects)*4))
		# fig = myfuncs.correlation_plot2(X=conf, Y=ppr, labelX='confidence', labelY='pupil', xlim=(-0.2, 0.8), ylim=(0,4))
		fig = myfuncs.correlation_plot(conf, ppr)
		plt.xlim(0,5)
		plt.ylim(ymax=0.75)
		plt.title('phasic pupil responses')
		plt.ylabel('pupil response (z)')
		plt.xticks( (1,2,3,4), ('--','-','+','++') )
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'prediction_error2_ppr.pdf'))
		
		d = {
		'subj_idx' : pd.Series(self.subj_idx),
		'correct' : pd.Series(np.array(self.parameters_joined['correct'], dtype=int)),
		'confidence' : pd.Series(np.array(self.parameters_joined['confidence'], dtype=int)),
		'pupil' : pd.Series(np.array(self.ppr_feed)),
		}
		data_response = pd.DataFrame(d)
		data_response.to_csv(os.path.join(self.project_directory, 'feedback_data_jw.csv'))
		
		
		
		
		
		
		
		
		
		
	# 	# FIGURE 1
	# 	MEANS_yes = np.array([np.mean(self.ppr[(self.hit+self.fa)*cond]) for cond in conditions])
	# 	SEMS_yes = np.array([sp.stats.sem(self.ppr[(self.hit+self.fa)*cond]) for cond in conditions])
	# 	MEANS_no = np.array([np.mean(self.ppr[(self.miss+self.cr)*cond]) for cond in conditions])
	# 	SEMS_no = np.array([sp.stats.sem(self.ppr[(self.miss+self.cr)*cond]) for cond in conditions])
	# 	fig = plt.figure(figsize=(4,3))
	# 	ax = fig.add_subplot(111)
	# 	ax.errorbar(ind, MEANS_yes, yerr=SEMS_yes, color = 'r', capsize = 0)
	# 	ax.errorbar(ind, MEANS_no, yerr=SEMS_no, color = 'b', capsize = 0)
	# 	simpleaxis(ax)
	# 	spine_shift(ax)
	# 	ax.set_xticklabels( ('--','-','+','++') )
	# 	ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
	# 	ax.tick_params(axis='x', which='major', labelsize=10)
	# 	ax.tick_params(axis='y', which='major', labelsize=10)
	# 	ax.set_ylim(ymin=0.2, ymax=1.6)
	# 	plt.gca().spines["bottom"].set_linewidth(.5)
	# 	plt.gca().spines["left"].set_linewidth(.5)
	# 	plt.xlabel('confidence')
	# 	plt.ylabel('pupil response (a.u.)')
	# 	plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
	# 	plt.tight_layout()
	# 	fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_choice_2_' + self.subject.initials + '.pdf'))
	#
	# 	# FIGURE 1
	# 	MEANS = np.array([np.mean(self.ppr[(self.hit+self.fa)*cond])-np.mean(self.ppr[(self.miss+self.cr)*cond]) for cond in conditions])
	# 	SEMS = np.array([(sp.stats.sem(self.ppr[(self.hit+self.fa)*cond])+sp.stats.sem(self.ppr[(self.miss+self.cr)*cond]))/2 for cond in conditions])
	# 	fig = plt.figure(figsize=(4,3))
	# 	ax = fig.add_subplot(111)
	# 	ax.errorbar(ind, MEANS, yerr=SEMS, color = 'k', capsize = 0)
	# 	simpleaxis(ax)
	# 	spine_shift(ax)
	# 	ax.set_xticklabels( ('--','-','+','++') )
	# 	ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
	# 	ax.tick_params(axis='x', which='major', labelsize=10)
	# 	ax.tick_params(axis='y', which='major', labelsize=10)
	# 	ax.set_ylim(ymin=-0.4, ymax=1.0)
	# 	plt.gca().spines["bottom"].set_linewidth(.5)
	# 	plt.gca().spines["left"].set_linewidth(.5)
	# 	plt.xlabel('confidence')
	# 	plt.ylabel('pupil response (a.u.)')
	# 	plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
	# 	plt.tight_layout()
	# 	fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_choice_' + self.subject.initials + '.pdf'))
	#
	# 	# FIGURE 2
	# 	MEANS = np.array([np.mean(self.ppr[(self.hit+self.cr)*cond])-np.mean(self.ppr[(self.fa+self.miss)*cond]) for cond in conditions])
	# 	SEMS = np.array([(sp.stats.sem(self.ppr[(self.hit+self.cr)*cond])+sp.stats.sem(self.ppr[(self.fa+self.miss)*cond]))/2 for cond in conditions])
	# 	fig = plt.figure(figsize=(4,3))
	# 	ax = fig.add_subplot(111)
	# 	for i in range(N):
	# 		ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = 'b', alpha = [.25,0.5,0.75,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
	# 	simpleaxis(ax)
	# 	spine_shift(ax)
	# 	ax.set_xticklabels( ('--','-','+','++') )
	# 	ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
	# 	ax.tick_params(axis='x', which='major', labelsize=10)
	# 	ax.tick_params(axis='y', which='major', labelsize=10)
	# 	plt.gca().spines["bottom"].set_linewidth(.5)
	# 	plt.gca().spines["left"].set_linewidth(.5)
	# 	plt.ylabel('pupil correctness effect (a.u.)')
	# 	plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
	# 	plt.tight_layout()
	# 	fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_correct_' + self.subject.initials + '.pdf'))
	#
	#
	# 	if self.experiment == 1:
	# 		# FIGURE 4
	# 		MEANS_yes = np.array([np.mean(self.ppr_feed[(self.hit+self.cr)*cond]) for cond in conditions])
	# 		SEMS_yes = np.array([sp.stats.sem(self.ppr_feed[(self.hit+self.cr)*cond]) for cond in conditions])
	# 		MEANS_no = np.array([np.mean(self.ppr_feed[(self.fa+self.miss)*cond]) for cond in conditions])
	# 		SEMS_no = np.array([sp.stats.sem(self.ppr_feed[(self.fa+self.miss)*cond]) for cond in conditions])
	# 		fig = plt.figure(figsize=(4,3))
	# 		ax = fig.add_subplot(111)
	# 		ax.errorbar(ind, MEANS_no-MEANS_yes, yerr=(SEMS_yes+SEMS_no)/2.0, color = 'k', capsize = 0)
	# 		simpleaxis(ax)
	# 		spine_shift(ax)
	# 		ax.set_xticklabels( ('--','-','+','++') )
	# 		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
	# 		ax.tick_params(axis='x', which='major', labelsize=10)
	# 		ax.tick_params(axis='y', which='major', labelsize=10)
	# 		# ax.set_ylim(ymin=0.2, ymax=1.6)
	# 		plt.gca().spines["bottom"].set_linewidth(.5)
	# 		plt.gca().spines["left"].set_linewidth(.5)
	# 		plt.xlabel('confidence')
	# 		plt.ylabel('prediction error (a.u.)')
	# 		plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
	# 		plt.tight_layout()
	# 		fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_prediction_error_' + self.subject.initials + '.pdf'))
	#
	#
	#
	# self.ppr
	# self.bpd
	#
	# conf1 = np.array(self.parameters_joined['confidence'] == 0)
	# conf2 = np.array(self.parameters_joined['confidence'] == 1)
	# conf3 = np.array(self.parameters_joined['confidence'] == 2)
	# conf4 = np.array(self.parameters_joined['confidence'] == 3)
	#
	# conditions = [conf1, conf2, conf3, conf4]
	
	def pupil_signal_presence(self):
		
		# shell()
		
		tp_h = np.zeros(len(self.subjects))
		tp_l = np.zeros(len(self.subjects))
		ta_h = np.zeros(len(self.subjects))
		ta_l = np.zeros(len(self.subjects))
		d_h = np.zeros(len(self.subjects))
		d_l = np.zeros(len(self.subjects))
		criterion_h = np.zeros(len(self.subjects))
		criterion_l = np.zeros(len(self.subjects))
		h = np.zeros(len(self.subjects))
		l = np.zeros(len(self.subjects))
		for i in range(len(self.subjects)):
			params = self.parameters_joined[self.subj_idx == i]
			params = params[-params['omissions']]
			high_pupil_ind = np.array(params['ppr_proj_lp']>np.median(params['ppr_proj_lp']))
			
			tp_h[i] = sum((params['yes']*params['present'])[high_pupil_ind]) / float(sum((params['present'])[high_pupil_ind]))
			tp_l[i] = sum((params['yes']*params['present'])[-high_pupil_ind]) / float(sum((params['present'])[-high_pupil_ind]))
			ta_h[i] = sum((params['yes']*-params['present'])[high_pupil_ind]) / float(sum((-params['present'])[high_pupil_ind]))
			ta_l[i] = sum((params['yes']*-params['present'])[-high_pupil_ind]) / float(sum((-params['present'])[-high_pupil_ind]))
			
			h[i] = (tp_h[i]+ta_h[i]) / 2.0
			l[i] = (tp_l[i]+ta_l[i]) / 2.0
			
			d_h[i] = myfuncs.SDT_measures(np.array(params['present'])[high_pupil_ind], np.array(params['hit'])[high_pupil_ind], np.array(params['fa'])[high_pupil_ind])[0]
			d_l[i] = myfuncs.SDT_measures(np.array(params['present'])[-high_pupil_ind], np.array(params['hit'])[-high_pupil_ind], np.array(params['fa'])[-high_pupil_ind])[0]
			
			criterion_h[i] = myfuncs.SDT_measures(np.array(params['present'])[high_pupil_ind], np.array(params['hit'])[high_pupil_ind], np.array(params['fa'])[high_pupil_ind])[1]
			criterion_l[i] = myfuncs.SDT_measures(np.array(params['present'])[-high_pupil_ind], np.array(params['hit'])[-high_pupil_ind], np.array(params['fa'])[-high_pupil_ind])[1]
		
		fig = myfuncs.correlation_plot(self.criterion, tp_h-tp_l)
		plt.title('signal present')
		plt.xlabel('criterion (c)')
		plt.ylabel('fraction yes high pupil - low pupil')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_split_signal_present.pdf'))
		
		fig = myfuncs.correlation_plot(self.criterion, ta_h-ta_l)
		plt.title('signal absent')
		plt.xlabel('criterion (c)')
		plt.ylabel('fraction yes high pupil - low pupil')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_split_signal_absent.pdf'))
		
		fig = myfuncs.correlation_plot(self.criterion, criterion_h-criterion_l)
		plt.xlabel('criterion (c)')
		plt.ylabel('criterion high pupil - low pupil')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_split_criterion.pdf'))
		
		fig = myfuncs.correlation_plot(self.criterion, d_h-d_l)
		plt.xlabel('criterion (c)')
		plt.ylabel('d prime high pupil - low pupil')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_split_d_prime.pdf'))
		

	def drift_diffusion(self):
		
		# data:
		d = {
		'subj_idx' : pd.Series(self.subj_idx)+18,
		'response' : pd.Series(np.array(self.parameters_joined['correct'], dtype=int)),
		'rt' : pd.Series(np.array(self.parameters_joined['rt']))/1000.0,
		'pupil' : pd.Series(np.array(self.ppr)),
		'pupil_b' : pd.Series(np.array(self.bpd)),
		}
		data_accuracy = pd.DataFrame(d)
		data_accuracy.to_csv(os.path.join(self.project_directory, 'data_accuracy.csv'))
		
		d = {
		'subj_idx' : pd.Series(self.subj_idx)+18,
		'stimulus' : pd.Series(np.array(self.parameters_joined['present'], dtype=int)),
		'response' : pd.Series(np.array(self.parameters_joined['yes'], dtype=int)),
		'rt' : pd.Series(np.array(self.parameters_joined['rt']))/1000.0,
		'pupil' : pd.Series(np.array(self.ppr)),
		'pupil_b' : pd.Series(np.array(self.bpd)),
		}
		data_response = pd.DataFrame(d)
		data_response.to_csv(os.path.join(self.project_directory, 'data_response.csv'))
		
	
	def grand_average_pupil_response(self,):
		
		# shell()
		
		nr_runs = [12,12,16,11,13,11,12,12,12]
		
		kernel_cue_low = []
		kernel_cue_high = []
		kernel_choice_low = []
		kernel_choice_high = []
		
		parameters_joined2 = []
		for i, s in enumerate(self.subjects):
			
			aliases = ['detection_{}'.format(r+1) for r in range(nr_runs[i])]
			
			self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
			self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
			self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
			parameters_joined2.append(self.ho.read_session_data('', 'parameters_joined'))
			
			downsample_rate = 50 # 50
			new_sample_rate = 1000 / downsample_rate

			# load data:
			parameters = []
			pupil = []
			time = []
			cue_times = []
			choice_times = []
			blink_times = []
			time_to_add = 0
			for alias in aliases:
				parameters.append(self.ho.read_session_data(alias, 'parameters2'))

				self.alias = alias
				self.trial_times = self.ho.read_session_data(alias, 'trials')

				# load pupil:
				self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
				self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
				self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
				pupil.append(self.pupil_bp / np.std(self.pupil_bp))
				
				# load times:
				self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
				self.time = np.array(self.pupil_data['time']) - self.session_start
				time.append( self.time + time_to_add)
				self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
				cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
				choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
				# load blinks:
				self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
				blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )

				time_to_add += self.time[-1]

			# join over runs:
			ppr = np.array(parameters_joined2[i]['ppr_mean_lp'])
			pupil_l_ind = (ppr <= np.percentile(ppr, 33))
			pupil_h_ind = (ppr >= np.percentile(ppr, 66))
			
			parameters_joined = pd.concat(parameters)
			pupil = np.concatenate(pupil)
			time = np.concatenate(time)
			cue_times = np.concatenate(cue_times) / 1000.0
			choice_times = np.concatenate(choice_times) / 1000.0
			blink_times = np.concatenate(blink_times) / 1000.0
			omissions = np.array(parameters_joined.omissions, dtype=bool)
			correct = np.array(parameters_joined.correct, dtype=bool)*-omissions
			error = -np.array(parameters_joined.correct, dtype=bool)*-omissions
			hit = np.array(parameters_joined.hit, dtype=bool)*-omissions
			fa = np.array(parameters_joined.fa, dtype=bool)*-omissions
			miss = np.array(parameters_joined.miss, dtype=bool)*-omissions
			cr = np.array(parameters_joined.cr, dtype=bool)*-omissions
			
			
			
			
			# event related averages:
			interval = 10

			# stimulus locked:
			# ---------------

			# output:
			kernel_cue_high.append(np.mean(np.vstack([pupil[floor(i):floor(i)+10000] for i in (cue_times[pupil_h_ind]-1)*1000]), axis=0))
			kernel_cue_low.append(np.mean(np.vstack([pupil[floor(i):floor(i)+10000] for i in (cue_times[pupil_l_ind]-1)*1000]), axis=0))
			# kernel_cue_error_A = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (cue_times[error*drug]-0.5)*1000]), axis=0)
			# kernel_cue_error_B = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (cue_times[error*-drug]-0.5)*1000]), axis=0)

			# choice locked:
			# -------------

			# output:
			kernel_choice_high.append(np.mean(np.vstack([pupil[floor(i):floor(i)+10000] for i in (choice_times[pupil_h_ind]-4)*1000]), axis=0))
			kernel_choice_low.append(np.mean(np.vstack([pupil[floor(i):floor(i)+10000] for i in (choice_times[pupil_l_ind]-4)*1000]), axis=0))
			# kernel_choice_error_A = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (choice_times[error*drug]-2)*1000]), axis=0)
			# kernel_choice_error_B = np.mean(np.vstack([pupil[floor(i):floor(i)+5000] for i in (choice_times[error*-drug]-2)*1000]), axis=0)
		
		
		pupil_l_ind = np.concatenate([self.parameters_joined.ppr_mean_lp[self.parameters_joined.subject == subj_idx] <= np.percentile(self.parameters_joined.ppr_mean_lp[self.parameters_joined.subject == subj_idx], 33) for subj_idx in self.subjects])
		pupil_h_ind = np.concatenate([self.parameters_joined.ppr_mean_lp[self.parameters_joined.subject == subj_idx] >= np.percentile(self.parameters_joined.ppr_mean_lp[self.parameters_joined.subject == subj_idx], 66) for subj_idx in self.subjects])
		
		sns.set_context("paper", font_scale=0.5)
		sns.set(style="ticks")
		
		fig = plt.figure(figsize=(4,4))
		ax1 = plt.subplot2grid((2,5), (0, 0), colspan=3)
		ax3 = plt.subplot2grid((2,5), (1, 0), colspan=3)
		ax2 = plt.subplot2grid((2,5), (0, 3), colspan=2)
		ax4 = plt.subplot2grid((2,5), (1, 3), colspan=2)
		axes = [ax1, ax2, ax3, ax4]
		ax_nr = 0
		
		for time_locked in ['stim_locked', 'resp_locked']:
			
			if time_locked == 'stim_locked':
				kernels_pupil_l = np.vstack(kernel_cue_low)
				kernels_pupil_h = np.vstack(kernel_cue_high)
				xlim = [-1,9]
			if time_locked == 'resp_locked':
				kernels_pupil_l = np.vstack(kernel_choice_low)
				kernels_pupil_h = np.vstack(kernel_choice_high)
				xlim = [-4,6]
		
			# length kernel:
			kernel_length = kernels_pupil_l.shape[1]
			
			# step size:
			if time_locked == 'stim_locked':
				step = pd.Series(np.linspace(xlim[0], xlim[1], kernel_length), name='time from cue (s)')
			if time_locked == 'resp_locked':
				step = pd.Series(np.linspace(xlim[0], xlim[1], kernel_length), name='time from report (s)')
		
			xlim_indices = np.array(step >= xlim[0]) * np.array(step <= xlim[1])
		
			# baseline
			# --------
			if time_locked == 'stim_locked':
				time_of_interest_b = [-0.5,0]
				time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step <= time_of_interest_b[1])
			if time_locked == 'resp_locked':
				time_of_interest_b = [-2.5,-2.0]
				time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step <= time_of_interest_b[1])
			
			# get baseline scalar amplitudes
			lin_projections_b = []
			for (kernel1, kernel2) in zip(kernels_pupil_h, kernels_pupil_l):
				lin_projections_b.append([np.mean(kernel1[time_of_interest_b_ind]), np.mean(kernel2[time_of_interest_b_ind])])
			lin_projections_b = np.array(lin_projections_b)
			means = np.vstack((lin_projections_b[:,0], lin_projections_b[:,1])).mean(axis=0)
			p_bars_b = myfuncs.permutationTest(lin_projections_b[:,0]-means, lin_projections_b[:,1]-means)[1]
		
			# baseline the kernels:
			kernels_pupil_h = np.array(kernels_pupil_h[:,:] - np.mat(kernels_pupil_h[:,time_of_interest_b_ind].mean(axis=1)).T)
			kernels_pupil_l = np.array(kernels_pupil_l[:,:] - np.mat(kernels_pupil_l[:,time_of_interest_b_ind].mean(axis=1)).T)
		
			# phasic responses:
			# -----------------
		
			# stats scalar amplitudes:
			if time_locked == 'stim_locked':
				time_of_interest = [1,2.5]
				time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step <= time_of_interest[1])
			if time_locked == 'resp_locked':
				time_of_interest = [-1,1.5]
				time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step <= time_of_interest[1])
		
			lin_projections = []
			# for (template, kernel1, kernel2) in zip((kernels_pupil_h+kernels_pupil_l)/2.0, kernels_pupil_h, kernels_pupil_l):
			# 	lin_projections.append([np.dot(template[time_of_interest], kernel1[time_of_interest])/(np.linalg.norm(template[time_of_interest])**2), np.dot(template[time_of_interest], kernel2[time_of_interest])/(np.linalg.norm(template[time_of_interest])**2)])
			for (kernel1, kernel2) in zip(kernels_pupil_h, kernels_pupil_l):
				lin_projections.append([np.mean(kernel1[time_of_interest_ind]), np.mean(kernel2[time_of_interest_ind])])
			lin_projections = np.array(lin_projections)
		
			means = np.vstack((lin_projections[:,0], lin_projections[:,1])).mean(axis=0)
			p_bars = myfuncs.permutationTest(lin_projections[:,0]-means, lin_projections[:,1]-means, nrand=5000)[1]
		
			# combine baseline and phasic scalars:
			lin_projections = np.hstack((lin_projections_b, lin_projections))
		
			# shell()
		
			# # stats across time series:
			# # -------------------------
			# p1 = np.zeros(kernels_h_hit.shape[1])
			# for i in range(kernels_h_hit.shape[1]):
			# 	means = np.vstack((kernels_pupil_h[:,i], kernels_pupil_l[:,i])).mean(axis=0)
			# 	p1[i] = myfuncs.permutationTest(kernels_pupil_h[:,i]-means, kernels_pupil_l[:,i]-means, nrand=2000)[1]
			# 	# p1[i] = sp.stats.ttest_rel(kernels_pupil_h[:,i], kernels_pupil_l[:,i])[1]
			# 	# print myfuncs.permutationTest2(kernels_pupil_h[:,i]-means, kernels_pupil_l[:,i]-means)
			# p = mne.stats.fdr_correction(p1, 0.05,)
		
			# plot:
			# -----
		
			# timeseries:
			ax = axes[ax_nr]
			ax.axvline(0, lw=0.25, alpha=0.5, color = 'k')
			ax.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		
			# shell()
		
			colors = ['black']
			conditions = pd.Series(['high pupil'], name='trial type')
			# sns.tsplot(kernels_pupil_h, time=step, condition=conditions, value='fMRI response\n(% signal change)', color='red', ci=66, lw=1, ls='-', ax=ax)
			sns.tsplot(kernels_pupil_h[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response (Z)', color='red', ci=66, lw=1, ls='-', ax=ax)
			conditions = pd.Series(['medium pupil'], name='trial type')
			# sns.tsplot(kernels_pupil_m, time=step, condition=conditions, value='fMRI response (Z)', color='black', ci=False, lw=1, ls=':', ax=ax, alpha=0.5)
			conditions = pd.Series(['low pupil'], name='trial type')
			sns.tsplot(kernels_pupil_l[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response (Z)', color='blue', ci=66, lw=1, ls='-', ax=ax)
			# ax.set_ylim(ylim)
			ax.set_xlim(xlim)
			ax.legend(loc='upper left', fontsize=6)
			# for i in range(kernels_h_hit.shape[1]):
			# 	if p[1][i] < 0.05:
			# 		ax.plot(step[i], ylim[0]+0.1, 'o', color='g', marker='o', markeredgecolor='w', ms=3)
			
			if time_locked == 'stim_locked':
				sns.distplot(np.array(self.rt[pupil_h_ind])/1000.0, bins=20, hist=False, kde_kws={"shade": True}, color='r', ax=ax)
				sns.distplot(np.array(self.rt[pupil_l_ind])/1000.0, bins=20, hist=False, kde_kws={"shade": True}, color='b', ax=ax)
			if time_locked == 'resp_locked':
				sns.distplot(0-(np.array(self.rt[pupil_h_ind])/1000.0), bins=20, hist=False, kde_kws={"shade": True}, color='r', ax=ax)
				sns.distplot(0-(np.array(self.rt[pupil_l_ind])/1000.0), bins=20, hist=False, kde_kws={"shade": True}, color='b', ax=ax)
			
			
			for axis in ['top','bottom','left','right']:
				ax.spines[axis].set_linewidth(0.5)
		
		
			ax_nr+=1
		
			# bars:
			ax = axes[ax_nr]
		
			ax.axhline(0, lw=0.25, alpha=0.5, color = 'k')
			bar_width = 0.50

			my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
			for i in range(lin_projections.shape[1]):
			
				# shell()
			
				ax.bar([0.5,1.25, 2.5,3.25][i], lin_projections.mean(axis = 0)[i], width = bar_width, yerr = (lin_projections.std(axis = 0)/sqrt(lin_projections.shape[0]))[i], color = ['r','b','r','b'][i], alpha = [1,1,1,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center',)
				ax.set_xticks([0.5,1.25, 2.5,3.25])
				ax.set_xticklabels(['bh', 'bl', 'ph', 'pl'])
			# ax.set_ylim(ylim)
		
			# shell()
		
			ax.text(s='p = {}'.format(round(p_bars_b, 3)), x=0.5, y=0, size=6)
			ax.text(s='p = {}'.format(round(p_bars, 3)), x=2, y=0, size=6)
			# ax.set_ylabel('fMRI response\n(% signal change)')
		
			for axis in ['top','bottom','left','right']:
				ax.spines[axis].set_linewidth(0.5)
		
			ax_nr+=1

		sns.despine(offset=10, trim=True)

		axes[1].yaxis.set_visible(False)
		axes[3].yaxis.set_visible(False)

		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'grand_mean.pdf'))
		
	




