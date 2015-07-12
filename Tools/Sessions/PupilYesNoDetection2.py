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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
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

import rpy2.robjects as robjects
import rpy2.rlike.container as rlc

import hddm
import mne

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

sns.set(style='ticks', font='Arial', font_scale=1, rc={
	'axes.linewidth': 0.50, 
	'axes.labelsize': 7, 
	'axes.titlesize': 7, 
	'xtick.labelsize': 6, 
	'ytick.labelsize': 6, 
	'legend.fontsize': 6, 
	'xtick.major.width': 0.25, 
	'ytick.major.width': 0.25,
	'text.color': 'Black',
	'axes.labelcolor':'Black',
	'xtick.color':'Black',
	'ytick.color':'Black',} )
sns.plotting_context()


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
		self.resolution = (800,600)
		
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
			self.ho.edf_gaze_data_to_hdf(alias=alias, pupil_hp=0.05, pupil_lp=10.0)
	
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
			indices = (self.time > self.cue_times[t]) * (self.time < self.choice_times[t])
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
			if sum((self.blink_start_times > self.cue_times[t] - 700) * (self.blink_end_times < self.choice_times[t] + 500)) > 0: # as in MEG
			# if sum((self.blink_start_times > self.cue_times[t] - 500) * (self.blink_end_times < self.choice_times[t] + 1000)) > 0:
				self.omission_indices_blinks[t] = True
		
		self.omission_indices_rt = np.zeros(self.nr_trials, dtype=bool)
		for t in range(self.nr_trials):
			if self.rt[t] < 100:
				self.omission_indices_rt[t] = True
		
		self.omission_indices_first = np.zeros(self.nr_trials, dtype=bool)
		self.omission_indices_first[0:2] = True
		if self.experiment == 1:
			self.omission_indices_subject = np.array(self.parameters['confidence'] == -1)
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
		present = np.array(self.parameters['signal_present'] == 1)
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
		self.parameters['drug'] = self.drug
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
		print
		print 'omissions'
		print '---------'
		print '{} omissions'.format(sum(self.omission_indices))
		print
		print
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
		
		# print
		# print answers
		# print
		
		# fix:
		for i in problem_trials:
			
			events = self.events[(self.events.EL_timestamp > self.trial_starts[i]) * (self.events.EL_timestamp < self.trial_ends[i])]
			response_times = np.array(events[((events.key == 275) + (events.key == 276)) * (events.up_down == 'Down')].EL_timestamp)
			
			if len(response_times) != 0:
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
			else:
				pass
			
	def normalize_gaze(self):
		
		self.gaze_x = self.gaze_x + (self.resolution[0]/2) - np.median(self.gaze_x)
		self.gaze_y = self.gaze_y + (self.resolution[1]/2) - np.median(self.gaze_y)
	
	def process_runs(self, alias, drug=None):
		print 'subject {}; {}'.format(self.subject.initials, alias)
		print '##############################'
		
		# load data:
		self.alias = alias
		self.drug = drug
		self.events = self.ho.read_session_data(alias, 'events')
		self.parameters = self.ho.read_session_data(alias, 'parameters')
		self.nr_trials = len(self.parameters['trial_nr'])
		
		self.trial_times = self.ho.read_session_data(alias, 'trials')
		self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
		self.trial_starts = np.array(self.trial_times['trial_start_EL_timestamp'])
		self.trial_ends = np.array(self.trial_times['trial_end_EL_timestamp'])
		self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
		self.cue_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)])
		if self.experiment == 1:
			self.choice_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 4)])
			self.confidence_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 5)])
			self.feedback_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 6)])
		if self.experiment == 2:
			self.choice_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)])
		self.problem_trials()
		self.rt = self.choice_times - self.cue_times
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
		self.gaze_x = np.array(self.pupil_data[(self.eye + '_gaze_x_int')])
		self.gaze_y = np.array(self.pupil_data[(self.eye + '_gaze_y_int')])
		
		self.normalize_gaze()
		self.compute_omission_indices()
		self.trial_params()
		# self.pupil_zscore()
		# self.create_timelocked_arrays()
		
		# plot interpolated gaze time series:
		heatmap, xedges, yedges = np.histogram2d(self.gaze_x, self.gaze_y, bins=(99), range=[[0,self.resolution[0]],[0,self.resolution[1]]],)
		# heatmap, xedges, yedges = np.histogram2d(self.gaze_x, self.gaze_y, bins=(111), range=[[300,500],[200,400]],)

		extent = (xedges[0],xedges[-1],yedges[0],yedges[-1])
		fig = plt.figure()
		plt.imshow(heatmap, extent=extent, shape=self.resolution, cmap='OrRd', vmax=1000)
		plt.colorbar()
		plt.ylabel('gaze y')
		plt.xlabel('gaze x')
		fig.savefig(os.path.join(self.base_directory, 'figs', 'gaze_preprocess_' + self.alias + '.pdf'))
		
		# shell()
		
	
	def process_across_runs(self, aliases):
		
		downsample_rate = 50 # 50
		new_sample_rate = 1000 / downsample_rate
		
		# load data:
		parameters = []
		pupil = []
		pupil_diff = []
		time = []
		cue_times = []
		choice_times = []
		confidence_times = []
		blink_times = []
		time_to_add = 0
		for alias in aliases:
			parameters.append(self.ho.read_session_data(alias, 'parameters2'))
			
			self.alias = alias
			self.trial_times = self.ho.read_session_data(alias, 'trials')
			
			# load pupil:
			self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
			self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
			# pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_lp_psc')]))
			
			p = np.array(self.pupil_data[(self.eye + '_pupil_lp_psc')])
			# pupil.append( (p - p.mean()) / p.std())
			pupil.append( p )
			
			pupil_diff.append( np.array(self.pupil_data[(self.eye + '_pupil_lp_diff')]) )
			
			# load times:
			self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
			self.time = np.array(self.pupil_data['time']) - self.session_start
			time.append( self.time + time_to_add)
			self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
			cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
			if self.experiment == 1:
				choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 4)]) - self.session_start + time_to_add )
				confidence_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 6)]) - self.session_start + time_to_add )
			if self.experiment == 2:
				choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
			# load blinks:
			self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
			blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
			
			time_to_add += self.time[-1]
		
		# shell()
		
		# join over runs:
		parameters_joined = pd.concat(parameters)
		pupil = np.concatenate(pupil)
		pupil_diff = np.concatenate(pupil_diff)
		time = np.concatenate(time)
		cue_times = np.concatenate(cue_times) / 1000.0
		choice_times = np.concatenate(choice_times) / 1000.0
		blink_times = np.concatenate(blink_times) / 1000.0
		if self.experiment == 1:
			confidence_times = np.concatenate(confidence_times) / 1000.0
		omissions = np.array(parameters_joined.omissions, dtype=bool)
		correct = np.array(parameters_joined.correct, dtype=bool)*-omissions
		error = -np.array(parameters_joined.correct, dtype=bool)*-omissions
		hit = np.array(parameters_joined.hit, dtype=bool)*-omissions
		fa = np.array(parameters_joined.fa, dtype=bool)*-omissions
		miss = np.array(parameters_joined.miss, dtype=bool)*-omissions
		cr = np.array(parameters_joined.cr, dtype=bool)*-omissions
		
		# drug = (np.array(np.array(parameters_joined['drug']) == 'B', dtype=bool) + np.array(np.array(parameters_joined['drug']) == 'D', dtype=bool))
		# placebo = (np.array(np.array(parameters_joined['drug']) == 'A', dtype=bool) + np.array(np.array(parameters_joined['drug']) == 'C', dtype=bool))
		drug = (np.array(np.array(parameters_joined['drug']) == 'B', dtype=bool) + np.array(np.array(parameters_joined['drug']) == 'D', dtype=bool))*-omissions
		placebo = (np.array(np.array(parameters_joined['drug']) == 'A', dtype=bool) + np.array(np.array(parameters_joined['drug']) == 'C', dtype=bool))*-omissions
		
		# pupil measures:
		bpd_lp = np.array([np.mean(pupil[floor(i-500):floor(i)]) for i in (cue_times)*1000])
		ppr_mean_lp = np.array([np.mean(pupil[floor(i-500):floor(i+750)]) for i in (choice_times)*1000]) - bpd_lp
		ppr_mean_lp_fb = np.array([np.mean(pupil[floor(i+750):floor(i+1500)]) for i in (choice_times)*1000]) - bpd_lp
		ppr_mean_lp_trial = np.array([np.mean(pupil[floor(i-500):floor(i+1500)]) for i in (choice_times)*1000]) - bpd_lp
		
		print
		print
		print self.subject.initials
		print 'drug effect = {}'.format(round(np.mean(ppr_mean_lp[drug])-np.mean(ppr_mean_lp[placebo]),3)) 
		print
		print
		
		
		
		# downsample:
		deconvolution = False
		if deconvolution:
			
			# new_nr_samples = pupil.shape[0] / downsample_rate
			# pupil = sp.signal.resample(pupil, new_nr_samples)
			# pupil_diff = sp.signal.resample(pupil_diff, new_nr_samples)
			
			pupil = sp.signal.decimate(pupil, downsample_rate)
			pupil_diff = sp.signal.decimate(pupil_diff, downsample_rate)
			
			# deconvolution:
			interval = 5
			
			if self.experiment == 1:
				
				# stimulus locked:
				# ---------------
				events = [cue_times[hit]-0.5, cue_times[fa]-0.5, cue_times[miss]-0.5, cue_times[cr]-0.5, confidence_times[correct]-0.5, confidence_times[-correct]-0.5, blink_times]
				do = ArrayOperator.DeconvolutionOperator( inputObject=pupil, eventObject=events, TR=1.0/new_sample_rate, deconvolutionSampleDuration=1.0/new_sample_rate, deconvolutionInterval=interval, run=True )
				do2 = ArrayOperator.DeconvolutionOperator( inputObject=pupil_diff, eventObject=events, TR=1.0/new_sample_rate, deconvolutionSampleDuration=1.0/new_sample_rate, deconvolutionInterval=interval, run=True )
				
				# output:
				output = do.deconvolvedTimeCoursesPerEventType
				kernel_cue_hit = output[0,:,0]
				kernel_cue_fa = output[1,:,0]
				kernel_cue_miss = output[2,:,0]
				kernel_cue_cr = output[3,:,0]
				kernel_conf_correct = output[4,:,0]
				kernel_conf_error = output[5,:,0]

				output = do2.deconvolvedTimeCoursesPerEventType
				kernel_cue_hit2 = output[0,:,0]
				kernel_cue_fa2 = output[1,:,0]
				kernel_cue_miss2 = output[2,:,0]
				kernel_cue_cr2 = output[3,:,0]
				kernel_conf_correct2 = output[4,:,0]
				kernel_conf_error2 = output[5,:,0]

				# plots:
				fig = plt.figure()
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_hit.shape[0]), kernel_cue_hit, 'r', label='hit')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_fa.shape[0]), kernel_cue_fa, 'r', alpha=0.5, label='fa')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_miss.shape[0]), kernel_cue_miss, 'b', alpha=0.5, label='miss')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_cr.shape[0]), kernel_cue_cr, 'b', label='cr')
				plt.xlabel('time from stimulus onset (s)')
				plt.ylabel('pupil size (Z)')
				plt.legend()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'stim_locked_{}.pdf'.format(self.subject.initials)))

				fig = plt.figure()
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_conf_correct.shape[0]), kernel_conf_correct, 'g', label='correct')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_conf_error.shape[0]), kernel_conf_error, 'r', label='error')
				plt.xlabel('time from confidence report (s)')
				plt.ylabel('pupil size (Z)')
				plt.legend()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'confidence_locked_{}.pdf'.format(self.subject.initials)))

				####

				# plots:
				fig = plt.figure()
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_hit2.shape[0]), kernel_cue_hit2, 'r', label='hit')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_fa2.shape[0]), kernel_cue_fa2, 'r', alpha=0.5, label='fa')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_miss2.shape[0]), kernel_cue_miss2, 'b', alpha=0.5, label='miss')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_cr2.shape[0]), kernel_cue_cr2, 'b', label='cr')
				plt.xlabel('time from stimulus onset (s)')
				plt.ylabel('pupil size (Z)')
				plt.legend()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'stim_locked_2{}.pdf'.format(self.subject.initials)))

				fig = plt.figure()
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_conf_correct2.shape[0]), kernel_conf_correct2, 'g', label='correct')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_conf_error2.shape[0]), kernel_conf_error2, 'r', label='error')
				plt.xlabel('time from confidence report (s)')
				plt.ylabel('pupil size (Z)')
				plt.legend()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'confidence_locked_2{}.pdf'.format(self.subject.initials)))

				# plots:
				fig = plt.figure()
				plt.plot(np.linspace(-0.5, interval-0.5, np.cumsum(kernel_cue_hit2).shape[0]), np.cumsum(kernel_cue_hit2), 'r', label='hit')
				plt.plot(np.linspace(-0.5, interval-0.5, np.cumsum(kernel_cue_fa2).shape[0]), np.cumsum(kernel_cue_fa2), 'r', alpha=0.5, label='fa')
				plt.plot(np.linspace(-0.5, interval-0.5, np.cumsum(kernel_cue_miss2).shape[0]), np.cumsum(kernel_cue_miss2), 'b', alpha=0.5, label='miss')
				plt.plot(np.linspace(-0.5, interval-0.5, np.cumsum(kernel_cue_cr2).shape[0]), np.cumsum(kernel_cue_cr2), 'b', label='cr')
				plt.xlabel('time from stimulus onset (s)')
				plt.ylabel('pupil size (Z)')
				plt.legend()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'stim_locked_3{}.pdf'.format(self.subject.initials)))

				fig = plt.figure()
				plt.plot(np.linspace(-0.5, interval-0.5, np.cumsum(kernel_conf_correct2).shape[0]), np.cumsum(kernel_conf_correct2), 'g', label='correct')
				plt.plot(np.linspace(-0.5, interval-0.5, np.cumsum(kernel_conf_error2).shape[0]), np.cumsum(kernel_conf_error2), 'r', label='error')
				plt.xlabel('time from confidence report (s)')
				plt.ylabel('pupil size (Z)')
				plt.legend()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'confidence_locked_3{}.pdf'.format(self.subject.initials)))
				
				# # response locked:
				# # ---------------
				# events = [choice_times[hit]-2, choice_times[fa]-2, choice_times[miss]-2, choice_times[cr]-2, confidence_times[correct]-0.5, confidence_times[-correct]-0.5, blink_times]
				# do = ArrayOperator.DeconvolutionOperator( inputObject=pupil, eventObject=events, TR=1.0/new_sample_rate, deconvolutionSampleDuration=1.0/new_sample_rate, deconvolutionInterval=interval, run=True )
				#
				# # output:
				# output = do.deconvolvedTimeCoursesPerEventType
				# kernel_choice_hit = output[0,:,0]
				# kernel_choice_fa = output[1,:,0]
				# kernel_choice_miss = output[2,:,0]
				# kernel_choice_cr = output[3,:,0]
				# kernel_conf_correct = output[4,:,0]
				# kernel_conf_error = output[5,:,0]
				#
				# # plots:
				# fig = plt.figure()
				# plt.plot(np.linspace(-2, interval-2, kernel_cue_hit.shape[0]), kernel_choice_hit, 'r', label='hit')
				# plt.plot(np.linspace(-2, interval-2, kernel_cue_fa.shape[0]), kernel_choice_fa, 'r', alpha=0.5, label='fa')
				# plt.plot(np.linspace(-2, interval-2, kernel_cue_miss.shape[0]), kernel_choice_miss, 'b', alpha=0.5, label='miss')
				# plt.plot(np.linspace(-2, interval-2, kernel_cue_cr.shape[0]), kernel_choice_cr, 'b', label='cr')
				# plt.xlabel('time from choice (s)')
				# plt.ylabel('pupil size (Z)')
				# plt.legend()
				# fig.savefig(os.path.join(self.project_directory, 'figures', 'choice_locked_{}.pdf'.format(self.subject.initials)))
				#
				# fig = plt.figure()
				# plt.plot(np.linspace(-0.5, interval-0.5, kernel_conf_correct.shape[0]), kernel_conf_correct, 'g', label='correct')
				# plt.plot(np.linspace(-0.5, interval-0.5, kernel_conf_error.shape[0]), kernel_conf_error, 'r', label='error')
				# plt.xlabel('time from confidence report (s)')
				# plt.ylabel('pupil size (Z)')
				# plt.legend()
				# fig.savefig(os.path.join(self.project_directory, 'figures', 'confidence_locked_{}.pdf'.format(self.subject.initials)))
				
			if self.experiment == 2:
				
				# stimulus locked:
				# ---------------
				events = [cue_times[correct*drug]-0.5, cue_times[correct*-drug]-0.5, cue_times[error*drug]-0.5, cue_times[error*-drug]-0.5, cue_times[omissions]-0.5, blink_times]
				do = ArrayOperator.DeconvolutionOperator( inputObject=pupil, eventObject=events, TR=1.0/new_sample_rate, deconvolutionSampleDuration=1.0/new_sample_rate, deconvolutionInterval=interval, run=True )
				
				# output:
				output = do.deconvolvedTimeCoursesPerEventType
				kernel_cue_correct_A = output[0,:,0]
				kernel_cue_correct_B = output[1,:,0]
				kernel_cue_error_A = output[2,:,0]
				kernel_cue_error_B = output[3,:,0]
				
				# choice locked:
				# -------------
				events = [choice_times[correct*drug]-2, choice_times[correct*-drug]-2, choice_times[error*drug]-2, choice_times[error*-drug]-2, choice_times[omissions]-2, blink_times]
				do = ArrayOperator.DeconvolutionOperator( inputObject=pupil, eventObject=events, TR=1.0/new_sample_rate, deconvolutionSampleDuration=1.0/new_sample_rate, deconvolutionInterval=interval, run=True )
				
				# output:
				output = do.deconvolvedTimeCoursesPerEventType
				kernel_choice_correct_A = output[0,:,0]
				kernel_choice_correct_B = output[1,:,0]
				kernel_choice_error_A = output[2,:,0]
				kernel_choice_error_B = output[3,:,0]
				
				# save:
				np.save(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_{}.npy'.format(self.subject.initials)), np.vstack((kernel_cue_correct_A, kernel_cue_correct_B, kernel_cue_error_A, kernel_cue_error_B)))
				np.save(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_{}.npy'.format(self.subject.initials)), np.vstack((kernel_choice_correct_A, kernel_choice_correct_B, kernel_choice_error_A, kernel_choice_error_B)))
				
				# plot:
				# -----
				fig = plt.figure(figsize=(10,10))
				
				ax = fig.add_subplot(221)
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A, 'g', label='correct, A')
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_B, 'g', ls='--', label='correct, B')
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_A, 'r', label='error, A')
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_B, 'r', ls='--', label='error, B')
				plt.axvline(0, color='k', ls='--', lw=0.5)
				plt.axvline(np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
				ax.set_xlim(-0.5, interval-0.5)
				ax.set_ylim(-1.5, 1.5)
				ax.set_xlabel('time from stimulus onset (s)')
				ax.set_ylabel('pupil size (Z)')
				ax.legend()
				
				ax = fig.add_subplot(222)
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A-kernel_cue_error_A, 'b', label='difference wave, A')
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_B-kernel_cue_error_B, 'b', ls='--', label='difference wave, B')
				plt.axvline(0, color='k', ls='--', lw=0.5)
				plt.axvline(np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
				plt.axhline(0, lw=0.5)
				ax.set_xlim(-0.5, interval-0.5)
				ax.set_ylim(-1.5, 1.5)
				ax.set_xlabel('time from stimulus onset (s)')
				ax.set_ylabel('correct - error')
				ax.legend()
				
				ax = fig.add_subplot(223)
				ax.plot(np.linspace(-2, interval-2, kernel_choice_correct_A.shape[0]), kernel_choice_correct_A, 'g', label='correct, A')
				ax.plot(np.linspace(-2, interval-2, kernel_choice_correct_A.shape[0]), kernel_choice_correct_B, 'g', ls='--', label='correct, B')
				ax.plot(np.linspace(-2, interval-2, kernel_choice_error_A.shape[0]), kernel_choice_error_A, 'r', label='error, A')
				ax.plot(np.linspace(-2, interval-2, kernel_choice_error_A.shape[0]), kernel_choice_error_B, 'r', ls='--', label='error, B')
				plt.axvline(0, color='k', ls='--', lw=0.5)
				plt.axvline(-np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
				ax.set_xlim(-2, interval-2)
				ax.set_ylim(-1.5, 1.5)
				ax.set_xlabel('time from choice (s)')
				ax.set_ylabel('pupil size (Z)')
				ax.legend()
				
				ax = fig.add_subplot(224)
				ax.plot(np.linspace(-2, interval-2, kernel_choice_error_A.shape[0]), kernel_choice_correct_A-kernel_choice_error_A, 'b', label='difference wave, A')
				ax.plot(np.linspace(-2, interval-2, kernel_choice_error_A.shape[0]), kernel_choice_correct_B-kernel_choice_error_B, 'b', ls='--', label='difference wave, B')
				plt.axvline(0, color='k', ls='--', lw=0.5)
				plt.axvline(-np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
				plt.axhline(0, lw=0.5)
				ax.set_xlim(-2, interval-2)
				ax.set_ylim(-1.5, 1.5)
				ax.set_xlabel('time from choice (s)')
				ax.set_ylabel('correct - error')
				ax.legend()
				
				plt.tight_layout()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_responses_{}.pdf'.format(self.subject.initials)))
				
		
		else:
			
			interval = 3
			
			kernel_cue_A = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] - np.mean(bpd_lp[placebo]) for i in (cue_times[placebo]-0.5)*1000]), axis=0), downsample_rate)
			kernel_cue_B = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] - np.mean(bpd_lp[drug]) for i in (cue_times[drug]-0.5)*1000]), axis=0), downsample_rate)
			
			kernel_choice_A = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] - np.mean(bpd_lp[placebo]) for i in (choice_times[placebo]-1.5)*1000]), axis=0), downsample_rate)
			kernel_choice_B = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] - np.mean(bpd_lp[drug]) for i in (choice_times[drug]-1.5)*1000]), axis=0), downsample_rate)
			
			np.save(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_drug_{}.npy'.format(self.subject.initials)), np.vstack((kernel_cue_A, kernel_cue_B)))
			np.save(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_drug_{}.npy'.format(self.subject.initials)), np.vstack((kernel_choice_A, kernel_choice_B)))
			
			for d in range(2):
				diff = np.array(parameters_joined.difficulty == d)
			
				# event related averages:
				# downsample_rate = 20
				
				# stimulus locked:
				# ---------------
		
				# output:
				kernel_cue_correct_A = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] for i in (cue_times[correct*placebo*diff]-0.5)*1000]), axis=0) - np.mean(bpd_lp[correct*placebo*diff]), downsample_rate)
				kernel_cue_correct_B = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] for i in (cue_times[correct*drug*diff]-0.5)*1000]), axis=0) - np.mean(bpd_lp[correct*drug*diff]), downsample_rate)
				kernel_cue_error_A = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] for i in (cue_times[error*placebo*diff]-0.5)*1000]), axis=0) - np.mean(bpd_lp[error*placebo*diff]), downsample_rate)
				kernel_cue_error_B = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] for i in (cue_times[error*drug*diff]-0.5)*1000]), axis=0) - np.mean(bpd_lp[error*drug*diff]), downsample_rate)
				
				# save:
				np.save(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_{}_{}.npy'.format(self.subject.initials, d)), np.vstack((kernel_cue_correct_A, kernel_cue_correct_B, kernel_cue_error_A, kernel_cue_error_B)))
				
				
				# choice locked:
				# -------------
		
				# output:
				kernel_choice_correct_A = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] for i in (choice_times[correct*placebo*diff]-1.5)*1000]), axis=0) - np.mean(bpd_lp[correct*placebo*diff]), downsample_rate)
				kernel_choice_correct_B = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] for i in (choice_times[correct*drug*diff]-1.5)*1000]), axis=0) - np.mean(bpd_lp[correct*drug*diff]), downsample_rate)
				kernel_choice_error_A = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] for i in (choice_times[error*placebo*diff]-1.5)*1000]), axis=0) - np.mean(bpd_lp[error*placebo*diff]), downsample_rate)
				kernel_choice_error_B = sp.signal.decimate(np.mean(np.vstack([pupil[floor(i):floor(i)+(interval*1000)] for i in (choice_times[error*drug*diff]-1.5)*1000]), axis=0) - np.mean(bpd_lp[error*drug*diff]), downsample_rate)
				
				# save:
				np.save(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_{}_{}.npy'.format(self.subject.initials, d)), np.vstack((kernel_choice_correct_A, kernel_choice_correct_B, kernel_choice_error_A, kernel_choice_error_B)))
				
				# plot:
				# -----
				fig = plt.figure(figsize=(10,10))
				
				ax = fig.add_subplot(221)
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A, 'g', label='correct, A')
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_B, 'g', ls='--', label='correct, B')
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_A, 'r', label='error, A')
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_B, 'r', ls='--', label='error, B')
				plt.axvline(0, color='k', ls='--', lw=0.5)
				plt.axvline(np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
				ax.set_xlim(-0.5, interval-0.5)
				# ax.set_ylim(-1.5, 1.5)
				ax.set_xlabel('time from stimulus onset (s)')
				ax.set_ylabel('pupil size (Z)')
				ax.legend()
		
				ax = fig.add_subplot(222)
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A-kernel_cue_error_A, 'b', label='difference wave, A')
				ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_B-kernel_cue_error_B, 'b', ls='--', label='difference wave, B')
				plt.axvline(0, color='k', ls='--', lw=0.5)
				plt.axvline(np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
				plt.axhline(0, lw=0.5)
				ax.set_xlim(-0.5, interval-0.5)
				# ax.set_ylim(-1.5, 1.5)
				ax.set_xlabel('time from stimulus onset (s)')
				ax.set_ylabel('correct - error')
				ax.legend()
		
				ax = fig.add_subplot(223)
				ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_correct_A.shape[0]), kernel_choice_correct_A, 'g', label='correct, A')
				ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_correct_A.shape[0]), kernel_choice_correct_B, 'g', ls='--', label='correct, B')
				ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_error_A.shape[0]), kernel_choice_error_A, 'r', label='error, A')
				ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_error_A.shape[0]), kernel_choice_error_B, 'r', ls='--', label='error, B')
				plt.axvline(0, color='k', ls='--', lw=0.5)
				plt.axvline(-np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
				ax.set_xlim(-2, interval-2)
				# ax.set_ylim(-1.5, 1.5)
				ax.set_xlabel('time from choice (s)')
				ax.set_ylabel('pupil size (Z)')
				ax.legend()
		
				ax = fig.add_subplot(224)
				ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_error_A.shape[0]), kernel_choice_correct_A-kernel_choice_error_A, 'b', label='difference wave, A')
				ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_error_A.shape[0]), kernel_choice_correct_B-kernel_choice_error_B, 'b', ls='--', label='difference wave, B')
				plt.axvline(0, color='k', ls='--', lw=0.5)
				plt.axvline(-np.mean(parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
				plt.axhline(0, lw=0.5)
				ax.set_xlim(-2, interval-2)
				# ax.set_ylim(-1.5, 1.5)
				ax.set_xlabel('time from choice (s)')
				ax.set_ylabel('correct - error')
				ax.legend()
		
				plt.tight_layout()
				
				sns.despine()
				
				
				fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_responses_avg_{}_{}.pdf'.format(self.subject.initials, d)))
		
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
		# parameters_joined['bpd_bp'] = bpd_bp
		# parameters_joined['ppr_peak_lp'] = ppr_peak_lp
		# parameters_joined['ppr_peak_bp'] = ppr_peak_bp
		parameters_joined['ppr_mean_lp'] = ppr_mean_lp
		parameters_joined['ppr_mean_lp_fb'] = ppr_mean_lp_fb
		parameters_joined['ppr_mean_lp_trial'] = ppr_mean_lp_trial
		# parameters_joined['ppr_mean_bp'] = ppr_mean_bp
		# parameters_joined['ppr_proj_lp'] = ppr_proj_lp
		# parameters_joined['ppr_proj_bp'] = ppr_proj_bp
		# parameters_joined['sign_template_lp'] = sign_template_lp
		# parameters_joined['sign_template_bp'] = sign_template_bp
		parameters_joined['d_prime'] = d
		parameters_joined['criterion'] = c
		for i in range(len(aliases)):
			parameters_joined['d_prime_' + str(i)] = d_run[i]
			parameters_joined['criterion_' + str(i)] = c_run[i]
		# if self.experiment == 1:
		# 	parameters_joined['bpd_feed_lp'] = bpd_lp_feed
		# 	parameters_joined['bpd_feed_bp'] = bpd_bp_feed
		# 	parameters_joined['ppr_peak_feed_lp'] = ppr_peak_feed_lp
		# 	parameters_joined['ppr_peak_feed_bp'] = ppr_peak_feed_bp
		# 	parameters_joined['ppr_mean_feed_lp'] = ppr_mean_feed_lp
		# 	parameters_joined['ppr_mean_feed_bp'] = ppr_mean_feed_bp
		# 	parameters_joined['ppr_proj_feed_lp'] = ppr_proj_feed_lp
		# 	parameters_joined['ppr_proj_feed_bp'] = ppr_proj_feed_bp
		parameters_joined['subject'] = self.subject.initials
		self.ho.data_frame_to_hdf('', 'parameters_joined', parameters_joined)
		
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
		
		self.parameters_joined = self.ho.read_session_data('', 'parameters_joined')
		
		self.rt = self.parameters_joined['rt']
		
		self.omissions = np.array(self.parameters_joined['omissions'])
		self.yes = np.array(self.parameters_joined['yes']) * -self.omissions
		self.no = -self.yes
		self.correct = np.array(self.parameters_joined['correct']) * -self.omissions
		self.error = -self.correct
		self.hit = np.array(self.parameters_joined['hit']) * -self.omissions
		self.fa = np.array(self.parameters_joined['fa']) * -self.omissions
		self.miss = np.array(self.parameters_joined['miss']) * -self.omissions
		self.cr = np.array(self.parameters_joined['cr']) * -self.omissions
		
		self.contrast = np.array(self.parameters_joined['signal_contrast'])
		if self.experiment == 1:
			self.staircase = np.array(self.parameters_joined['staircase'])
		
		self.bpd = np.array(self.parameters_joined['bpd_lp'])
		self.ppr = np.array(self.parameters_joined['ppr_mean_lp'])
		#
		# self.criterion = np.array(self.parameters_joined['criterion'])[0]
		#
		# if self.experiment == 1:
		# 	self.ppr_feed = np.array(self.parameters_joined['ppr_peak_feed_lp'])
		# 	self.bpd_feed = np.array(self.parameters_joined['bpd_feed_lp'])
	
	def trial_wise_pupil(self,):
		
		for session in ['A', 'B', 'C', 'D']:
			
			params = self.parameters_joined[self.parameters_joined.drug == session]
			
			nr_runs = (pd.Series(np.array(params.trial_nr))==0).sum()
			start_run = np.where(pd.Series(np.array(params.trial_nr))==0)[0]
			run_nr = np.ones(len(np.array(params.trial_nr)))
			for i in range(nr_runs):
				if i != (nr_runs-1):
					run_nr[start_run[i]:start_run[i+1]] = i
				if i == (nr_runs-1): 
					run_nr[start_run[i]:] = i
		
			# data:
			d = {
			'trial_nr' : pd.Series(np.array(params.trial_nr)) + 1,
			'run_nr' : pd.Series(run_nr) + 1,
			'pupil_baseline' : pd.Series(np.array(params.bpd_lp)),
			'pupil_decision' : pd.Series(np.array(params.ppr_mean_lp)),
			'pupil_feedback' : pd.Series(np.array(params.ppr_mean_lp_fb)),
			'pupil_trial' : pd.Series(np.array(params.ppr_mean_lp_trial)),
			'rt' : pd.Series(np.array(params.rt)/1000.0),
			}
			data = pd.DataFrame(d)
			data.to_csv(os.path.join(self.project_directory, '{}_{}_pupil_data.csv'.format(self.subject.initials, session)))
	
	def psychometric_curve(self, plotting=True):
		
		nr_staircases = 2
		
		fig = pl.figure(figsize = (8,10))
		s1 = fig.add_subplot(2, 1, 1)
		s2 = fig.add_subplot(2,1, 2)
		plot_range = [min(self.contrast)-0.2, max(self.contrast)+0.2]
		
		tresholds = []
		for s in range(nr_staircases):
			
			indices = (self.staircase==s) * (-self.omissions)
			
			contrasts = self.contrast[indices]
			
			# bin:
			contrasts_bin = np.linspace(min(contrasts), max(contrasts), 10)
			contrasts_c = np.digitize(contrasts, contrasts_bin)
			contrasts = np.array([contrasts_bin[c] for c in contrasts_c-1])
			
			contrasts_unique = np.unique(contrasts)
			answers = self.yes[indices]
			corrects = self.correct[indices]
			
			print contrasts_unique
			print str(answers.shape[0]) + ' trials to be analyzed'
		
			# # delete staircases if neccesary:
			# staircase_to_delete = None
			# staircase_to_delete2 = None
			# if staircase_to_delete!=None:
			# 	ad = ad[(ad[:,-2])!=staircase_to_delete]
			# if staircase_to_delete2!=None:
			# 	ad = ad[(ad[:,-2])!=staircase_to_delete2]
			# else:
			# 	ad = ad
			
			
			pf, corrects_grouped = myfuncs.psychometric_curve(contrasts=contrasts, answers=answers, corrects=corrects,)
			corrects_grouped_sum = [c.sum() for c in corrects_grouped]
			nr_samples = [float(c.shape[0]) for c in corrects_grouped]
			tresholds.append(pf.getThres())
			
			# fig = plt.figure()
			# plt.plot((10**contrasts_unique)*100, nr_samples)
			# plt.ylabel('nr trails')
			# plt.xlabel('signal strength (% contrast)')
			#
			#
			# shell()
			
			# traces:
			trial_nr = np.arange(len(contrasts))
			if s == 0:
				s1.plot(trial_nr, contrasts, 'r--', linewidth = 1.25, alpha = 0.75)
			if s == 1:
				s1.plot(trial_nr, contrasts, 'b--', linewidth = 1.25, alpha = 0.75)
			s1.axis([-1,len(contrasts),plot_range[0], plot_range[1]])
			s1.set_title("staircase history " + self.subject.initials)
			s1.set_xlabel('trials', fontsize=10)
			s1.set_ylabel('log contrast', fontsize=10)
	
			# weibul fit:
			if s == 0:
				s2.scatter(contrasts_unique, np.array(corrects_grouped_sum) / np.array(nr_samples), s = np.array(nr_samples) + 5, facecolor = (1.0,1.0,1.0), edgecolor = 'r', alpha = 1.0, linewidth = 1.25)
				s2.plot(np.linspace(plot_range[0],plot_range[1], 200), pf.evaluate(np.linspace(plot_range[0],plot_range[1], 200)), 'r--', linewidth = 1.75, alpha = 0.75, label='noise redraw every 1 frame')
				s2.axvline(x=pf.getThres(), c = 'r', alpha = 0.7, linewidth = 2.25)
				# s2.axvline(x=pf.getCI(0.75)[0], c = 'r', alpha = 0.25, linewidth = 1.25)
				# s2.axvline(x=pf.getCI(0.75)[1], c = 'r', alpha = 0.25, linewidth = 1.25)
			if s == 1:
				s2.scatter(contrasts_unique, np.array(corrects_grouped_sum) / np.array(nr_samples), s = np.array(nr_samples) + 5, facecolor = (1.0,1.0,1.0), edgecolor = 'b', alpha = 1.0, linewidth = 1.25)
				s2.plot(np.linspace(plot_range[0],plot_range[1], 200), pf.evaluate(np.linspace(plot_range[0],plot_range[1], 200)), 'b--', linewidth = 1.75, alpha = 0.75, label='noise redraw every 4 frame')
				s2.axvline(x=pf.getThres(), c = 'b', alpha = 0.7, linewidth = 2.25)
				# s2.axvline(x=pf.getCI(0.75)[0], c = 'b', alpha = 0.25, linewidth = 1.25)
				# s2.axvline(x=pf.getCI(0.75)[1], c = 'b', alpha = 0.25, linewidth = 1.25)
			for c in contrasts_unique:
				s2.axvline(c, color='k', linewidth=0.5, alpha=0.2)
			s2.axis([plot_range[0],plot_range[1],0,1])
			s2.set_xlabel('log contrast', fontsize=10)
			s2.set_ylabel('proportion correct', fontsize=10)
			
			# shell()
	
		s2.legend(loc=2)
		s2.set_title("thresholds: " + str(round(tresholds[0],3)) + ' and ' + str(round(tresholds[1],3)))
		plt.tight_layout()
		fig.savefig(os.path.join(self.base_directory, 'figs', "behavior_psychometric_curve_" + self.subject.initials + '.pdf'))
		
		
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
		ax2.set_ylabel('criterion')
		ax.set_xlabel('confidence')
		# ax.set_ylim(ymin=0, ymax=2)
		# ax2.set_ylim(ymin=-0.7, ymax=0.7)
		ax2.legend()
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
	def __init__(self, subjects, experiment_name, experiment_nr, project_directory, sample_rate_new=50):
		
		self.subjects = subjects
		self.nr_subjects = len(self.subjects)
		self.experiment_name = experiment_name
		self.experiment = experiment_nr
		self.project_directory = project_directory
		self.sample_rate_new = int(sample_rate_new)
		self.downsample_rate = int(1000 / sample_rate_new)
		
		parameters = []
		self.omissions_per_s = []
		for s in self.subjects:
			self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
			self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
			self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
			
			parameters.append(self.ho.read_session_data('', 'parameters_joined'))
			self.omissions_per_s.append(np.array(self.ho.read_session_data('', 'parameters_joined')['omissions'], dtype=bool))
		self.parameters_joined = pd.concat(parameters)
		self.omissions = np.array(self.parameters_joined['omissions'], dtype=bool)
		
		self.parameters_joined = self.parameters_joined[-self.omissions]
		
		self.rt = np.array(self.parameters_joined['rt'])
		
		self.hit = np.array(self.parameters_joined['hit'], dtype=bool)
		self.fa = np.array(self.parameters_joined['fa'], dtype=bool)
		self.miss = np.array(self.parameters_joined['miss'], dtype=bool)
		self.cr = np.array(self.parameters_joined['cr'], dtype=bool)
		
		self.yes = np.array(self.parameters_joined['yes'], dtype=bool)
		self.no = -np.array(self.parameters_joined['yes'], dtype=bool)
		self.present = np.array(self.parameters_joined['signal_present'], dtype=bool)
		self.absent = -np.array(self.parameters_joined['signal_present'], dtype=bool)
		self.correct = np.array(self.parameters_joined['correct'], dtype=bool)
		self.error = -np.array(self.parameters_joined['correct'], dtype=bool)
		
		self.bpd = np.array(self.parameters_joined['bpd_lp'])
		self.ppr = np.array(self.parameters_joined['ppr_mean_lp'])
		self.ppr_fb = np.array(self.parameters_joined['ppr_mean_lp_fb'])
		self.ppr_trial = np.array(self.parameters_joined['ppr_mean_lp_trial'])
		
		self.subj_idx = np.concatenate(np.array([np.repeat(i, sum(self.parameters_joined['subject'] == self.subjects[i])) for i in range(len(self.subjects))]))
		
		self.criterion = np.array([np.array(self.parameters_joined[self.parameters_joined['subject']==subj]['criterion'])[0] for subj in self.subjects])
		
		# shell()
		
		self.pupil_l_ind = np.concatenate([self.ppr[np.array(self.parameters_joined.subject == subj_idx)] <= np.percentile(self.ppr[np.array(self.parameters_joined.subject == subj_idx)], 50) for subj_idx in self.subjects])
		self.pupil_h_ind = np.concatenate([self.ppr[np.array(self.parameters_joined.subject == subj_idx)] > np.percentile(self.ppr[np.array(self.parameters_joined.subject == subj_idx)], 50) for subj_idx in self.subjects])
		self.pupil_rest_ind = -(self.pupil_h_ind + self.pupil_l_ind)
		
		if self.experiment == 2:
			# A&C = 0 = placebo;
			# B&D = 1 = atomoxetine
			self.drug = np.array(np.array(self.parameters_joined['drug']) == 'B', dtype=bool) + np.array(np.array(self.parameters_joined['drug']) == 'D', dtype=bool)
			self.simon = np.array(np.array(self.parameters_joined['drug']) == 'C', dtype=bool) + np.array(np.array(self.parameters_joined['drug']) == 'D', dtype=bool)
			self.difficulty = np.array(self.parameters_joined['difficulty'], dtype=bool)
			
		# initialize behavior operator:
		if self.experiment == 2:
			d = {
			'subj_idx' : pd.Series(self.subj_idx),
			'correct' : pd.Series(np.array(self.parameters_joined['correct'], dtype=int)),
			'choice' : pd.Series(np.array(self.parameters_joined['yes'], dtype=int)),
			'stimulus' : pd.Series(np.array(self.parameters_joined['signal_present'], dtype=int)),
			'rt' : pd.Series(np.array(self.rt)) / 1000.0,
			'pupil' : pd.Series(np.array(self.ppr)),
			'pupil_b' : pd.Series(np.array(self.bpd)),
			'pupil_high' : pd.Series(self.pupil_h_ind),
			'split' : pd.Series(self.drug),
			}
		self.df = pd.DataFrame(d)
		# self.df = self.df[self.df.correct != -1]
		self.behavior = myfuncs.behavior(self.df)
		
		# SPLIT:
		pupil_b_drug = np.zeros(len(self.subjects))
		pupil_b_placebo = np.zeros(len(self.subjects))
		pupil_drug = np.zeros(len(self.subjects))
		pupil_placebo = np.zeros(len(self.subjects))
		pupil_fb_drug = np.zeros(len(self.subjects))
		pupil_fb_placebo = np.zeros(len(self.subjects))
		pupil_trial_drug = np.zeros(len(self.subjects))
		pupil_trial_placebo = np.zeros(len(self.subjects))
		for i, s in enumerate(self.subjects):
			pupil_b_drug[i] = np.mean(self.bpd[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_b_placebo[i] = np.mean(self.bpd[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_drug[i] = np.mean(self.ppr[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_placebo[i] = np.mean(self.ppr[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_fb_drug[i] = np.mean(self.ppr_fb[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_fb_placebo[i] = np.mean(self.ppr_fb[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_trial_drug[i] = np.mean(self.ppr_trial[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_trial_placebo[i] = np.mean(self.ppr_trial[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
		
		# SPLIT 1
		self.split_1 = pupil_placebo >= np.median(pupil_placebo) # ind --> large phasic pupils
		
		# SPLIT 2
		diff = pupil_drug - pupil_placebo
		self.split_2 = diff >= np.median(diff) # ind --> smaller pupils under atomox
		
		# SPLIT 3:
		self.split_3 = abs(self.criterion) >= np.median(abs(self.criterion)) # ind --> large absolute criterion
		
	def behavior_rt(self):
		
		import rpy2.robjects as robjects
		import rpy2.rlike.container as rlc
		
		print 'median RT = {}'.format(round(np.mean([np.median(self.df.rt[np.array(self.df.subj_idx == subj_idx)]) for subj_idx in range(len(self.subjects))]),5))
		for j in range(2):
			
			# normalize RT:
			df = self.df[self.difficulty == j]
			
			df.rt = np.concatenate([(df.rt[np.array(df.subj_idx == subj_idx)] - np.mean(df.rt[np.array(df.subj_idx == subj_idx)])) / np.std(df.rt[np.array(df.subj_idx == subj_idx)]) for subj_idx in range(len(self.subjects))])
			
			# run behavior operator:
			self.behavior = myfuncs.behavior(df)
			rt_correct_0, rt_error_0, rt_correct_std_0, rt_error_std_0 = self.behavior.rt_mean_var(split_by='split', split_target=0)
			rt_correct_1, rt_error_1, rt_correct_std_1, rt_error_std_1 = self.behavior.rt_mean_var(split_by='split', split_target=1)
			
			# ANOVA:
			data = np.concatenate((rt_correct_0, rt_correct_1, rt_error_0, rt_error_1))
			subject = np.concatenate((np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects))))
			correct = np.concatenate((np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects))))
			drug = np.concatenate((np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects))))
			
			d = rlc.OrdDict([('correct', robjects.IntVector(list(correct.ravel()))), ('drug', robjects.IntVector(list(drug.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
			robjects.r.assign('dataf', robjects.DataFrame(d))
			robjects.r('attach(dataf)')
			statres = robjects.r('res = summary(aov(data ~ as.factor(correct)*as.factor(drug) + Error(as.factor(subject)), dataf))')
			p1 = statres[-1][0][4][0]	# we will log-transform and min p values
			p2 = statres[-1][0][4][1]	# we will log-transform and min p values
			p3 = statres[-1][0][4][2]	# we will log-transform and min p values
			
			print
			print statres
			
			data = np.concatenate((rt_correct_std_0, rt_correct_std_1, rt_error_std_0, rt_error_std_1))
			subject = np.concatenate((np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects))))
			correct = np.concatenate((np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects))))
			drug = np.concatenate((np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects))))
			
			d = rlc.OrdDict([('correct', robjects.IntVector(list(correct.ravel()))), ('drug', robjects.IntVector(list(drug.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
			robjects.r.assign('dataf', robjects.DataFrame(d))
			robjects.r('attach(dataf)')
			statres = robjects.r('res = summary(aov(data ~ as.factor(correct)*as.factor(drug) + Error(as.factor(subject)), dataf))')
			p4 = statres[-1][0][4][0]	# we will log-transform and min p values
			p5 = statres[-1][0][4][1]	# we will log-transform and min p values
			p6 = statres[-1][0][4][2]	# we will log-transform and min p values
			
			print
			print statres
			
			
			
			# plot:
			N = 2
			ind = np.linspace(0,N/2,N)
		
			fig = plt.figure(figsize=(1.5,2))
			ax = plt.subplot(111)
			ax.plot(ind, np.array([np.mean(rt_correct_0), np.mean(rt_correct_1)]), color='g', label='correct')
			ax.plot(ind, np.array([np.mean(rt_error_0), np.mean(rt_error_1)]), color='r', label='error')
			ax.errorbar(ind, np.array([np.mean(rt_correct_0), np.mean(rt_correct_1)]), yerr=np.array([sp.stats.sem(rt_correct_0), sp.stats.sem(rt_correct_1)]), marker='o', color='g', markeredgecolor='w', markeredgewidth=1, markersize=5, ecolor = 'g', elinewidth=0.5)
			ax.errorbar(ind, np.array([np.mean(rt_error_0), np.mean(rt_error_1)]), yerr=np.array([sp.stats.sem(rt_error_0), sp.stats.sem(rt_error_1)]), marker='o', color='r', markeredgecolor='w', markeredgewidth=1, markersize=5, ecolor = 'r', elinewidth=0.5 )
			ax.set_xlim(xmin=ind[0]-0.5, xmax=ind[1]+0.5)
			# ax.legend()
			ax.set_xticks( (ind) )
			ax.set_xticklabels( ('placebo', 'drug') )
			plt.title('main correct: p = {}\nmain drug: p = {}\ninteraction: p = {}'.format(round(p1,4),round(p2,4),round(p3,4)))
			plt.ylabel('mean RT (Z)')
			sns.despine(offset=10, trim=True)
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_mean_{}.pdf'.format(j)))
		
		
			fig = plt.figure(figsize=(1.5,2))
			ax = plt.subplot(111)
			ax.plot(ind, np.array([np.mean(rt_correct_std_0), np.mean(rt_correct_std_1)]), color='g', label='correct')
			ax.plot(ind, np.array([np.mean(rt_error_std_0), np.mean(rt_error_std_1)]), color='r', label='error')
			ax.errorbar(ind, np.array([np.mean(rt_correct_std_0), np.mean(rt_correct_std_1)]), yerr=np.array([sp.stats.sem(rt_correct_std_0), sp.stats.sem(rt_correct_std_1)]), marker='o', color='g', markeredgecolor='w', markeredgewidth=1, markersize=5, ecolor = 'g', elinewidth=0.5)
			ax.errorbar(ind, np.array([np.mean(rt_error_std_0), np.mean(rt_error_std_1)]), yerr=np.array([sp.stats.sem(rt_error_std_0), sp.stats.sem(rt_error_std_1)]), marker='o', color='r', markeredgecolor='w', markeredgewidth=1, markersize=5, ecolor = 'r', elinewidth=0.5 )
			ax.set_xlim(xmin=ind[0]-0.5, xmax=ind[1]+0.5)
			# ax.legend()
			ax.set_xticks( (ind) )
			ax.set_xticklabels( ('placebo', 'drug') )
			plt.title('main correct: p = {}\nmain drug: p = {}\ninteraction: p = {}'.format(round(p4,4),round(p5,4),round(p6,4)))
			plt.ylabel('standard deviation RT (Z)')
			sns.despine(offset=10, trim=True)
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_var_{}.pdf'.format(j)))
		
		
		
		
	def behavior_rt_kde(self):
		
		# RT histograms:
		# --------------

		x_grid = [0, 3, 100]
		c0_pdf, c1_pdf, c_correct_pdf, c_error_pdf, c0_correct_pdf, c0_error_pdf, c1_correct_pdf, c1_error_pdf = self.behavior.rt_kernel_densities(x_grid=x_grid, bandwidth=0.1)

		yes = np.vstack(c0_pdf)
		no = np.vstack(c1_pdf)
		correct = np.vstack(c_correct_pdf)
		error = np.vstack(c_error_pdf)
		cr = np.vstack(c0_correct_pdf)
		miss = np.vstack(c0_error_pdf)
		hit = np.vstack(c1_correct_pdf)
		fa = np.vstack(c1_error_pdf)

		step = pd.Series(np.linspace(x_grid[0], x_grid[1], x_grid[2]), name='rt (s)')

		# Make the plt.plot
		fig = plt.figure(figsize=(2, 3))
		ax = plt.subplot(211)
		conditions = pd.Series(['hits'], name='trial type')
		sns.tsplot(hit, time=step, condition=conditions, value='kde', color='red', ci=66, lw=1, ls='-', ax=ax)
		conditions = pd.Series(['miss'], name='trial type')
		sns.tsplot(miss, time=step, condition=conditions, value='kde', color='blue', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
		ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.stimulus==1) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
		ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.stimulus==1) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
		ax = plt.subplot(212)
		conditions = pd.Series(['cr'], name='trial type')
		sns.tsplot(cr, time=step, condition=conditions, value='kde', color='blue', ci=66, lw=1, ls='-', ax=ax)
		conditions = pd.Series(['fa'], name='trial type')
		sns.tsplot(fa, time=step, condition=conditions, value='kde', color='red', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
		ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.stimulus==0) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
		ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.stimulus==0) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists.pdf'))
		
		# shell()
		
		for j in range(2):
			
			# RT histograms:
			# --------------

			x_grid = [0, 3, 100]
			c0_pdf, c1_pdf, c_correct_pdf, c_error_pdf, c0_correct_pdf, c0_error_pdf, c1_correct_pdf, c1_error_pdf = self.behavior.rt_kernel_densities(x_grid=x_grid, bandwidth=0.1, split_by='split', split_target=j)

			yes = np.vstack(c0_pdf)
			no = np.vstack(c1_pdf)
			correct = np.vstack(c_correct_pdf)
			error = np.vstack(c_error_pdf)
			cr = np.vstack(c0_correct_pdf)
			miss = np.vstack(c0_error_pdf)
			hit = np.vstack(c1_correct_pdf)
			fa = np.vstack(c1_error_pdf)

			step = pd.Series(np.linspace(x_grid[0], x_grid[1], x_grid[2]), name='rt (s)')

			# Make the plt.plot
			fig = plt.figure(figsize=(2,3))
			ax = plt.subplot(211)
			conditions = pd.Series(['hits'], name='trial type')
			sns.tsplot(hit, time=step, condition=conditions, value='kde', color='red', ci=66, lw=1, ls='-', ax=ax)
			conditions = pd.Series(['miss'], name='trial type')
			sns.tsplot(miss, time=step, condition=conditions, value='kde', color='blue', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
			ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.stimulus==1) & (self.df.subj_idx==i) & (self.df.split==j)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
			ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.stimulus==1) & (self.df.subj_idx==i) & (self.df.split==j)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
			ax = plt.subplot(212)
			conditions = pd.Series(['cr'], name='trial type')
			sns.tsplot(cr, time=step, condition=conditions, value='kde', color='blue', ci=66, lw=1, ls='-', ax=ax)
			conditions = pd.Series(['fa'], name='trial type')
			sns.tsplot(fa, time=step, condition=conditions, value='kde', color='red', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
			ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.stimulus==0) & (self.df.subj_idx==i) & (self.df.split==j)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
			ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.stimulus==0) & (self.df.subj_idx==i) & (self.df.split==j)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
			sns.despine(offset=10, trim=True)
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_{}.pdf'.format(j)))
			
			
			# Make the plt.plot
			fig = plt.figure(figsize=(2, 1.5))
			ax = plt.subplot(111)
			conditions = pd.Series(['correct'], name='trial type')
			sns.tsplot(correct, time=step, condition=conditions, value='kde', color='green', ci=66, lw=1, ls='-', ax=ax)
			conditions = pd.Series(['error'], name='trial type')
			sns.tsplot(error, time=step, condition=conditions, value='kde', color='red', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
			ax.axvline(np.mean([np.median(self.df.rt[(self.df.correct==1) & (self.df.subj_idx==i) & (self.df.split==j)]) for i in range(self.nr_subjects)]), color='g', linestyle='--', lw=1)
			ax.axvline(np.mean([np.median(self.df.rt[(self.df.correct==0) & (self.df.subj_idx==i) & (self.df.split==j)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
			sns.despine(offset=10, trim=True)
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_correct_{}.pdf'.format(j)))
			
			
	
	def behavior_choice(self):
		
		# SDT fractions:
		correct_f, error_f, yes_f, no_f, cr_f, miss_f, hit_f, fa_f = self.behavior.choice_fractions()
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
		fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_fractions.pdf'))
		
		# d-prime and criterion:
		# ----------------------
		
		rt, acc, d, c = self.behavior.behavior_measures()
		MEANS = (d.mean(), c.mean())
		SEMS = (sp.stats.sem(d), sp.stats.sem(c))
		N = 2
		ind = np.linspace(0,N/2,N)
		bar_width = 0.50
		fig = plt.figure(figsize=(2,3))
		ax = fig.add_subplot(111)
		for i in range(N):
			ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = 'k', alpha = [1,0.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		ax.set_title('N={}'.format(self.nr_subjects), size=8)
		ax.set_xticks( (ind) )
		ax.set_xticklabels( ("d'", 'c') )
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures.pdf'))
		
		# measures:
		
		measures_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		titles = ['rt', 'acc', 'd', 'c']
		ylim_max = [1.5,1,2,0.5]
		ylim_min = [0,0,0,-0.5]
		
		for m in range(len(measures_0)):
		
			MEANS = (measures_0[m].mean(), measures_1[m].mean())
			SEMS = (sp.stats.sem(measures_0[m]), sp.stats.sem(measures_1[m]))
			N = 2
			ind = np.linspace(0,N/2,N)
			bar_width = 0.50
			fig = plt.figure(figsize=(1.5,2.5))
			ax = fig.add_subplot(111)
			for i in range(N):
				ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['b', 'r'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
			ax.set_ylim( ylim_min[m],ylim_max[m] )
			ax.set_title('N={}'.format(self.nr_subjects), size=7)
			ax.set_ylabel(titles[m], size=7)
			ax.set_xticks( (ind) )
			ax.set_xticklabels( ('placebo', 'drug') )
			plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(round(sp.stats.ttest_rel(measures_0[m], measures_1[m])[1],3)), horizontalalignment='center')
			sns.despine(offset=10, trim=True)
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_{}.pdf'.format(titles[m])))

		
		# separate per difficulty:
		
		for j in range(2):
			
			self.behavior = myfuncs.behavior(self.df[(self.difficulty == j)])
			measures_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
			measures_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
			for m in range(len(measures_0)):
				
				MEANS = (measures_0[m].mean(), measures_1[m].mean())
				SEMS = (sp.stats.sem(measures_0[m]), sp.stats.sem(measures_1[m]))
				N = 2
				ind = np.linspace(0,N/2,N)
				bar_width = 0.50
				fig = plt.figure(figsize=(1.5,2.5))
				ax = fig.add_subplot(111)
				for i in range(N):
					ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['b', 'r'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
				ax.set_ylim( ylim_min[m],ylim_max[m] )
				ax.set_title('N={}'.format(self.nr_subjects), size=7)
				ax.set_ylabel(titles[m], size=7)
				ax.set_xticks( (ind) )
				ax.set_xticklabels( ('placebo', 'drug') )
				plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(round(sp.stats.ttest_rel(measures_0[m], measures_1[m])[1],3)), horizontalalignment='center')
				sns.despine(offset=10, trim=True)
				plt.tight_layout()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_{}_{}.pdf'.format(titles[m], j)))
			
		# separate per difficulty and split:
		# SPLIT 1
		split_diff = self.split_1
		subj = np.unique(self.subj_idx)
		group_ind1 = subj[split_diff]
		indd = np.zeros(self.drug.shape, dtype=int)
		for s in subj:
			if s in group_ind1:
				indd[self.subj_idx==s] = 1
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 0)*(indd==0)])
		measures_0_0_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_0_0_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 1)*(indd==0)])
		measures_0_1_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_0_1_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 0)*(indd==1)])
		measures_1_0_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_1_0_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 1)*(indd==1)])
		measures_1_1_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_1_1_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		# ANOVA:
		for m in range(len(measures_0_0_0)):
			dv = np.concatenate((measures_0_0_0[m], measures_0_0_1[m], measures_0_1_0[m], measures_0_1_1[m], measures_1_0_0[m], measures_1_0_1[m], measures_1_1_0[m], measures_1_1_1[m]))
			spli = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.zeros(len(measures_0_0_1[m])), np.zeros(len(measures_0_1_0[m])), np.zeros(len(measures_0_1_1[m])), np.ones(len(measures_1_0_0[m])), np.ones(len(measures_1_0_1[m])), np.ones(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			diff = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.zeros(len(measures_0_0_1[m])), np.ones(len(measures_0_1_0[m])), np.ones(len(measures_0_1_1[m])), np.zeros(len(measures_1_0_0[m])), np.zeros(len(measures_1_0_1[m])), np.ones(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			drug = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.ones(len(measures_0_0_1[m])), np.zeros(len(measures_0_1_0[m])), np.ones(len(measures_0_1_1[m])), np.zeros(len(measures_1_0_0[m])), np.ones(len(measures_1_0_1[m])), np.zeros(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			subject = np.concatenate((subj[-split_diff], subj[-split_diff], subj[-split_diff], subj[-split_diff], subj[split_diff], subj[split_diff], subj[split_diff], subj[split_diff]))
			d = rlc.OrdDict([('spli', robjects.IntVector(list(spli.ravel()))), ('diff', robjects.IntVector(list(diff.ravel()))), ('drug', robjects.IntVector(list(drug.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(dv.ravel())))])
			robjects.r.assign('dataf', robjects.DataFrame(d))
			robjects.r('attach(dataf)')
			statres = robjects.r('res = summary(aov(data ~ as.factor(spli)*as.factor(diff)*as.factor(drug) + Error(as.factor(subject)), dataf))')
			p1 = statres[-1][0][4][0]	# we will log-transform and min p values
			p2 = statres[-1][0][4][1]	# we will log-transform and min p values
			p3 = statres[-1][0][4][2]	# we will log-transform and min p values
			
			text_file = open(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_ANOVA_split_1_{}.txt'.format(titles[m])), 'w')
			for string in statres:
				text_file.write(str(string))
			text_file.close()
			
		for j in range(2):
			for k in range(2):
				for m in range(len(measures_0_0_0)):
					exec('MEANS = (measures_{}_{}_0[m].mean(), measures_{}_{}_1[m].mean())'.format(k, j, k, j))
					exec('SEMS = (sp.stats.sem(measures_{}_{}_0[m]), sp.stats.sem(measures_{}_{}_1[m]))'.format(k, j, k, j))
					N = 2
					ind = np.linspace(0,N/2,N)
					bar_width = 0.50
					fig = plt.figure(figsize=(1.5,2.5))
					ax = fig.add_subplot(111)
					for i in range(N):
						ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['b', 'r'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
					ax.set_ylim( ylim_min[m],ylim_max[m] )
					ax.set_title('N={}'.format(self.nr_subjects), size=7)
					ax.set_ylabel(titles[m], size=7)
					ax.set_xticks( (ind) )
					ax.set_xticklabels( ('placebo', 'drug') )
					exec('p = round(sp.stats.ttest_rel(measures_{}_{}_0[m], measures_{}_{}_1[m])[1], 3)'.format(k, j, k, j))
					plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(p), horizontalalignment='center')
					sns.despine(offset=10, trim=True)
					plt.tight_layout()
					fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_split_{}_{}_{}.pdf'.format(k, titles[m], j)))
		
		# SPLIT 2:
		split_diff = self.split_2
		subj = np.unique(self.subj_idx)
		group_ind1 = subj[split_diff]
		indd = np.zeros(self.drug.shape, dtype=int)
		for s in subj:
			if s in group_ind1:
				indd[self.subj_idx==s] = 1
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 0)*(indd==0)])
		measures_0_0_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_0_0_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 1)*(indd==0)])
		measures_0_1_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_0_1_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 0)*(indd==1)])
		measures_1_0_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_1_0_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 1)*(indd==1)])
		measures_1_1_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_1_1_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		# ANOVA:
		for m in range(len(measures_0_0_0)):
			dv = np.concatenate((measures_0_0_0[m], measures_0_0_1[m], measures_0_1_0[m], measures_0_1_1[m], measures_1_0_0[m], measures_1_0_1[m], measures_1_1_0[m], measures_1_1_1[m]))
			spli = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.zeros(len(measures_0_0_1[m])), np.zeros(len(measures_0_1_0[m])), np.zeros(len(measures_0_1_1[m])), np.ones(len(measures_1_0_0[m])), np.ones(len(measures_1_0_1[m])), np.ones(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			diff = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.zeros(len(measures_0_0_1[m])), np.ones(len(measures_0_1_0[m])), np.ones(len(measures_0_1_1[m])), np.zeros(len(measures_1_0_0[m])), np.zeros(len(measures_1_0_1[m])), np.ones(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			drug = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.ones(len(measures_0_0_1[m])), np.zeros(len(measures_0_1_0[m])), np.ones(len(measures_0_1_1[m])), np.zeros(len(measures_1_0_0[m])), np.ones(len(measures_1_0_1[m])), np.zeros(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			subject = np.concatenate((subj[-split_diff], subj[-split_diff], subj[-split_diff], subj[-split_diff], subj[split_diff], subj[split_diff], subj[split_diff], subj[split_diff]))
			d = rlc.OrdDict([('spli', robjects.IntVector(list(spli.ravel()))), ('diff', robjects.IntVector(list(diff.ravel()))), ('drug', robjects.IntVector(list(drug.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(dv.ravel())))])
			robjects.r.assign('dataf', robjects.DataFrame(d))
			robjects.r('attach(dataf)')
			statres = robjects.r('res = summary(aov(data ~ as.factor(spli)*as.factor(diff)*as.factor(drug) + Error(as.factor(subject)), dataf))')
			p1 = statres[-1][0][4][0]	# we will log-transform and min p values
			p2 = statres[-1][0][4][1]	# we will log-transform and min p values
			p3 = statres[-1][0][4][2]	# we will log-transform and min p values
			text_file = open(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_ANOVA_split_2_{}.txt'.format(titles[m])), 'w')
			for string in statres:
				text_file.write(str(string))
			text_file.close()
		
		for j in range(2):
			for k in range(2):
				for m in range(len(measures_0_0_0)):
					exec('MEANS = (measures_{}_{}_0[m].mean(), measures_{}_{}_1[m].mean())'.format(k, j, k, j))
					exec('SEMS = (sp.stats.sem(measures_{}_{}_0[m]), sp.stats.sem(measures_{}_{}_1[m]))'.format(k, j, k, j))
					N = 2
					ind = np.linspace(0,N/2,N)
					bar_width = 0.50
					fig = plt.figure(figsize=(1.5,2.5))
					ax = fig.add_subplot(111)
					for i in range(N):
						ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['b', 'r'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
					ax.set_ylim( ylim_min[m],ylim_max[m] )
					ax.set_title('N={}'.format(self.nr_subjects), size=7)
					ax.set_ylabel(titles[m], size=7)
					ax.set_xticks( (ind) )
					ax.set_xticklabels( ('placebo', 'drug') )
					exec('p = round(sp.stats.ttest_rel(measures_{}_{}_0[m], measures_{}_{}_1[m])[1], 3)'.format(k, j, k, j))
					plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(p), horizontalalignment='center')
					sns.despine(offset=10, trim=True)
					plt.tight_layout()
					fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_split_{}_{}_{}.pdf'.format(k+2, titles[m], j)))
			
		# SPLIT 3:
		split_diff = self.split_3
		subj = np.unique(self.subj_idx)
		group_ind1 = subj[split_diff]
		indd = np.zeros(self.drug.shape, dtype=int)
		for s in subj:
			if s in group_ind1:
				indd[self.subj_idx==s] = 1
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 0)*(indd==0)])
		measures_0_0_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_0_0_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 1)*(indd==0)])
		measures_0_1_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_0_1_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 0)*(indd==1)])
		measures_1_0_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_1_0_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		self.behavior = myfuncs.behavior(self.df[(self.difficulty == 1)*(indd==1)])
		measures_1_1_0 = self.behavior.behavior_measures(split_by='split', split_target=0)
		measures_1_1_1 = self.behavior.behavior_measures(split_by='split', split_target=1)
		
		# ANOVA:
		for m in range(len(measures_0_0_0)):
			dv = np.concatenate((measures_0_0_0[m], measures_0_0_1[m], measures_0_1_0[m], measures_0_1_1[m], measures_1_0_0[m], measures_1_0_1[m], measures_1_1_0[m], measures_1_1_1[m]))
			spli = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.zeros(len(measures_0_0_1[m])), np.zeros(len(measures_0_1_0[m])), np.zeros(len(measures_0_1_1[m])), np.ones(len(measures_1_0_0[m])), np.ones(len(measures_1_0_1[m])), np.ones(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			diff = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.zeros(len(measures_0_0_1[m])), np.ones(len(measures_0_1_0[m])), np.ones(len(measures_0_1_1[m])), np.zeros(len(measures_1_0_0[m])), np.zeros(len(measures_1_0_1[m])), np.ones(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			drug = np.concatenate((np.zeros(len(measures_0_0_0[m])), np.ones(len(measures_0_0_1[m])), np.zeros(len(measures_0_1_0[m])), np.ones(len(measures_0_1_1[m])), np.zeros(len(measures_1_0_0[m])), np.ones(len(measures_1_0_1[m])), np.zeros(len(measures_1_1_0[m])), np.ones(len(measures_1_1_1[m]))))
			subject = np.concatenate((subj[-split_diff], subj[-split_diff], subj[-split_diff], subj[-split_diff], subj[split_diff], subj[split_diff], subj[split_diff], subj[split_diff]))
			d = rlc.OrdDict([('spli', robjects.IntVector(list(spli.ravel()))), ('diff', robjects.IntVector(list(diff.ravel()))), ('drug', robjects.IntVector(list(drug.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(dv.ravel())))])
			robjects.r.assign('dataf', robjects.DataFrame(d))
			robjects.r('attach(dataf)')
			statres = robjects.r('res = summary(aov(data ~ as.factor(spli)*as.factor(diff)*as.factor(drug) + Error(as.factor(subject)), dataf))')
			p1 = statres[-1][0][4][0]	# we will log-transform and min p values
			p2 = statres[-1][0][4][1]	# we will log-transform and min p values
			p3 = statres[-1][0][4][2]	# we will log-transform and min p values
			text_file = open(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_ANOVA_split_3_{}.txt'.format(titles[m])), 'w')
			for string in statres:
				text_file.write(str(string))
			text_file.close()
			
		for j in range(2):
			for k in range(2):
				for m in range(len(measures_0_0_0)):
					exec('MEANS = (measures_{}_{}_0[m].mean(), measures_{}_{}_1[m].mean())'.format(k, j, k, j))
					exec('SEMS = (sp.stats.sem(measures_{}_{}_0[m]), sp.stats.sem(measures_{}_{}_1[m]))'.format(k, j, k, j))
					N = 2
					ind = np.linspace(0,N/2,N)
					bar_width = 0.50
					fig = plt.figure(figsize=(1.5,2.5))
					ax = fig.add_subplot(111)
					for i in range(N):
						ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['b', 'r'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
					ax.set_ylim( ylim_min[m],ylim_max[m] )
					ax.set_title('N={}'.format(self.nr_subjects), size=7)
					ax.set_ylabel(titles[m], size=7)
					ax.set_xticks( (ind) )
					ax.set_xticklabels( ('placebo', 'drug') )
					exec('p = round(sp.stats.ttest_rel(measures_{}_{}_0[m], measures_{}_{}_1[m])[1], 3)'.format(k, j, k, j))
					plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(p), horizontalalignment='center')
					sns.despine(offset=10, trim=True)
					plt.tight_layout()
					fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_split_{}_{}_{}.pdf'.format(k+4, titles[m], j)))
			
			
			
	def behavior_omissions(self):
		
		# count omissions:
		fractions = np.zeros(len(self.subjects))
		nr_trials = np.zeros(len(self.subjects))
		for i in range(len(self.subjects)):
			fractions[i] = sum(self.omissions_per_s[i]) / float(len(self.omissions_per_s[i]))
			nr_trials[i] = sum(-self.omissions_per_s[i])
		
		fig = plt.figure()
		ind = np.arange(len(self.subjects))
		ax = plt.subplot(111)
		for i, f in enumerate(fractions):
			ax.bar(ind[i], f, align='center')
		plt.xticks(ind, self.subjects)
		plt.ylabel('fraction of trials deleted due\nto blinks / saccades')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_omissions_1.pdf'))
		
		fig = plt.figure()
		ind = np.arange(len(self.subjects))
		ax = plt.subplot(111)
		for i, n in enumerate(nr_trials):
			ax.bar(ind[i], n, align='center')
		plt.xticks(ind, self.subjects)
		plt.ylabel('number of clean trials')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_omissions_2.pdf'))
		
	
	def psychometric_fits(self):
		
		plot_range = [min(self.parameters_joined['signal_contrast'])-0.2, max(self.parameters_joined['signal_contrast'])+0.2]
		x = np.linspace(plot_range[0],plot_range[1], 200)
		pf1 = []
		pf2 = []
		
		c_unique = np.unique(self.parameters_joined['signal_contrast'])
		
		for noise_redraw in (1,4):
			
			parameters = self.parameters_joined[self.parameters_joined['noise_redraw'] == noise_redraw]
			
			for i in xrange(self.nr_subjects):
				ind = np.array(parameters['subject']==self.subjects[i])
				data_subj = parameters[ind]
			
				contrasts = np.array(data_subj['signal_contrast'])
				
				# bin:
				contrasts_bin = np.linspace(min(contrasts), max(contrasts), 10)
				contrasts_c = np.digitize(contrasts, contrasts_bin)
				contrasts = np.array([contrasts_bin[c] for c in contrasts_c-1])
				
				contrasts_unique = np.unique(contrasts)
				answers = np.array(data_subj['yes'])
				corrects = np.array(data_subj['correct'])
				
				print contrasts_unique
				print str(answers.shape[0]) + ' trials to be analyzed'
				
				pf, corrects_grouped = myfuncs.psychometric_curve(contrasts=contrasts, answers=answers, corrects=corrects)
				if noise_redraw == 1:
					pf1.append(pf.evaluate(x))
				if noise_redraw == 4:
					pf2.append(pf.evaluate(x))
		
		
		pf1_mean = np.mean(np.vstack(pf1), axis=0)
		pf1_sem = sp.stats.sem(np.vstack(pf1), axis=0)
		pf2_mean = np.mean(np.vstack(pf2), axis=0)
		pf2_sem = sp.stats.sem(np.vstack(pf2), axis=0)
		
		# shell()
		
		# x2 = (10**x)
		x2 = x
		
		fig = plt.figure(figsize = (6,3))
		ax = plt.subplot(121)
		for c in c_unique:
			# plt.axvline((10**c)*100, color='k', linewidth=0.5, alpha=0.2)
			plt.axvline(c, color='k', linewidth=0.5, alpha=0.2)
		for i in range(self.nr_subjects):
			ax.plot(x2, np.vstack(pf1)[i,:], 'r--', linewidth = 1.75, alpha = 1, label='noise redraw every 1 frame')
			ax.plot(x2, np.vstack(pf2)[i,:], 'b--', linewidth = 1.75, alpha = 1, label='noise redraw every 4 frame')
		ax.set_xlabel(r'contrast ($10^1\!$%)', fontsize=10)
		ax.set_ylabel('proportion correct', fontsize=10)
		ax = plt.subplot(122)
		ax.hist(np.array(self.parameters_joined['signal_contrast']), bins=10, color='gray', alpha=0.25, lw=1)
		ax2 = plt.twinx(ax)
		ax2.plot(x2, pf1_mean, 'r--', linewidth = 1.75, alpha = 1, label='noise redraw every 1 frame')
		ax2.fill_between(x2, pf1_mean+pf1_sem, pf1_mean-pf1_sem, color='r', alpha=0.25)
		ax2.plot(x2, pf2_mean, 'b--', linewidth = 1.75, alpha = 1, label='noise redraw every 4 frame')
		ax2.fill_between(x2, pf2_mean+pf2_sem, pf2_mean-pf2_sem, color='b', alpha=0.25)
		ax.set_xlabel(r'contrast ($10^1\!$%)', fontsize=10)
		ax.set_ylabel('nr trials', fontsize=10)
		ax2.set_ylabel('proportion correct', fontsize=10)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_psychometric_curves.pdf'))
		
		# shell()
		
		# # The data are form a 2afc task
		# nafc = 2
		#
		# # Now we get the data
		# stimulus_intensities = [0.0,2.0,4.0,6.0,8.0,10.0]
		# number_of_correct = [34,32,40,48,50,48]
		# number_of_trials  = [50]*len(stimulus_intensities)
		# data = zip(stimulus_intensities,number_of_correct,number_of_trials)
		#
		# # Constraints for the parameters
		# constraints = ( 'unconstrained', 'unconstrained', 'Uniform(0,0.1)' )
		#
		# # Determine point estimate
		# # this uses the default values, i.e. a logistic sigmoid and the ab-core
		# # resulting in a parameterization of the psychometric function of the
		# # form
		# #
		# # psi ( x ) = gamma + (1-gamma-lambda) / ( 1 + exp ( - (x-alpha)/beta ) )
		# #
		# # With a parameter vector (alpha,beta,lambda) and gamma=1/2.
		# B = BootstrapInference ( data, priors=constraints, nafc=nafc )
		#
		# # Now we perform bootstrap sampling to obtain confidence regions and goodness
		# # of fit statistics.
		# #
		# # Again the default values are used which is: 2000 bootstrap samples are generated
		# # by parametric bootstrap
		# B.sample ()
		#
		# # We generate a summary of the goodness of fit statistics
		# GoodnessOfFit ( B )
		#
		# # We plot information about the parameters and their distributions
		# ParameterPlot ( B )
		#
		# # information about thresholds and their distributions
		# ThresholdPlot ( B )
		#
		# # Now we print the confidence intervals of the 0.5-threshold before and after
		# # sensitivity analysis (where we perform the sensitivity analsis implicitely
		# # by calling plotSensitivity)
		# print "CI_0 =",B.getCI(1)
		# fig = figure()
		# ax = fig.add_axes ( [.1,.1,.8,.8] )
		# plotSensitivity ( B, ax )
		# print "CI_1 =",B.getCI(1)
		#
		# # Show all figures
		# show()
		
	def rt_distributions(self, bins=25):
		
		# shell()
		
		quantiles = [0.5, 10, 30, 50, 70, 90, 99.5]
		
		data = self.parameters_joined
		pupil = 'ppr_proj_lp'
		
		data.rt = data.rt / 1000.0
		# if self.experiment == 1:
			# data.rt = data.rt - 1.0
		
		plot_width = self.nr_subjects * 4
		
		# plot 1 -- rt combined
		max_xlim = max(data.rt)+1
		plot_nr = 1
		fig = plt.figure(figsize=(plot_width,4))
		for i in xrange(self.nr_subjects):
			
			ax1 = fig.add_subplot(1,self.nr_subjects,plot_nr)
			data_subj = data[data.subject==self.subjects[i]]
			rt = np.array(data_subj.rt)
			myfuncs.hist_q2(rt, bins=bins, corrects=self.correct[self.subj_idx==i], quantiles=quantiles, ax=ax1, xlim=(-max_xlim,max_xlim))
			ax1.set_xlabel('rt (s)')
			ax1.set_ylabel('# trials')
			plot_nr += 1
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt.pdf'))
		
		if self.experiment == 2:
			# plot 2 -- rt split by difficulty:
			max_xlim = max(data.rt)+1
			plot_nr = 1
			fig = plt.figure(figsize=(plot_width,8))
			for i in xrange(self.nr_subjects):
			
				# shell()
			
				data_subj = data[data.subject==self.subjects[i]]
				max_ylim = max(max(np.histogram(np.array(data_subj.rt[data_subj['difficulty']==1]), bins=bins)[0]), max(np.histogram(np.array(data_subj.rt[data_subj['difficulty']==1]), bins=bins)[0]))

				ax1 = plt.subplot(2,self.nr_subjects,plot_nr)
				rt = np.array(data_subj.rt[data_subj['difficulty']==1])
				corrects = self.correct[self.subj_idx==i][np.array(data_subj['difficulty']==1)]
				myfuncs.hist_q2(rt, bins=bins, corrects=corrects, quantiles=quantiles, ax=ax1, ylim=(0,max_ylim), xlim=(-max_xlim,max_xlim))
				plt.title('{}; hard {}% correct; {} trials'.format(self.subjects[i], round(sum(corrects)/float(len(corrects)), 2)*100, len(corrects)))
			
			
				ax2 = plt.subplot(2,self.nr_subjects,plot_nr+self.nr_subjects)
				rt = np.array(data_subj.rt[data_subj['difficulty']==0])
				corrects = self.correct[self.subj_idx==i][np.array(data_subj['difficulty']==0)]
				myfuncs.hist_q2(rt, bins=bins, corrects=corrects, quantiles=quantiles, ax=ax2, ylim=(0,max_ylim), xlim=(-max_xlim,max_xlim))
				plt.title('{}; easy {}% correct; {} trials'.format(self.subjects[i], round(sum(corrects)/float(len(corrects)), 2)*100, len(corrects)))
			
				# ax1.set_xlabel('rt')
				ax2.set_xlabel('rt (s)')
				ax1.set_ylabel('# trials')
				ax2.set_ylabel('# trials')
				plot_nr += 1
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_split_difficulty.pdf'))
		
		# shell()

		# # plot 2 -- rt split by pupil
		# plot_nr = 1
		# fig = plt.figure(figsize=(plot_width,8))
		# for i in xrange(self.nr_subjects):
		#
		# 	data_subj = data[data.subject==self.subjects[i]]
		# 	max_ylim = max(max(np.histogram(np.array(data_subj.rt[data_subj[pupil] < np.median(data_subj[pupil])]), bins=bins)[0]), max(np.histogram(np.array(data_subj.rt[data_subj[pupil] > np.median(data_subj[pupil])]), bins=bins)[0]))
		#
		# 	ax1 = plt.subplot(2,self.nr_subjects,plot_nr)
		# 	rt = np.array(data_subj.rt[data_subj[pupil] < np.median(data_subj[pupil])])
		# 	myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax1, quantiles_color='k', alpha=0.75, xlim=(0,4), ylim=(0,max_ylim))
		# 	ax2 = plt.subplot(2,self.nr_subjects,plot_nr+self.nr_subjects)
		# 	rt = np.array(data_subj.rt[data_subj[pupil] > np.median(data_subj[pupil])])
		# 	myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax2, quantiles_color='k', xlim=(0,4), ylim=(0,max_ylim))
		# 	ax1.set_xlabel('rt')
		# 	ax2.set_xlabel('rt')
		# 	plot_nr += 1
		# plt.tight_layout()
		# fig.savefig(os.path.join(self.project_directory, 'figures', 'rt_split_' + pupil + '.pdf'))
		
		
		
		# # plot 3 -- rt split by SDT trial types
		# plot_nr = 1
		# fig = plt.figure(figsize=(plot_width,16))
		# for i in xrange(self.nr_subjects):
		#
		# 	data_subj = data[data.subject==self.subjects[i]]
		# 	max_ylim = max(max(np.histogram(np.array(data_subj.rt[data_subj.hit]), bins=bins)[0]), max(np.histogram(np.array(data_subj.rt[data_subj.cr]), bins=bins)[0]))
		#
		# 	ax1 = plt.subplot(4, self.nr_subjects,plot_nr)
		# 	rt = np.array(data_subj.rt[data_subj.hit])
		# 	myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax1, quantiles_color='r', xlim=(0,4), ylim=(0,max_ylim))
		# 	ax2 = plt.subplot(4, self.nr_subjects,plot_nr+self.nr_subjects)
		# 	rt = np.array(data_subj.rt[data_subj.fa])
		# 	myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax2, quantiles_color='r', alpha=0.5, xlim=(0,4), ylim=(0,max_ylim))
		# 	ax3 = plt.subplot(4, self.nr_subjects,plot_nr+(2*self.nr_subjects))
		# 	rt = np.array(data_subj.rt[data_subj.miss])
		# 	myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax3, quantiles_color='b', alpha=0.5, xlim=(0,4), ylim=(0,max_ylim))
		# 	ax4 = plt.subplot(4, self.nr_subjects,plot_nr+(3*self.nr_subjects))
		# 	rt = np.array(data_subj.rt[data_subj.cr])
		# 	myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax4, quantiles_color='b', xlim=(0,4), ylim=(0,max_ylim))
		# 	ax1.set_xlabel('rt')
		# 	ax2.set_xlabel('rt')
		# 	ax3.set_xlabel('rt')
		# 	ax4.set_xlabel('rt')
		# 	plot_nr += 1
		# plt.tight_layout()
		#
		# fig.savefig(os.path.join(self.project_directory, 'figures', 'rt_split_answer.pdf'))
	
	def correlation_PPRa_BPD(self):
		
		correlations = np.zeros((2,len(self.subjects)))
		
		for d in range(2):
			
			fig = plt.figure(figsize=(15,12))
			for i in range(len(self.subjects)):
		
				varX = self.bpd[(self.subj_idx == i)*(self.drug==d)]
				varY = self.ppr[(self.subj_idx == i)*(self.drug==d)]
			
				slope, intercept, r_value, p_value, std_err = stats.linregress(varX,varY)
				(m,b) = sp.polyfit(varX, varY, 1)
				regression_line = sp.polyval([m,b], varX)
				
				correlations[d,i] = r_value
				
				ax = fig.add_subplot(4,5,i+1)
				ax.plot(varX, varY, 'o', color='k', marker='o', markeredgecolor='w', markeredgewidth=0.5, rasterized=True)
				ax.plot(varX,regression_line, color = 'r', linewidth = 1.5)
				ax.set_title('subj.' + str(i+1) + ' (r = ' + str(round(r_value, 3)) + ')', size = 12)
				ax.set_ylabel('phasic response (% signal change)', size = 10)
				ax.set_xlabel('baseline (% signal change)', size = 10)
				plt.tick_params(axis='both', which='major', labelsize=10)
				# if round(p_value,5) < 0.005:
				#	 ax.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
				# else:	
				#	 ax.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
				
				sns.despine(offset=10, trim=True)
				plt.tight_layout()
				# plt.gca().spines["bottom"].set_linewidth(.5)
				# plt.gca().spines["left"].set_linewidth(.5)
		
			fig.savefig(os.path.join(self.project_directory, 'figures', 'correlation_bpd_ppr_{}.pdf'.format(d)))
		# shell()
		
	def pupil_bars(self):
		
		shell()
		
		
		pupil_b_drug = np.zeros(len(self.subjects))
		pupil_b_placebo = np.zeros(len(self.subjects))
		pupil_drug = np.zeros(len(self.subjects))
		pupil_placebo = np.zeros(len(self.subjects))
		pupil_fb_drug = np.zeros(len(self.subjects))
		pupil_fb_placebo = np.zeros(len(self.subjects))
		pupil_trial_drug = np.zeros(len(self.subjects))
		pupil_trial_placebo = np.zeros(len(self.subjects))
		for i, s in enumerate(self.subjects):
			pupil_b_drug[i] = np.mean(self.bpd[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_b_placebo[i] = np.mean(self.bpd[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_drug[i] = np.mean(self.ppr[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_placebo[i] = np.mean(self.ppr[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_fb_drug[i] = np.mean(self.ppr_fb[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_fb_placebo[i] = np.mean(self.ppr_fb[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_trial_drug[i] = np.mean(self.ppr_trial[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_trial_placebo[i] = np.mean(self.ppr_trial[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
		print sp.stats.ttest_rel(pupil_b_drug, pupil_b_placebo)
		print sp.stats.ttest_rel(pupil_drug, pupil_placebo)
		print sp.stats.ttest_rel(pupil_fb_drug, pupil_fb_placebo)	
		print sp.stats.ttest_rel(pupil_trial_drug, pupil_trial_placebo)	
		
		d = {
		'subj_idx' : pd.Series(self.subjects),
		'pupil_b_drug' : pd.Series(pupil_b_drug),
		'pupil_b_placebo' : pd.Series(pupil_b_placebo),
		'pupil_decision_drug' : pd.Series(pupil_drug),
		'pupil_decision_placebo' : pd.Series(pupil_placebo),
		'pupil_feedback_drug' : pd.Series(pupil_fb_drug),
		'pupil_feedback_placebo' : pd.Series(pupil_fb_placebo),
		'pupil_trial_drug' : pd.Series(pupil_trial_drug),
		'pupil_trial_placebo' : pd.Series(pupil_trial_placebo),
		}
		data_accuracy = pd.DataFrame(d)
		data_accuracy.to_csv(os.path.join(self.project_directory, 'pupil_data.csv'))
		
		
		
		
		# d, c = self.behavior.sdt_measures()
		#
		#
		# MEANS = (d.mean(), c.mean())
		# SEMS = (sp.stats.sem(d), sp.stats.sem(c))
		# N = 2
		# ind = np.linspace(0,N/2,N)
		# bar_width = 0.50
		# fig = plt.figure(figsize=(2,3))
		# ax = fig.add_subplot(111)
		# for i in range(N):
		# 	ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = 'k', alpha = [1,0.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		# ax.set_title('N={}'.format(self.nr_subjects), size=8)
		# ax.set_xticks( (ind) )
		# ax.set_xticklabels( ("d'", 'c') )
		# plt.gca().spines["bottom"].set_linewidth(.5)
		# plt.gca().spines["left"].set_linewidth(.5)
		# sns.despine(offset=10, trim=True)
		# plt.tight_layout()
		# fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures.pdf'))
		#
		#
		#
		#
		# for p in ['pupil', 'pupil_b']:
		# 	for t in [0,1]:
		#
		# 		correct_mean, error_mean, no_mean, yes_mean, cr_mean, miss_mean, hit_mean, fa_mean = self.behavior.pupil_bars(pupil=p, split_by='drug', split_target=t)
		#
		# 		my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
		#
		# 		N = 4
		# 		ind = np.linspace(0,2,N)  # the x locations for the groups
		# 		bar_width = 0.6   # the width of the bars
		# 		spacing = [0, 0, 0, 0]
		#
		# 		# shell()
		#
		# 		# FIGURE 1
		# 		# shell()
		# 		# p_values = np.array([myfuncs.permutationTest(ppr_hit, ppr_miss)[1], myfuncs.permutationTest(ppr_fa, ppr_cr)[1]])
		# 		p_values = np.array([sp.stats.ttest_rel(hit_mean, miss_mean)[1], sp.stats.ttest_rel(fa_mean, cr_mean)[1]])
		# 		ppr = [hit_mean, miss_mean, fa_mean, cr_mean]
		# 		MEANS = np.array([np.mean(values) for values in ppr])
		# 		SEMS = np.array([sp.stats.sem(values) for values in ppr])
		# 		fig = plt.figure(figsize=(4,3))
		# 		ax = fig.add_subplot(111)
		# 		for i in range(N):
		# 			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','r','b'][i], alpha=[1,0.5,0.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		# 		simpleaxis(ax)
		# 		spine_shift(ax)
		# 		ax.set_xticklabels( ('H','M','FA','CR') )
		# 		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		# 		ax.tick_params(axis='x', which='major', labelsize=10)
		# 		ax.tick_params(axis='y', which='major', labelsize=10)
		# 		plt.gca().spines["bottom"].set_linewidth(.5)
		# 		plt.gca().spines["left"].set_linewidth(.5)
		# 		plt.ylabel('pupil response (a.u.)')
		# 		plt.text(x=np.mean((ind[0],ind[1])), y=0.1, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
		# 		plt.text(x=np.mean((ind[2],ind[3])), y=0.1, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
		# 		plt.tight_layout()
		# 		fig.savefig(os.path.join(self.project_directory, 'figures', 'STD1_{}_{}.pdf'.format(p, t)))
		#
		# 		# FIGURE 2
		# 		# p_values = np.array([myfuncs.permutationTest(ppr_yes, ppr_no)[1], myfuncs.permutationTest(ppr_correct, ppr_error)[1]])
		# 		p_values = np.array([sp.stats.ttest_rel(yes_mean, no_mean)[1], sp.stats.ttest_rel(correct_mean, error_mean)[1]])
		# 		ppr = [yes_mean, no_mean, correct_mean, error_mean]
		# 		MEANS = np.array([np.mean(values) for values in ppr])
		# 		SEMS = np.array([sp.stats.sem(values) for values in ppr])
		# 		fig = plt.figure(figsize=(4,3))
		# 		ax = fig.add_subplot(111)
		# 		for i in range(N):
		# 			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','k','k'][i], alpha=[1,1,1,0.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		# 		simpleaxis(ax)
		# 		spine_shift(ax)
		# 		ax.set_xticklabels( ('YES','NO','CORRECT','ERROR') )
		# 		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		# 		ax.tick_params(axis='x', which='major', labelsize=10)
		# 		ax.tick_params(axis='y', which='major', labelsize=10)
		# 		plt.gca().spines["bottom"].set_linewidth(.5)
		# 		plt.gca().spines["left"].set_linewidth(.5)
		# 		plt.ylabel('pupil response (a.u.)')
		# 		plt.text(x=np.mean((ind[0],ind[1])), y=0.1, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
		# 		plt.text(x=np.mean((ind[2],ind[3])), y=0.1, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
		# 		plt.tight_layout()
		# 		fig.savefig(os.path.join(self.project_directory, 'figures', 'STD2_{}_{}.pdf'.format(p, t)))
		
	def quantile_prob_plot(self):
		
		# self.rt = np.concatenate([(self.rt[np.array(self.subj_idx == subj_idx)] - np.mean(self.rt[np.array(self.subj_idx == subj_idx)])) / np.std(self.rt[np.array(self.subj_idx == subj_idx)]) for subj_idx in range(len(self.subjects))])
		self.rt = self.rt / 1000.0
		
		fig = plt.figure(figsize=(2,2))
		ax = fig.add_subplot(111)
		myfuncs.quantile_plot(conditions=self.difficulty[-self.drug], rt=self.rt[-self.drug], corrects=self.correct[-self.drug], subj_idx=self.subj_idx[-self.drug], ax=ax, fmt='o', color='b')
		myfuncs.quantile_plot(conditions=self.difficulty[self.drug], rt=self.rt[self.drug], corrects=self.correct[self.drug], subj_idx=self.subj_idx[self.drug], ax=ax, fmt='^', color='r')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'quantile_plot.pdf'))
		
		
		# now after splitting by:
		
		# shell()
		
		pupil_b_drug = np.zeros(len(self.subjects))
		pupil_b_placebo = np.zeros(len(self.subjects))
		pupil_drug = np.zeros(len(self.subjects))
		pupil_placebo = np.zeros(len(self.subjects))
		pupil_fb_drug = np.zeros(len(self.subjects))
		pupil_fb_placebo = np.zeros(len(self.subjects))
		pupil_trial_drug = np.zeros(len(self.subjects))
		pupil_trial_placebo = np.zeros(len(self.subjects))
		for i, s in enumerate(self.subjects):
			pupil_b_drug[i] = np.mean(self.bpd[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_b_placebo[i] = np.mean(self.bpd[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_drug[i] = np.mean(self.ppr[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_placebo[i] = np.mean(self.ppr[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_fb_drug[i] = np.mean(self.ppr_fb[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_fb_placebo[i] = np.mean(self.ppr_fb[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
			
			pupil_trial_drug[i] = np.mean(self.ppr_trial[(np.array(self.parameters_joined['subject'])==s)*np.array(self.drug,dtype=bool)])
			pupil_trial_placebo[i] = np.mean(self.ppr_trial[(np.array(self.parameters_joined['subject'])==s)*-np.array(self.drug,dtype=bool)])
		
		
		shell()
		
		# SPLIT 1
		subj = np.unique(self.subj_idx)
		group_ind1 = subj[self.split_1]
		ind = np.zeros(self.drug.shape, dtype=bool)
		for s in subj:
			if s in group_ind1:
				ind[self.subj_idx==s] = True
		
		fig = plt.figure(figsize=(2,2))
		ax = fig.add_subplot(111)
		myfuncs.quantile_plot(conditions=self.difficulty[-self.drug & ind], rt=self.rt[-self.drug & ind], corrects=self.correct[-self.drug & ind], subj_idx=self.subj_idx[-self.drug & ind], ax=ax, fmt='o', color='b')
		myfuncs.quantile_plot(conditions=self.difficulty[self.drug & ind], rt=self.rt[self.drug & ind], corrects=self.correct[self.drug & ind], subj_idx=self.subj_idx[self.drug & ind], ax=ax, fmt='^', color='r')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'quantile_plot_split_0.pdf'))
		
		fig = plt.figure(figsize=(2,2))
		ax = fig.add_subplot(111)
		myfuncs.quantile_plot(conditions=self.difficulty[-self.drug & -ind], rt=self.rt[-self.drug & -ind], corrects=self.correct[-self.drug & -ind], subj_idx=self.subj_idx[-self.drug & -ind], ax=ax, fmt='o', color='b')
		myfuncs.quantile_plot(conditions=self.difficulty[self.drug & -ind], rt=self.rt[self.drug & -ind], corrects=self.correct[self.drug & -ind], subj_idx=self.subj_idx[self.drug & -ind], ax=ax, fmt='^', color='r')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'quantile_plot_split_1.pdf'))

		# SPLIT 2
		subj = np.unique(self.subj_idx)
		group_ind1 = subj[self.split_2]
		ind = np.zeros(self.drug.shape, dtype=bool)
		for s in subj:
			if s in group_ind1:
				ind[self.subj_idx==s] = True
		
		fig = plt.figure(figsize=(2,2))
		ax = fig.add_subplot(111)
		myfuncs.quantile_plot(conditions=self.difficulty[-self.drug & ind], rt=self.rt[-self.drug & ind], corrects=self.correct[-self.drug & ind], subj_idx=self.subj_idx[-self.drug & ind], ax=ax, fmt='o', color='b')
		myfuncs.quantile_plot(conditions=self.difficulty[self.drug & ind], rt=self.rt[self.drug & ind], corrects=self.correct[self.drug & ind], subj_idx=self.subj_idx[self.drug & ind], ax=ax, fmt='^', color='r')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'quantile_plot_split_2.pdf'))
		
		fig = plt.figure(figsize=(2,2))
		ax = fig.add_subplot(111)
		myfuncs.quantile_plot(conditions=self.difficulty[-self.drug & -ind], rt=self.rt[-self.drug & -ind], corrects=self.correct[-self.drug & -ind], subj_idx=self.subj_idx[-self.drug & -ind], ax=ax, fmt='o', color='b')
		myfuncs.quantile_plot(conditions=self.difficulty[self.drug & -ind], rt=self.rt[self.drug & -ind], corrects=self.correct[self.drug & -ind], subj_idx=self.subj_idx[self.drug & -ind], ax=ax, fmt='^', color='r')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'quantile_plot_split_3.pdf'))
		
		# SPLIT 3
		subj = np.unique(self.subj_idx)
		group_ind1 = subj[self.split_3]
		ind = np.zeros(self.drug.shape, dtype=bool)
		for s in subj:
			if s in group_ind1:
				ind[self.subj_idx==s] = True
		
		fig = plt.figure(figsize=(2,2))
		ax = fig.add_subplot(111)
		myfuncs.quantile_plot(conditions=self.difficulty[-self.drug & ind], rt=self.rt[-self.drug & ind], corrects=self.correct[-self.drug & ind], subj_idx=self.subj_idx[-self.drug & ind], ax=ax, fmt='o', color='b')
		myfuncs.quantile_plot(conditions=self.difficulty[self.drug & ind], rt=self.rt[self.drug & ind], corrects=self.correct[self.drug & ind], subj_idx=self.subj_idx[self.drug & ind], ax=ax, fmt='^', color='r')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'quantile_plot_split_4.pdf'))
		
		fig = plt.figure(figsize=(2,2))
		ax = fig.add_subplot(111)
		myfuncs.quantile_plot(conditions=self.difficulty[-self.drug & -ind], rt=self.rt[-self.drug & -ind], corrects=self.correct[-self.drug & -ind], subj_idx=self.subj_idx[-self.drug & -ind], ax=ax, fmt='o', color='b')
		myfuncs.quantile_plot(conditions=self.difficulty[self.drug & -ind], rt=self.rt[self.drug & -ind], corrects=self.correct[self.drug & -ind], subj_idx=self.subj_idx[self.drug & -ind], ax=ax, fmt='^', color='r')
		fig.savefig(os.path.join(self.project_directory, 'figures', 'quantile_plot_split_5.pdf'))
		
		# shell()
		
	
	def pupil_criterion(self):
		
		ppr_yes = np.array([np.mean(self.ppr[(self.subj_idx == i) & self.yes]) for i in range(self.nr_subjects)])
		ppr_no = np.array([np.mean(self.ppr[(self.subj_idx == i) & self.no]) for i in range(self.nr_subjects)])
		
		fig = plt.figure(figsize=(2,2))
		ax = fig.add_subplot(111)
		myfuncs.correlation_plot(self.criterion, ppr_yes-ppr_no, ax=ax, line=True)
		ax.yaxis.set_major_locator(MaxNLocator(5))
		ax.xaxis.set_major_locator(MaxNLocator(5))
		plt.xlabel('criterion')
		plt.ylabel('choice effect (right-left)')
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'criterion.pdf'))
		
		# per drug and placebo:
		for d in range(2):
			
			ppr_yes = np.array([np.mean(self.ppr[(self.subj_idx == i) & self.yes & (self.drug==d)]) for i in range(self.nr_subjects)])
			ppr_no = np.array([np.mean(self.ppr[(self.subj_idx == i) & self.no & (self.drug==d)]) for i in range(self.nr_subjects)])		
			
			fig = plt.figure(figsize=(2,2))
			ax = fig.add_subplot(111)
			myfuncs.correlation_plot(self.criterion, ppr_yes-ppr_no, ax=ax, line=True)
			ax.yaxis.set_major_locator(MaxNLocator(5))
			ax.xaxis.set_major_locator(MaxNLocator(5))
			plt.xlabel('criterion')
			plt.ylabel('choice effect (right-left)')
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'criterion_{}.pdf'.format(d)))
		
		
	def pupil_prediction_error(self):
		
		self.confidence = np.array(self.parameters_joined['confidence'])
		self.ppr_feed = np.array(self.parameters_joined['ppr_peak_feed_lp'])
		
		ppr_correct_0 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.hit+self.cr)*(self.confidence==0))[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_error_0 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.miss+self.fa)*(self.confidence==0))[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_correct_1 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.hit+self.cr)*(self.confidence==1))[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_error_1 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.miss+self.fa)*(self.confidence==1))[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_correct_2 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.hit+self.cr)*(self.confidence==2))[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_error_2 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.miss+self.fa)*(self.confidence==2))[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_correct_3 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.hit+self.cr)*(self.confidence==3))[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		ppr_error_3 = np.array([np.mean(self.ppr_feed[self.subj_idx == i][((self.miss+self.fa)*(self.confidence==3))[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
		
		my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

		N = 4
		ind = np.linspace(0,2,N)  # the x locations for the groups
		bar_width = 0.6   # the width of the bars
		spacing = [0, 0, 0, 0]
		
		# FIGURE 1
		ppr = [ppr_error_0-ppr_correct_0, ppr_error_1-ppr_correct_1, ppr_error_2-ppr_correct_2, ppr_error_3-ppr_correct_3]
		MEANS = np.array([np.mean(values) for values in ppr])
		SEMS = np.array([sp.stats.sem(values) for values in ppr])
		fig = plt.figure(figsize=(4,3))
		ax = fig.add_subplot(111)
		ax.errorbar(ind, MEANS, yerr=SEMS, color = 'k', capsize = 0)
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('--','-','+','++') )
		ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
		ax.tick_params(axis='x', which='major', labelsize=10)
		ax.tick_params(axis='y', which='major', labelsize=10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		plt.title('phasic pupil responses')
		plt.ylabel('pupil response (z)')
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'prediction_error_ppr.pdf'))
		
		# FIGURE 2
		ppr = np.concatenate([ppr_error_0-ppr_correct_0, ppr_error_1-ppr_correct_1, ppr_error_2-ppr_correct_2, ppr_error_3-ppr_correct_3])
		conf = np.concatenate((np.ones(self.nr_subjects), np.ones(self.nr_subjects)*2, np.ones(self.nr_subjects)*3, np.ones(self.nr_subjects)*4))
		fig = myfuncs.correlation_plot2(conf, ppr)
		plt.xlim(0,5)
		plt.ylim(ymax=0.75)
		plt.title('phasic pupil responses')
		plt.ylabel('pupil response (z)')
		plt.xticks( (1,2,3,4), ('--','-','+','++') )
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'prediction_error2_ppr.pdf'))
		
		
		
		
		
		
		
		
		
		
		
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
		
	
	def mean_timelocked(self):
	
		interval = 3
	
		cue_A = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_drug_{}.npy'.format(s)))[0,:] for s in self.subjects])
		cue_B = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_drug_{}.npy'.format(s)))[1,:] for s in self.subjects])
		choice_A = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_drug_{}.npy'.format(s)))[0,:] for s in self.subjects])
		choice_B = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_drug_{}.npy'.format(s)))[1,:] for s in self.subjects])
		
		# shell()
		
		kernel_cue_A = np.mean(cue_A, axis=0)
		kernel_cue_B = np.mean(cue_B, axis=0)
		kernel_cue_diff = kernel_cue_B - kernel_cue_A
		kernel_choice_A = np.mean(choice_A, axis=0)
		kernel_choice_B = np.mean(choice_B, axis=0)
		kernel_choice_diff = kernel_choice_B - kernel_choice_A
	
		kernel_cue_A_sem = sp.stats.sem(cue_A, axis=0)
		kernel_cue_B_sem = sp.stats.sem(cue_B, axis=0)
		# kernel_cue_A_sem = sp.stats.sem(cue_A - ((cue_A+cue_B)/2), axis=0)
		# kernel_cue_B_sem = sp.stats.sem(cue_B - ((cue_A+cue_B)/2), axis=0)
		kernel_cue_diff_sem = sp.stats.sem(cue_B - cue_A, axis=0)
		kernel_choice_A_sem = sp.stats.sem(choice_A, axis=0)
		kernel_choice_B_sem = sp.stats.sem(choice_B, axis=0)
		# kernel_choice_A_sem = sp.stats.sem(choice_A - ((choice_A+choice_B)/2), axis=0)
		# kernel_choice_B_sem = sp.stats.sem(choice_B - ((choice_A+choice_B)/2), axis=0)
		kernel_choice_diff_sem = sp.stats.sem(choice_B - choice_A, axis=0)
		
		p = np.zeros(kernel_cue_A.shape[0])
		for i in range(kernel_cue_A.shape[0]):
			p[i] = sp.stats.ttest_rel(cue_A[:,i], cue_B[:,i])[1]
		# p = mne.stats.fdr_correction(p, alpha=0.05)[1]
		p[-1] = 1
		sig_indices = np.array(p < 0.05, dtype=int)
		sig_indices[0] = 0
		sig_indices[-1] = 0
		s_bar_cue = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0]+1)
		
		p = np.zeros(kernel_cue_A.shape[0])
		for i in range(kernel_cue_A.shape[0]):
			p[i] = sp.stats.ttest_1samp(cue_B[:,i] - cue_A[:,i],0)[1]
		# p = mne.stats.fdr_correction(p, alpha=0.05)[1]
		p[-1] = 1
		sig_indices = np.array(p < 0.05, dtype=int)
		sig_indices[0] = 0
		sig_indices[-1] = 0
		s_bar_cue_diff = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0]+1)
		
		p = np.zeros(kernel_cue_A.shape[0])
		for i in range(kernel_cue_A.shape[0]):
			p[i] = sp.stats.ttest_rel(choice_A[:,i], choice_B[:,i])[1]
		# p = mne.stats.fdr_correction(p, alpha=0.05)[1]
		p[-1] = 1
		sig_indices = np.array(p < 0.05, dtype=int)
		sig_indices[0] = 0
		sig_indices[-1] = 0
		s_bar_choice = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0]+1)
		
		p = np.zeros(kernel_cue_A.shape[0])
		for i in range(kernel_cue_A.shape[0]):
			p[i] = sp.stats.ttest_1samp(choice_B[:,i] - choice_A[:,i],0)[1]
		# p = mne.stats.fdr_correction(p, alpha=0.05)[1]
		p[-1] = 1
		sig_indices = np.array(p < 0.05, dtype=int)
		sig_indices[0] = 0
		sig_indices[-1] = 0
		s_bar_choice_diff = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0]+1)
		
		
		# ----- all subjects:
	
		fig = plt.figure(figsize=(10,5))
	
		ax = fig.add_subplot(231)
		for response in cue_A:
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0]), response, 'b', alpha=0.25)
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
		# ax.set_xlim(-0.25, 3)
		# ax.set_ylim(-0.4, 0.3)
		ax.set_xlabel('time from stimulus onset (s)')
		ax.set_ylabel('pupil size (% signal change)')
		ax.set_title('placebo')
		ax = fig.add_subplot(232)
		for response in cue_B:
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0]), response, 'r', alpha=0.25)
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
		plt.axhline(0, lw=0.5)
		# ax.set_xlim(-0.25, 3)
		# ax.set_ylim(-0.4, 0.3)
		ax.set_xlabel('time from stimulus onset (s)')
		ax.set_ylabel('pupil size (% signal change)')
		ax.set_title('drug')
		ax = fig.add_subplot(233)
		for response_A, response_B in zip(cue_A, cue_B):
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0]), response_B-response_A, 'k', alpha=0.25)
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
		plt.axhline(0, lw=0.5)
		# ax.set_xlim(-0.25, 3)
		# ax.set_ylim(-0.4, 0.3)
		ax.set_xlabel('time from stimulus onset (s)')
		ax.set_ylabel('pupil size (% signal change)')
		ax.set_title('difference')
		ax = fig.add_subplot(234)
		for response in choice_A:
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), response, 'b', alpha=0.25)
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(-np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
		# ax.set_xlim(-1.25, 2)
		# ax.set_ylim(-0.5, 1)
		ax.set_xlabel('time from choice (s)')
		ax.set_ylabel('pupil size (% signal change)')
		ax.set_title('placebo')
		ax = fig.add_subplot(235)
		for response in choice_B:
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), response, 'r', alpha=0.25)
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(-np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
		plt.axhline(0, lw=0.5)
		# ax.set_xlim(-1.25, 2)
		# ax.set_ylim(-0.5, 1)
		ax.set_xlabel('time from choice (s)')
		ax.set_ylabel('pupil size (% signal change)')
		ax.set_title('drug')
		ax = fig.add_subplot(236)
		for response_A, response_B in zip(choice_A, choice_B):
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), response_B-response_A, 'k', alpha=0.25)
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(-np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
		plt.axhline(0, lw=0.5)
		# ax.set_xlim(-1.25, 2)
		# ax.set_ylim(-0.5, 1)
		ax.set_xlabel('time from choice (s)')
		ax.set_ylabel('pupil size (% signal change)')
		ax.set_title('difference')
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_responses_across_all_subjects.pdf'))
	
	
		# -----
		fig = plt.figure(figsize=(7.5,5))
		ax = fig.add_subplot(221)
		ax.fill_between(np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0]), kernel_cue_A+kernel_cue_A_sem, kernel_cue_A-kernel_cue_A_sem, color='b', alpha=0.25)
		ax.fill_between(np.linspace(-0.5, interval-0.5, kernel_cue_B.shape[0]), kernel_cue_B+kernel_cue_B_sem, kernel_cue_B-kernel_cue_B_sem, color='r', alpha=0.25)
		ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0]), kernel_cue_A, 'b', label='placebo')
		ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_B.shape[0]), kernel_cue_B, 'r', label='atomoxetin')
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
		ax.set_xlim(-0.5, 2.5)
		# ax.set_ylim(-0.4, 0.3)
		ax.set_xlabel('time from stimulus onset (s)')
		ax.set_ylabel('pupil size (% signal change)')
		ax.legend()
		for sig in s_bar_cue:
			ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0)+ax.get_ylim()[0], np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0])[int(sig[0])], np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0])[int(sig[1])], lw=2, color='g')
		ax = fig.add_subplot(222)
		ax.fill_between(np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0]), kernel_cue_diff+kernel_cue_diff_sem, kernel_cue_diff-kernel_cue_diff_sem, color='k', alpha=0.25)
		ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0]), kernel_cue_diff, 'k', label='difference wave')
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
		plt.axhline(0, lw=0.5)
		ax.set_xlim(-0.5, 2.5)
		# ax.set_ylim(-0.4, 0.3)
		ax.set_xlabel('time from stimulus onset (s)')
		ax.set_ylabel('atomoxetine - placebo')
		ax.legend()
		for sig in s_bar_cue_diff:
			ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0)+ax.get_ylim()[0], np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0])[int(sig[0])], np.linspace(-0.5, interval-0.5, kernel_cue_A.shape[0])[int(sig[1])], lw=2, color='g')
		ax = fig.add_subplot(223)
		ax.fill_between(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), kernel_choice_A+kernel_choice_A_sem, kernel_choice_A-kernel_choice_A_sem, color='b', alpha=0.25)
		ax.fill_between(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), kernel_choice_B+kernel_choice_B_sem, kernel_choice_B-kernel_choice_B_sem, color='r', alpha=0.25)
		ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), kernel_choice_A, 'b', label='placebo')
		ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), kernel_choice_B, 'r', label='atomoxetine')
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(-np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
		ax.set_xlim(-1.5, 1.5)
		# ax.set_ylim(-0.5, 1)
		ax.set_xlabel('time from choice (s)')
		ax.set_ylabel('pupil size (% signal change)')
		ax.legend()
		for sig in s_bar_choice:
			ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0)+ax.get_ylim()[0], np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0])[int(sig[0])], np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0])[int(sig[1])], lw=2, color='g')
		ax = fig.add_subplot(224)
		ax.fill_between(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), kernel_choice_diff+kernel_choice_diff_sem, kernel_choice_diff-kernel_choice_diff_sem, color='k', alpha=0.25)
		ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0]), kernel_choice_diff, 'k', label='difference wave')
		plt.axvline(0, color='k', ls='--', lw=0.5)
		plt.axvline(-np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
		plt.axhline(0, lw=0.5)
		ax.set_xlim(-1.5, 1.5)
		# ax.set_ylim(-0.5, 1)
		ax.set_xlabel('time from choice (s)')
		ax.set_ylabel('atomoxetin - placebo')
		ax.legend()
		for sig in s_bar_choice_diff:
			ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0)+ax.get_ylim()[0], np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0])[int(sig[0])], np.linspace(-1.5, interval-1.5, kernel_choice_A.shape[0])[int(sig[1])], lw=2, color='g')
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_responses_across_drug_effect.pdf'))
		
		
		# shell()
		
		for d in range(2):
			
			cue_correct_A = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_{}_{}.npy'.format(s,d)))[0,:] for s in self.subjects])
			cue_correct_B = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_{}_{}.npy'.format(s,d)))[1,:] for s in self.subjects])
			cue_error_A = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_{}_{}.npy'.format(s,d)))[2,:] for s in self.subjects])
			cue_error_B = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_cue_locked_avg_{}_{}.npy'.format(s,d)))[3,:] for s in self.subjects])
			choice_correct_A = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_{}_{}.npy'.format(s,d)))[0,:] for s in self.subjects])
			choice_correct_B = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_{}_{}.npy'.format(s,d)))[1,:] for s in self.subjects])
			choice_error_A = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_{}_{}.npy'.format(s,d)))[2,:] for s in self.subjects])
			choice_error_B = np.vstack([np.load(os.path.join(self.project_directory, 'across_data', 'deconv_choice_locked_avg_{}_{}.npy'.format(s,d)))[3,:] for s in self.subjects])
		
			kernel_cue_correct_A = np.mean(cue_correct_A, axis=0)
			kernel_cue_correct_B = np.mean(cue_correct_B, axis=0)
			kernel_cue_error_A = np.mean(cue_error_A, axis=0)
			kernel_cue_error_B = np.mean(cue_error_B, axis=0)
			kernel_choice_correct_A = np.mean(choice_correct_A, axis=0)
			kernel_choice_correct_B = np.mean(choice_correct_B, axis=0)
			kernel_choice_error_A = np.mean(choice_error_A, axis=0)
			kernel_choice_error_B = np.mean(choice_error_B, axis=0)
		
			kernel_cue_correct_sem_A = sp.stats.sem(cue_correct_A, axis=0)
			kernel_cue_correct_sem_B = sp.stats.sem(cue_correct_B, axis=0)
			kernel_cue_error_sem_A = sp.stats.sem(cue_error_A, axis=0)
			kernel_cue_error_sem_B = sp.stats.sem(cue_error_B, axis=0)
			kernel_choice_correct_sem_A = sp.stats.sem(choice_correct_A, axis=0)
			kernel_choice_correct_sem_B = sp.stats.sem(choice_correct_B, axis=0)
			kernel_choice_error_sem_A = sp.stats.sem(choice_error_A, axis=0)
			kernel_choice_error_sem_B = sp.stats.sem(choice_error_B, axis=0)
			
			# shell()
			
			# plot:
			# -----
			fig = plt.figure(figsize=(7.5,5))
			
			ax = fig.add_subplot(221)
			
			ax.fill_between(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A+kernel_cue_correct_sem_A, kernel_cue_correct_A-kernel_cue_correct_sem_A, color='g', alpha=0.25)
			ax.fill_between(np.linspace(-0.5, interval-0.5, kernel_cue_correct_B.shape[0]), kernel_cue_correct_B+kernel_cue_correct_sem_B, kernel_cue_correct_B-kernel_cue_correct_sem_B, color='g', alpha=0.25)
			ax.fill_between(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_A+kernel_cue_error_sem_A, kernel_cue_error_A-kernel_cue_error_sem_A, color='r', alpha=0.25)
			ax.fill_between(np.linspace(-0.5, interval-0.5, kernel_cue_error_B.shape[0]), kernel_cue_error_B+kernel_cue_error_sem_B, kernel_cue_error_B-kernel_cue_error_sem_B, color='r', alpha=0.25)
			
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A, 'g', ls='--', label='correct, placebo')
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_B, 'g', label='correct, drug')
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_A, 'r', ls='--', label='error, placebo')
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error_A.shape[0]), kernel_cue_error_B, 'r', label='error, drug')
			plt.axvline(0, color='k', ls='--', lw=0.5)
			plt.axvline(np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
			ax.set_xlim(-0.5, 2.5)
			# ax.set_ylim(-0.4, 0.3)
			ax.set_xlabel('time from stimulus onset (s)')
			ax.set_ylabel('pupil size (% signal change)')
			ax.legend()
		
			ax = fig.add_subplot(222)
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_A-kernel_cue_error_A, 'k', ls='--', label='difference wave, placebo')
			ax.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct_A.shape[0]), kernel_cue_correct_B-kernel_cue_error_B, 'k', label='difference wave, drug')
			plt.axvline(0, color='k', ls='--', lw=0.5)
			plt.axvline(np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5, alpha=0.25)
			plt.axhline(0, lw=0.5)
			ax.set_xlim(-0.5, 1.5)
			# ax.set_ylim(-0.4, 0.3)
			ax.set_xlabel('time from stimulus onset (s)')
			ax.set_ylabel('correct - error')
			ax.legend()
		
			ax = fig.add_subplot(223)
			
			ax.fill_between(np.linspace(-2, interval-2, kernel_cue_correct_A.shape[0]), kernel_choice_correct_A+kernel_choice_correct_sem_A, kernel_choice_correct_A-kernel_choice_correct_sem_A, color='g', alpha=0.25)
			ax.fill_between(np.linspace(-2, interval-2, kernel_cue_correct_B.shape[0]), kernel_choice_correct_B+kernel_choice_correct_sem_B, kernel_choice_correct_B-kernel_choice_correct_sem_B, color='g', alpha=0.25)
			ax.fill_between(np.linspace(-2, interval-2, kernel_cue_error_A.shape[0]), kernel_choice_error_A+kernel_choice_error_sem_A, kernel_choice_error_A-kernel_choice_error_sem_A, color='r', alpha=0.25)
			ax.fill_between(np.linspace(-2, interval-2, kernel_cue_error_B.shape[0]), kernel_choice_error_B+kernel_choice_error_sem_B, kernel_choice_error_B-kernel_choice_error_sem_B, color='r', alpha=0.25)
			
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_correct_A.shape[0]), kernel_choice_correct_A, 'g', ls='--', label='correct, placebo')
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_correct_A.shape[0]), kernel_choice_correct_B, 'g', label='correct, drug')
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_error_A.shape[0]), kernel_choice_error_A, 'r', ls='--', label='error, placebo')
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_error_A.shape[0]), kernel_choice_error_B, 'r', label='error, drug')
			plt.axvline(0, color='k', ls='--', lw=0.5)
			plt.axvline(-np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
			ax.set_xlim(-1.5, 1.5)
			# ax.set_ylim(-0.5, 1)
			ax.set_xlabel('time from choice (s)')
			ax.set_ylabel('pupil size (% signal change)')
			ax.legend()
		
			ax = fig.add_subplot(224)
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_error_A.shape[0]), kernel_choice_correct_A-kernel_choice_error_A, 'k', ls='--', label='difference wave, placebo')
			ax.plot(np.linspace(-1.5, interval-1.5, kernel_choice_error_A.shape[0]), kernel_choice_correct_B-kernel_choice_error_B, 'k', label='difference wave, drug')
			plt.axvline(0, color='k', ls='--', lw=0.5)
			plt.axvline(-np.mean(self.parameters_joined.rt)/1000.0, color='k', ls='--', lw=0.5 , alpha=0.25)
			plt.axhline(0, lw=0.5)
			ax.set_xlim(-1.5, 1.5)
			# ax.set_ylim(-0.5, 1)
			ax.set_xlabel('time from choice (s)')
			ax.set_ylabel('correct - error')
			ax.legend()
			
			sns.despine(offset=10, trim=True)
			plt.tight_layout()
			fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_responses_across_{}.pdf'.format(d)))
			# fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_responses_across_avg.pdf'))
		
	def drift_diffusion(self, all_trials=False):
		
		if all_trials:
			parameters = []
			for s in self.subjects:
				self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
				self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
				self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
			
				parameters.append(self.ho.read_session_data('', 'parameters_joined'))
			self.parameters_joined = pd.concat(parameters)
			omissions = np.array(self.parameters_joined.correct == -1)
			self.parameters_joined = self.parameters_joined[-omissions]
		else:
			pass
		
		self.parameters_joined['rt'] = self.parameters_joined['rt'] / 1000.0
		
		
		# drug B = 1 --> atomoxetine!
		
		# data:
		d = {
		'subj_idx' : pd.Series(self.subj_idx),
		'response' : pd.Series(np.array(self.parameters_joined['correct'], dtype=int)),
		'right' : pd.Series(np.array(self.parameters_joined['yes'], dtype=int)),
		'simon' : pd.Series(self.simon),
		'rt' : pd.Series(np.array(self.parameters_joined['rt'])),
		'difficulty' : pd.Series(np.array(self.parameters_joined['difficulty'], dtype=int)),
		'split' : pd.Series(self.drug, dtype=int),
		'pupil' : pd.Series(np.array(self.ppr)),
		'pupil_b' : pd.Series(np.array(self.bpd)),
		}
		data_accuracy = pd.DataFrame(d)
		data_accuracy.to_csv(os.path.join(self.project_directory, 'data_accuracy.csv'))
		
		split_diff = self.split_2
		subj = np.unique(self.subj_idx)
		group_ind1 = subj[split_diff]
		indd = np.zeros(self.drug.shape, dtype=int)
		for s in subj:
			if s in group_ind1:
				indd[self.subj_idx==s] = 1
		
		data_accuracy_split_0 = data_accuracy[indd==0]
		data_accuracy_split_0.to_csv(os.path.join(self.project_directory, 'data_accuracy_0.csv'))
		
		data_accuracy_split_1 = data_accuracy[indd==1]
		data_accuracy_split_1.to_csv(os.path.join(self.project_directory, 'data_accuracy_1.csv'))
		
		# d = {
		# 'subj_idx' : pd.Series(self.subj_idx),
		# 'stimulus' : pd.Series(np.array(self.parameters_joined['present'], dtype=int)),
		# 'response' : pd.Series(np.array(self.parameters_joined['yes'], dtype=int)),
		# 'rt' : pd.Series(np.array(self.parameters_joined['rt']/1000.0)),
		# 'pupil' : pd.Series(np.array(self.ppr)),
		# 'pupil_b' : pd.Series(np.array(self.bpd)),
		# }
		# data_response = pd.DataFrame(d)
		# data_response.to_csv(os.path.join(self.project_directory, 'data_response.csv'))
		
	def reward_rate(self):
		
		def autocorr_2(x, lags):
			result = np.zeros(len(lags))
			for i, l in enumerate(lags):
				result[i] = np.corrcoef(np.array([x[0:len(x)-l], x[l:len(x)]]))[0,1]
			return result
		
		rt = self.rt / 1000.0
		drug = np.array(self.drug, dtype=bool)
		
		
		# overall:
		reward_rate_drug = np.zeros(len(self.subjects))
		reward_rate_placebo = np.zeros(len(self.subjects))
		for i in range(len(self.subjects)):
			reward_rate_drug[i] = np.mean(self.correct[(self.subj_idx==i)*drug]) / np.mean(rt[(self.subj_idx==i)*drug])
			reward_rate_placebo[i] = np.mean(self.correct[(self.subj_idx==i)*-drug]) / np.mean(rt[(self.subj_idx==i)*-drug])
		print sp.stats.ttest_rel(reward_rate_drug, reward_rate_placebo)
		
		
		# ttest:
		std_placebo = np.zeros(len(self.subjects))
		std_drug = np.zeros(len(self.subjects))
		for i in range(len(self.subjects)):
			std_placebo[i] = np.std(reward_rate_placebo_moving_average[i])
			std_drug[i] = np.std(reward_rate_drug_moving_average[i])
		print sp.stats.ttest_rel(std_placebo, std_drug)
		
		# figure:
		fig = plt.figure(figsize=(6,len(self.subjects)))
		for i in range(len(self.subjects)):
			min_y = min(min(reward_rate_placebo_moving_average[i]), min(reward_rate_drug_moving_average[i]))
			max_y = max(max(reward_rate_placebo_moving_average[i]), max(reward_rate_drug_moving_average[i]))
			ax = fig.add_subplot(len(self.subjects),2,(i*2)+1)
			ax.plot(reward_rate_placebo_moving_average[i], 'b')
			ax.xaxis.set_major_locator(MultipleLocator(500))
			ax.set_ylim(min_y,max_y)
			ax = fig.add_subplot(len(self.subjects),2,(i*2)+2)
			ax.plot(reward_rate_drug_moving_average[i], 'r')
			ax.set_ylim(min_y,max_y)
			ax.xaxis.set_major_locator(MultipleLocator(500))
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'reward_rates.pdf'))
		
		
		
		shell()

		# all trials::
		window_len = 40
		reward_rate_drug_all = []
		reward_rate_placebo_all = []
		for i in range(len(self.subjects)):
			# reward_rate_drug_all.append(self.correct[(self.subj_idx==i)*drug]/rt[(self.subj_idx==i)*drug])
			# reward_rate_placebo_all.append(self.correct[(self.subj_idx==i)*-drug]/rt[(self.subj_idx==i)*-drug])
			reward_rate_drug_all.append(rt[(self.subj_idx==i)*drug])
			reward_rate_placebo_all.append(rt[(self.subj_idx==i)*-drug])
		
		lags = 80
		reward_rate_placebo_all_auto_cor = np.zeros((len(self.subjects),lags))
		reward_rate_drug_all_auto_cor = np.zeros((len(self.subjects),lags))
		for i in range(len(self.subjects)):
			reward_rate_placebo_all_auto_cor[i,:] = autocorr_2(reward_rate_placebo_all[i], lags=range(lags))
			reward_rate_drug_all_auto_cor[i,:] = autocorr_2(reward_rate_drug_all[i], lags=range(lags))
		print sp.stats.ttest_rel(reward_rate_drug_all_auto_cor[:,-1], reward_rate_placebo_all_auto_cor[:,-1])
		
		fig = plt.figure(figsize=(2,3))
		for i in range(len(self.subjects)):
			plt.plot(reward_rate_placebo_all_auto_cor[i,:], alpha=0.2, lw=0.5, color='b')
			plt.plot(reward_rate_drug_all_auto_cor[i,:], alpha=0.25, lw=0.5, color= 'r')
		plt.plot(reward_rate_placebo_all_auto_cor.mean(axis=0), lw=2, color='b')
		plt.plot(reward_rate_drug_all_auto_cor.mean(axis=0), lw=2, color= 'r')
		plt.xlabel('lags')
		plt.ylabel('autocorrelation')
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'reward_rates_1_autocorrelation.pdf'))
		
		
		
		# moving average:
		window_len = 40
		reward_rate_drug_moving_average = []
		reward_rate_placebo_moving_average = []
		for i in range(len(self.subjects)):
			
			# drug:
			total_drug_trials = sum((self.subj_idx==i)*drug)
			reward_rate_drug_m = np.zeros(total_drug_trials-window_len)
			for t in range(total_drug_trials-window_len):
				reward_rate_drug_m[t] = np.mean(rt[(self.subj_idx==i)*drug][t:t+window_len])
				# reward_rate_drug_m[t] = np.mean(self.correct[(self.subj_idx==i)*drug][t:t+window_len] / rt[(self.subj_idx==i)*drug][t:t+window_len])
			reward_rate_drug_moving_average.append(reward_rate_drug_m)
			
			# placebo:
			total_placebo_trials = sum((self.subj_idx==i)*-drug)
			reward_rate_placebo_m = np.zeros(total_placebo_trials-window_len)
			for t in range(total_placebo_trials-window_len):
				reward_rate_placebo_m[t] = np.mean(rt[(self.subj_idx==i)*-drug][t:t+window_len])
				# reward_rate_placebo_m[t] = np.mean(self.correct[(self.subj_idx==i)*-drug][t:t+window_len] / rt[(self.subj_idx==i)*-drug][t:t+window_len])
			reward_rate_placebo_moving_average.append(reward_rate_placebo_m)
			
		lags = 40
		reward_rate_placebo_moving_average_auto_cor = np.zeros((len(self.subjects),lags))
		reward_rate_drug_moving_average_auto_cor = np.zeros((len(self.subjects),lags))
		for i in range(len(self.subjects)):
			reward_rate_placebo_moving_average_auto_cor[i,:] = autocorr_2(reward_rate_placebo_moving_average[i], lags=range(lags))
			reward_rate_drug_moving_average_auto_cor[i,:] = autocorr_2(reward_rate_drug_moving_average[i], lags=range(lags))
		print sp.stats.ttest_rel(reward_rate_drug_moving_average_auto_cor[:,-1], reward_rate_placebo_moving_average_auto_cor[:,-1])
		
		fig = plt.figure(figsize=(2,3))
		for i in range(len(self.subjects)):
			plt.plot(reward_rate_placebo_moving_average_auto_cor[i,:], alpha=0.2, lw=0.5, color='b')
			plt.plot(reward_rate_drug_moving_average_auto_cor[i,:], alpha=0.25, lw=0.5, color= 'r')
		plt.plot(reward_rate_placebo_moving_average_auto_cor.mean(axis=0), lw=2, color='b')
		plt.plot(reward_rate_drug_moving_average_auto_cor.mean(axis=0), lw=2, color= 'r')
		plt.xlabel('lags')
		plt.ylabel('autocorrelation')
		sns.despine(offset=10, trim=True)
		plt.tight_layout()
		fig.savefig(os.path.join(self.project_directory, 'figures', 'reward_rates_2_autocorrelation.pdf'))

		
		# print sp.stats.ttest_rel(reward_rate_drug_moving_average_auto_cor[:,-1], reward_rate_placebo_moving_average_auto_cor[:,-1])
		
		
		
		
		# def autocorr_1(x):
		# 	result = np.correlate(x, x, mode='full')
		# 	return result[result.size/2:]
		
		

		
		
		
				
				
			
		
		
		
		
		
		
		
		
		
		pass
		

		
		
		
		
		

