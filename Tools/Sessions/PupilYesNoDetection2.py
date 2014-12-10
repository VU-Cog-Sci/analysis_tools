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
			if sum((self.blink_start_times > self.cue_times[t] - 700) * (self.blink_end_times < self.choice_times[t] + 500)) > 0:
				self.omission_indices_blinks[t] = True
		
		self.omission_indices_rt = np.zeros(self.nr_trials, dtype=bool)
		for t in range(self.nr_trials):
			if self.rt[t] < 250:
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
		
	def process_runs(self, alias):
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
		
		downsample_rate = 100
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
			self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
			pupil.append(self.pupil_bp / np.std(self.pupil_bp))
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
		
		# join over runs:
		parameters_joined = pd.concat(parameters)
		pupil = np.concatenate(pupil)
		pupil_diff = np.concatenate(pupil_diff)
		time = np.concatenate(time)
		cue_times = np.concatenate(cue_times)[-np.array(parameters_joined.omissions, dtype=bool)] / 1000.0
		choice_times = np.concatenate(choice_times)[-np.array(parameters_joined.omissions, dtype=bool)] / 1000.0
		blink_times = np.concatenate(blink_times) / 1000.0
		correct = np.array(parameters_joined.correct, dtype=bool)[-np.array(parameters_joined.omissions, dtype=bool)]
		hit = np.array(parameters_joined.hit, dtype=bool)[-np.array(parameters_joined.omissions, dtype=bool)]
		fa = np.array(parameters_joined.fa, dtype=bool)[-np.array(parameters_joined.omissions, dtype=bool)]
		miss = np.array(parameters_joined.miss, dtype=bool)[-np.array(parameters_joined.omissions, dtype=bool)]
		cr = np.array(parameters_joined.cr, dtype=bool)[-np.array(parameters_joined.omissions, dtype=bool)]
		if self.experiment == 1:
			confidence_times = np.concatenate(confidence_times)[-np.array(parameters_joined.omissions, dtype=bool)] / 1000.0
		
		# 
		
		# downsample:
		
		deconvolution = None
		if deconvolution:
		
			new_nr_samples = pupil.shape[0] / downsample_rate
			pupil = sp.signal.resample(pupil, new_nr_samples)
			pupil_diff = sp.signal.resample(pupil_diff, new_nr_samples)
			# time = time[::downsample_rate]
			

			# deconvolution:
			interval = 4
			
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
				events = [cue_times[correct]-0.5, cue_times[-correct]-0.5, blink_times]
				do = ArrayOperator.DeconvolutionOperator( inputObject=pupil, eventObject=events, TR=1.0/new_sample_rate, deconvolutionSampleDuration=1.0/new_sample_rate, deconvolutionInterval=interval, run=True )

				# output:
				output = do.deconvolvedTimeCoursesPerEventType
				kernel_cue_correct = output[0,:,0]
				kernel_cue_error = output[1,:,0]

				# plot:
				fig = plt.figure()
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_correct.shape[0]), kernel_cue_correct, 'g', label='correct')
				plt.plot(np.linspace(-0.5, interval-0.5, kernel_cue_error.shape[0]), kernel_cue_error, 'r', label='error')
				plt.xlabel('time from stimulus onset (s)')
				plt.ylabel('pupil size (Z)')
				plt.legend()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'stim_locked_{}.pdf'.format(self.subject.initials)))

				# choice locked:
				# -------------
				events = [choice_times[correct]-0.5, choice_times[-correct]-0.5, blink_times]
				do = ArrayOperator.DeconvolutionOperator( inputObject=pupil, eventObject=events, TR=1.0/new_sample_rate, deconvolutionSampleDuration=1.0/new_sample_rate, deconvolutionInterval=interval, run=True )

				# output:
				output = do.deconvolvedTimeCoursesPerEventType
				kernel_choice_correct = output[0,:,0]
				kernel_choice_error = output[1,:,0]

				fig = plt.figure()
				plt.plot(np.linspace(-2, interval-2, kernel_choice_correct.shape[0]), kernel_choice_correct, 'g', label='correct')
				plt.plot(np.linspace(-2, interval-2, kernel_choice_error.shape[0]), kernel_choice_error, 'r', label='error')
				plt.xlabel('time from choice (s)')
				plt.ylabel('pupil size (Z)')
				plt.legend()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'choice_locked_{}.pdf'.format(self.subject.initials)))
			
		# # baseline pupil measures:
		# start = 500
		# end = 1000
		# bpd_lp = myfuncs.pupil_scalar_mean(cue_locked_array_lp_joined, start, end)
		# bpd_bp = myfuncs.pupil_scalar_mean(cue_locked_array_bp_joined, start, end)
		# if self.experiment == 1:
		# 	bpd_lp_feed = myfuncs.pupil_scalar_mean(feedback_locked_array_lp_joined, start, end)
		# 	bpd_bp_feed = myfuncs.pupil_scalar_mean(feedback_locked_array_bp_joined, start, end)
		#
		# # phasic pupil responses
		# start = 3000
		# end = 5500
		# ppr_peak_lp = myfuncs.pupil_scalar_peak(choice_locked_array_lp_joined_b, start, end)
		# ppr_mean_lp = myfuncs.pupil_scalar_mean(choice_locked_array_lp_joined_b, start, end)
		# template = np.mean(choice_locked_array_lp_joined_b[-np.array(parameters_joined['omissions']),start:end], axis=0)
		# sign_template_lp = np.sign(sum(template))
		# ppr_proj_lp = myfuncs.pupil_scalar_lin_projection(choice_locked_array_lp_joined_b, start, end, template)
		# ppr_peak_bp = myfuncs.pupil_scalar_peak(choice_locked_array_bp_joined_b, start, end)
		# ppr_mean_bp = myfuncs.pupil_scalar_mean(choice_locked_array_bp_joined_b, start, end)
		# template = np.mean(choice_locked_array_bp_joined_b[-np.array(parameters_joined['omissions']),start:end], axis=0)
		# sign_template_bp = np.sign(sum(template))
		# ppr_proj_bp = myfuncs.pupil_scalar_lin_projection(choice_locked_array_bp_joined_b, start, end, template)
		# if self.experiment == 1:
		# 	start = 1000
		# 	end = 3000
		# 	ppr_peak_feed_lp = myfuncs.pupil_scalar_peak(feedback_locked_array_lp_joined_b, start, end)
		# 	ppr_mean_feed_lp = myfuncs.pupil_scalar_mean(feedback_locked_array_lp_joined_b, start, end)
		# 	template = np.mean(feedback_locked_array_lp_joined_b[-np.array(parameters_joined['omissions']),start:end], axis=0)
		# 	ppr_proj_feed_lp = myfuncs.pupil_scalar_lin_projection(feedback_locked_array_lp_joined_b, start, end, template)
		# 	ppr_peak_feed_bp = myfuncs.pupil_scalar_peak(feedback_locked_array_bp_joined_b, start, end)
		# 	ppr_mean_feed_bp = myfuncs.pupil_scalar_mean(feedback_locked_array_bp_joined_b, start, end)
		# 	template = np.mean(feedback_locked_array_bp_joined_b[-np.array(parameters_joined['omissions']),start:end], axis=0)
		# 	ppr_proj_feed_bp = myfuncs.pupil_scalar_lin_projection(feedback_locked_array_bp_joined_b, start, end, template)
		
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
		# parameters_joined['bpd_lp'] = bpd_lp
		# parameters_joined['bpd_bp'] = bpd_bp
		# parameters_joined['ppr_peak_lp'] = ppr_peak_lp
		# parameters_joined['ppr_peak_bp'] = ppr_peak_bp
		# parameters_joined['ppr_mean_lp'] = ppr_mean_lp
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
		
		# self.bpd = np.array(self.parameters_joined['bpd_lp'])
		#
		# self.ppr = np.array(self.parameters_joined['ppr_proj_lp'])
		#
		# self.criterion = np.array(self.parameters_joined['criterion'])[0]
		#
		# if self.experiment == 1:
		# 	self.ppr_feed = np.array(self.parameters_joined['ppr_peak_feed_lp'])
		# 	self.bpd_feed = np.array(self.parameters_joined['bpd_feed_lp'])
	
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
		'pupil' : pd.Series(np.array(self.parameters_joined.ppr_mean_lp)),
		# 'pupil' : pd.Series(np.array(self.ppr)),
		'pupil_b' : pd.Series(np.array(self.bpd)),
		'hit' : pd.Series(np.array(self.hit)),
		'fa' : pd.Series(np.array(self.fa)),
		'miss' : pd.Series(np.array(self.miss)),
		'cr' : pd.Series(np.array(self.cr)),
		'rt' : pd.Series(np.array(self.rt)/1000.0),
		}
		data = pd.DataFrame(d)
		data.to_csv(os.path.join(self.project_directory, 'data', self.subject.initials, 'pupil_data.csv'))
	
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
		
		self.rt = self.parameters_joined['rt']
		
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
		
		# self.bpd = np.array(self.parameters_joined['bpd_lp'])
		# self.ppr = np.array(self.parameters_joined['ppr_proj_lp'])
		
		self.subj_idx = np.concatenate(np.array([np.repeat(i, sum(self.parameters_joined['subject'] == self.subjects[i])) for i in range(len(self.subjects))]))
		
		# self.criterion = np.array([np.array(self.parameters_joined[self.parameters_joined['subject']==subj]['criterion'])[0] for subj in self.subjects])
	
	
	def behavior(self):
		
		if self.experiment == 1:
		
			MEAN_d_prime = []
			MEAN_criterion = []
			MEAN_confidence = []
			SEM_confidence = []
		
			for j in range(2):
			
				if self.experiment == 1:
					if j == 0:
						ind = np.array(self.parameters_joined['noise_redraw'] == 1)
					if j == 1:
						ind = np.array(self.parameters_joined['noise_redraw'] == 4)
				if self.experiment == 2:
					ind = np.ones(len(self.parameters_joined), dtype=bool)
			
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
				sns.set_style("white")
				fig = plt.figure(figsize=(2,3))
				ax = fig.add_subplot(111)
				for i in range(N):
					ax.bar(locs[i], height=MEANS_correct[i], width = bar_width, yerr = SEMS_correct[i], color = ['r', 'b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
				for i in range(N):
					ax.bar(locs[i], height=MEANS_error[i], bottom = MEANS_correct[i], width = bar_width, color = ['b', 'r'][i], alpha = 0.5, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
				ax.axhline(np.mean(MEANS_correct), color='g', ls='--')
			
				print np.mean(MEANS_correct)
			
				simpleaxis(ax)
				spine_shift(ax)
				ax.set_ylabel('fraction of trials', size = 10)
				ax.set_title('SDT fractions', size = 12)
				ax.set_xticks( (locs) )
				ax.set_xticklabels( ('signal+\nnoise', 'noise') )
				ax.set_ylim((0,1))
				plt.gca().spines["bottom"].set_linewidth(.5)
				plt.gca().spines["left"].set_linewidth(.5)
				plt.tight_layout()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_fractions_{}.pdf'.format(j)))
		
				# RT histograms:
				# --------------
				RESPONSE_TIME = np.array(self.rt)/1000.0
				max_y = max( max(plt.hist(RESPONSE_TIME[self.hit*ind], bins=30)[0]), max(plt.hist(RESPONSE_TIME[self.cr*ind], bins=30)[0]) )
				y1,binEdges1 = np.histogram(RESPONSE_TIME[self.hit*ind],bins=30)
				bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
				y2,binEdges2 = np.histogram(RESPONSE_TIME[self.fa*ind],bins=30)
				bincenters2 = 0.5*(binEdges2[1:]+binEdges2[:-1])
				y3,binEdges3 = np.histogram(RESPONSE_TIME[self.miss*ind],bins=30)
				bincenters3 = 0.5*(binEdges3[1:]+binEdges3[:-1])
				y4,binEdges4 = np.histogram(RESPONSE_TIME[self.cr*ind],bins=30)
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
				a.axvline(np.median(RESPONSE_TIME[self.hit]), color='r', linestyle='--')
				a.axvline(np.median(RESPONSE_TIME[self.miss]), color='b', linestyle='--')
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
				b.axvline(np.median(RESPONSE_TIME[self.fa]), color='r', linestyle='--')
				b.axvline(np.median(RESPONSE_TIME[self.cr]), color='b', linestyle='--')
				plt.gca().spines["bottom"].set_linewidth(.5)
				plt.gca().spines["left"].set_linewidth(.5)
				plt.tight_layout()
				fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_{}.pdf'.format(j)))
			
			# Confidence:
			# --------------
			
			# # SDT measures, confidence:
			# # SDT fractions:
			# # --------------
			# d_prime = np.zeros(len(self.subjects))
			# criterion = np.zeros(len(self.subjects))
			# confidence = np.zeros(len(self.subjects))
			# for i in range(len(self.subjects)):
			# 	d_prime[i] = myfuncs.SDT_measures( self.present[(self.subj_idx==i)*ind], self.hit[(self.subj_idx==i)*ind], self.fa[(self.subj_idx==i)*ind])[0]
			# 	criterion[i] = myfuncs.SDT_measures( self.present[(self.subj_idx==i)*ind], self.hit[(self.subj_idx==i)*ind], self.fa[(self.subj_idx==i)*ind])[1]
			# 	confidence[i] = np.mean(self.parameters_joined['confidence'][(self.subj_idx==i)*ind])
			#
			# MEAN_d_prime.append(np.mean(d_prime))
			# MEAN_criterion.append(np.mean(criterion))
			# MEAN_confidence.append(np.mean(confidence))
			# SEM_confidence.append(sp.stats.sem(confidence))
			#
			# MEANS_correct = (np.mean(hit_fraction), np.mean(cr_fraction))
			# SEMS_correct = (sp.stats.sem(hit_fraction), sp.stats.sem(cr_fraction))
			# MEANS_error = (np.mean(miss_fraction), np.mean(fa_fraction))
			# SEMS_error = (sp.stats.sem(miss_fraction), sp.stats.sem(fa_fraction))
			
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
		if self.experiment == 1:
			data.rt = data.rt - 1.0
		
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
		
		sns.set_style("white")
		
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
		

	def drift_diffusion(self):
		
		shell()
		
		# data:
		d = {
		'subj_idx' : pd.Series(self.subj_idx),
		'response' : pd.Series(np.array(self.parameters_joined['correct'], dtype=int)),
		'rt' : pd.Series(np.array(self.parameters_joined['rt']/1000.0)),
		'difficulty' : pd.Series(np.array(self.parameters_joined['difficulty'], dtype=int)),
		'version' : pd.Series(np.ones(len(self.subj_idx)), dtype=int),
		# 'pupil' : pd.Series(np.array(self.ppr)),
		# 'pupil_b' : pd.Series(np.array(self.bpd)),
		}
		data_accuracy = pd.DataFrame(d)
		data_accuracy.to_csv(os.path.join(self.project_directory, 'data_accuracy.csv'))
		
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
		
		
		
		
		

