#!/usr/bin/env python
# encoding: utf-8
"""
EyeLinkSession.py

Created by Tomas Knapen on 2011-04-27.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import os, sys, pickle, math
from subprocess import *

import scipy as sp
import scipy.stats as stats
import numpy as np
import matplotlib.pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

from pypsignifit import *
from tables import *

import pp
import logging, logging.handlers, logging.config

from ..log import *
from ..Run import *
from ..Subjects.Subject import *
from ..Operators.Operator import *
from ..Operators.CommandLineOperator import *
from ..Operators.ImageOperator import *
from ..Operators.BehaviorOperator import *
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from ..circularTools import *


class EyeLinkSession(object):
	def __init__(self, ID, subject, project_name, experiment_name, base_directory, wildcard, loggingLevel = logging.DEBUG):
		self.ID = ID
		self.subject = subject
		self.project_name = project_name
		self.experiment_name = experiment_name
		self.wildcard = wildcard
		try:
			os.mkdir(os.path.join( base_directory, project_name, experiment_name, self.subject.initials ))
		except OSError:
			pass
		self.base_directory = os.path.join( base_directory, project_name, experiment_name, self.subject.initials )
		
		self.create_folder_hierarchy()
		self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
		
		self.saccade_dtype = np.dtype([('peak_velocity', '<f8'), ('start_time', '<f8'), ('end_time', '<f8'), ('start_point', '<f8', (2)), ('vector', '<f8', (2)), ('end_point', '<f8', (2)), ('amplitude', '<f8'), ('duration', '<f8'), ('direction', '<f8'), ('end_timestamp', '<f8')])
		
		# add logging for this session
		# sessions create their own logging file handler
		self.loggingLevel = loggingLevel
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(self.loggingLevel)
		addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.base_directory, 'log', 'sessionLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		self.logger.info('starting analysis of session ' + str(self.ID))
	
	def create_folder_hierarchy(self):
		"""docstring for createFolderHierarchy"""
		# check for basic directory
		for p in ['raw','processed','figs','log']:
			try:
				os.mkdir(os.path.join(self.base_directory, p))
			except OSError:
				pass
	
	def import_raw_data(self, original_data_directory):
		"""docstring for import_raw_data"""
		os.chdir(original_data_directory)
		behavior_files = subprocess.Popen('ls ' + self.wildcard + '*_outputDict.pickle', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('importing files ' + str(behavior_files) + ' from ' + original_data_directory)
		eye_files = [f.split('_outputDict.pickle')[0] + '.edf' for f in behavior_files]
		
		# put output dicts and eyelink files next to one another - that's what the eyelinkoperator expects.
		for i in range(len(behavior_files)):
			ExecCommandLine('cp ' + behavior_files[i] + ' ' + os.path.join(self.base_directory, 'raw', behavior_files[i]) )
			ExecCommandLine('cp ' + eye_files[i] + ' ' + os.path.join(self.base_directory, 'raw', eye_files[i]) )
	
	def convert_edf(self, check_answers = True, compute_velocities = True):
		"""docstring for convert_edf"""
		os.chdir(os.path.join(self.base_directory, 'raw'))
		# what files are there to analyze?
		edf_files = subprocess.Popen('ls ' + self.wildcard + '*.edf', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0:-1]
		msg_files = subprocess.Popen('ls ' + self.wildcard + '*.msg', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0:-1]
		
		edf_files_no_ext = [f.split('.edf')[0] for f in edf_files]
		msg_files_no_ext = [f.split('.msg')[0] for f in msg_files]
		
		eyelink_fos = []
		for f in edf_files_no_ext:
			# check which files are already split - and get their data. 
			# if they haven't been split, split then and then import their data
			if f not in msg_files_no_ext:
				elo = EyelinkOperator( inputObject = f+'.edf', split = True )
			else:
				elo = EyelinkOperator( inputObject = f+'.edf', split = False )
			eyelink_fos.append(elo)
			
			# make sure to throw out trials that do not have answers
			elo.findAll(check_answers = check_answers)
		
		# enter the files into the hdf5 file in order
		order = np.argsort(np.array([int(elo.timeStamp.strftime("%Y%m%d%H%M%S")) for elo in eyelink_fos]))
		for i in order:
			eyelink_fos[i].processIntoTable(self.hdf5_filename, name = self.wildcard + '_run_' + str(i), compute_velocities = compute_velocities )
			eyelink_fos[i].clean_data()
	
	def import_parameters(self, run_name = 'run_'):
		parameter_data = []
		h5f = openFile(self.hdf5_filename, mode = "r" )
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if run_name in r._v_name:
				# try to take care of the problem that parameter composition of runs may change over time - we choose the common denominator for now.
				# perhaps later a more elegant solution is possible
				this_dtype = np.array(r.trial_parameters.read().dtype.names)
				if len(parameter_data) == 0:	# if the first run, we construct a dtype_array
					dtype_array = this_dtype
				else:	# common denominator by intersection
					dtype_array = np.intersect1d(dtype_array, this_dtype)
				parameter_data.append(np.array(r.trial_parameters.read()))
		parameter_data = [p[:][dtype_array] for p in parameter_data]
		self.parameter_data = np.concatenate(parameter_data)
		self.logger.info('imported parameter data from ' + str(self.parameter_data.shape[0]) + ' trials')
		h5f.close()
	
	def get_EL_samples_per_trial(self, run_index = 0, trial_ranges = [[0,-1]], trial_phase_range = [0,-1], data_type = 'smoothed_velocity'):
		h5f = openFile(self.hdf5_filename, mode = "r" )
		run = None
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if self.wildcard + '_run_' + str(run_index) == r._v_name:
				run = r
				break
		if run == None:
			self.logger.error('No run named ' + self.wildcard + '_run_' + str(run_index) + ' in this session\'s hdf5 file ' + self.hdf5_filename )
		timings = run.trial_times.read()
		gaze_timestamps = run.gaze_data.read()[:,0]
		
		# select data_type
		if data_type == 'smoothed_velocity':
			all_data_of_requested_type = run.smoothed_velocity_data.read()[:,-1]
		elif data_type == 'smoothed_velocity_x':
			all_data_of_requested_type = run.smoothed_velocity_data.read()[:,0]
		elif data_type == 'smoothed_velocity_y':
			all_data_of_requested_type = run.smoothed_velocity_data.read()[:,1]
		elif data_type == 'smoothed_velocity_xy':
			all_data_of_requested_type = run.smoothed_velocity_data.read()[:,[0,1]]
		elif data_type == 'velocity':
			all_data_of_requested_type = run.velocity_data.read()[:,-1]
		elif data_type == 'velocity_x':
			all_data_of_requested_type = run.velocity_data.read()[:,0]
		elif data_type == 'velocity_y':
			all_data_of_requested_type = run.velocity_data.read()[:,1]
		elif data_type == 'gaze_xy':
			all_data_of_requested_type = run.gaze_data.read()[:,[1,2]]
		elif data_type == 'gaze_x':
			all_data_of_requested_type = run.gaze_data.read()[:,1]
		elif data_type == 'gaze_y':
			all_data_of_requested_type = run.gaze_data.read()[:,2]
		elif data_type == 'smoothed_gaze_xy':
			all_data_of_requested_type = run.smoothed_gaze_data.read()[:,[0,1]]
		elif data_type == 'smoothed_gaze_x':
			all_data_of_requested_type = run.smoothed_gaze_data.read()[:,0]
		elif data_type == 'smoothed_gaze_y':
			all_data_of_requested_type = run.smoothed_gaze_data.read()[:,1]
		elif data_type == 'pupil_size':
			all_data_of_requested_type = run.gaze_data.read()[:,3]
		
		# run for loop for actual data
		export_data = []
		for (i, trial_range) in zip(range(len(trial_ranges)), trial_ranges):
			export_data.append([])
			for t in timings[trial_range[0]:trial_range[1]]:
				phase_timestamps = np.concatenate((np.array([t['trial_start_EL_timestamp']]), t['trial_phase_timestamps'][:,0], np.array([t['trial_end_EL_timestamp']])))
				which_samples = (gaze_timestamps >= phase_timestamps[trial_phase_range[0]]) * (gaze_timestamps <= phase_timestamps[trial_phase_range[1]])
				export_data[-1].append(np.vstack((gaze_timestamps[which_samples].T, all_data_of_requested_type[which_samples].T)).T)
		# clean-up
		h5f.close()
		return export_data
	
	def get_EL_events_per_trial(self, run_index = 0, trial_ranges = [[0,-1]], trial_phase_range = [0,-1], data_type = 'saccades'):
		h5f = openFile(self.hdf5_filename, mode = "r" )
		run = None
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if self.wildcard + '_run_' + str(run_index) == r._v_name:
				run = r
				break
		if run == None:
			self.logger.error('No run named ' + self.wildcard + '_run_' + str(run_index) + ' in this session\'s hdf5 file ' + self.hdf5_filename )
		timings = run.trial_times.read()
		
		if data_type == 'saccades':
			table = run.saccades_from_EL
		elif data_type == 'fixations':
			table = run.fixations_from_EL
		elif data_type == 'blinks':
			table = run.blinks_from_EL
		
		# run for loop for actual data
		export_data = []
		for (i, trial_range) in zip(range(len(trial_ranges)), trial_ranges):
			export_data.append([])
			for t in timings[trial_range[0]:trial_range[1]]:
				phase_timestamps = np.concatenate((np.array([t['trial_start_EL_timestamp']]), t['trial_phase_timestamps'][:,0], np.array([t['trial_end_EL_timestamp']])))
				where_statement = '(start_timestamp >= ' + str(phase_timestamps[trial_phase_range[0]]) + ') & (start_timestamp < ' + str(phase_timestamps[trial_phase_range[1]]) + ')' 
				export_data[-1].append(np.array([s[:] for s in table.where(where_statement) ], dtype = table.dtype))
		h5f.close()
		return export_data
	
	def detect_saccade_from_data(self, xy_data = None, xy_velocity_data = None, l = 5, sample_times = None, pixels_per_degree = 26.365):
		"""
		detect_saccade_from_data takes a sequence (2 x N) of xy gaze position or velocity data and uses the engbert & mergenthaler algorithm (PNAS 2006) to detect saccades.
		L determines the threshold - standard set at 5 median-based standard deviations from the median
		"""
		minimum_saccade_duration = 6 # in ms, as we assume the sampling to be
		
		if xy_velocity_data == None:
			vel_data = np.zeros(xydata.shape)
			vel_data[1:] = np.diff(xydata, axis = 0)
		else:
			vel_data = xy_velocity_data
		
		if sample_times == None:
			sample_times = np.arange(vel_data.shape[1])
			
		# median-based standard deviation
		med = np.median(vel_data, axis = 0)
		scaled_vel_data = vel_data/np.mean(np.sqrt(((vel_data - med)**2)), axis = 0)
		
		# when are we above the threshold, and when were the crossings
		over_threshold = (np.array([np.linalg.norm(s) for s in scaled_vel_data]) > l)
		# integers instead of bools preserve the sign of threshold transgression
		over_threshold_int = np.array(over_threshold, dtype = np.int16)
		
		# crossings come in pairs
		threshold_crossings_int = np.concatenate([[0], np.diff(over_threshold_int)])
		threshold_crossing_indices = np.arange(threshold_crossings_int.shape[0])[threshold_crossings_int != 0]
		
		# check for shorter saccades and gaps
		tci = []
		sacc_on = False
		for i in range(0, threshold_crossing_indices.shape[0]):
			# last transgression, is an offset of a saccade
			if i == threshold_crossing_indices.shape[0]-1:
				if threshold_crossings_int[threshold_crossing_indices[i]] == -1:
					tci.append(threshold_crossing_indices[i])
					sacc_on = False # be complete
				else: pass
			# first transgression, start of a saccade
			elif i == 0:
				if threshold_crossings_int[threshold_crossing_indices[i]] == 1:
					tci.append(threshold_crossing_indices[i])
					sacc_on = True
				else: pass
			elif threshold_crossings_int[threshold_crossing_indices[i]] == 1 and sacc_on == False: # start of a saccade that occurs without a prior saccade en route
				tci.append(threshold_crossing_indices[i])
				sacc_on = True
			# don't want to add any point that borders on a too-short interval
			elif (threshold_crossing_indices[i+1] - threshold_crossing_indices[i] <= minimum_saccade_duration):
				if threshold_crossings_int[threshold_crossing_indices[i]] == -1: # offset but the next is too short - disregard offset
					pass
				elif threshold_crossings_int[threshold_crossing_indices[i]] == 1: # onset but the next is too short - disregard offset if there is already a previous saccade going on
					if sacc_on: # there already is a saccade going on - no need to include this afterbirth
						pass
					else:	# this should have been caught earlier
						tci.append(threshold_crossing_indices[i])
						sacc_on = True
			elif (threshold_crossing_indices[i] - threshold_crossing_indices[i-1] <= minimum_saccade_duration):
				if threshold_crossings_int[threshold_crossing_indices[i]] == -1: # offset but the previous one is too short - use offset offset
					if sacc_on:
						tci.append(threshold_crossing_indices[i])
						sacc_on = False
			# but add anything else
			else:
				tci.append(threshold_crossing_indices[i])
				if threshold_crossings_int[threshold_crossing_indices[i]] == 1:
					sacc_on = True
				else:
					sacc_on = False
		
		threshold_crossing_indices = np.array(tci)
		
		if threshold_crossing_indices.shape[0] > 0:
			saccades = np.zeros( (floor(sample_times[threshold_crossing_indices].shape[0]/2.0)) , dtype = self.saccade_dtype )
		
			# construct saccades:
			for i in range(0,sample_times[threshold_crossing_indices].shape[0]-1,2):
				j = i/2
				saccades[j]['start_time'] = sample_times[threshold_crossing_indices[i]] - sample_times[0]
				saccades[j]['end_time'] = sample_times[threshold_crossing_indices[i+1]] - sample_times[0]
				saccades[j]['start_point'][:] = xy_data[threshold_crossing_indices[i],:]
				saccades[j]['end_point'][:] = xy_data[threshold_crossing_indices[i+1],:]
				saccades[j]['duration'] = saccades[j]['end_time'] - saccades[j]['start_time']
				saccades[j]['vector'] = saccades[j]['end_point'] - saccades[j]['start_point']
				saccades[j]['amplitude'] = np.linalg.norm(saccades[j]['vector'])
				saccades[j]['direction'] = math.atan(saccades[j]['vector'][0] / (saccades[j]['vector'][1] + 0.00001))
				saccades[j]['peak_velocity'] = vel_data[threshold_crossing_indices[i]:threshold_crossing_indices[i+1]].max()
		else: saccades = np.array([])
			
		return saccades
		
	

class TAESession(EyeLinkSession):
	def preprocess_behavioral_data(self):
		"""docstring for preprocess_behavioral_data"""
		# rectify answers and test orientations
		self.rectified_answers = (1-self.parameter_data['answer'] * np.sign(self.parameter_data['adaptation_orientation'])) / 2.0
		self.rectified_test_orientations = self.parameter_data['test_orientation']*np.sign(self.parameter_data['adaptation_orientation'])
		
		# get values that are unique for conditions and tests
		self.adaptation_frequencies = np.unique(self.parameter_data['phase_redraw_period'])
		self.adaptation_durations = np.unique(self.parameter_data['adaptation_duration'])
		self.test_orientations = np.unique(self.parameter_data['test_orientation'])
		
		self.test_orientation_indices = [self.parameter_data['test_orientation'] == t for t in self.test_orientations]
		self.rectified_test_orientation_indices = [self.rectified_test_orientations == t for t in self.test_orientations]
		
		
	
	def fit_condition(self, boolean_array, sub_plot, title, plot_range = [-5,5], x_label = '', y_label = ''):
		"""fits the data in self.parameter_data[boolean_array] with a standard TAE psychometric curve and plots the data and result in sub_plot. It sets the title of the subplot, too."""
		rectified_answers = [self.rectified_answers[boolean_array * self.rectified_test_orientation_indices[i]] for i in range(self.test_orientations.shape[0])]
		nr_ones, nr_samples = [r.sum() for r in rectified_answers], [r.shape[0] for r in rectified_answers]
		fit_data = zip(self.test_orientations, nr_ones, nr_samples)
		
		# and we fit the data
		pf = BootstrapInference(fit_data, sigmoid = 'gauss', core = 'ab', nafc = 1, cuts = [0.25,0.5,0.75], gammaislambda = True)
		# and let the package do a bootstrap sampling of the resulting fits
		pf.sample()
		
		# scatter plot of the actual data points
		sub_plot.scatter(self.test_orientations, np.array(nr_ones) / np.array(nr_samples), facecolor = (1.0,1.0,1.0), edgecolor = 'k', alpha = 1.0, linewidth = 1.25)
		
		# line plot of the fitted curve
		sub_plot.plot(np.linspace(plot_range[0],plot_range[1], 500), pf.evaluate(np.linspace(plot_range[0],plot_range[1], 500)), 'k--', linewidth = 1.75)
		
		# TEAE value as a red line in the plot
		sub_plot.axvline(x=pf.estimate[0], c = 'r', alpha = 0.7, linewidth = 2.25)
		# and the boundaries of the confidence interval on this point
		sub_plot.axvline(x=pf.getCI(1)[0], c = 'r', alpha = 0.55, linewidth = 1.25)
		sub_plot.axvline(x=pf.getCI(1)[1], c = 'r', alpha = 0.55, linewidth = 1.25)
		
		sub_plot.set_title(title, fontsize=9)
		sub_plot.axis([plot_range[0], plot_range[1], -0.025, 1.025])
		
		# archive these results in the object's list variables.
		self.psychometric_data.append(fit_data)
		self.TAEs.append(pf.estimate[0])
		self.pfs.append(pf)
	
	def plot_confidence(self, boolean_array, sub_plot, normalize_confidence = True, plot_range = [-5,5], y_label = ''):
		"""plots the confidence data in self.parameter_data[boolean_array] in sub_plot. It doesn't set the title of the subplot."""
		rectified_confidence = [self.parameter_data[boolean_array * self.rectified_test_orientation_indices[i]]['confidence'] for i in range(self.test_orientations.shape[0])]
		mm = [np.min(self.parameter_data[:]['confidence']),np.max(self.parameter_data[:]['confidence'])]
		sub_plot = sub_plot.twinx()
		
		if normalize_confidence:
			# the confidence ratings are normalized
			conf_grouped_mean = np.array([np.mean((c-mm[0])/(mm[1]-mm[0])) for c in rectified_confidence])
			sub_plot.plot(self.test_orientations, conf_grouped_mean, 'g--' , alpha = 0.5, linewidth = 1.75)
			sub_plot.axis([plot_range[0],plot_range[1],-0.025,1.025])
			
		else:
			# raw confidence is used
			conf_grouped_mean = np.array([np.mean(c) for c in rectified_confidence])
			sub_plot.plot(self.test_orientations, conf_grouped_mean, 'g--' , alpha = 0.5, linewidth = 1.75)
			sub_plot.axis([plot_range[0], plot_range[1], mm[0], mm[1]])
			
		self.confidence_ratings.append(conf_grouped_mean)
		
		return sub_plot
	
	def run_temporal_conditions(self):
		"""
		run across conditions and adaptation durations
		"""
		# prepare some lists for further use
		self.psychometric_data = []
		self.TAEs = []
		self.pfs = []
		self.confidence_ratings = []
		self.conditions = []
		
		fig = pl.figure(figsize = (15,4))
		fig.subplots_adjust(wspace = 0.2, hspace = 0.3, left = 0.05, right = 0.95, bottom = 0.1)
		pl_nr = 1
		# across conditions
		for c in self.adaptation_frequencies:
			c_array = self.parameter_data[:]['phase_redraw_period'] == c
			# across adaptation durations:
			for a in self.adaptation_durations:
				a_array = self.parameter_data[:]['adaptation_duration'] == a
				# combination of conditions and durations
				this_condition_array = c_array * a_array
				sub_plot = fig.add_subplot(self.adaptation_frequencies.shape[0], self.adaptation_durations.shape[0], pl_nr)
				self.fit_condition(this_condition_array, sub_plot, 'adapt period ' + str(c) + ', adapt duration ' + str(a) )
				if c == self.adaptation_frequencies[-1]:
					sub_plot.set_xlabel('orientation [deg]', fontsize=9)
				if a == self.adaptation_durations[0]:
					sub_plot.set_ylabel('p(tilt seen in adapt direction)', fontsize=9)
					if c == self.adaptation_frequencies[0]:
						sub_plot.annotate(self.subject.firstName, (-4,1), va="top", ha="left", size = 14)
				
				sub_plot = self.plot_confidence(this_condition_array, sub_plot)
				if a == self.adaptation_durations[-1]:
					sub_plot.set_ylabel('confidence', fontsize=9)
				
				pl_nr += 1
				self.conditions.append([c, a])
				
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adaptation_psychometric_curves_' + str(self.wildcard) + '.pdf'))
		
		self.TAEs = np.array(self.TAEs).reshape((self.adaptation_frequencies.shape[0],self.adaptation_durations.shape[0]))
		
		# one big figure
		fig = pl.figure(figsize = (6,3))
		s = fig.add_subplot(111)
		s.axhline(y=0.0, c = 'k', marker = '.', alpha = 0.55, linewidth = 0.5)
		s.plot(self.adaptation_durations, self.TAEs[0], 'b--', linewidth = 1.75, alpha = 0.5)
		s.scatter(self.adaptation_durations, self.TAEs[0], facecolor = (1.0,1.0,1.0), edgecolor = 'b', alpha = 1.0, linewidth = 1.75)
		s.plot(self.adaptation_durations, self.TAEs[1], 'g--', linewidth = 1.75, alpha = 0.5)
		s.scatter(self.adaptation_durations, self.TAEs[1], facecolor = (1.0,1.0,1.0), edgecolor = 'g', alpha = 1.0, linewidth = 1.75)
		s.set_ylabel('TAE [deg]', fontsize = 9)
		s.set_xlabel('Adapt duration [s]', fontsize = 9)
		s.set_title('Subject ' + self.subject.firstName, fontsize = 9)
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adapt_summary_' + str(self.wildcard) + '.pdf'))
		
		# correlate TAE values against minimum confidence points
		self.confidence_ratings = np.array(self.confidence_ratings).reshape((self.adaptation_frequencies.shape[0],self.adaptation_durations.shape[0],-1))
		# construct minimum confidence points
		self.confidence_minima = np.zeros((self.adaptation_frequencies.shape[0],self.adaptation_durations.shape[0]))
		fig = pl.figure(figsize = (5,5))
		s = fig.add_subplot(111, aspect='equal')
		s.plot(np.linspace(-1.0, 2.75,40), np.linspace(-1.0, 2.75, 40), 'k--', alpha = 0.95, linewidth = 1.75)
		for i in range(self.adaptation_frequencies.shape[0]):
			for j in range(self.adaptation_durations.shape[0]):
				self.confidence_minima[i,j] = self.test_orientations[self.confidence_ratings[i,j] == np.min(self.confidence_ratings[i,j])].mean()
				s.scatter(self.TAEs[i,j], self.confidence_minima[i,j], s = 20 + 20 * j, facecolor = (1.0,1.0,1.0), edgecolor = ['b','g'][i], alpha = 1.0 - (0.75 * j) / float(self.adaptation_durations.shape[0]), linewidth = 1.75)
		s.axis([-1.0, 2.75, -1.0, 2.75])
		s.set_xlabel('TAE [deg]', fontsize = 9)
		s.set_ylabel('Conf Min [deg]', fontsize = 9)
		s.set_title('Subject ' + self.subject.firstName, fontsize = 9)
		s.grid()
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adapt_conf_corr_' + str(self.wildcard) + '.pdf'))
		
		# rotate and histogram for upper right corner
		fig = pl.figure(figsize = (7,3.5))
		s = fig.add_subplot(111)
		
		# rotate those and make a histogram:
		rotatedDistances = np.zeros((self.TAEs.shape[0], self.TAEs[0].ravel().shape[0], 2))
		for i in range(self.adaptation_frequencies.shape[0]):
			rotatedDistances[i] = rotateCartesianPoints(np.array([self.TAEs[i].ravel(),self.confidence_minima[i].ravel()]), -45.0, indegrees = True)
			pl.hist(rotatedDistances[i][:,0], edgecolor = (1.0,1.0,1.0), facecolor = ['b','g'][i], label = ['fast','slow'][i], alpha = 0.5, range = [-sqrt(2),sqrt(2)], normed = True, rwidth = 0.5)
		s.legend()
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adapt_conf_corr_' + str(self.wildcard) + '_hist.pdf'))
	
	def run_temporal_conditions_joined(self):
		"""
		run across conditions and adaptation durations
		"""
		# prepare some lists for further use
		self.psychometric_data = []
		self.TAEs = []
		self.pfs = []
		self.confidence_ratings = []
		self.conditions = []
		
		fig = pl.figure(figsize = (15,3))
		fig.subplots_adjust(wspace = 0.2, hspace = 0.3, left = 0.05, right = 0.95, bottom = 0.1)
		pl_nr = 1
		# across adaptation durations:
		for a in self.adaptation_durations:
			a_array = self.parameter_data[:]['adaptation_duration'] == a
			# combination of conditions and durations
			this_condition_array = a_array
			sub_plot = fig.add_subplot(1, self.adaptation_durations.shape[0], pl_nr)
			self.fit_condition(this_condition_array, sub_plot, 'adapt duration ' + str(a) )
			sub_plot.set_xlabel('orientation [deg]', fontsize=9)
			if a == self.adaptation_durations[0]:
				sub_plot.set_ylabel('p(tilt seen in adapt direction)', fontsize=9)
				sub_plot.annotate(self.subject.firstName, (-4,1), va="top", ha="left", size = 14)
					
			sub_plot = self.plot_confidence(this_condition_array, sub_plot)
			if a == self.adaptation_durations[-1]:
				sub_plot.set_ylabel('confidence', fontsize=9)
				
			pl_nr += 1
			self.conditions.append(a)
			
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adaptation_psychometric_curves_' + str(self.wildcard) + '_joined.pdf'))
		
		self.TAEs = np.array(self.TAEs).reshape((self.adaptation_durations.shape[0]))
		
		# one big figure
		fig = pl.figure(figsize = (6,3))
		s = fig.add_subplot(111)
		s.axhline(y=0.0, c = 'k', marker = '.', alpha = 0.55, linewidth = 0.5)
		s.plot(self.adaptation_durations, self.TAEs, 'r--', linewidth = 1.75, alpha = 0.5)
		s.scatter(self.adaptation_durations, self.TAEs, facecolor = (1.0,1.0,1.0), edgecolor = 'r', alpha = 1.0, linewidth = 1.75)
		s.set_ylabel('TAE [deg]', fontsize = 9)
		s.set_xlabel('Adapt duration [s]', fontsize = 9)
		s.set_title('Subject ' + self.subject.firstName, fontsize = 9)
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adapt_summary_' + str(self.wildcard) + '_joined.pdf'))
	
	def run_orientation_noise_conditions(self):
		self.noise_widths = np.unique(self.parameter_data[:]['adaptation_orientation_standard_deviation'])
		
		# prepare some lists for further use
		self.psychometric_data = []
		self.TAEs = []
		self.pfs = []
		self.confidence_ratings = []
		self.conditions = []
		
		fig = pl.figure(figsize = (4,10))
		fig.subplots_adjust(wspace = 0.2, hspace = 0.4, left = 0.1, right = 0.9, bottom = 0.025, top = 0.975)
		pl_nr = 1
		# across adaptation durations:
		for nw in self.noise_widths:
			a_array = self.parameter_data[:]['adaptation_orientation_standard_deviation'] == nw
			this_condition_array = a_array
			sub_plot = fig.add_subplot(self.noise_widths.shape[0], 1, pl_nr)
			self.fit_condition(this_condition_array, sub_plot, 'adapt spread ' + str(nw) )
			sub_plot.set_xlabel('orientation [deg]', fontsize=9)
			if nw == self.noise_widths[0]:
				sub_plot.set_ylabel('p(tilt seen in adapt direction)', fontsize=9)
				sub_plot.annotate(self.subject.firstName, (-4,1), va="top", ha="left", size = 14)
					
			sub_plot = self.plot_confidence(this_condition_array, sub_plot)
			if nw == self.noise_widths[-1]:
				sub_plot.set_ylabel('confidence', fontsize=9)
				
			pl_nr += 1
			self.conditions.append(nw)
			
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adaptation_psychometric_curves_' + str(self.wildcard) + '_joined.pdf'))
		
		self.TAEs = np.array(self.TAEs).reshape((self.noise_widths.shape[0]))
		
		# one big figure
		fig = pl.figure(figsize = (6,3))
		s = fig.add_subplot(111)
		s.axhline(y=0.0, c = 'k', marker = '.', alpha = 0.55, linewidth = 0.5)
		s.plot(self.noise_widths, self.TAEs, 'r--', linewidth = 1.75, alpha = 0.5)
		s.scatter(self.noise_widths, self.TAEs, facecolor = (1.0,1.0,1.0), edgecolor = 'r', alpha = 1.0, linewidth = 1.75)
		s.set_ylabel('TAE [deg]', fontsize = 9)
		s.set_xlabel('Adapt spread [$\sigma$ in deg]', fontsize = 9)
		s.set_title('Subject ' + self.subject.firstName, fontsize = 9)
		s.axis([-1,22,-0.3,2.0])
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adapt_summary_' + str(self.wildcard) + '_joined.pdf'))
	
	def save_fit_results(self, suffix = ''):
		"""docstring for save_fit_results"""
		if not len(self.psychometric_data) > 0 or not len(self.confidence_ratings) > 0:
			self.run_conditions()
		else:
			h5f = openFile(self.hdf5_filename, mode = "a" )
			if 'results_' + str(self.wildcard) + suffix in [g._v_name for g in h5f.listNodes(where = '/', classname = 'Group')]:
				h5f.removeNode(where = '/', name = 'results_' + str(self.wildcard) + suffix, recursive = True)
				
			resultsGroup = h5f.createGroup('/', 'results_' + str(self.wildcard) + suffix, 'results created at ' + datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			if hasattr(self, 'psychometric_data'): h5f.createArray(resultsGroup, 'psychometric_data', np.array(self.psychometric_data, dtype = np.float64),'Psychometric Data')
			if hasattr(self, 'TAEs'): h5f.createArray(resultsGroup, 'TAEs', np.array(self.TAEs, dtype = np.float64), 'Tilt after-effects')
			if hasattr(self, 'confidence_ratings'): h5f.createArray(resultsGroup, 'confidence_ratings', np.array(self.confidence_ratings, dtype = np.float64), 'Confidence_ratings')
			if hasattr(self, 'conditions'): h5f.createArray(resultsGroup, 'conditions', np.array(self.conditions, dtype = np.float64), 'Conditions - Adaptation frequencies and Durations')
			if hasattr(self, 'confidence_minima'): h5f.createArray(resultsGroup, 'confidence_minima', np.array(self.confidence_minima, dtype = np.float64), 'Confidence_rating minima')
			
			h5f.close()
	
	def import_distilled_behavioral_data(self, run_name = 'run_', results_name = ''):
		super(TAESession, self).import_parameters( run_name = run_name )
		h5f = openFile(self.hdf5_filename, mode = "r" )
		if results_name == None:
			results_name = 'results_' + self.wildcard
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if 'results_' + str(self.wildcard) + results_name == r._v_name:
				self.psychometric_data = r.psychometric_data.read()
				self.TAEs = r.TAEs.read()
				self.confidence_ratings = r.confidence_ratings.read()
				self.conditions = r.conditions.read()
				if hasattr(r, 'confidence_minima'):
					self.confidence_minima = r.confidence_minima.read()
		self.logger.info('imported behavioral distilled results from ' + 'results_' + str(self.wildcard) + results_name)
		h5f.close()
	

class TEAESession(EyeLinkSession):
	"""TEAESession analyzes the results of TEAE experiments"""
	def preprocess_behavioral_data(self):
		"""docstring for preprocess_behavioral_data"""
		# compute the relevant test contrasts
		self.test_contrasts = np.log10(self.parameter_data[:]['first_test_contrast'] + self.parameter_data[:]['last_test_contrast'])
		self.adaptation_orientations = np.unique(self.parameter_data[:]['adaptation_orientation'])
		self.test_orientations = np.unique(self.parameter_data[:]['test_orientation'])
		self.adaptation_durations = np.unique(self.parameter_data[:]['adaptation_duration'])
		self.phase_redraw_periods = np.unique(self.parameter_data[:]['phase_redraw_period'])
	
	def fit_condition(self, boolean_array, sub_plot, title, plot_range = [-2.8,-1.5], x_label = '', y_label = '', colors = ['r','k']):
		"""fits the data in self.parameter_data[boolean_array] with a standard TAE psychometric curve and plots the data and result in sub_plot. It sets the title of the subplot, too."""
		
		# psychometric curve settings
		# gumbel_l makes the fit a weibull given that the x-data are log-transformed
		# logistic and gauss (core = ab) deliver similar results
		nafc = 2
		sig = 'gumbel_l'	
		core = 'ab'
		
		tested_contrasts = np.unique(self.test_contrasts[boolean_array])
		tested_contrast_indices = [self.test_contrasts == tc for tc in tested_contrasts]
		
		corrects = [self.parameter_data[boolean_array * tested_contrast_indices[i]]['correct'] for i in range(len(tested_contrast_indices))]
		nr_ones, nr_samples = [r.sum() for r in corrects], [r.shape[0] for r in corrects]
		fit_data = zip(tested_contrasts, nr_ones, nr_samples)
		
		# and we fit the data
		pf = BootstrapInference(fit_data, sigmoid = sig, nafc = nafc, core = core, priors = ( 'unconstrained', 'unconstrained', 'Uniform(0,0.1)' ))
		# and let the package do a bootstrap sampling of the resulting fits
		pf.sample()
		
		# scatter plot of the actual data points
		sub_plot.scatter(tested_contrasts, np.array(nr_ones) / np.array(nr_samples), facecolor = (1.0,1.0,1.0), edgecolor = colors[1], alpha = 1.0, linewidth = 1.25, s = nr_samples)
		
		# line plot of the fitted curve
		sub_plot.plot(np.linspace(plot_range[0],plot_range[1], 500), pf.evaluate(np.linspace(plot_range[0],plot_range[1], 500)), c = colors[1], linewidth = 1.75)
		# line plot of chance
		sub_plot.plot(np.linspace(plot_range[0],plot_range[1], 500), np.ones((500)) * 0.5, 'y--', linewidth = 0.75, alpha = 0.5)
		
		# TEAE value as a red line in the plot
		sub_plot.axvline(x=pf.estimate[0], c = colors[0], alpha = 0.7, linewidth = 2.25)
		# and the boundaries of the confidence interval on this point
		sub_plot.axvline(x=pf.getCI(1)[0], c = colors[0], alpha = 0.55, linewidth = 1.25)
		sub_plot.axvline(x=pf.getCI(1)[1], c = colors[0], alpha = 0.55, linewidth = 1.25)
		
		sub_plot.set_title(title, fontsize=9)
		sub_plot.axis([plot_range[0], plot_range[1], -0.025, 1.025])
		
		# archive these results in the object's list variables.
		self.psychometric_data.append(fit_data)
		self.TEAEs.append(pf.estimate[0])
		self.pfs.append(pf)		
	
	def plot_staircases_for_condition(self, boolean_array, sub_plot, color = 'b', plot_range = [-2.8,-1.5]):
		staircases_in_this_condition = np.unique(self.parameter_data[boolean_array]['staircase'])
		staircase_indices = [self.parameter_data[:]['staircase'] == st for st in staircases_in_this_condition]
		
		sub_plot = sub_plot.twinx()
		
		for i in range(len(staircase_indices)):
			sub_plot.plot(self.test_contrasts[staircase_indices * boolean_array], c = 'b', linewidth = 1.75, alpha = 0.5, linestyle = '--')
		sub_plot.axis([0, self.test_contrasts[staircase_indices * boolean_array].shape[0]+1, plot_range[0], plot_range[1]])
		
		return sub_plot
	
	def run_training_analysis(self, run_nr = 0):
		self.logger.debug('starting training analysis for run ' + str(run_nr))
		# the training runs are called '3'
		self.import_parameters( run_name = self.subject.initials + '_' + str(3) + '_run_' + str(run_nr) )
		self.preprocess_behavioral_data()
		
		self.psychometric_data = []
		self.conditions = []
		self.TEAEs = []
		self.pfs = []
		
		fig = pl.figure(figsize = (11,3))
		s = fig.add_subplot(1,1,1)
		self.fit_condition(boolean_array = np.ones((len(self.parameter_data)), dtype = 'Bool'), sub_plot = s, title = 'training ' + self.wildcard, x_label = 'log contrast', y_label = 'p (correct)')
		pl.show()
		pl.savefig(os.path.join(self.base_directory, 'figs', 'training_psychometric_curves_' + str(run_nr) + '_' + str(self.wildcard) + '.pdf'))
	
	def run_time_analysis(self, run_nr = None):
		self.logger.debug('starting adaptation analysis for run ' + str(run_nr))
		# the training runs are called '3'
		if run_nr == None:
			self.import_parameters( run_name = self.subject.initials + '_' + str(2) + '_run_' ) # run_name = self.subject.initials + '_' + str(2) + '_run_'
			run_nr = 'all'
		else:
			self.import_parameters( run_name = self.subject.initials + '_' + str(2) + '_run_' + str(run_nr) )
		self.preprocess_behavioral_data()
		
		self.psychometric_data = []
		self.conditions = []
		self.TEAEs = []
		self.pfs = []
		
		equal_opposite_orientations = np.array([((self.parameter_data[:]['test_orientation'] - self.parameter_data[:]['adaptation_orientation']) == 0) == v for v in [True, False]])
		
		fig_nr = 1
		fig = pl.figure(figsize = (15,4))
		for i in range(self.phase_redraw_periods.shape[0]):
			this_prp_indices = self.parameter_data[:]['phase_redraw_period'] == self.phase_redraw_periods[i]
			for j in range(self.adaptation_durations.shape[0]):
				s = fig.add_subplot(self.phase_redraw_periods.shape[0],self.adaptation_durations.shape[0],fig_nr)
				for k in range(equal_opposite_orientations.shape[0]):
					this_ad_indices = self.parameter_data[:]['adaptation_duration'] == self.adaptation_durations[j]
					self.fit_condition(boolean_array = this_prp_indices * this_ad_indices * equal_opposite_orientations[k], sub_plot = s, title = 'adaptation ' + str(self.adaptation_durations[j]) + ' ' +  str(i), x_label = 'log contrast', y_label = 'p (correct)', colors = [['r','r'], ['k','k']][k])
					self.conditions.append([self.phase_redraw_periods[i], self.adaptation_durations[j], [0,1][k]])
				fig_nr += 1
				
		pl.savefig(os.path.join(self.base_directory, 'figs', 'training_psychometric_curves_' + str(run_nr) + '_' + str(self.wildcard) + '.pdf'))
		
		self.TEAEs = np.array(self.TEAEs).reshape((self.phase_redraw_periods.shape[0], self.adaptation_durations.shape[0],equal_opposite_orientations.shape[0]))
		
		# one big figure
		fig = pl.figure(figsize = (6,3))
		s = fig.add_subplot(111)
		for i in range(self.phase_redraw_periods.shape[0]):
			s.axhline(y=0.0, c = 'k', marker = '.', alpha = 0.55, linewidth = 0.5)
			s.plot(self.adaptation_durations, self.TEAEs[i,:,0]-self.TEAEs[i,:,1], ['g--','b--'][i], linewidth = 1.75, alpha = 0.5)
			s.scatter(self.adaptation_durations, self.TEAEs[i,:,0]-self.TEAEs[i,:,1], facecolor = (1.0,1.0,1.0), edgecolor = ['g','b'][i], alpha = 1.0, linewidth = 1.75)
			s.set_ylabel('TEAE [log contrast]', fontsize = 9)
			s.set_xlabel('Adaptation duration [s]', fontsize = 9)
			s.set_title('Subject ' + self.subject.firstName, fontsize = 9)
			pl.savefig(os.path.join(self.base_directory, 'figs', 'adapt_summary_' + str(self.wildcard) + '_joined.pdf'))
	

class SASession(EyeLinkSession):
	"""Saccade adaptation session"""
	def plot_velocity_per_trial_for_run(self, run_index = 0, trial_phase_range = [1,4], trial_ranges = [[25,125],[125,185],[185,245]], colors = ['b','g','r','c','m','y','k'], nr_plot_points = 1000):
		"""create a single - file pdf plotting the normed velocity of the eye position in all trials"""
		
		vel_data = self.get_EL_samples_per_trial(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range, data_type = 'smoothed_velocity')
		sacc_data = self.get_EL_events_per_trial(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range, data_type = 'saccades')
		
		durations = []
		saccade_latencies = []
		max_index = 0
		
		fig = pl.figure(figsize = (12,8))
		fig.subplots_adjust(wspace = 0.2, hspace = 0.3, left = 0.05, right = 0.95, bottom = 0.1)
		
		s = fig.add_subplot(411)
		for (i, trial_block_vel_data, trial_block_sacc_data) in zip(range(len(vel_data)), vel_data, sacc_data):
			durations.append([])
			saccade_latencies.append([])
			for (j, trial_vel_data, trial_sacc_data) in zip(range(len(trial_block_vel_data)), trial_block_vel_data, trial_block_sacc_data):
				if max_index < np.min([nr_plot_points, trial_vel_data.shape[0]]):
					max_index = np.min([nr_plot_points, trial_vel_data.shape[0]])
				s.plot( trial_vel_data[:np.min([nr_plot_points, trial_vel_data.shape[0]]),0] - trial_vel_data[0,0], trial_vel_data[:np.min([nr_plot_points, trial_vel_data.shape[0]]),1], c = colors[i], linewidth = 1.5, alpha = 0.25 )
				# take eyelink's saccade start times in this trial
				if len(trial_sacc_data) > 0:
					sacc_timestamps = np.array([sacc['start_timestamp'] for sacc in trial_sacc_data])
					saccade_latencies[-1].append( sacc_timestamps - trial_vel_data[0,0])
				durations[-1].append(trial_vel_data[-1,0] - trial_vel_data[0,0] - 750)
			s.axis([0, trial_vel_data[np.min([nr_plot_points, trial_vel_data.shape[0]]),0] - trial_vel_data[0,0], 0, 600])
		
		# find our own saccades
		self_saccades = self.find_saccades_per_trial_for_run(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range)
		s = fig.add_subplot(412)
		bin_range = [0,trial_vel_data[max_index,0]-trial_vel_data[0,0]]
		for i in range(len(vel_data)):
			#pl.hist(np.array(durations[i]), range = bin_range, bins = 90, alpha = 0.5, normed = True, histtype = 'stepfilled', color = colors[i] )
			pl.plot(np.sort(np.array(durations[i])), np.linspace(0,1, np.array(durations[i]).shape[0]), alpha = 0.5, color = colors[i], linewidth = 1.25 )
		for i in range(len(vel_data)):
			#pl.hist(np.array([ss[0] for ss in saccade_latencies[i]]), range = bin_range, bins = 90, alpha = 0.25, normed = True, histtype = 'step', linewidth = 2.5, color = colors[i] )
			pl.plot(np.sort(np.array([ss[0] for ss in saccade_latencies[i]])), np.linspace(0,1,np.array([ss[0] for ss in saccade_latencies[i]]).shape[0]), alpha = 0.675, color = colors[i], linewidth = 1.5 )
		for i in range(len(vel_data)):
			#pl.hist(np.array([ss[0]['start_time'] for ss in self_saccades[i]]), range = bin_range, bins = 90, alpha = 0.75, normed = True, histtype = 'step', linewidth = 2.5, color = colors[i] )
			pl.plot(np.sort(np.array([ss[0]['start_time'] for ss in self_saccades[i]])), np.linspace(0,1,np.array([ss[0]['start_time'] for ss in self_saccades[i]]).shape[0]), alpha = 0.75, color = colors[i], linewidth = 1.75 )
		s.axis([0,500,0,1])
		s.axis([0, trial_vel_data[np.min([nr_plot_points, trial_vel_data.shape[0]]),0] - trial_vel_data[0,0], 0, 1])
		
		s = fig.add_subplot(413)
		for i in range(len(vel_data)):
			pl.scatter(np.arange(len(vel_data[i])), np.array([ss[0]['amplitude'] for ss in self_saccades[i]]), facecolor = (1.0,1.0,1.0), edgecolor = colors[i], alpha = 0.5, linewidth = 1.25 )
			smooth_width = 10
			sm_signal = np.convolve( np.array([ss[0]['amplitude'] for ss in self_saccades[i]]), np.ones((smooth_width))/float(smooth_width), 'valid' )
			pl.plot( np.arange(sm_signal.shape[0]) + smooth_width/2, sm_signal, c = colors[i], alpha = 0.5, linewidth = 1.25 )
		s.axis([0, np.max([len(rr) for rr in vel_data]) + 1, 0, 15])
		
		self.import_parameters(run_name = self.wildcard + '_run_' + str(run_index))
		all_these_saccades_from_parameters = self.distill_saccades_from_parameters()[0]
		par_sacs = [all_these_saccades_from_parameters[tr[0]:tr[1]] for tr in trial_ranges]
		
		s = fig.add_subplot(414)
		for i in range(len(vel_data)):
			pl.scatter(np.arange(len(vel_data[i])), np.array([ss[0]['amplitude'] for ss in self_saccades[i]]) / par_sacs[i]['amplitude'], facecolor = (1.0,1.0,1.0), edgecolor = colors[i], alpha = 0.5, linewidth = 1.25 )
			smooth_width = 10
			sm_signal = np.convolve( np.array([ss[0]['amplitude'] for ss in self_saccades[i]]) / par_sacs[i]['amplitude'], np.ones((smooth_width))/float(smooth_width), 'valid' )
			pl.plot( np.arange(sm_signal.shape[0]) + smooth_width/2, sm_signal, c = colors[i], alpha = 0.5, linewidth = 1.25 )
		s.axis([0, np.max([len(rr) for rr in vel_data]) + 1, 0, 1.75])
		
		pl.savefig(os.path.join(self.base_directory, 'figs', 'trial_' + 'smoothed_velocity' + '_' + str(self.wildcard) + '_run_' + str(run_index) + '.pdf'))
		
		return self_saccades
	
	def plot_all_saccades_for_run(self, run_index = 0, trial_phase_range = [1,4], trial_ranges = [[25,125],[125,185],[185,245]], colors = ['b','g','r','c','m','y','k'], nr_plot_points = 1000):
		from matplotlib.backends.backend_pdf import PdfPages
		pp = PdfPages(os.path.join(self.base_directory, 'figs', 'all_saccades' + '_' + str(self.wildcard) + '_run_' + str(run_index) + '.pdf'))
		
		max_index = 0
		self.import_parameters(run_name = self.wildcard + '_run_' + str(run_index))
		vel_data = self.get_EL_samples_per_trial(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range, data_type = 'smoothed_velocity')
		xy_data = self.get_EL_samples_per_trial(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range, data_type = 'gaze_xy')
		sacc_data = self.get_EL_events_per_trial(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range, data_type = 'saccades')
		
		self_saccades = self.find_saccades_per_trial_for_run(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range)
		ps = [self.parameter_data[tr[0]:tr[1]] for tr in trial_ranges]
		
		print 'data gathered, entering loops'
		
		for (i, trial_block_vel_data, trial_block_sacc_data, trial_block_xy_data, trial_block_ps) in zip(range(len(vel_data)), vel_data, sacc_data, xy_data, ps):
			print 'i:' + str(i)
			for (j, trial_vel_data, trial_sacc_data, trial_xy_data, trial_ps) in zip(range(len(trial_block_vel_data)), trial_block_vel_data, trial_block_sacc_data, trial_block_xy_data, trial_block_ps):
				print 'j:' + str(j)
				if max_index < np.min([nr_plot_points, trial_vel_data.shape[0]]):
					max_index = np.min([nr_plot_points, trial_vel_data.shape[0]])
				f = pl.figure(figsize = (12,3))
				f.subplots_adjust(wspace = 0.2, hspace = 0.3, left = 0.05, right = 0.95, bottom = 0.1)
				
				# take eyelink's saccade start times in this trial
				if len(trial_sacc_data) > 0:
					sacc_timestamps = np.array([sacc['start_timestamp'] for sacc in trial_sacc_data])
					el_saccade_latency = sacc_timestamps[0] - trial_vel_data[0,0]
					sacc_startpoint = [trial_sacc_data['start_x'][0], trial_sacc_data['start_y'][0]]
					sacc_endpoint = [trial_sacc_data['end_x'][0], trial_sacc_data['end_y'][0]]
				else:
					el_saccade_latency = 0.0
					
				s = f.add_subplot(121)
				s.plot( trial_xy_data[:np.min([nr_plot_points, trial_vel_data.shape[0]]),1], trial_xy_data[:np.min([nr_plot_points, trial_vel_data.shape[0]]),2], c = colors[i], linewidth = 2.5, alpha = 0.35 )
				if len(trial_sacc_data) > 0:
					s.plot(np.array([sacc_startpoint[0], sacc_endpoint[0]]), np.array([sacc_startpoint[1], sacc_endpoint[1]]), colors[i] + 'o', mew = 2.5, alpha = 0.75, mec = 'w', ms = 10)
				s.plot(np.array([trial_ps['saccade_endpoint_x']]), np.array([trial_ps['saccade_endpoint_y']]), colors[i] + 's', mew = 2.5, alpha = 0.5, mec = 'w', ms = 8)
				s.plot(np.array([trial_ps['fixation_target_x'], trial_ps['saccade_target_x']]), np.array([trial_ps['fixation_target_y'], trial_ps['saccade_target_y']]), colors[i] + 'D', mew = 2.5, alpha = 0.65, mec = 'w', ms = 8)
				# fit the screen
				s.axis([0,800,0,600])
				s.set_title('gaze in screen coordinates')
				s.annotate('Trial # ' + str(trial_ps['trial_nr']) + ' block ' + str(trial_ps['trial_block']), (20,550), va="top", ha="left", size = 6 )
				s.annotate('Saccade amplitude pre-step: ' + str(trial_ps['amplitude_pre']) + ' or ' + str(np.linalg.norm(np.array([trial_ps['fixation_target_x'], trial_ps['fixation_target_y']]) - np.array([trial_ps['saccade_target_x'], trial_ps['saccade_target_y']]))), (20,500), va="top", ha="left", size = 6 )
				s.annotate('Saccade amplitude post-step: ' + str(np.linalg.norm(np.array([trial_ps['fixation_target_x'], trial_ps['fixation_target_y']]) - np.array([trial_ps['saccade_endpoint_x'], trial_ps['saccade_endpoint_y']]))), (20,450), va="top", ha="left", size = 6 )
				s.annotate('Gain of step: ' + str(np.linalg.norm(np.array([trial_ps['fixation_target_x'], trial_ps['fixation_target_y']]) - np.array([trial_ps['saccade_target_x'], trial_ps['saccade_target_y']]))/np.linalg.norm(np.array([trial_ps['fixation_target_x'], trial_ps['fixation_target_y']]) - np.array([trial_ps['saccade_endpoint_x'], trial_ps['saccade_endpoint_y']]))), (20,400), va="top", ha="left", size = 6 )
				s.annotate('Amplitude of actual saccade: ' + str(np.linalg.norm(np.array(sacc_startpoint)-np.array(sacc_endpoint))) + ' gain: ' + str(np.linalg.norm(np.array(sacc_startpoint)-np.array(sacc_endpoint))/np.linalg.norm(np.array([trial_ps['fixation_target_x'], trial_ps['fixation_target_y']]) - np.array([trial_ps['saccade_target_x'], trial_ps['saccade_target_y']]))), (20,350), va="top", ha="left", size = 6 )
				s = f.add_subplot(122)
				s.plot( trial_vel_data[:np.min([nr_plot_points, trial_vel_data.shape[0]]),0] - trial_vel_data[0,0], trial_vel_data[:np.min([nr_plot_points, trial_vel_data.shape[0]]),1], c = colors[i], linewidth = 2.5, alpha = 0.25 )
					
				if len(trial_sacc_data) > 0:
					s.axvline(el_saccade_latency, c = colors[i], alpha = 0.7, linewidth = 1.25)
				s.axvline(trial_vel_data[-1,0] - trial_vel_data[0,0] - 750, c = colors[i], alpha = 0.7, linewidth = 1.25, linestyle = '--')
				s.axis([0,500,0,500])
				s.set_title('velocity')
				pp.savefig()
			pl.close()
		pp.close()
	
	def find_saccades_per_trial_for_run(self, run_index = 0, trial_phase_range = [1,4], trial_ranges = [[25,125],[125,185],[185,245]]):
		"""
		finds saccades in a session 
		"""
		gaze_data = self.get_EL_samples_per_trial(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range, data_type = 'smoothed_gaze_xy')
		vel_data = self.get_EL_samples_per_trial(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range, data_type = 'smoothed_velocity_xy')
		sacc_data = self.get_EL_events_per_trial(run_index = run_index, trial_ranges = trial_ranges, trial_phase_range = trial_phase_range, data_type = 'saccades')
		
		self.import_parameters(run_name = self.wildcard + '_run_' + str(run_index))
		
		saccades = []
		for (i, trial_block_vel_data, trial_block_sacc_data, trial_block_gaze_data) in zip(range(len(vel_data)), vel_data, sacc_data, gaze_data):
			saccades.append([])
			for (j, trial_vel_data, trial_sacc_data, trial_gaze_data) in zip(range(len(trial_block_vel_data)), trial_block_vel_data, trial_block_sacc_data, trial_block_gaze_data):
				saccs = self.detect_saccade_from_data(xy_data = trial_gaze_data[:,1:], xy_velocity_data = trial_vel_data[:,1:], l = 5, sample_times = trial_gaze_data[:,0], pixels_per_degree = self.parameter_data[0]['pixels_per_degree'])
				if saccs.shape[0] > 0:
					saccades[-1].append(saccs)
				else:
					saccades[-1].append(np.zeros((1), dtype = self.saccade_dtype))
		self.logger.debug('Detected saccades from run # ' + str(run_index))
		return saccades
	
	def plot_velocity_per_trial_all_runs(self, trial_phase_range = [1,4], trial_ranges = [[25,125],[125,185],[185,245]], colors = ['b','g','r','c','m','y','k'], nr_plot_points = 1000):
		h5f = openFile(self.hdf5_filename, mode = "r" )
		runs = []
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if self.wildcard + '_run_' in r._v_name:
				runs.append( int(r._v_name.split('_')[-1]) )
		h5f.close()
		sacs = []
		if len(runs) != 0:
			for r in runs:
				sacs.append(self.plot_velocity_per_trial_for_run(run_index = r, trial_phase_range = trial_phase_range, trial_ranges = trial_ranges, colors = colors, nr_plot_points = nr_plot_points))
			return sacs
		else:
			return
	
	def analyze_saccades_all_runs(self, trial_phase_range = [1,4], trial_ranges = [[25,125],[125,185],[185,245]], smooth_width = 15, colors = ['b','g','r','c','m','y','k'] ):
		h5f = openFile(self.hdf5_filename, mode = "r" )
		runs = []
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if self.wildcard + '_run_' in r._v_name:
				runs.append( int(r._v_name.split('_')[-1]) )
		h5f.close()
		runs = np.sort(runs)
		
		sacs = []
		par_sacs = []
		
		if len(runs) != 0:
			fig = pl.figure(figsize = (15,3))
			s = fig.add_subplot(1,1,1)
			for r in runs:
				self.import_parameters(run_name = self.wildcard + '_run_' + str(r))
				sacs.append(self.find_saccades_per_trial_for_run(run_index = r, trial_phase_range = trial_phase_range, trial_ranges = trial_ranges))
				all_these_saccades_from_parameters = self.distill_saccades_from_parameters()[0]
				par_sacs.append([all_these_saccades_from_parameters[tr[0]:tr[1]] for tr in trial_ranges])
			
			blocks_multiple_runs_saccades = [[s[i] for s in sacs] for i in range(len(trial_ranges))]
			blocks_multiple_runs_parameter_saccades = [[s[i] for s in par_sacs] for i in range(len(trial_ranges))]
			
			for (i, bs, bp) in zip(range(len(blocks_multiple_runs_saccades)), blocks_multiple_runs_saccades, blocks_multiple_runs_parameter_saccades):
				minimal_block_length = np.array([len(br) for br in bs]).min()
				first_saccades = np.array([[st[0] for st in br[:minimal_block_length]] for br in bs], dtype = self.saccade_dtype)
				intended_saccades = np.array([br[:minimal_block_length] for br in bp], dtype = self.saccade_dtype)
				amps = np.array([first_saccades['amplitude'], intended_saccades['amplitude'], first_saccades['amplitude'] / intended_saccades['amplitude']])
				
				pl.plot( np.mean(amps[:,0], axis = 0), colors[i] + '--', alpha = 0.5, linewidth = 1.25 )
				pl.plot( np.mean(amps[:,1], axis = 0), colors[i] + '+', alpha = 0.5, linewidth = 1.25 )
				pl.plot( np.mean(amps[:,2], axis = 0), colors[i] + 'v', alpha = 0.5, linewidth = 1.25 )
				
#				kern =  stats.norm.pdf( np.linspace(-3.25,3.25,smooth_width) )
#				sm_signal = np.convolve( np.mean(amps, axis = 0), kern / kern.sum(), 'valid' )
#				pl.plot( np.arange(sm_signal.shape[0]) + smooth_width/2.0,  sm_signal, c = colors[i], alpha = 0.75, linewidth = 1.75 )
#				alpha = 0.2
#				for a in amps:
#					sm_signal = np.convolve( a, kern / kern.sum(), 'valid' )
#					pl.plot( np.arange(sm_signal.shape[0]) + smooth_width/2.0,  sm_signal, c = colors[i], alpha = alpha, linewidth = 0.75 )
#					alpha += 0.05
					
				pl.draw()
				
			pl.savefig(os.path.join(self.base_directory, 'figs', 'averaged_sacc_ampl_per_block_across_runs_' + str(self.wildcard) + '.pdf'))
	
	def distill_saccades_from_parameters(self, parameter_data = None):
		
		if parameter_data == None: parameter_data = self.parameter_data
		
		pre_step_saccades = np.zeros( (parameter_data.shape[0]) , dtype = self.saccade_dtype )
		post_step_saccades = np.zeros( (parameter_data.shape[0]) , dtype = self.saccade_dtype )
		
		# construct saccades:
		for (i, p) in zip(range(parameter_data.shape[0]), parameter_data):
			pre_step_saccades[i]['start_point'][:] = [p['fixation_target_x'], p['fixation_target_y']]
			pre_step_saccades[i]['end_point'][:] = [p['saccade_target_x'], p['saccade_target_y']]
			pre_step_saccades[i]['vector'] = pre_step_saccades[i]['end_point'] - pre_step_saccades[i]['start_point']
			pre_step_saccades[i]['amplitude'] = np.linalg.norm(pre_step_saccades[i]['vector']) / p['pixels_per_degree']
			pre_step_saccades[i]['direction'] = math.atan(pre_step_saccades[i]['vector'][0] / (pre_step_saccades[i]['vector'][1] + 0.00001))
			
			post_step_saccades[i] = pre_step_saccades[i]
			post_step_saccades[i]['end_point'][:] = [p['saccade_target_x'], p['saccade_target_y']]
			post_step_saccades[i]['vector'] = post_step_saccades[i]['end_point'] - post_step_saccades[i]['start_point']
			post_step_saccades[i]['amplitude'] = np.linalg.norm(post_step_saccades[i]['vector']) / p['pixels_per_degree']
			post_step_saccades[i]['direction'] = math.atan(post_step_saccades[i]['vector'][0] / (post_step_saccades[i]['vector'][1] + 0.00001))
			
		return [pre_step_saccades, post_step_saccades]
