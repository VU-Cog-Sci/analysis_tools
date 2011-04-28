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
import numpy as np
import matplotlib.pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
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

class EyelinkSession(object):
	def __init__(self, ID, subject, project_name, experiment_name, base_directory, wild_card, loggingLevel = logging.DEBUG):
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
		eye_files = [f.split('_outputDict.pickle') + '.edf' for f in behavior_files]
		
		# put output dicts and eyelink files next to one another - that's what the eyelinkoperator expects.
		for i in range(len(behavior_files)):
			ExecCommandLine('cp ' + behavior_files[i] + ' ' + os.path.join(self.base_directory, 'raw', behavior_files[i]) )
			ExecCommandLine('cp ' + eye_files[i] + ' ' + os.path.join(self.base_directory, 'raw', eye_files[i]) )
	
	def convert_edf(self):
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
			elo.findAll()
			for r in range(len(elo.parameters)):
				if 'answer' not in elo.parameters[r].keys():
					print 'no answer in run # ' + elo.gazeFile + ' trial # ' + str(r)
					elo.parameters.pop(r)
		
		# enter the files into the hdf5 file in order
		order = np.argsort(np.array([int(elo.timeStamp.strftime("%Y%m%d%H%M%S")) for elo in eyelink_fos]))
		for i in order:
			eyelink_fos[i].processIntoTable(self.hdf5_filename, name = 'run_' + str(i))
	
	def import_behavioral_data(self)
		behavioral_data = []
		h5f = openFile(self.hdf5_filename, mode = "r" )
		for r in h5f.walkGroups(where = '/'):
			behavioral_data.append(r.trial_parameters.read())
		self.behavioral_data = np.concatenate(behavioral_data)
		self.logger.info('imported behavioral data from ' + str(self.behavioral_data.shape[0]) + ' trials')
	
class TAESession(EyelinkSession):
	def preprocess_behavioral_data(self):
		"""docstring for preprocess_behavioral_data"""
		# rectify answers and test orientations
		self.rectified_answers = (1-self.behavioral_data['answer'] * np.sign(self.behavioral_data['adaptation_orientation'])) / 2.0
		self.rectified_test_orientations = self.behavioral_data['test_orientation']*np.sign(self.behavioral_data['adaptation_orientation'])
		
		# get values that are unique for conditions and tests
		self.adaptation_frequencies = np.unique(self.behavioral_data['phase_redraw_period'])
		self.adaptation_durations = np.unique(self.behavioral_data['adaptation_duration'])
		self.test_orientations = np.unique(self.behavioral_data['test_orientation'])
		
		self.test_orientation_indices = [self.behavioraldata['test_orientation'] == t for t in self.test_orientations]
		self.rectified_test_orientation_indices = [rectified_test_orientations == t for t in self.test_orientations]
		
		# prepare some lists for further use
		self.psychometric_data = []
		self.TAEs = []
		self.pfs = []
		
	
	def fit_condition(self, boolean_array, sub_plot, title, plot_range = [-5,5], x_label = '', y_label = ''):
		"""fits the data in self.behavioral_data[boolean_array] with a standard TAE psychometric curve and plots the data and result in sub_plot. It sets the title of the subplot, too."""
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
		sub_plot.axis([plot_range[0], plot_range[0], -0.025, 1.025])
		
		# archive these results in the object's list variables.
		self.psychometric_data.append(fit_data)
		self.TAEs.append(pf.estimate[0])
		self.pfs.append(pf)
		
	def plot_confidence(self, boolean_array, sub_plot, normalize_confidence = True, plot_range = [-5,5], y_label = ''):
		"""plots the confidence data in self.behavioral_data[boolean_array] in sub_plot. It doesn't set the title of the subplot."""
		rectified_confidence = [self.behavioral_data[boolean_array * self.rectified_test_orientation_indices[i]]['confidence'] for i in range(self.test_orientations.shape[0])]
		mm = [np.min(self.behavioral_data[:]['confidence']),np.max(self.behavioral_data[:]['confidence'])]
		sub_plot = sub_plot.twinx()
		
		if normalize_confidence:
			# the confidence ratings are normalized
			conf_grouped_mean = np.array([np.mean((c-mm[0])/(mm[1]-mm[0])) for c in rectified_confidence])
			sub_plot.plot(self.test_orientations, conf_grouped_mean, 'g--' , alpha = 0.5, linewidth = 0.75)
			sub_plot.axis([plot_range[0],plot_range[1],-0.05,1.05])
			
		else:
			# raw confidence is used
			conf_grouped_mean = np.array([np.mean(c) for c in rectified_confidence])
			sub_plot.plot(self.test_orientations, conf_grouped_mean, 'g--' , alpha = 0.5, linewidth = 0.75)
			sub_plot.axis([plot_range[0], plot_range[1], mm[0], mm[1]])
	
	def run_conditions(self):
		"""
		run across conditions and adaptation durations
		"""
		
		fig = pl.figure(figsize = (15,4))
		pl_nr = 1
		# across conditions
		for c in self.adaptation_frequencies:
			c_array = self.behavioral_data[:]['phase_redraw_period'] == c
			# across adaptation durations:
			for a in self.adaptation_durations:
				a_array = self.behavioral_data[:]['adaptation_duration'] == a
				# combination of conditions and durations
				this_condition_array = c_array * a_array
				sub_plot = fig.add_subplot(self.adaptation_frequencies.shape[0], self.adaptation_durations.shape[0], pl_nr)
				self.fit_condition(this_condition_array, sub_plot, 'adapt period ' + str(c) + ', adapt duration ' + str(a) )
				self.plot_confidence(this_condition_array, sub_plot)
				pl_nr += 1
		
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adaptation_psychometric_curves.pdf')