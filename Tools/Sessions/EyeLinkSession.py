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
			eyelink_fos[i].clean_data()
	
	def import_behavioral_data(self):
		behavioral_data = []
		h5f = openFile(self.hdf5_filename, mode = "r" )
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if 'run_' in r._v_name:
				behavioral_data.append(r.trial_parameters.read())
		self.behavioral_data = np.concatenate(behavioral_data)
		self.logger.info('imported behavioral data from ' + str(self.behavioral_data.shape[0]) + ' trials')
		h5f.close()


class TAESession(EyeLinkSession):
	def preprocess_behavioral_data(self):
		"""docstring for preprocess_behavioral_data"""
		# rectify answers and test orientations
		self.rectified_answers = (1-self.behavioral_data['answer'] * np.sign(self.behavioral_data['adaptation_orientation'])) / 2.0
		self.rectified_test_orientations = self.behavioral_data['test_orientation']*np.sign(self.behavioral_data['adaptation_orientation'])
		
		# get values that are unique for conditions and tests
		self.adaptation_frequencies = np.unique(self.behavioral_data['phase_redraw_period'])
		self.adaptation_durations = np.unique(self.behavioral_data['adaptation_duration'])
		self.test_orientations = np.unique(self.behavioral_data['test_orientation'])
		
		self.test_orientation_indices = [self.behavioral_data['test_orientation'] == t for t in self.test_orientations]
		self.rectified_test_orientation_indices = [self.rectified_test_orientations == t for t in self.test_orientations]
		
		
	
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
		sub_plot.axis([plot_range[0], plot_range[1], -0.025, 1.025])
		
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
			sub_plot.plot(self.test_orientations, conf_grouped_mean, 'g--' , alpha = 0.5, linewidth = 1.75)
			sub_plot.axis([plot_range[0],plot_range[1],-0.025,1.025])
			
		else:
			# raw confidence is used
			conf_grouped_mean = np.array([np.mean(c) for c in rectified_confidence])
			sub_plot.plot(self.test_orientations, conf_grouped_mean, 'g--' , alpha = 0.5, linewidth = 1.75)
			sub_plot.axis([plot_range[0], plot_range[1], mm[0], mm[1]])
			
		self.confidence_ratings.append(conf_grouped_mean)
		
		return sub_plot
	
	def run_conditions(self):
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
			c_array = self.behavioral_data[:]['phase_redraw_period'] == c
			# across adaptation durations:
			for a in self.adaptation_durations:
				a_array = self.behavioral_data[:]['adaptation_duration'] == a
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
		tr = Affine2D().scale(3, 1).rotate_deg(-45)
		grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(-2, 2, 0, .7))
		s = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper) 
		# fig.add_subplot(s)
		s = fig.add_subplot(111)
		
		grid_helper.grid_finder.grid_locator1._nbins = 4 
		grid_helper.grid_finder.grid_locator2._nbins = 4
		
		# rotate those and make a histogram:
		rotatedDistances = np.zeros((self.TAEs.shape[0], self.TAEs[0].ravel().shape[0], 2))
		for i in range(self.adaptation_frequencies.shape[0]):
			rotatedDistances[i] = rotateCartesianPoints(np.array([self.TAEs[i].ravel(),self.confidence_minima[i].ravel()]), -45.0, indegrees = True)
			pl.hist(rotatedDistances[i][:,0], edgecolor = (1.0,1.0,1.0), facecolor = ['b','g'][i], label = ['fast','slow'][i], alpha = 0.5, range = [-sqrt(2),sqrt(2)], normed = True, rwidth = 0.5)
		s.legend()
		pl.savefig(os.path.join(self.base_directory, 'figs', 'adapt_conf_corr_' + str(self.wildcard) + '_hist.pdf'))
	
	def save_fit_results(self):
		"""docstring for save_fit_results"""
		if not len(self.psychometric_data) > 0 or not len(self.confidence_ratings) > 0:
			self.run_conditions()
		else:
			h5f = openFile(self.hdf5_filename, mode = "a" )
			if 'results' in [g._v_name for g in h5f.listNodes(where = '/', classname = 'Group')]:
				h5f.removeNode(where = '/', name = 'results', recursive = True)
				
			resultsGroup = h5f.createGroup('/', 'results', 'results created at ' + datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			h5f.createArray(resultsGroup, 'psychometric_data', np.array(self.psychometric_data, dtype = np.float64),'Psychometric Data')
			h5f.createArray(resultsGroup, 'TAEs', np.array(self.TAEs, dtype = np.float64), 'Tilt after-effects')
			h5f.createArray(resultsGroup, 'confidence_ratings', np.array(self.confidence_ratings, dtype = np.float64), 'Confidence_ratings')
			h5f.createArray(resultsGroup, 'conditions', np.array(self.conditions, dtype = np.float64), 'Conditions - Adaptation frequencies and Durations')
			h5f.createArray(resultsGroup, 'confidence_minima', np.array(self.confidence_minima, dtype = np.float64), 'Confidence_rating minima')
			
			h5f.close()
	
	def import_distilled_behavioral_data(self):
		super(TAESession, self).import_behavioral_data()
		h5f = openFile(self.hdf5_filename, mode = "r" )
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if 'results' == r._v_name:
				self.psychometric_data = r.psychometric_data.read()
				self.TAEs = r.TAEs.read()
				self.confidence_ratings = r.confidence_ratings.read()
				self.conditions = r.conditions.read()
				if hasattr(r, 'confidence_minima'):
					self.confidence_minima = r.confidence_minima.read()
		self.logger.info('imported behavioral distilled results')
		h5f.close()
