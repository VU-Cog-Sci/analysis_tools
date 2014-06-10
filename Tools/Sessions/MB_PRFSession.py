#!/usr/bin/env python
# encoding: utf-8
"""
MB_PRFSession.py

Created by Tomas HJ Knapen on 2014-04-13.
Copyright (c) 2014 TK. All rights reserved.
"""

import os, sys, pickle, math
from subprocess import *
import datetime

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
from nifti import *

import pp
import logging, logging.handlers, logging.config

from ..log import *
from ..Run import *
from ..Subjects.Subject import *
from ..Operators.Operator import *
from ..Operators.CommandLineOperator import *
from ..Operators.ImageOperator import *
from ..Operators.BehaviorOperator import *
# from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from IPython import embed as shell

from PopulationReceptiveFieldMappingSession import PopulationReceptiveFieldMappingSession 

class TrialEventSequence(object):
	def __init__(self, parameters, events, index = 0, run_start_time = 0.0):
		self.parameters = parameters
		self.events = events
		self.run_start_time = run_start_time
		self.index = index
	
	def convert_events(self):
		"""convert the string-based event-array from the pickle file 
		to a set of lists that we can work with 
		and are easily interpretable."""
		rec_button = re.compile('trial %i event (\S+) at (-?\d+\.?\d*)' % self.index)
		self.button_events = [re.findall(rec_button, e)[0] for e in self.events if 'trial %i event' % self.index in e]
		self.button_events = [[b[0], float(b[1]) - self.run_start_time] for b in self.button_events]
		self.button_events = [be for be in self.button_events if (be[1] > 0) and ((be[0] == 'b') or (be[0] == 'y'))]
		
		rec_phase = re.compile('trial %d phase (\d+) started at (-?\d+\.?\d*)' % self.index)
		self.phase_events = [re.findall(rec_phase, e)[0] for e in self.events if 'trial %d phase'% self.index in e]
		self.phase_events = [[p[0], float(p[1]) - self.run_start_time] for p in self.phase_events]
		

class MB_PRFOperator(PopulationReceptiveFieldBehaviorOperator):
	def __init__(self, inputObject, **kwargs):
		"""docstring for __init__"""
		super(PopulationReceptiveFieldBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		with open( self.inputFileName ) as f:
			file_data = pickle.load(f)
		self.events = file_data['eventArray']
		self.parameters = file_data['parameterArray']
		
		run_start_time_string = [e for e in self.events[0] if e[:len('trial 0 phase 1')] == 'trial 0 phase 1']
		self.run_start_time = float(run_start_time_string[0].split(' ')[-1])
	
	def convert_events(self):
		self.trials = []
		for p, e, i in zip(self.parameters, self.events, range(len(self.parameters))):
			tes = TrialEventSequence(p, e, i, self.run_start_time)
			tes.convert_events()
			self.trials.append(tes)
		
	def trial_times(self):
		if not hasattr(self, 'trials'):
			self.convert_events()
		self.trial_times = []
		self.all_button_times = []
		for i, t in enumerate(self.trials):
			# shell()
			stim_on_time = [p[1] for p in t.phase_events if p[0] == '2']
			stim_off_time = [p[1] for p in t.phase_events if p[0] == '3' ]
			if len(stim_off_time) != 1:
				continue
			self.trial_times.append([self.parameters[i]['task'], stim_on_time[0], stim_off_time[0]])
			self.all_button_times.append([[bt[1]] for bt in t.button_events])
		# self.all_button_times = np.concatenate(self.all_button_times)
		
		

class MB_PRFSession(PopulationReceptiveFieldMappingSession):
	"""
	Class for population receptive field mapping sessions analysis.
	"""
	def transplant_headers_to_MB_niftis(self):
		"""docstring for transplant_headers_to_MB_niftis"""
		MB_conditions = [c for c in self.conditionDict.keys() if 'MB' in c]
		first_non_MB_condition = [c for c in self.conditionDict.keys() if 'MB' not in c][0]
		standard_file = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[first_non_MB_condition][0]], postFix = ['mcf']) )
		self.logger.info('using %s as a standard image header for header transplant to MB files' % self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[first_non_MB_condition][0]], postFix = ['mcf']))
		for condition in MB_conditions:
			for r in [self.runList[i] for i in self.conditionDict[condition]]:
				self.logger.info('transplanting header to %s' % self.runFile(stage = 'processed/mri', run = r))
				this_MB_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r) )
				os.system('cp %s %s'%(self.runFile(stage = 'processed/mri', run = r), self.runFile(stage = 'processed/mri', run = r, postFix = ['noheader']) ))
				this_MB_file.header = standard_file.header
				this_MB_file.rtime = standard_file.rtime / r.MB_factor
				this_MB_file.save(self.runFile(stage = 'processed/mri', run = r))
		
	def stimulus_timings(self):
		"""stimulus_timings uses behavior operators to distil:
		- the times at which stimulus presentation began and ended per task type
		- the times at which the task buttons were pressed. 
		"""
		for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:
			bO = MB_PRFOperator(self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' ))
			bO.trial_times() # sets up all behavior  
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'] ))
			duration = nii_file.rtime * nii_file.timepoints
			# correct for the scan stopping before the behavior
			valid_trials = np.arange(len(bO.trial_times))[np.array([bO.trial_times[i][2] for i in range(len(bO.trial_times))]) < duration]
			bO.trial_times = [bO.trial_times[v] for v in valid_trials]
			bO.all_button_times = [bO.all_button_times[v] for v in valid_trials]
			bO.trial_times = [bO.trial_times[v] for v in valid_trials]
			bO.parameters = [bO.parameters[v] for v in valid_trials]
			bO.trials = [bO.trials[v] for v in valid_trials]
				
			r.trial_times = bO.trial_times
			r.all_button_times = bO.all_button_times
			r.parameters = bO.parameters
			r.tasks = [t.parameters['task'] for t in bO.trials]
			r.orientations = [t.parameters['orientation'] for t in bO.trials]
			for i, task in enumerate(np.unique([t.parameters['task'] for t in bO.trials])):
				these_trials = np.array([[tt[1], tt[2] - tt[1], 1.0] for j, tt in enumerate(r.trial_times) if r.tasks[j] == task])
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]), these_trials, fmt = '%3.2f', delimiter = '\t')
				these_buttons = np.concatenate([[[float(b[0]), 0.5, 1.0] for b in bt] for j, bt in enumerate(r.all_button_times) if (r.tasks[j] == task) and (len(bt) > 0)])
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['button', task]), these_buttons, fmt = '%3.2f', delimiter = '\t')
			all_buttons = np.concatenate([[[float(b[0]), 0.5, 1.0] for b in bt] for j, bt in enumerate(r.all_button_times) if (len(bt) > 0)])
			np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['button', 'all']), all_buttons, fmt = '%3.2f', delimiter = '\t')
	
	def create_occipital_region_from_aparc_2009(self, 
			list_of_regions = ['Pole_occipital','S_oc_sup_and_transversal','S_oc_middle_and_Lunatus','S_intrapariet_and_P_trans','G_and_S_occipital_inf', 'G_occipital_middle']):
		"""docstring for create_occipital_region_from_aparc_2009"""
		# make a directory for the subject's occipital region label
		try:
			os.mkdir(os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', 'occip'))
		except OSError:
			pass
		
		for hemi in ['lh', 'rh']:
			cmd = 'mri_mergelabels '
			for reg in list_of_regions:
				cmd += '-i ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', 'aparc.a2009s', hemi + '.' + reg + '.label ')
			cmd += '-o ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', 'occip', '%s.occip.label' % hemi)
			ExecCommandLine(cmd)
			