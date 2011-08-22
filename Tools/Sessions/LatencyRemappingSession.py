#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import * 
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from ..circularTools import *
import matplotlib.cm as cm
from pylab import *
from nifti import *

class LatencyRemappingSession(Session):
	def saccade_latency_analysis_all_runs(self):
		self.mapper_saccade_data = []
		for r in [self.runList[i] for i in self.conditionDict['Mapper']]:
			self.mapper_saccade_data.append(self.saccade_latency_analysis_one_run(r))
		self.remapping_saccade_data = []
		for r in [self.runList[i] for i in self.conditionDict['Remapping']]:
			self.remapping_saccade_data.append(self.saccade_latency_analysis_one_run(r))
	
	def parameter_analysis_all_runs(self):
		self.mapper_parameter_data = []
		for r in [self.runList[i] for i in self.conditionDict['Mapper']]:
			self.mapper_parameter_data.append(self.parameter_analysis_one_run(r))
		self.remapping_parameter_data = []
		for r in [self.runList[i] for i in self.conditionDict['Remapping']]:
			self.remapping_parameter_data.append(self.parameter_analysis_one_run(r))
	
	def saccade_latency_analysis_one_run(self, run):
		elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
		elO.import_parameters(run_name = 'bla')
		el_saccades = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'saccades')
	
	def parameter_analysis_one_run(self, run):
		elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
		elO.import_parameters(run_name = 'bla')
		print run.condition
		if run.condition == 'Mapper':
			####
			####	run retinotopic area mapping. do the main pattern-based GLM internally. 
			####
			trial_types = np.sign((elO.parameter_data[:]['contrast_L'] - elO.parameter_data[:]['contrast_R']) * elO.parameter_data[:]['stim_eccentricity'])
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf','hpf'], extension = '.nii.gz'))
			tr, nrsamples = niiFile.rtime, niiFile.timepoints
			stim_durations = np.ones((trial_types.shape[0])) * 2
			stim_onsets = np.arange(0,nrsamples*tr,tr*2)
			design = np.vstack((stim_onsets, stim_durations, trial_types))
			stim_locations = np.unique(trial_types)
			for i in range(3):
				this_location_design = design[:,design[2] == stim_locations[i]]
				print this_location_design
				this_location_design[2] = 1
				np.savetxt( self.runFile(stage = 'processed/mri', run = run, postFix = ['design',str(i)], extension = '.txt'), this_location_design.T, fmt = '%3.1f', delimiter = '\t')
		
		