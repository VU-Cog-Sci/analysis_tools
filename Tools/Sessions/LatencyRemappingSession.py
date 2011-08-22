#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import * 
from ..circularTools import *
import matplotlib.cm as cm
from pylab import *

class LatencyRemappingSession(Session):
	def saccade_latency_analysis_all_runs(self):
		self.mapper_saccade_data = []
		for r in self.runList(self.conditionDict['Mapper']):
			self.mapper_saccade_data.append(saccade_latency_analysis_one_run(r))
		self.remapping_saccade_data = []
		for r in self.runList(self.conditionDict['Remapping']):
			self.remapping_saccade_data.append(saccade_latency_analysis_one_run(r))
	
	def parameter_analysis_all_runs(self):
		self.mapper_parameter_data = []
		for r in self.runList(self.conditionDict['Mapper']):
			self.mapper_parameter_data.append(parameter_analysis_one_run(r))
		self.remapping_saccade_data = []
		for r in self.runList(self.conditionDict['Remapping']):
			self.remapping_parameter_data.append(parameter_analysis_one_run(r))
		
	def saccade_latency_analysis_one_run(self, run):
		elO = EyeLinkOperator(self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'))
		el_saccades = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'saccades')
		
		
		
		
	def parameter_analysis_one_run(self, run):
		elO = EyeLinkOperator(self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'))
		elO.import_parameters(run_name = 'bla')
		