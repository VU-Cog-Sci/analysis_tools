#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Sessions import * 
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from ..circularTools import *
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle, re

class TrialEventSequence(object):
	def __init__(self, parameters, events, index = 0, run_start_time = 0.0)
		self.parameters = parameters
		self.events = events
		self.reference_time = reference_time
		self.index = index
	
	def convert_events(self):
		"""convert the string-based event-array from the pickle file 
		to a set of lists that we can work with 
		and are easily interpretable."""
		rec_button = re.compile('trial %i key: (\S+) at time: (-?\d+\.?\d*) for task [\S+]' % self.index)
		self.button_events = [re.findall(rec_button, e)[0] for e in self.events if e[:13] == 'trial %i key:' % self.index]
		self.button_events = [[b[0], float(b[1]) - self.run_start_time] for b in self.button_events]
		
		rec_phase = re.compile('trial %d phase (\d+) started at (-?\d+\.?\d*)' % self.index)
		self.phase_events = [re.findall(rec_phase, e)[0] for e in self.events if e[:5] == 'trial']
		self.phase_events = [[p[0], float(p[1]) - self.run_start_time] for p in self.phase_events]
		
		self.find_task()	# need to know the task this trial to be able to extract.
		rec_signal = re.compile('signal in task (\S+) at (-?\d+\.?\d*) value 1.0')
		self.signal_events = [re.findall(rec_signal, e)[0] for e in self.events if e[:6] == 'signal']
		self.signal_events = [[s[0], float(s[1]) - self.run_start_time] for s in self.signal_events]
		self.task_signal_events = np.array([s[0] == self.task for s in self.signal_events])
		self.task_signal_times = np.array(self.signal_events)[self.task_signal_events,1]
		
	
	def find_task(self):
		"""find the task from parameters, 
		if the 'y' button has been pressed, 
		this means the fixation task."""
		self.task = self.parameters['task']
		
		if np.array([b[0] == 'y' for b in self.button_events]).sum() > 0
			self.task = 'fix'
			self.task_button_event_times = np.array([b[1] for b in self.button_events if b[0] == 'y'])
		else:
			self.task_button_event_times = np.array([b[1] for b in self.button_events if b[0] == 'b'])
	
	def check_answers(self, maximal_reaction_time = 1.0):
		"""take the task signal events and the button presses
		and evaluate which signals were responded to and which were not.
		maximal_reaction_time defines the window in which button presses are counted."""
		
		response_delays = np.array([self.task_button_event_times - s for s in self.task_signal_times])
		response_delays_indices = [rd > 0 for rd in response_delays]
		[True for rdi in response_delays_indices if rdi.sum() > 0 else 0]
		
		
		

class PopulationReceptiveFieldMappingSession(Session):
	"""
	Class for population receptive field mapping sessions analysis.
	"""
	def event_analysis_run(self, run):
		"""docstring for event_analysis_run"""
		with open(self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' )) as f:
			file_data = pickle.load(f)
		run.events = file_data['eventArray']
		run.parameters = file_data['parameterArray']
		
		run_start_time_string = [e for e in run.events[0] if e[:len('trial 0 phase 1')] == 'trial 0 phase 1']
		run.run_start_time = float(expt_start_time_string[0].split(' ')[-1])
		
		trials = []
		for p, e, i in zip(run.parameters, run.events, range(len(run.parameters))):
			tes = TrialEventSequence(p, e, i, run.run_start_time)
			tes.convert_events()
			# do full psychophysics analysis per trial here.
			
	
	# def signal_analysis(self, run):
	# 	"""docstring for signal_analysis"""
	# 	
	# 	signal_events = [[re.findall(rec, e)[0] for e in run.events[i] if e[:6] == 'signal'] for i in range(len(run.events))]
	# 	stim_signal_events = [[[s[0], float(s[1]) - run.run_start_time] for s in signal_events[i]] for i in range(len(signal_events))]
	# 	
	# 	run.target_signals_per_trial = [pt['task'] for pt in run.parameters]
	# 	run.which_signals_are_targets_per_trial = [[np.array(np.array(stim_signal_events[i])[np.array(stim_signal_events[i])[:,0] == np.array(run.target_signals_per_trial[i])][:,1], dtype = float), np.array(np.array(stim_signal_events[i])[np.array(stim_signal_events[i])[:,0] != np.array(run.target_signals_per_trial[i])][:,1], dtype = float)] for i in range(len(stim_signal_events))]
		
		
		
		
		
		
		



	pass