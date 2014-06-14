#!/usr/bin/env python
# encoding: utf-8
"""
EDFOperator.py

Created by Tomas Knapen on 2014-2-19.
Copyright (c) 2010 VU. All rights reserved.
"""

import os, sys, subprocess, re
import tempfile, logging
import pickle
from datetime import *

from math import *
import numpy as np
import matplotlib.pylab as pl
import scipy as sp

from itertools import chain

from CommandLineOperator import EDF2ASCOperator
from Operator import Operator

from IPython import embed as shell

class EDFOperator( Operator ):
	"""docstring for EDFOperator"""
	def __init__(self, inputObject, **kwargs):
		super(EDFOperator, self).__init__(inputObject = inputObject, **kwargs)
		if self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
		self.logger.info('started with ' +os.path.split(self.inputFileName)[-1])
		
		# ** DATE: Tue Feb  4 10:19:06 2014
		if os.path.splitext(self.inputObject)[-1] == '.edf':
			self.inputFileName = self.inputObject
			# in Kwargs there's a variable that we can set to 
			eac = EDF2ASCOperator(self.inputFileName)
			eac.configure()
			self.message_file = eac.messageOutputFileName
			self.gaze_file = eac.gazeOutputFileName
			if not os.path.isfile(eac.messageOutputFileName):
				eac.execute()
		else:
			self.logger.error('input file has to be of edf type.')
	
	def clean_gaze_information(self):
		"""clean_gaze_information takes out non-numeric signs from the gaze data file."""
		self.logger.info('cleaning gaze information from %s'%self.gaze_file)
		if os.path.isfile(self.gaze_file + '.gz'):
			self.logger.error('gaze information already cleaned')
			return
		
		with open(self.gaze_file, 'r') as f:
			gaze_string = f.read()
		
		# optimize this so that it doesn't delete the periods in the float time, for example.
		# first clean out those C and R occurrences. No letters allowed.
		gaze_string = re.sub(re.compile('[A-Z]*'), '', gaze_string)
		gaze_string = re.sub(re.compile('\t+\.+'), '', gaze_string)
		# # check for these really weird character shit in the final columns of the output.
		# self.workingStringClean = re.sub(re.compile('C.'), '', self.workingStringClean)
		
		# clean the older gaze file up a bit by compressing it
		os.system('gzip "' + self.gaze_file + '"')
		
		of = open(self.gaze_file, 'w')
		of.write(gaze_string)
		of.close()
	
	def get_message_string(self):
		"""message_data loads the message data into an internal string variable"""
		if not hasattr(self, 'message_string'):
			with open(self.message_file, 'r') as mfd:
				self.message_string = mfd.read()
	
	def read_session_information(self):
		"""
		read_session_information takes the message file and reads the information pertaining to this session.
		this information is then stored in this operator's internal variables.
		"""
		self.logger.info('reading session information from %s'%self.message_file)
		
		with open(self.message_file, 'r') as mfd:
			self.header = ''.join([mfd.next() for x in xrange(10)])
		date_string = ' '.join([h for h in self.header.split('\n')[1].split(' ') if h not in ('', '**', 'DATE:')][1:])
		self.recording_datetime = datetime.strptime(date_string, "%b %d %H:%M:%S %Y") # %a 
		
		self.logger.info(self.header)
		
	
	def identify_blocks(self, minimal_time_gap = 50.0):
		"""
		identify separate recording blocks in eyelink file
		
		identify_blocks looks into the gaze file and searches for discontinuities in the sample times of at least minimal_time_gap milliseconds. 
		The message file is used to look at the recorded eyes and samplerate per discontinuous block.
		The resulting internal variable is blocks, which is a dictionary for each of the encountered blocks.
		"""
		self.logger.info('identifying recording blocks from %s'%self.gaze_file)
		# check whether the gaze data has been cleaned up
		if not os.path.isfile(self.gaze_file + '.gz'):
			self.logger.error('gaze information not cleaned before running block identification')
			self.clean_gaze_information()
			pass
		
		gaze_times = np.loadtxt(self.gaze_file, usecols = (0,))
		block_edge_indices = np.array([np.arange(gaze_times.shape[0])[np.roll(np.r_[True,np.diff(gaze_times) > minimal_time_gap], 0)], 										
			np.arange(gaze_times.shape[0])[np.roll(np.r_[True,np.diff(gaze_times) > minimal_time_gap], -1)]]).T
		block_edge_times = [gaze_times[i] for i in block_edge_indices]
		
		self.get_message_string()
		sample_re = 'MSG\t[\d\.]+\t!MODE RECORD CR (\d+) \d+ \d+ (\S+)'
		block_sample_occurrences = re.findall(re.compile(sample_re), self.message_string)
		
		screen_re = 'MSG\t[\d\.]+\tGAZE_COORDS (\d+.\d+) (\d+.\d+) (\d+.\d+) (\d+.\d+)'
		block_screen_occurrences = re.findall(re.compile(screen_re), self.message_string)
		
		self.blocks = [{'block_start_index': h[0], 'block_end_index': h[1], 
						'block_start_timestamp': i[0], 'block_end_timestamp': i[1], 
						'sample_rate': int(j[0]), 'eye_recorded': j[1], 
						'screen_x_pix': float(k[2])-float(k[0]), 'screen_y_pix': float(k[3])-float(k[1]),  }
					for h,i,j,k in zip(block_edge_indices, block_edge_times, block_sample_occurrences, block_screen_occurrences)]
		
		# now, we know what the different columns must mean per block, so we can create them
		for bl in self.blocks:
			if bl['eye_recorded'] == 'LR':
				bl.update({'data_columns': ['time','L_gaze_x','L_gaze_y','L_pupil','R_gaze_x','R_gaze_y','R_pupil','L_vel_x','L_vel_y','R_vel_x','R_vel_y']})
			elif bl['eye_recorded'] == 'R':
				bl.update({'data_columns': ['time','R_gaze_x','R_gaze_y','R_pupil','R_vel_x','R_vel_y']})
			elif bl['eye_recorded'] == 'L':
				bl.update({'data_columns': ['time','L_gaze_x','L_gaze_y','L_pupil','L_vel_x','L_vel_y']})
		
		self.logger.info('%i blocks discovered' % len(self.blocks))
		
	
	def read_all_messages(self):
		"""
		read_messages reads the messages sent to the eyelink by the experiment computer.
		
		it does this with all of the different types of information that can be listed in a message file.
		read_all_messages works independently of the identify_blocks method.
		"""
		self.get_message_string()
		self.read_trials()
		self.read_key_events()
		self.read_eyelink_events()
		self.read_sound_events()
	
	def read_trials(self, 
		start_re = 'MSG\t([\d\.]+)\ttrial (\d+) started at (\d+.\d)', 
		stop_re = 'MSG\t([\d\.]+)\ttrial (\d+) stopped at (\d+.\d)', 
		phase_re = 'MSG\t([\d\.]+)\ttrial X phase (\d+) started at (\d+.\d)',
		parameter_re = 'MSG\t[\d\.]+\ttrial X parameter[\t ]*(\S*?)\s+: ([-\d\.]*|[\w]*)'):
		
		"""read_trials reads in trials from the message file, constructing timings and parameters for each of the trials and their phases. """
		self.logger.info('reading trials from %s', os.path.split(self.message_file)[-1])
		self.get_message_string()
		
		#
		# read the trials themselves
		#
		self.start_trial_strings = re.findall(re.compile(start_re), self.message_string)
		self.stop_trial_strings = re.findall(re.compile(stop_re), self.message_string)
		
		if len(self.start_trial_strings) > 0:	# check whether there are any trials here. 
			self.trial_starts = np.array([[float(s[0]), int(s[1]), float(s[2])] for s in self.start_trial_strings])
			self.trial_ends = np.array([[float(s[0]), int(s[1]), float(s[2])] for s in self.stop_trial_strings])
			
			if len(self.trial_starts) != len(self.trial_ends):
				self.trial_ends = self.trial_ends[::2]
			
			self.nr_trials = len(self.stop_trial_strings)
			self.trials = np.hstack((self.trial_starts, self.trial_ends))
			
			# create a dictionary for the types of timing informations we'd like to look at
			self.trial_type_dictionary = [('trial_start_EL_timestamp', np.float64), ('trial_start_index',np.int32), ('trial_start_exp_timestamp',np.float64), ('trial_end_EL_timestamp',np.float64), ('trial_end_index',np.int32), ('trial_end_exp_timestamp',np.float64)]
			
			self.trials = [{'trial_start_EL_timestamp': tr[0], 'trial_start_index': tr[1], 'trial_start_exp_timestamp': tr[2], 'trial_end_EL_timestamp': tr[3], 'trial_end_index': tr[4], 'trial_end_exp_timestamp': tr[5]} for tr in self.trials]
			
			self.trial_type_dictionary = np.dtype(self.trial_type_dictionary)
			#
			# trial phases 
			#
			self.trial_phases = []
			for i in range(self.nr_trials):
				this_trial_re = phase_re.replace(' X ', ' ' + str(i) + ' ')
				phase_strings = re.findall(re.compile(this_trial_re), self.message_string)
				self.trial_phases.append([[int(i), float(s[0]), int(s[1]), float(s[2])] for s in phase_strings])
			self.trial_phases = list(chain.from_iterable(self.trial_phases))
			self.trial_phases = [{'trial_phase_trial': tr[0], 'trial_phase_EL_timestamp': tr[1], 'trial_phase_index': tr[2], 'trial_phase_exp_timestamp': tr[3]} for tr in self.trial_phases]
			self.nr_trial_phases = len(self.trial_phases)
			
			self.trial_phase_type_dictionary = [('trial_phase_trial', np.float64), ('trial_phase_EL_timestamp',np.int32), ('trial_phase_index',np.float64), ('trial_phase_exp_timestamp',np.float64)]
			self.trial_phase_type_dictionary = np.dtype(self.trial_phase_type_dictionary)
			
			# now adjust the trial type dictionary and convert into a numpy dtype
			# self.trial_type_dictionary.append(('trial_phase_timestamps', np.float64, (self.nr_phase_starts.max(), 3)))
		else:
			self.logger.info('no trial or phase information in edf file %s'%self.inputFileName)
			self.nr_trials = 0
		
		#
		# parameters 
		#
		parameters = []
		for i in range(self.nr_trials):
			this_re = parameter_re.replace(' X ', ' ' + str(i) + ' ')
			parameter_strings = re.findall(re.compile(this_re), self.message_string)
			if len(parameter_strings) > 0:
				# assuming all these parameters are numeric
				this_trial_parameters = {'trial_nr': float(i)}
				for s in parameter_strings:
					try:
						this_trial_parameters.update({s[0]: float(s[1])})
					except ValueError:
						pass
				parameters.append(this_trial_parameters)
		
		if len(parameters) > 0:		# there were parameters in the edf file
			self.parameters = parameters
			
			ptd = [(k, np.float64) for k in np.unique(np.concatenate([k.keys() for k in self.parameters]))]
			self.parameter_type_dictionary = np.dtype(ptd)
		else: # we have to take the parameters from the output_dict pickle file of the same name as the edf file. 
			self.logger.info('no parameter information in edf file')
			
	
	def read_key_events(self, 
		key_re = 'MSG\t([\d\.]+)\ttrial X event \<Event\((\d)-Key(\S*?) {\'scancode\': (\d+), \'key\': (\d+)(, \'unicode\': u\'\S*?\',|,) \'mod\': (\d+)}\)\> at (\d+.\d)'):
		"""read_key_events reads experimental events from the message file"""
		self.logger.info('reading key_events from %s', os.path.split(self.message_file)[-1])
		self.get_message_string()
		if not hasattr(self, 'nr_trials'):
			self.read_trials()
		
		events = []
		this_length = 0
		for i in range(self.nr_trials):
			this_key_re = key_re.replace(' X ', ' ' + str(i) + ' ')
			event_strings = re.findall(re.compile(this_key_re), self.message_string)
			if len(event_strings) > 0:
				if len(event_strings[0]) == 8:
					events.append([{'EL_timestamp':float(e[0]),'event_type':int(e[1]),'up_down':e[2],'scancode':int(e[3]),'key':int(e[4]),'modifier':int(e[6]), 'exp_timestamp':float(e[7])} for e in event_strings])
					this_length = 8
				elif len(event_strings[0]) == 3:
					events.append([{'EL_timestamp':float(e[0]),'event_type':int(e[1]), 'exp_timestamp':float(e[2])} for e in event_strings])
					this_length = 3
		if len(events) > 0:
			self.events = list(chain.from_iterable(events))
			#
			# add types to eventTypeDictionary that specify the relevant trial and time in trial for this event - per run.
			#
			if this_length == 8:
				self.event_type_dictionary = np.dtype([('EL_timestamp', np.float64), ('event_type', np.float64), ('up_down', '|S25'), ('scancode', np.float64), ('key', np.float64), ('modifier', np.float64), ('exp_timestamp', np.float64)])
			elif this_length == 3:
				self.event_type_dictionary = np.dtype([('EL_timestamp', np.float64), ('event_type', np.float64), ('exp_timestamp', np.float64)])
	
	def read_eyelink_events(self,
		sacc_re = 'ESACC\t(\S+)[\s\t]+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+.?\d+)', 
		fix_re = 'EFIX\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?', 
		blink_re = 'EBLINK\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d?.?\d*)?'):
		"""
		read_key_events reads experimental events from the message file. 
		
		Examples:
		ESACC	R	2347313	2347487	174	  621.8	  472.4	  662.0	  479.0	   0.99	 
		EFIX	R	2340362.0	2347312.0	6950	  650.0	  480.4	   5377
		EBLINK	R	2347352	2347423	71
		"""
		self.logger.info('reading eyelink events from %s', os.path.split(self.message_file)[-1])
		self.get_message_string()
		saccade_strings = re.findall(re.compile(sacc_re), self.message_string)
		fix_strings = re.findall(re.compile(fix_re), self.message_string)
		blink_strings = re.findall(re.compile(blink_re), self.message_string)
		
		if len(saccade_strings) > 0:
			self.saccades_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3]),'start_x':float(e[4]),'start_y':float(e[5]),'end_x':float(e[6]),'end_y':float(e[7]), 'peak_velocity':float(e[7])} for e in saccade_strings]
			self.fixations_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3]),'x':float(e[4]),'y':float(e[5]),'pupil_size':float(e[6])} for e in fix_strings]
			self.blinks_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3])} for e in blink_strings]
		
			self.saccade_type_dictionary = np.dtype([(s , np.array(self.saccades_from_message_file[0][s]).dtype) for s in self.saccades_from_message_file[0].keys()])
			self.fixation_type_dictionary = np.dtype([(s , np.array(self.fixations_from_message_file[0][s]).dtype) for s in self.fixations_from_message_file[0].keys()])
			if len(self.blinks_from_message_file) > 0:
				self.blink_type_dictionary = np.dtype([(s , np.array(self.blinks_from_message_file[0][s]).dtype) for s in self.blinks_from_message_file[0].keys()])
	
	def read_sound_events(self, 
		sound_re = 'MSG\t([\d\.]+)\treward (\d+) at (\d+.\d)'):
		"""
		read_sound_events reads sounds from the message file.
		
		Example:
		MSG	1885375.0	sound 0 at 29.3498238511
		"""
		self.logger.info('reading sounds from %s', os.path.split(self.message_file)[-1])
		self.get_message_string()
		if not hasattr(self, 'nr_trials'):
			self.read_trials()
		sounds = []
		this_length = 0
			sound_strings = re.findall(re.compile(this_sound_re), self.message_string)
			sounds.append([{'EL_timestamp':float(s[0]),'sound_type':int(s[1]), 'exp_timestamp':float(s[2])} for s in sound_strings])
		self.sounds = list(chain.from_iterable(sounds))
		#
		# add types to eventTypeDictionary that specify the relevant trial and time in trial for this event - per run.
		#
		self.sound_type_dictionary = np.dtype([('EL_timestamp', np.float64), ('sound_type', np.float64), ('exp_timestamp', np.float64)])
	
	def take_gaze_data_for_blocks(self):
		"""docstring for take_gaze_data"""
		# check for gaze data and blocks
		self.clean_gaze_information()
		self.identify_blocks()
		
		with open(self.gaze_file) as gfd:
			txt_data = gfd.readlines()
			float_data = [[float(i) for i in line.split('\t')] for line in txt_data]
			
			for i, block in enumerate(self.blocks):
				block_data = float_data[block['block_start_index']:block['block_end_index']]
				block['block_data'] = np.array(block_data, dtype = np.float32)
				self.logger.info('found data from block %i of shape %s'%(i, str(block['block_data'].shape)))
		
		
	