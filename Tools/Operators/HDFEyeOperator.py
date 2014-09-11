import os, sys, subprocess, re
import tempfile, logging
import pickle
from datetime import *

from math import *
import numpy as np
import numpy.linalg as LA
import matplotlib.pylab as pl
import scipy as sp
from tables import *
import pandas as pd

from EDFOperator import EDFOperator
from Operator import Operator
from EyeSignalOperator import EyeSignalOperator

from IPython import embed as shell 

class HDFEyeOperator(Operator):
	"""docstring for HDFEyeOperator"""
	def __init__(self, inputObject, **kwargs):
		super(HDFEyeOperator, self).__init__(inputObject = inputObject, **kwargs)
		"""inputObject is the name of the hdf5 file that this operator will create"""
	
	def add_edf_file(self, edf_file_name):
		"""docstring for add_edf_file"""
		self.edf_operator = EDFOperator(edf_file_name)
		# now create all messages
		self.edf_operator.read_all_messages()
		
		# set up blocks for the floating-point data
		self.edf_operator.take_gaze_data_for_blocks()
	
	def open_hdf_file(self, mode = "a"):
		"""docstring for open_hdf_file"""
		self.h5f = open_file(self.inputObject, mode = mode )
	
	def close_hdf_file(self):
		"""docstring for close_hdf_file"""
		self.h5f.close()
	
	def add_table_to_hdf(self, run_group, type_dict, data, name = 'bla'):
		"""
		add_table_to_hdf adds a data table to the hdf file
		"""
		this_table = self.h5f.createTable(run_group, name, type_dict, '%s in file %s' % (name, self.edf_operator.inputFileName))
		
		row = this_table.row
		for r in data:
			for par in r.keys():
				row[par] = r[par]
			row.append()
		this_table.flush()
	
	def edf_message_data_to_hdf(self, alias = None, mode = 'a'):
		"""docstring for edf_message_data_to_hdf"""
		if not hasattr(self, 'edf_operator'):
			self.add_edf_file(edf_file_name = alias)
		
		if alias == None:
			alias = os.path.split(self.edf_operator.inputFileName)[-1]
		self.open_hdf_file( mode = mode )
		self.logger.info('Adding message data from %s to group  %s to %s' % (os.path.split(self.edf_operator.inputFileName)[-1], alias, self.inputObject))
		thisRunGroup = self.h5f.create_group("/", alias, 'Run ' + alias +' imported from ' + self.edf_operator.inputFileName)
		
		#
		#	trials and trial phases
		#
		
		if hasattr(self.edf_operator, 'trials'):
			# create a table for the parameters of this run's trials
			self.add_table_to_hdf(thisRunGroup, self.edf_operator.trial_type_dictionary, self.edf_operator.trials, 'trials')
		
		if hasattr(self.edf_operator, 'trial_phases'):
			# create a table for the parameters of this run's trials
			self.add_table_to_hdf(thisRunGroup, self.edf_operator.trial_phase_type_dictionary, self.edf_operator.trial_phases, 'trial_phases')
		
		#
		#	supporting data types
		#
		
		if hasattr(self.edf_operator, 'parameters'):
			# create a table for the parameters of this run's trials
			self.add_table_to_hdf(thisRunGroup, self.edf_operator.parameter_type_dictionary, self.edf_operator.parameters, 'parameters')
		
		if hasattr(self.edf_operator, 'events'):
			# create a table for the events of this run's trials
			self.add_table_to_hdf(thisRunGroup, self.edf_operator.event_type_dictionary, self.edf_operator.events, 'events')
		
		if hasattr(self.edf_operator, 'sounds'):
			# create a table for the events of this run's trials
			self.add_table_to_hdf(thisRunGroup, self.edf_operator.sound_type_dictionary, self.edf_operator.sounds, 'sounds')
		
		#
		#	eyelink data types
		#
		
		if hasattr(self.edf_operator, 'saccades_from_message_file'):
			# create a table for the saccades from the eyelink of this run's trials
			self.add_table_to_hdf(thisRunGroup, self.edf_operator.saccade_type_dictionary, self.edf_operator.saccades_from_message_file, 'saccades_from_message_file')
			self.add_table_to_hdf(thisRunGroup, self.edf_operator.blink_type_dictionary, self.edf_operator.blinks_from_message_file, 'blinks_from_message_file')
			self.add_table_to_hdf(thisRunGroup, self.edf_operator.fixation_type_dictionary, self.edf_operator.fixations_from_message_file, 'fixations_from_message_file')
		
		# first close the hdf5 file to write to it with pandas
		self.close_hdf_file()
	
	def edf_gaze_data_to_hdf(self, alias = None, which_eye = 0):
		"""docstring for edf_gaze_data_to_hdf"""
		
		if not hasattr(self, 'edf_operator'):
			self.add_edf_file(edf_file_name = alias)
		
		if alias == None:
			alias = os.path.split(self.edf_operator.inputFileName)[-1]
		self.logger.info('Adding gaze data from %s to group  %s to %s' % (os.path.split(self.edf_operator.inputFileName)[-1], alias, self.inputObject))
		
		#
		#	gaze data in blocks
		#
		with pd.get_store(self.inputObject) as h5_file:
			# recreate the non-gaze data for the block, that is, its sampling rate, eye of origin etc.
			blocks_data_frame = pd.DataFrame([dict([[i,self.edf_operator.blocks[j][i]] for i in self.edf_operator.blocks[0].keys() if i not in ('block_data', 'data_columns')]) for j in range(len(self.edf_operator.blocks))])
			h5_file.put("/%s/blocks"%alias, blocks_data_frame)
			
			# gaze data per block
			if not 'block_data' in self.edf_operator.blocks[0].keys():
				self.edf_operator.take_gaze_data_for_blocks()
			for i, block in enumerate(self.edf_operator.blocks):
				bdf = pd.DataFrame(block['block_data'], columns = block['data_columns'])
				
				#
				# preprocess pupil:
				#

				for eye in blocks_data_frame.eye_recorded[i]: # this is a string with one or two letters, 'L', 'R' or 'LR'
				# create dictionairy of data per block:
					gazeXY = bdf[[s%'gaze' for s in [eye+'_%s_x', eye+'_%s_y',]]]
					pupil = bdf[[s%'pupil' for s in [eye+'_%s']]]
					eye_dict = {'timepoints':bdf.time, 'gazeXY':gazeXY, 'pupil':pupil,}
					
					# create instance of class EyeSignalOperator, and include the blink data as detected by the Eyelink 1000:
					if hasattr(self.edf_operator, 'blinks_from_message_file'):
						blink_dict = self.read_session_data(alias, 'blinks_from_message_file')
						blink_dict[blink_dict['eye'] == eye]
						eso = EyeSignalOperator(inputObject=eye_dict, eyelink_blink_data=blink_dict)
					else:
						eso = EyeSignalOperator(inputObject=eye_dict)
					# detect blinks (coalese period in samples):
					eso.blink_detection_pupil(coalesce_period=250)
					# interpolate blinks:
					eso.interpolate_blinks(method='linear')
					# low-pass and band-pass pupil data:
					eso.filter_pupil(hp=0.01, lp=6.0)
					# z-score filtered pupil data:
					eso.zscore_pupil()
					# add to existing dataframe:
					bdf[eye+'_pupil_int'] = eso.interpolated_pupil
					bdf[eye+'_pupil_lp'] = eso.lp_filt_pupil
					bdf[eye+'_pupil_bp'] = eso.bp_filt_pupil
					bdf[eye+'_pupil_hp'] = eso.hp_filt_pupil
					
					bdf[eye+'_gaze_x_int'] = eso.interpolated_x
					bdf[eye+'_gaze_y_int'] = eso.interpolated_y
					
				# put in HDF5:
				h5_file.put("/%s/block_%i"%(alias, i), bdf)
	
	def data_frame_to_hdf(self, alias, name, data_frame):
		"""docstring for data_frame_to_hdf"""
		with pd.get_store(self.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(alias, name), data_frame)
	
	#
	#	we also have to get the data from the hdf5 file. 
	#	first, based on simply a EL timestamp period
	#
	
	def sample_in_block(self, sample, block_table):
		"""docstring for sample_in_block"""
		return np.arange(block_table['block_end_timestamp'].shape[0])[np.array(block_table['block_end_timestamp'] > float(sample), dtype=bool)][0]
	
	def data_from_time_period(self, time_period, alias, columns = None):
		"""data_from_time_period delivers a set of data of type data_type for a given timeperiod"""
		# find the block in which the data resides, based on just the first time of time_period
		with pd.get_store(self.inputObject) as h5_file:
			period_block_nr = self.sample_in_block(sample = time_period[0], block_table = h5_file['%s/blocks'%alias]) 
			table = h5_file['%s/block_%i'%(alias, period_block_nr)]
			if columns == None:
				columns = table.keys()
		if 'L_vel' in columns:
			columns = table.keys()
		return table[(table['time'] > float(time_period[0])) & (table['time'] < float(time_period[1]))][columns]
	
	def eye_during_period(self, time_period, alias):
		"""docstring for eye_during_period"""
		with pd.get_store(self.inputObject) as h5_file:
			period_block_nr = self.sample_in_block(sample = time_period[0], block_table = h5_file['%s/blocks'%alias])
			return h5_file['%s/blocks'%alias]['eye_recorded'][period_block_nr]
	
	def eye_during_trial(self, trial_nr, alias):
		"""docstring for signal_from_trial"""
		with pd.get_store(self.inputObject) as h5_file:
			table = h5_file['%s/trials'%alias]
			time_period = np.array(table[table['trial_start_index'] == trial_nr][['trial_start_EL_timestamp', 'trial_end_EL_timestamp']])
		return self.eye_during_period(time_period[0], alias)

	def screen_dimensions_during_period(self, time_period, alias):
		"""docstring for eye_during_period"""
		with pd.get_store(self.inputObject) as h5_file:
			period_block_nr = self.sample_in_block(sample = time_period[0], block_table = h5_file['%s/blocks'%alias])
			return np.array(h5_file['%s/blocks'%alias][['screen_x_pix','screen_y_pix']][period_block_nr:period_block_nr+1]).squeeze()
	
	def sample_rate_during_period(self, time_period, alias):
		"""docstring for eye_during_period"""
		with pd.get_store(self.inputObject) as h5_file:
			period_block_nr = self.sample_in_block(sample = time_period[0], block_table = h5_file['%s/blocks'%alias])
			return h5_file['%s/blocks'%alias]['sample_rate'][period_block_nr]
	
	def sample_rate_during_trial(self, trial_nr, alias):
		"""docstring for signal_from_trial"""
		with pd.get_store(self.inputObject) as h5_file:
			table = h5_file['%s/trials'%alias]
			time_period = np.array(table[table['trial_start_index'] == trial_nr][['trial_start_EL_timestamp', 'trial_end_EL_timestamp']])
		return float(self.sample_rate_during_period(time_period[0], alias))
	
	def signal_during_period(self, time_period, alias, signal, requested_eye = 'L'):
		"""docstring for gaze_during_period"""
		recorded_eye = self.eye_during_period(time_period, alias)
		if requested_eye == 'LR' and recorded_eye == 'LR':
			if signal == 'gaze':
				columns = [s%signal for s in ['L_%s_x', 'L_%s_y', 'R_%s_x', 'R_%s_y']]
			elif signal == 'time':
				columns = [s%signal for s in ['%s']]
			else:
				columns = [s%signal for s in ['L_%s', 'R_%s']]
		elif requested_eye in recorded_eye:
			if signal == 'gaze':
				columns = [s%signal for s in [requested_eye + '_%s_x', requested_eye + '_%s_y']]
			elif signal == 'time':
				columns = [s%signal for s in ['%s']]
			else:
				columns = [s%signal for s in [requested_eye + '_%s']]
		else:
			with pd.get_store(self.inputObject) as h5_file:
				self.logger.error('requested eye %s not found in block %i' % (requested_eye, self.sample_in_block(time_period[0], block_table = h5_file['%s/blocks'%alias])))
			return None	# assert something, dammit!
		return self.data_from_time_period(time_period, alias, columns)
	
	
	#
	#	second, based also on trials, using the above functionality
	#
	
	def signal_from_trial(self, trial_nr, alias, signal, requested_eye = 'L', time_extensions = [0,0]):
		"""docstring for signal_from_trial"""
		with pd.get_store(self.inputObject) as h5_file:
			table = h5_file['%s/trials'%alias]
			time_period = np.array([
				table[table['trial_start_index'] == trial_nr]['trial_start_EL_timestamp'] + time_extensions[0],
				table[table['trial_start_index'] == trial_nr]['trial_end_EL_timestamp'] + time_extensions[1]
			]).squeeze()
		return self.signal_during_period(time_period, alias, signal, requested_eye = requested_eye)
	
	def signal_from_trial_phases(self, trial_nr, trial_phases, alias, signal, requested_eye = 'L', time_extensions = [0,0]):
		"""docstring for signal_from_trial"""
		with pd.get_store(self.inputObject) as h5_file:
			table = h5_file['%s/trial_phases'%alias]
			# check whether one of the trial phases is the end or the beginning of the trial.
			# if so, then supplant the time of that phase with its trial's end or start time.
			if trial_phases[0] == 0:
				start_time = table[table['trial_start_index'] == trial_nr]['trial_start_EL_timestamp']
			else:
				start_time = table[((table['trial_phase_index'] == trial_phases[0]) * (table['trial_phase_trial'] == trial_nr))]['trial_phase_EL_timestamp']
			if trial_phases[-1] == -1:
				end_time = table[table['trial_start_index'] == trial_nr]['trial_end_EL_timestamp']
			else:
				end_time = table[((table['trial_phase_index'] == trial_phases[1]) * (table['trial_phase_trial'] == trial_nr))]['trial_phase_EL_timestamp']
			time_period = np.array([np.array(start_time) + np.array(time_extensions)[0], np.array(end_time) + np.array(time_extensions)[1]]).squeeze()
		return self.signal_during_period(time_period, alias, signal, requested_eye = requested_eye)
	
	def saccades_from_trial_phases(self, trial_nr, trial_phases, alias, requested_eye = 'L', time_extensions = [0,0]):
		with pd.get_store(self.inputObject) as h5_file:
			table = h5_file['%s/trial_phases'%alias]
			if trial_phases[0] == 0:
				start_time = table[table['trial_start_index'] == trial_nr]['trial_start_EL_timestamp']
			else:
				start_time = table[((table['trial_phase_index'] == trial_phases[0]) * (table['trial_phase_trial'] == trial_nr))]['trial_phase_EL_timestamp']
			if trial_phases[-1] == -1:
				end_time = table[table['trial_start_index'] == trial_nr]['trial_end_EL_timestamp']
			else:
				end_time = table[((table['trial_phase_index'] == trial_phases[1]) * (table['trial_phase_trial'] == trial_nr))]['trial_phase_EL_timestamp']
			time_period = np.array([start_time + time_extensions[0], end_time + time_extensions[1]]).squeeze()
			
	#
	#	read whole dataframes
	#
	
	def read_session_data(self, alias, name):
		with pd.get_store(self.inputObject) as h5_file:
			session_data = h5_file['%s/%s'%(alias, name)]
		return session_data
