#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Session import * 
from RewardSession import * 
from ...Operators.ArrayOperator import *
from ...Operators.EyeOperator import *
from ...Operators import ArrayOperator, EDFOperator, HDFEyeOperator, EyeSignalOperator
from ...other_scripts.circularTools import *
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle
from scipy.stats import *
import pandas as pd
import os

class SingleRewardSession(RewardSession):
	"""
	Analyses for visual reward sessions
	"""
	def __init__(self, ID, date, project, subject, session_label = 'first', parallelize = True, loggingLevel = logging.DEBUG):
		super(SingleRewardSession, self).__init__(ID, date, project, subject, session_label = session_label, parallelize = parallelize, loggingLevel = loggingLevel)
	
	def create_feat_event_files_one_run(self, run, minimum_blink_duration = 0.01):
		"""
		creates feat analysis event files for reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		if not hasattr(self, 'dual_pilot'):
			if run.condition == 'reward':
				# get EL Data
				elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
				elO.import_parameters(run_name = 'bla')
				el_blinks = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,255]], trial_phase_range = [0,4], data_type = 'blinks')[0] # just a single list instead of more than one....
				el_blinks = np.concatenate(el_blinks)
			
				# stimulus onsets are the trial phase indexed by 1
				# 'trial' onsets are indexed by 0
				experiment_start_time = (elO.timings['trial_phase_timestamps'][0,0,0] / 1000.0)
			
				# blinks
				blink_times = (el_blinks['start_timestamp'] / 1000.0) - experiment_start_time 
				blinks_during_experiment = blink_times > 0.0
				minimum_blink_duration_indices = (el_blinks['duration'] / 1000.0) > minimum_blink_duration
				run.blink_durations, run.blink_times = (el_blinks['duration'][blinks_during_experiment * minimum_blink_duration_indices] / 1000.0), blink_times[blinks_during_experiment * minimum_blink_duration_indices]
			
				try:
					os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']))
				except OSError:
					pass
				# shell()
				a = np.ones((run.blink_times.shape[0], 3))
				a[:,0] = run.blink_times; a[:,1] = run.blink_durations;
				np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']), a, fmt = '%3.2f', delimiter = '\t')
				
				# stimulus onset thingies
				run.stimulus_onset_times = (elO.timings['trial_phase_timestamps'][:,1,0] / 1000) - experiment_start_time
				# save stimulus_onset_times to separate text file to be used for per-trial glm analyses
				try:
					os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials']))
				except OSError:
					pass
				np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials']), np.array([run.stimulus_onset_times, np.ones((run.stimulus_onset_times.shape[0])), np.ones((run.stimulus_onset_times.shape[0]))]).T, fmt = '%3.2f', delimiter = '\t')
				
				
			
				# trials are separated on 'sound' and 'contrast' parameters, and we parcel in the reward scheme here, since not every subject receives the same reward and zero sounds
				run.sound_trials, run.visual_trials = np.array((self.which_reward + elO.parameter_data['sound']) % 2, dtype = 'bool'), np.array(elO.parameter_data['contrast'], dtype = 'bool')
			
				run.condition_labels = ['visual_sound', 'visual_silence', 'blank_silence', 'blank_sound']
				
				# conditions are made of boolean combinations
				run.visual_sound_trials = run.sound_trials * run.visual_trials
				run.visual_silence_trials = run.visual_trials * (-run.sound_trials)
				run.blank_silence_trials = -(run.visual_trials + run.sound_trials)
				run.blank_sound_trials = (-run.visual_trials) * run.sound_trials
				
				for (cond, label) in zip([run.visual_sound_trials, run.visual_silence_trials, run.blank_silence_trials, run.blank_sound_trials], run.condition_labels):
					try:
						os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]))
					except OSError:
						pass
					np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]), np.array([run.stimulus_onset_times[cond], np.ones((cond.sum())), np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
				pickle.dump(run, file(self.runFile(stage = 'processed/mri', run = run, extension = '.pickle', postFix = ['run']), 'w'))
				
				for (cond, label) in zip([run.sound_trials, run.visual_trials, -run.sound_trials, -run.visual_trials], ['reward', 'stimulus', 'no_reward', 'no_stimulus']):
					try:
						os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]))
					except OSError:
						pass
					np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]), np.array([run.stimulus_onset_times[cond], np.ones((cond.sum())), np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
				
		elif self.dual_pilot == 1:
			# get EL Data, do blink and timings irrespective of mapper or reward runs
			elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
			elO.import_parameters(run_name = 'bla')
			el_blinks = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,255]], trial_phase_range = [0,4], data_type = 'blinks')[0] # just a single list instead of more than one....
			el_blinks = np.concatenate(el_blinks)
			
			# stimulus onsets are the trial phase indexed by 1
			# 'trial' onsets are indexed by 0
			experiment_start_time = (elO.timings['trial_phase_timestamps'][0,0,0] / 1000)
			
			# blinks
			blink_times = (el_blinks['start_timestamp'] / 1000) - experiment_start_time 
			blinks_during_experiment = blink_times > 0.0
			minimum_blink_duration_indices = (el_blinks['duration'] / 1000) > minimum_blink_duration
			blink_durations, blink_times = (el_blinks['duration'][blinks_during_experiment * minimum_blink_duration_indices] / 1000), blink_times[blinks_during_experiment * minimum_blink_duration_indices]
			
			try:
				os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']))
			except OSError:
				pass
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']), np.array([blink_times, blink_durations, np.ones((blink_times.shape[0]))]).T, fmt = '%3.2f', delimiter = '\t')
			
			# stimulus onset thingies
			stimulus_onset_times = (elO.timings['trial_phase_timestamps'][:,1,0] / 1000) - experiment_start_time
			
			if run.condition == 'mapper':
				left_none_right = np.sign(elO.parameter_data['x_position'])
				CW_none_CCW = np.sign(elO.parameter_data['orientation'])
				
				condition_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW', 'left', 'right']
				left_CW_trials = (left_none_right == -1) * (CW_none_CCW == -1)
				left_CCW_trials = (left_none_right == -1) * (CW_none_CCW == 1)
				right_CW_trials = (left_none_right == 1) * (CW_none_CCW == -1)
				right_CCW_trials = (left_none_right == 1) * (CW_none_CCW == 1)
				
				left_trials = left_CW_trials + left_CCW_trials
				right_trials = right_CW_trials + right_CCW_trials
				
				for (cond, label) in zip([left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials, left_trials, right_trials], condition_labels):
					try:
						os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]))
					except OSError:
						pass
					np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]), np.array([stimulus_onset_times[cond], np.ones((cond.sum())), np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
				# make an all_trials txt file
				all_stimulus_trials_sum = np.array([left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials], dtype = bool).sum(axis = 0, dtype = bool)
				try:
					os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials']))
				except OSError:
					pass
				np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials']), np.array([stimulus_onset_times[all_stimulus_trials_sum], np.ones((all_stimulus_trials_sum.sum())), np.ones((all_stimulus_trials_sum.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
			
			
			elif run.condition == 'reward':
				# trials are separated on 'sound' and 'contrast' parameters, and we parcel in the reward scheme here, since not every subject receives the same reward and zero sounds
				left_none_right = np.sign(elO.parameter_data['x_position'])
				CW_none_CCW = np.sign(elO.parameter_data['orientation'])
			
				condition_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW']
				left_CW_trials = (left_none_right == -1) * (CW_none_CCW == -1)
				left_CCW_trials = (left_none_right == -1) * (CW_none_CCW == 1)
				right_CW_trials = (left_none_right == 1) * (CW_none_CCW == -1)
				right_CCW_trials = (left_none_right == 1) * (CW_none_CCW == 1)
				
				all_stimulus_trials = [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials]
				all_reward_trials = np.array(elO.parameter_data['sound'], dtype = bool)
				
				which_trials_rewarded = np.array([(trials * all_reward_trials).sum() > 0 for trials in [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials]])
				which_stimulus_rewarded = np.arange(4)[which_trials_rewarded]
				stim_trials_rewarded = np.squeeze(np.array(all_stimulus_trials)[which_trials_rewarded])
				
				blank_trials_rewarded = all_reward_trials - stim_trials_rewarded
				blank_trials_silence = -np.array(np.abs(left_none_right) + blank_trials_rewarded, dtype = bool)
				
				# identify rewarded condition by label
				# condition_labels[which_stimulus_rewarded] += '_rewarded'
				run.which_stimulus_rewarded = which_stimulus_rewarded
				
				# reorder the conditions
				# all_stimulus_trials.pop(which_stimulus_rewarded)
				# all_stimulus_trials.append(stim_trials_rewarded)
				# reorder the condition labels
				# reward_label = condition_labels.pop(which_stimulus_rewarded)
				# condition_labels.append(reward_label)
				
				condition_labels.extend( ['blank_silence','blank_rewarded'] )
				all_stimulus_trials.extend( [blank_trials_silence, blank_trials_rewarded] )
				
				run.condition_labels = condition_labels
				run.all_stimulus_trials = all_stimulus_trials
				for (cond, label) in zip(all_stimulus_trials, condition_labels):
					try:
						os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]))
					except OSError:
						pass
					np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]), np.array([stimulus_onset_times[cond], np.ones((cond.sum())), np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
				
				# make an all_trials txt file
				all_stimulus_trials_sum = np.array([all_stimulus_trials[i] for i in range(6)], dtype = bool).sum(axis = 0, dtype = bool)
				try:
					os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials']))
				except OSError:
					pass
				np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials']), np.array([stimulus_onset_times[all_stimulus_trials_sum], np.ones((all_stimulus_trials_sum.sum())), np.ones((all_stimulus_trials_sum.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
	
	def feat_reward_analysis(self, version = '', postFix = ['mcf'], run_feat = True, feat_file = 'reward_more_contrasts.fsf', waitForExecute = False):
		"""
		Runs feat analysis for all reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		if not hasattr(self, 'dual_pilot'):
			for r in [self.runList[i] for i in self.conditionDict['reward']]:
				# self.create_feat_event_files_one_run(r)
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks','T']), np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks'])).T, fmt = '%3.2f', delimiter = '\t')
				if run_feat:
					try:
						self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.fsf'))
					except OSError:
						pass
			
					# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
					# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
					if 'sara' in os.uname() or 'aeneas' in os.uname():
						thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward/first/fsf/' + feat_file
					else:
						thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward/first/fsf/' + feat_file
				
					REDict = {
					'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
					'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
					'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks','T']), 	
					'---BLANK_SILENCE_FILE---': self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blank_silence']), 	
					'---BLANK_SOUND_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blank_sound']), 
					'---VISUAL_SILENCE_FILE---':self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['visual_silence']), 	
					'---VISUAL_SOUND_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['visual_sound']), 
					}
					featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.fsf')
					featOp = FEATOperator(inputObject = thisFeatFile)
					# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
					if r == [self.runList[i] for i in self.conditionDict['reward']][-1]:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
					else:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
					self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
					# run feat
					featOp.execute()
			if 'mapper' in self.conditionDict.keys():
				for r in [self.runList[i] for i in self.conditionDict['mapper']]:
					if run_feat:
						try:
							self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
							os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
							os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.fsf'))
						except OSError:
							pass
			
						# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
						# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
						if 'sara' in os.uname():
							thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward/first/fsf/mapper.fsf'
						else:
							thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward/first/fsf/mapper.fsf'
						REDict = {
						'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
						'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
						}
						featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.fsf')
						featOp = FEATOperator(inputObject = thisFeatFile)
						if r == [self.runList[i] for i in self.conditionDict['mapper']][-1]:
							featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
						else:
							featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = waitForExecute )
						self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
						# run feat
						featOp.execute()
		elif self.dual_pilot == 1:
			for r in [self.runList[i] for i in self.conditionDict['reward']]:
				# create_feat_event_files_one_run will create condition_labels and all_stimulus_trials variables for this run in the run object, to be used later on.
				self.create_feat_event_files_one_run(r)
			
				if run_feat:
					feat_post_fix = postFix + [version]
					try:
						self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = feat_post_fix, extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = feat_post_fix, extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = feat_post_fix, extension = '.fsf'))
					except OSError:
						pass
					
					# now segment according to reward condition.
					# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
					# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
					if 'sara' in os.uname():
						thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward/dual/fsf/reward_dual_pilot_orientation_reward.fsf'
					else:
						thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward/dual/fsf/reward_dual_pilot_orientation_reward.fsf'
					REDict = {
					'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
					'---OUTPUT_DIR---': 		self.runFile(stage = 'processed/mri', run = r, postFix = feat_post_fix), 
					'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']), 	
					'---LEFT_CW_FILE---': self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['left_CW']), 
					'---LEFT_CCW_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['left_CCW']), 
					'---RIGHT_CW_FILE---':self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['right_CW']), 
					'---RIGHT_CCW_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['right_CCW']), 
					'---BLANK_REWARD_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blank_rewarded']), 
					'---BLANK_SILENCE_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blank_silence']), 
					}
					# adapt reward contrast to the stimulus which was rewarded
					contrast_values_this_run = -np.ones(4)
					contrast_values_this_run[r.which_stimulus_rewarded] = 2.0
					contrast_keys = ['---L_CW_REWARD_CONTRAST_VALUE---','---L_CCW_REWARD_CONTRAST_VALUE---','---R_CW_REWARD_CONTRAST_VALUE---','---R_CCW_REWARD_CONTRAST_VALUE---']
					for i in range(4):
						REDict.update({contrast_keys[i]: str(contrast_values_this_run[i])})
					# over to the actual feat analysis
					featFileName = self.runFile(stage = 'processed/mri', run = r, postFix = feat_post_fix, extension = '.fsf')
					featOp = FEATOperator(inputObject = thisFeatFile)
					if r == [self.runList[i] for i in self.conditionDict['reward']][-1]:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
					else:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = waitForExecute )
					self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
					# run feat
					featOp.execute()
					
			
			for r in [self.runList[i] for i in self.conditionDict['mapper']]:
				self.create_feat_event_files_one_run(r)
				if run_feat:
					if 'orientation' == version:
						version_postFix = postFix + [version]
						try:
							self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix, extension = '.feat'))
							os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix, extension = '.feat'))
							os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix, extension = '.fsf'))
						
						except OSError:
							pass
			
						# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
						# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
						if 'sara' in os.uname():
							thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_dual_pilot_orientation_mapper.fsf'
						else:
							thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward/dual/fsf/reward_dual_pilot_orientation_mapper.fsf'
						REDict = {
						'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
						'---OUTPUT_DIR---': 		self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix), 
						'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']), 	
						'---LEFT_CW_FILE---': self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['left_CW']), 
						'---LEFT_CCW_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['left_CCW']), 
						'---RIGHT_CW_FILE---':self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['right_CW']), 
						'---RIGHT_CCW_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['right_CCW']), 
						}
						featFileName = self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix, extension = '.fsf')
						featOp = FEATOperator(inputObject = thisFeatFile)
						if r == [self.runList[i] for i in self.conditionDict['mapper']][-1]:
							featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
						else:
							featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = waitForExecute )
						self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
						# run feat
						featOp.execute()
					
					elif 'location' == version:
						version_postFix = postFix + [version]
						try:
							self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix, extension = '.feat'))
							os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix, extension = '.feat'))
							os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix, extension = '.fsf'))
						
						except OSError:
							pass
			
						# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
						# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
						if 'sara' in os.uname():
							thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_dual_pilot_location_mapper.fsf'
						else:
							thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward/dual/fsf/reward_dual_pilot_location_mapper.fsf'
						REDict = {
						'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
						'---OUTPUT_DIR---': 		self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix), 
						'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']), 	
						'---LEFT_FILE---': 			self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['left']), 
						'---RIGHT_FILE---':			self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['right']), 
						}
						featFileName = self.runFile(stage = 'processed/mri', run = r, postFix = version_postFix, extension = '.fsf')
						featOp = FEATOperator(inputObject = thisFeatFile)
						if r == [self.runList[i] for i in self.conditionDict['mapper']][-1]:
							featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
						else:
							featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = waitForExecute )
						self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
						# run feat
						featOp.execute()

			
	
	def project_stats(self, which_file = 'zstat', postFix = ['mcf']):
		if not hasattr(self, 'dual_pilot'):
			for r in [self.runList[i] for i in self.conditionDict['reward']]:
				this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
				visual_results_file = os.path.join(this_feat, 'stats', which_file + '1.nii.gz')
				reward_results_file = os.path.join(this_feat, 'stats', which_file + '2.nii.gz')
				silent_fix_results_file = os.path.join(this_feat, 'stats', which_file + '3.nii.gz')
				reward_fix_results_file = os.path.join(this_feat, 'stats', which_file + '4.nii.gz')
				silent_visual_results_file = os.path.join(this_feat, 'stats', which_file + '5.nii.gz')
				reward_visual_results_file = os.path.join(this_feat, 'stats', which_file + '6.nii.gz')
				fix_reward_silence_results_file = os.path.join(this_feat, 'stats', which_file + '7.nii.gz')
				visual_reward_silence_results_file = os.path.join(this_feat, 'stats', which_file + '8.nii.gz')
				visual_silence_fix_silence_results_file = os.path.join(this_feat, 'stats', which_file + '9.nii.gz')
				visual_reward_fix_reward_results_file = os.path.join(this_feat, 'stats', which_file + '10.nii.gz')
			
				for (label, f) in zip(
										['visual', 'reward', 'fix_silence', 'fix_reward', 'visual_silent', 'visual_reward', 'fix_reward-silence', 'visual_reward-silence', 'visual_silence-fix_silence', 'visual_reward-fix_reward'], 
										[visual_results_file, reward_results_file, silent_fix_results_file, reward_fix_results_file, silent_visual_results_file, reward_visual_results_file, fix_reward_silence_results_file, visual_reward_silence_results_file, visual_silence_fix_silence_results_file, visual_reward_fix_reward_results_file]
										):
					vsO = VolToSurfOperator(inputObject = f)
					ofn = self.runFile(stage = 'processed/mri/', run = r, base = which_file, postFix = [label] )
					ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
					vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute()
			# mappers also have 1 and 2 for stats files
			for r in [self.runList[i] for i in self.conditionDict['mapper']]:
				this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
				center_results_file = os.path.join(this_feat, 'stats', which_file + '1.nii.gz')
				surround_results_file = os.path.join(this_feat, 'stats', which_file + '2.nii.gz')
				center_surround_results_file = os.path.join(this_feat, 'stats', which_file + '3.nii.gz')
				surround_center_results_file = os.path.join(this_feat, 'stats', which_file + '4.nii.gz')
			
			
				for (label, f) in zip(['center', 'surround', 'center_surround', 'surround_center'], [center_results_file, surround_results_file, center_surround_results_file, surround_center_results_file]):
					vsO = VolToSurfOperator(inputObject = f)
					ofn = self.runFile(stage = 'processed/mri/', run = r, base = which_file, postFix = [label] )
					ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
					vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute()
		elif self.dual_pilot == 1:
			for r in [self.runList[i] for i in self.conditionDict['mapper']]:
				this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat') # to look at the locations, which is what we're doing here, add  + 'tf' + 'location' to postfix when calling this method.
				left_file = os.path.join(this_feat, 'stats', which_file + '1.nii.gz')
				right_file = os.path.join(this_feat, 'stats', which_file + '2.nii.gz')
				for (label, f) in zip(['left', 'right'], [left_file, right_file]):
					vsO = VolToSurfOperator(inputObject = f)
					ofn = self.runFile(stage = 'processed/mri/', run = r, base = which_file, postFix = [label] )
					ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
					vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute()
				
				
	
	def mask_stats_to_hdf(self, run_type = 'reward', postFix = ['mcf'], version = 'orientation'):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
		
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = open_file(self.hdf5_filename, mode = "w", title = run_type + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = open_file(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		if not hasattr(self, 'dual_pilot'):
			for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
				"""loop over runs, and try to open a group for this run's data"""
				this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
				try:
					thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
					self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + ' to this file')
					thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
			
				"""
				Now, take different stat masks based on the run_type
				"""
				this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
				if run_type == 'reward':
					stat_files = {
									'visual_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
									'visual_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
									'visual_cope': os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
								
									'reward_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
									'reward_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
									'reward_cope': os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
								
									'blinks': os.path.join(this_feat, 'stats', 'pe1.nii.gz'),
									'blank_silence': os.path.join(this_feat, 'stats', 'cope3.nii.gz'),
									'blank_sound': os.path.join(this_feat, 'stats', 'cope4.nii.gz'),
									'visual_silence': os.path.join(this_feat, 'stats', 'cope5.nii.gz'),
									'visual_sound': os.path.join(this_feat, 'stats', 'cope6.nii.gz'),
								
									'fix_reward_silence': os.path.join(this_feat, 'stats', 'cope7.nii.gz'),
									'visual_reward_silence': os.path.join(this_feat, 'stats', 'cope8.nii.gz'),
								
									'visual_silence_fix_silence': os.path.join(this_feat, 'stats', 'cope9.nii.gz'),
									'visual_reward_fix_reward': os.path.join(this_feat, 'stats', 'cope10.nii.gz'),
								
									}
				
				elif run_type == 'mapper':
					stat_files = {
									'center_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
									'center_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
									'center_cope': os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
									'center_pe': os.path.join(this_feat, 'stats', 'pe1.nii.gz'),
								
									'surround_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
									'surround_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
									'surround_cope': os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
									'surround_pe': os.path.join(this_feat, 'stats', 'pe3.nii.gz'),
								
									'center>surround_T': os.path.join(this_feat, 'stats', 'tstat3.nii.gz'),
									'center>surround_Z': os.path.join(this_feat, 'stats', 'zstat3.nii.gz'),
									'center>surround_cope': os.path.join(this_feat, 'stats', 'cope3.nii.gz'),
								
									'surround>center_T': os.path.join(this_feat, 'stats', 'tstat4.nii.gz'),
									'surround>center_Z': os.path.join(this_feat, 'stats', 'zstat4.nii.gz'),
									'surround>center_cope': os.path.join(this_feat, 'stats', 'cope4.nii.gz'),
									}
				# general info we want in all hdf files
				stat_files.update({
									'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
									'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf', 'psc']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									# for these final two, we need to pre-setup the retinotopic mapping data
									'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
									'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz')
				})
				
				if os.path.isfile(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco/'), 'residuals.nii.gz')):
					stat_files.update({
										'deco_residuals': os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco/'), 'residuals.nii.gz'),
					})
				
				stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
				for (roi, roi_name) in zip(rois, roinames):
					try:
						thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
					except NoSuchNodeError:
						# import actual data
						self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
						thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
					for (i, sf) in enumerate(stat_files.keys()):
						# loop over stat_files and rois
						# to mask the stat_files with the rois:
						imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
						these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
						h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
		else:
			version_postFix = postFix + ['orientation']
			for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
				"""loop over runs, and try to open a group for this run's data"""
				this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
				try:
					thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
					self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + ' to this file')
					thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
			
				"""
				Now, take different stat masks based on the run_type
				"""
				this_orientation_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['orientation'], extension = '.feat')
				this_location_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['location'], extension = '.feat')
				this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['orientation'], extension = '.feat')
				
				if run_type == 'reward':
					stat_files = {
									'left_CW_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
									'left_CW_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
									'left_CW_cope': os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
								
									'left_CCW_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
									'left_CCW_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
									'left_CCW_cope': os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
									
									'right_CW_T': os.path.join(this_feat, 'stats', 'tstat3.nii.gz'),
									'right_CW_Z': os.path.join(this_feat, 'stats', 'zstat3.nii.gz'),
									'right_CW_cope': os.path.join(this_feat, 'stats', 'cope3.nii.gz'),
									
									'right_CCW_T': os.path.join(this_feat, 'stats', 'tstat4.nii.gz'),
									'right_CCW_Z': os.path.join(this_feat, 'stats', 'zstat4.nii.gz'),
									'right_CCW_cope': os.path.join(this_feat, 'stats', 'cope4.nii.gz'),
									
									'reward_blank_T': os.path.join(this_feat, 'stats', 'tstat5.nii.gz'),
									'reward_blank_Z': os.path.join(this_feat, 'stats', 'zstat5.nii.gz'),
									'reward_blank_cope': os.path.join(this_feat, 'stats', 'cope5.nii.gz'),
								
									'reward_all_T': os.path.join(this_feat, 'stats', 'tstat6.nii.gz'),
									'reward_all_Z': os.path.join(this_feat, 'stats', 'zstat6.nii.gz'),
									'reward_all_cope': os.path.join(this_feat, 'stats', 'cope6.nii.gz'),
									
									}
				
				elif run_type == 'mapper':
					stat_files = {
									'left_CW_T': os.path.join(this_orientation_feat, 'stats', 'tstat1.nii.gz'),
									'left_CW_Z': os.path.join(this_orientation_feat, 'stats', 'zstat1.nii.gz'),
									'left_CW_cope': os.path.join(this_orientation_feat, 'stats', 'cope1.nii.gz'),
								
									'left_CCW_T': os.path.join(this_orientation_feat, 'stats', 'tstat2.nii.gz'),
									'left_CCW_Z': os.path.join(this_orientation_feat, 'stats', 'zstat2.nii.gz'),
									'left_CCW_cope': os.path.join(this_orientation_feat, 'stats', 'cope2.nii.gz'),
									
									'right_CW_T': os.path.join(this_orientation_feat, 'stats', 'tstat3.nii.gz'),
									'right_CW_Z': os.path.join(this_orientation_feat, 'stats', 'zstat3.nii.gz'),
									'right_CW_cope': os.path.join(this_orientation_feat, 'stats', 'cope3.nii.gz'),
									
									'right_CCW_T': os.path.join(this_orientation_feat, 'stats', 'tstat4.nii.gz'),
									'right_CCW_Z': os.path.join(this_orientation_feat, 'stats', 'zstat4.nii.gz'),
									'right_CCW_cope': os.path.join(this_orientation_feat, 'stats', 'cope4.nii.gz'),
									
									'left_T': os.path.join(this_location_feat, 'stats', 'tstat1.nii.gz'),
									'left_Z': os.path.join(this_location_feat, 'stats', 'zstat1.nii.gz'),
									'left_cope': os.path.join(this_location_feat, 'stats', 'cope1.nii.gz'),
								
									'right_T': os.path.join(this_location_feat, 'stats', 'tstat2.nii.gz'),
									'right_Z': os.path.join(this_location_feat, 'stats', 'zstat2.nii.gz'),
									'right_cope': os.path.join(this_location_feat, 'stats', 'cope2.nii.gz'),
									
									}
									
				# general info we want in all hdf files
				stat_files.update({
									'residuals': os.path.join(this_orientation_feat, 'stats', 'res4d.nii.gz'),
									'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf', 'psc']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									'hpf_data': os.path.join(this_orientation_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									# for these final two, we need to pre-setup the retinotopic mapping data
									'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
									'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz'),
									
									'center_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'tstat1.nii.gz'),
									'center_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'zstat1.nii.gz'),
									'center_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'cope1.nii.gz'),
									'center_pe': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'pe1.nii.gz'),
								
									'surround_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'tstat2.nii.gz'),
									'surround_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'zstat2.nii.gz'),
									'surround_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'cope2.nii.gz'),
									'surround_pe': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'pe3.nii.gz'),
								
									'center>surround_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'tstat3.nii.gz'),
									'center>surround_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'zstat3.nii.gz'),
									'center>surround_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'cope3.nii.gz'),
								
									'surround>center_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'tstat4.nii.gz'),
									'surround>center_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'zstat4.nii.gz'),
									'surround>center_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'stat', 'cope4.nii.gz'),
									
				})
				
				
				
				stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
				for (roi, roi_name) in zip(rois, roinames):
					try:
						thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
					except NoSuchNodeError:
						# import actual data
						self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
						thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
					for (i, sf) in enumerate(stat_files.keys()):
						# loop over stat_files and rois
						# to mask the stat_files with the rois:
						imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
						these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
						h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
			
		h5file.close()
	
	def hdf5_file(self, run_type, mode = 'r'):
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('no table file ' + self.hdf5_filename + 'found for stat mask')
			return None
		else:
			# self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = open_file(self.hdf5_filename, mode = mode, title = run_type + " file")
		return h5file
	
	
	def pupil_responses_one_run(self, run, frequency, sample_rate = 2000, postFix = ['mcf'], analysis_duration = 10):
		if run.condition == 'reward':
			# get EL Data
			
			h5f = open_file(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'), mode = "r" )
			r = None
			for item in h5f.iterNodes(where = '/', classname = 'Group'):
				if item._v_name == 'bla':
					r = item
					break
			if r == None:
				self.logger.error('No run named bla in this run\'s hdf5 file ' + self.hdf5_filename )
			
			trial_times = r.trial_times.read()
			gaze_timestamps = r.gaze_data.read()[:,0]
			raw_pupil_sizes = r.gaze_data.read()[:,3]
			trial_parameters = r.trial_parameters.read()
			blink_data = r.blinks_from_EL.read()
			
			from scipy import interpolate
			from scipy.signal import butter, filtfilt
			
			# lost data will appear as a 0.0 instead of anything sensible, and is not detected as blinks. 
			# we will detect them, string them together so that they coalesce into reasonable events, and
			# add them to the blink array.
			zero_edges = np.arange(raw_pupil_sizes.shape[0])[np.diff((raw_pupil_sizes < 0.1))]
			
			zero_edges = zero_edges[:int(2 * floor(zero_edges.shape[0]/2.0))].reshape(-1,2)
			new_ze = [zero_edges[0]]
			for ze in zero_edges[1:]:
				if (ze[0] - new_ze[-1][-1])/sample_rate < 0.2:
					new_ze[-1][1] = ze[1]
				else:
					new_ze.append(ze)
			zero_edges = np.array(new_ze)
			
			ze_to_blinks = np.zeros((zero_edges.shape[0]), dtype = blink_data.dtype)
			for (i, ze_d) in enumerate(ze_to_blinks):
				# make sure to convert times-indices back to times, as for blinks
				ze_d['duration'] = gaze_timestamps[zero_edges[i, 1]] - gaze_timestamps[zero_edges[i, 0]]
				ze_d['start_timestamp'] = gaze_timestamps[zero_edges[i, 0]]
				ze_d['end_timestamp'] = gaze_timestamps[zero_edges[i, 1]]
				ze_d['eye'] = 'R'
			blink_data = np.concatenate((blink_data, ze_to_blinks))
			# # re-sort
			blink_data = blink_data[np.argsort(blink_data['start_timestamp'])]
			
			# shell()
			
			# trials are separated on 'sound' and 'contrast' parameters, and we parcel in the reward scheme here, since not every subject receives the same reward and zero sounds
			sound_trials, visual_trials = np.array((self.which_reward + trial_parameters['sound']) % 2, dtype = 'bool'), np.array(trial_parameters['contrast'], dtype = 'bool')
			
			condition_labels = ['visual_sound', 'visual_silence', 'blank_silence', 'blank_sound']
			# conditions are made of boolean combinations
			visual_sound_trials = sound_trials * visual_trials
			visual_silence_trials = visual_trials * (-sound_trials)
			blank_silence_trials = -(visual_trials + sound_trials)
			blank_sound_trials = (-visual_trials) * sound_trials
			
			# these are the time points (defined in samples for now...) from which we take the levels on which to base the interpolation
			points_for_interpolation = np.array([[-150, -75],[75, 150]])
			interpolation_time_points = np.zeros((blink_data.shape[0], points_for_interpolation.ravel().shape[0]))
			interpolation_time_points[:,[0,1]] = np.tile(blink_data['start_timestamp'], 2).reshape((2,-1)).T + points_for_interpolation[0]
			interpolation_time_points[:,[2,3]] = np.tile(blink_data['end_timestamp'], 2).reshape((2,-1)).T + points_for_interpolation[1]
			
			# shell()
			
			# blinks may start or end before or after sampling has begun or stopped
			interpolation_time_points = np.where(interpolation_time_points < gaze_timestamps[-1], interpolation_time_points, gaze_timestamps[-1])
			interpolation_time_points = np.where(interpolation_time_points > gaze_timestamps[0], interpolation_time_points, gaze_timestamps[0])
			# apparently the above doesn't actually work, and resulting in nan interpolation results. We'll just throw these out.
			# this last-ditch attempt rule might work out the nan errors after interpolation
			interpolation_time_points = np.array([itp for itp in interpolation_time_points if ((itp == gaze_timestamps[0]).sum() == 0) and ((itp == gaze_timestamps[-1]).sum() == 0)])
			# itp = itp[(itp != gaze_timestamps[0]) + (itp != gaze_timestamps[-1])]
			
			# correct for the fucking eyelink not keeping track of fucking time
			# shell()
			# interpolation_time_points = np.array([[np.arange(gaze_timestamps.shape[0])[gaze_timestamps >= interpolation_time_points[i,j]][0] for j in range(points_for_interpolation.ravel().shape[0])] for i in range(interpolation_time_points.shape[0])])
			# convert everything to indices
			interpolation_time_points = np.array([[np.arange(gaze_timestamps.shape[0])[gaze_timestamps >= interpolation_time_points[i,j]][0] for j in range(points_for_interpolation.ravel().shape[0])] for i in range(interpolation_time_points.shape[0])])
			
			# print raw_pupil_sizes.mean()
			# print interpolation_time_points
			
			for itp in interpolation_time_points:
				# interpolate
				spline = interpolate.InterpolatedUnivariateSpline(itp,raw_pupil_sizes[itp])
				# replace with interpolated data
				raw_pupil_sizes[itp[0]:itp[-1]] = spline(np.arange(itp[0],itp[-1]))
			
			# print raw_pupil_sizes.mean()
			
			# band-pass filtering of signal, high pass first and then low-pass
			hp_frequency = 0.02
			hp_cof_sample = hp_frequency / (raw_pupil_sizes.shape[0] / (sample_rate / 2))
			bhp, ahp = butter(3, hp_cof_sample, btype = 'high')
			
			hp_c_pupil_size = filtfilt(bhp, ahp, raw_pupil_sizes)
			
			lp_frequency = 10.0
			lp_cof_sample = lp_frequency / (raw_pupil_sizes.shape[0] / (sample_rate / 2))
			blp, alp = butter(3, lp_cof_sample)
			
			lp_hp_c_filt_pupil_size = filtfilt(blp, alp, hp_c_pupil_size)
			
			pupil_zscore = (lp_hp_c_filt_pupil_size - np.array(lp_hp_c_filt_pupil_size).mean()) / lp_hp_c_filt_pupil_size.std() # Possible because vectorized.
			# trials ordered by trial type
			trial_phase_timestamps = [trial_times['trial_phase_timestamps'][:,1][cond,0] for cond in [blank_silence_trials, blank_sound_trials, visual_silence_trials, visual_sound_trials]]			
			tr_data = np.array([[pupil_zscore[(gaze_timestamps>tpt) * (gaze_timestamps<(tpt+500 + (analysis_duration * sample_rate)))][:(analysis_duration * sample_rate)] for tpt in trphts] for trphts in trial_phase_timestamps])
			tr_r_data = np.array([[raw_pupil_sizes[(gaze_timestamps>tpt) * (gaze_timestamps<(tpt+500 + (analysis_duration * sample_rate)))][:(analysis_duration * sample_rate)] for tpt in trphts] for trphts in trial_phase_timestamps])
			# trials ordered by occurrence time
			trial_phase_timestamps_timed = trial_times['trial_phase_timestamps'][:,1][:,0]
			tr_data_timed = np.array([pupil_zscore[(gaze_timestamps>tpt) * (gaze_timestamps<(tpt+500 + (analysis_duration * sample_rate)))][:(analysis_duration * sample_rate)] for tpt in trial_phase_timestamps_timed])
			# close edf hdf5 file
			h5f.close()
			
			# save this data to the joint hdf5 file
			# find or create file for this run
			h5file = self.hdf5_file(run_type = 'reward', mode = 'a')
			# in the file, create the appropriate group
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]
			try:
				thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix))
			
			# save parameter data to joint file
			import numpy.lib.recfunctions as rfn
			try: 
				h5file.remove_node(where = thisRunGroup, name = 'trial_parameters')
			except NoSuchNodeError:
				pass
			parTable = h5file.createTable(thisRunGroup, 'trial_parameters', trial_parameters.dtype, 'Parameters for trials in run ' + str(run.ID))
			# fill up the table
			trial = parTable.row
			for tr in trial_parameters:
				for par in rfn.get_names(trial_parameters.dtype):
					trial[par] = tr[par]
				trial.append()
			parTable.flush()
			try: 
				h5file.remove_node(where = thisRunGroup, name = 'trial_times')
			except NoSuchNodeError:
				pass
			timeTable = h5file.createTable(thisRunGroup, 'trial_times', trial_times.dtype, 'Timings for trials in run ' + str(run.ID))
			# fill up the table
			trial = timeTable.row
			for tr in trial_times:
				for par in rfn.get_names(trial_times.dtype):
					trial[par] = tr[par]
				trial.append()
			timeTable.flush()
			
			# save pupil data to joint file
			try: 
				h5file.remove_node(where = thisRunGroup, name = 'filtered_pupil_zscore')
				h5file.remove_node(where = thisRunGroup, name = 'per_trial_filtered_pupil_zscore')
				h5file.remove_node(where = thisRunGroup, name = 'per_condition_filtered_pupil_zscore')
			except NoSuchNodeError:
				pass
			h5file.create_array(thisRunGroup, 'filtered_pupil_zscore', np.vstack((gaze_timestamps, pupil_zscore)).T, 'filtered_pupil_zscore conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			h5file.create_array(thisRunGroup, 'per_trial_filtered_pupil_zscore', np.array(tr_data_timed), 'per_trial_filtered_pupil_zscore conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			h5file.create_array(thisRunGroup, 'per_condition_filtered_pupil_zscore', np.array([np.array(tr).mean(axis = 0) for tr in tr_data]), 'per_condition_filtered_pupil_zscore conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			h5file.close()
			
			# shell()
			fig = pl.figure(figsize = (10,7))
			s = fig.add_subplot(211)
			for i in range(tr_data.shape[0]):
				s.plot(np.array(tr_data[i]).T, 'k', linewidth = 1.75, alpha = 0.3)
			s = fig.add_subplot(212)
			for i in range(tr_data.shape[0]):
				pl.plot(np.array(tr_r_data[i]).T, 'r', linewidth = 1.75, alpha = 0.3)
			# save in two separate locations
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/eye/figs/'), os.path.split(self.runFile(stage = 'processed/eye', run = run, extension = '.pdf', postFix = ['pupil']))[1]))
			pl.savefig(self.runFile(stage = 'processed/eye', run = run, extension = '.pdf', postFix = ['pupil']))
			return tr_data
			
	def pupil_responses(self, sample_rate = 2000, save_all = True):
		"""docstring for pupil_responses"""
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		all_pupil_responses = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			all_pupil_responses.append(self.pupil_responses_one_run(run = r, frequency = 4, sample_rate = sample_rate))
		# self.all_pupil_responses_hs = np.array(all_pupil_responses)
		fig = pl.figure(figsize = (9,5))
		s = fig.add_subplot(2,1,1)
		s.set_ylabel('Z-scored Pupil Size')
		s.set_title('Pupil Size after stimulus onset')
		all_data_conditions = []
		all_data = []
		for i in range(4):
			all_data_this_condition = np.vstack([all_pupil_responses[j][i] for j in range(len(all_pupil_responses))])
			zero_points = all_data_this_condition[:,[0,1]].mean(axis = 1)
			all_data_this_condition = np.array([a - z for (a, z) in zip (all_data_this_condition, zero_points)])
			all_data_conditions.append(all_data_this_condition.mean(axis = 0))
			all_data.append(all_data_this_condition)
			rnge = np.linspace(0,all_data_conditions[-1].shape[0]/sample_rate, all_data_conditions[-1].shape[0])
			sems = 1.96 * (all_data_this_condition.std(axis = 0)/np.sqrt(all_data_this_condition.shape[0]))
			pl.plot(rnge, all_data_conditions[-1], ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = cond_labels[i])
			pl.fill_between(rnge, all_data_conditions[-1]+sems, all_data_conditions[-1]-sems, color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.75)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		s = fig.add_subplot(2,1,2)
		for i in range(0, 4, 2):
			diffs = -(all_data_conditions[i] - all_data_conditions[i+1])
			pl.plot(np.linspace(0,diffs.shape[0]/sample_rate, diffs.shape[0]), diffs, ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2])
			s.set_title('reward signal')
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.75)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		s.set_xlabel('time [s]')
		s.set_ylabel('$\Delta$ Z-scored Pupil Size')
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'pupil_evolution_per_condition.pdf'))
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/eye/figs/'), 'pupil_evolution_per_condition.pdf'))
		
		if save_all:
			# save all these data to the hdf5 file
			h5file = self.hdf5_file(run_type = 'reward', mode = 'a')
			try: 
				h5file.remove_node(where = '/', name = 'all_pupil_scores')
			except NoSuchNodeError:
				pass
			h5file.create_array('/', 'all_pupil_scores', np.array(all_data), '_'.join(cond_labels) + ' conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			h5file.close()
			# shell()
	
	def pupil_responses_interval(self, sample_rate = 2000, save_all = False):
		"""docstring for pupil_responses_interval"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		reward_h5file = self.hdf5_file('reward')
		
		event_data = []
		all_pupil_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			pupil_data = self.run_data_from_hdf(reward_h5file, r, 'per_trial_filtered_pupil_zscore')
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times')
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters')
			stim_onsets = trial_times['trial_phase_timestamps'][:,1,0]
			# trials are separated on 'sound' and 'contrast' parameters, and we parcel in the reward scheme here, since not every subject receives the same reward and zero sounds
			sound_trials, visual_trials = np.array((self.which_reward + parameter_data['sound']) % 2, dtype = 'bool'), np.array(parameter_data['contrast'], dtype = 'bool')
			blank_sound_trials = (-visual_trials) * sound_trials
			
			itis = np.diff(stim_onsets)[blank_sound_trials[1:]]
			trials_of_interest = pupil_data[1:]
			
			# do median split here
			all_pupil_data.append([trials_of_interest[itis < np.median(itis)], trials_of_interest[itis > np.median(itis)]])
			
			nr_runs += 1
		
		reward_h5file.close()
		
		lp, sp = np.concatenate([ap[0] for ap in all_pupil_data]), np.concatenate([ap[1] for ap in all_pupil_data])
		
		lpm, lps = lp.mean(axis = 0), lp.std(axis = 0) / sqrt(lp.shape[0])
		spm, sps = sp.mean(axis = 0), sp.std(axis = 0) / sqrt(sp.shape[0])		
		
		if save_all:
			# save all these data to the hdf5 file
			h5file = self.hdf5_file(run_type = 'reward', mode = 'a')
			try: 
				h5file.remove_node(where = '/', name = 'all_pupil_interval_scores')
			except NoSuchNodeError:
				pass
			h5file.create_array('/', 'all_pupil_interval_scores', np.array([spm, lpm]), ' conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			h5file.close()
			# shell()
		
	
	def correlate_copes_from_run(self, run, rois = ['V1', 'V2', 'V3', 'V4', 'V3AB'], copes = ['visual_cope','reward_cope'], plot = True):
		"""
		correlates two types of data from regions of interest with one another
		"""
		from scipy import stats
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		corrs = np.zeros((len(rois), len(copes)))
		if reward_h5file != None:
			# there was a file and it has data in it
			if plot:	
				fig = pl.figure(figsize = (len(rois)*3, 3))
			for roi in rois:
				if plot: 
					s = fig.add_subplot(1, len(rois), rois.index(roi) + 1)
					s.set_title(roi, fontsize=9)
				cope1 = self.roi_data_from_hdf(reward_h5file, run, roi, copes[0])
				cope2 = self.roi_data_from_hdf(reward_h5file, run, roi, copes[1])
				mapper_cope = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'center_pe')
				if cope1 != None and cope2 != None:
					if plot:
						pl.plot(mapper_cope, cope2, marker = 'o', ms = 3, mec = 'w', c = 'r', mew = 0.5, alpha = 0.75, linewidth = 0) # , alpha = 0.25
						pl.plot(mapper_cope, cope1, marker = 'o', ms = 3, mec = 'w', c = 'g', mew = 0.5, alpha = 0.75, linewidth = 0) # , alpha = 0.25
						s.set_xlabel('mapper visual cope', fontsize=9)
						if rois.index(roi) == 0:
							s.set_ylabel('/'.join(copes), fontsize=9)
					srs = [stats.spearmanr(mapper_cope, cope1), stats.spearmanr(mapper_cope, cope2)]
					corrs[rois.index(roi), 0] = srs[0][0]
					corrs[rois.index(roi), 1] = srs[1][0]
		if plot:
			pl.draw()
			pdf_file_name = os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'copes_' + str(run.ID) +  '_'.join(copes) + '.pdf')
			pl.savefig(pdf_file_name)
		reward_h5file.close()
		mapper_h5file.close()
		return corrs
	
	def correlate_reward_copes(self, rois = ['V1', 'V2d', 'V2v', 'V3d', 'V3v', 'V4', 'V3AB'], copes = ['visual_cope','reward_cope'], scatter_plots = False):
		"""
		correlate reward run cope values with one another from all reward runs separately.
		"""
		all_corrs = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			all_corrs.append(self.correlate_copes_from_run(run = r, rois = rois, copes = copes, plot = scatter_plots))
			
		cs = np.array(all_corrs)
		
		fig = pl.figure(figsize = (4, len(rois)*1.5))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		for roi in rois:
			s = fig.add_subplot(len(rois), 1, rois.index(roi) + 1)
			s.grid()
			pl.plot(np.arange(6), cs[:,rois.index(roi),0], 'g', linewidth = 2.5, alpha = 0.75)
			pl.plot(np.arange(6), cs[:,rois.index(roi),1], 'r', linewidth = 2.5, alpha = 0.75)
			s.set_title(roi, fontsize=9)
			if rois.index(roi) == len(rois)-1:
				s.set_xlabel('run number', fontsize=9)
			if rois.index(roi) == 3:
				s.set_ylabel('spearman $R$', fontsize=9)
			s.axis([-0.25,5.35,-1,1])
			# designate bad runs:
			s.axvspan(3.75, 4.25, facecolor='y', alpha=0.25, edgecolor = 'w')
			s.axvspan(1.75, 2.25, facecolor='y', alpha=0.25, edgecolor = 'w')
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'cope_spearman_rho_over_runs.pdf'))
		
		# average across runs
		meancs = cs.mean(axis = 0)
		sdcs = 1.96 * cs.std(axis = 0) / sqrt(6) 
		
		fig = pl.figure(figsize = (6, 3))
		s = fig.add_subplot(111)
		width = 0.35
		pl.plot([-1, 10], [0,0], 'k', linewidth = 0.5)
		rects1 = pl.bar(np.arange(meancs.shape[0]), height = meancs[:,0], width = width, yerr = sdcs[:,0], color='g', alpha = 0.7, edgecolor = (0.5, 0.5, 0.5), linewidth = 2.5, ecolor = (0.5, 0.5, 0.5), capsize = 0)
		rects2 = pl.bar(np.arange(meancs.shape[0])+width, height = meancs[:,1], width = width, yerr = sdcs[:,1], color='r', alpha = 0.7, edgecolor = (0.5, 0.5, 0.5), linewidth = 2.5, ecolor = (0.5, 0.5, 0.5), capsize = 0)
		pl.ylabel('Spearman correlation')
		pl.xticks(np.arange(len(rois))+width, rois )
		s.set_xlim(-0.5, meancs.shape[0]+2.5)
		leg = pl.legend( (rects1[0], rects2[0]), tuple(copes), fancybox = True)
		leg.get_frame().set_alpha(0.5)
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'cope_spearman_rho_bar_over_runs' + '_'.join(copes) + '.pdf'))
		
		# average across runs - but take out runs with lower confidence
		meancs = cs[[0,1,3,5]].mean(axis = 0)
		sdcs = 1.96 * cs[[0,1,3,5]].std(axis = 0) / sqrt(4) 
		
		fig = pl.figure(figsize = (6, 3))
		s = fig.add_subplot(111)
		width = 0.35
		pl.plot([-1, 10], [0,0], 'k', linewidth = 0.5)
		rects1 = pl.bar(np.arange(meancs.shape[0]), height = meancs[:,0], width = width, yerr = sdcs[:,0], color='g', alpha = 0.7, edgecolor = (0.5, 0.5, 0.5), linewidth = 2.5, ecolor = (0.5, 0.5, 0.5), capsize = 0)
		rects2 = pl.bar(np.arange(meancs.shape[0])+width, height = meancs[:,1], width = width, yerr = sdcs[:,1], color='r', alpha = 0.7, edgecolor = (0.5, 0.5, 0.5), linewidth = 2.5, ecolor = (0.5, 0.5, 0.5), capsize = 0)
		pl.ylabel('Spearman correlation')
		pl.xticks(np.arange(len(rois))+width, rois )
		s.set_xlim(-0.5, meancs.shape[0]+2.5)
		pl.legend( (rects1[0], rects2[0]), tuple(copes), fancybox = True)
		leg.get_frame().set_alpha(0.5)
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'cope_spearman_rho_bar_over_runs_high_conf' + '_'.join(copes) + '.pdf'))
		
		return all_corrs
	
	def deconvolve_roi(self, roi, threshold = 3.5, mask_type = 'center_surround_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'mean', data_type = 'psc_hpf_data'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		blink_events = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, data_type))
			if 'residuals' in data_type:
				roi_data[-1] = roi_data[-1] ** 2
			
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			nr_runs += 1
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		if signal_type in ['std', 'var']:
			timeseries = (timeseries - timeseries.mean() ) / timeseries.std()
				
		fig = pl.figure(figsize = (9, 5))
		s = fig.add_subplot(211)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		if analysis_type == 'deconvolution':
			interval = [0.0,16.0]
			# nuisance version?
			nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
			nuisance_design.configure(np.array([np.hstack(blink_events)]))
			deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
			deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix.T)
			deco.residuals()
			residuals = deco.residuals
			# shell()
			for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
				time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze())
				# shell()
				pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze()), ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			
			# the following commented code doesn't factor in blinks as nuisances
			# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
			# for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
			# 	pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			# 	time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
			s.set_title('deconvolution' + roi + ' ' + mask_type)
			deco_per_run = []
			for i, rd in enumerate(roi_data_per_run):
				event_data_this_run = event_data_per_run[i] - i * run_duration
				deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
				deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
				# deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
				# deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix[i*nr_trs*2:(i+1)*nr_trs*2])
				# deco_per_run.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance)
			deco_per_run = np.array(deco_per_run)
			mean_deco = deco_per_run.mean(axis = 0)
			std_deco = 1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))
			for i in range(0, mean_deco.shape[0]):
				# pl.plot(np.linspace(interval[0],interval[1],mean_deco.shape[1]), mean_deco[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
				s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), (np.array(time_signals[i]) + std_deco[i].T)[0], (np.array(time_signals[i]) - std_deco[i].T)[0], color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])
		
		else:
			interval = [-3.0,19.5]
			# zero_timesignals = eraO = EventRelatedAverageOperator(inputObject = np.array([timeseries]), eventObject = event_data[0], interval = interval)
			# zero_time_signal = eraO.run(binWidth = 3.0, stepSize = 1.5)
			for i in range(event_data.shape[0]):
				eraO = EventRelatedAverageOperator(inputObject = np.array([timeseries]), eventObject = event_data[i], TR = tr, interval = interval)
				time_signal = eraO.run(binWidth = 3.0, stepSize = 1.5)
				zero_zero_means = time_signal[:,1] - time_signal[time_signal[:,0] == 0,1]
				s.fill_between(time_signal[:,0], zero_zero_means + time_signal[:,2]/np.sqrt(time_signal[:,3]), zero_zero_means - time_signal[:,2]/np.sqrt(time_signal[:,3]), color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])
				pl.plot(time_signal[:,0], zero_zero_means, ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
				time_signals.append(zero_zero_means)
			s.set_title('event-related average ' + roi + ' ' + mask_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		
		s = fig.add_subplot(212)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		if analysis_type == 'deconvolution':
			for i in range(0, len(event_data), 2):
				ts_diff = -(time_signals[i] - time_signals[i+1])
				pl.plot(np.linspace(0,interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), np.array(ts_diff), ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
				s.set_title('reward signal ' + roi + ' ' + mask_type + ' ' + analysis_type)
		
		else:
			time_signals = np.array(time_signals)
			for i in range(0, event_data.shape[0], 2):
				ts_diff = -(time_signals[i] - time_signals[i+1])
				pl.plot(time_signal[:,0], ts_diff, ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
			s.set_title('reward signal ' + roi + ' ' + mask_type + ' ' + analysis_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('$\Delta$ % signal change')
		s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
			
		reward_h5file.close()
		mapper_h5file.close()
		
		# pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + data_type + '.pdf'))
		# shell()
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals), np.array(deco_per_run), residuals]
	
	def prepare_for_pupil(self):
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			subprocess.Popen('rm ' + self.runFolder(stage = 'processed/eye', run = r) + '/*.msg', shell=True, stdout=PIPE).communicate()[0].split('\n')[0] 
			subprocess.Popen('rm ' + self.runFolder(stage = 'processed/eye', run = r) + '/*.gaz', shell=True, stdout=PIPE).communicate()[0].split('\n')[0] 
			subprocess.Popen('rm ' + self.runFolder(stage = 'processed/eye', run = r) + '/*.gaz.gz', shell=True, stdout=PIPE).communicate()[0].split('\n')[0] 
			edf_file = subprocess.Popen('ls ' + self.runFolder(stage = 'processed/eye', run = r) + '/*.edf', shell=True, stdout=PIPE).communicate()[0].split('\n')[0] 
			self.ho.add_edf_file(edf_file)
			self.ho.edf_message_data_to_hdf(alias = str(r.indexInSession))
			self.ho.edf_gaze_data_to_hdf(alias = str(r.indexInSession), pupil_hp = 0.04, pupil_lp = 4)		

	def events_and_signals_in_time(self, data_type = 'pupil_bp'):
		"""events_and_signals_in_time takes all aliases' data from the hdf5 file.
		This results in variables that designate occurrences in seconds time, 
		in the time as useful for the variable self.pupil_data, which contains z-scored data_type type data and
		is still sampled at the original sample_rate. Note: the assumption is that all aliases are sampled at the same frequency. 
		events_and_signals_in_time further creates self.colour_indices and self.sound_indices variables that 
		index which trials (corresponding to _times indices) correspond to which sounds and which reward probabilities.
		"""
		event_data = []
		pupil_data = []
		blink_times = []

		session_time = 0

		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			alias = r.indexInSession
			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]

			trial_parameters = self.ho.read_session_data(alias, 'parameters')

			self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)
			self.sampled_eye = self.ho.eye_during_period([session_start_EL_time, session_stop_EL_time], alias)
			#load in blink data
			eyelink_blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
			eyelink_blink_data_L = eyelink_blink_data[eyelink_blink_data['eye'] == self.sampled_eye] #only select data from left eye
			b_start_times = np.array(eyelink_blink_data_L.start_timestamp)
			b_end_times = np.array(eyelink_blink_data_L.end_timestamp)

			#evaluate only blinks that occur after start and before end experiment
			b_indices = (b_start_times>session_start_EL_time)*(b_end_times<session_stop_EL_time) 
			b_start_times_t = (b_start_times[b_indices] - session_start_EL_time) #valid blinks (start times) 
			b_end_times_t = (b_end_times[b_indices] - session_start_EL_time) 
			blinks = np.array(b_start_times_t)
			blink_times.append(((blinks + session_time) / self.sample_rate ))

			pupil = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = data_type, requested_eye = self.sampled_eye))
			pupil_data.append((pupil - pupil.mean()) / pupil.std())

			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + session_time / 1000.0
			event_data.append(this_run_events)

			session_time += session_stop_EL_time - session_start_EL_time
			

		self.blink_times = np.concatenate(blink_times)
		self.pupil_data = np.concatenate(pupil_data)
		self.event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]

		#shell()

	def prepocessing_report(self, downsample_rate =20 ):
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			alias = str(r.indexInSession)
			# load times per session:
			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_times['trial_start_EL_timestamp'])[0]
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]

			sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)
			eye = self.ho.eye_during_period([session_start_EL_time, session_stop_EL_time], alias)

			pupil_raw = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil', requested_eye = eye))
			pupil_int = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_int', requested_eye = eye))

			pupil_bp = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_bp', requested_eye = eye))
			pupil_lp = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_lp', requested_eye = eye))
			pupil_hp = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_hp', requested_eye = eye))

			x = sp.signal.decimate(np.arange(len(pupil_raw)) / float(sample_rate), downsample_rate, 1)
			pup_raw_dec = sp.signal.decimate(pupil_raw, downsample_rate, 1)
			pup_int_dec = sp.signal.decimate(pupil_int, downsample_rate, 1)

			pupil_bp_dec = sp.signal.decimate(pupil_bp, downsample_rate, 1)
			pupil_lp_dec = sp.signal.decimate(pupil_lp, downsample_rate, 1)
			pupil_hp_dec = sp.signal.decimate(pupil_hp, downsample_rate, 1)

			# plot interpolated pupil:
			fig = pl.figure(figsize = (24,9))
			s = fig.add_subplot(311)
			pl.plot(x, pup_raw_dec, 'b'); pl.plot(x, pup_int_dec, 'g')
			pl.ylabel('pupil size'); pl.xlabel('time (s)')
			pl.legend(['raw pupil', 'blink interpolated pupil'])
			s.set_title(self.subject.initials)

			ymin = pupil_raw.min(); ymax = pupil_raw.max()
			tps = (list(trial_phase_times[trial_phase_times['trial_phase_index'] == 2]['trial_phase_EL_timestamp']) - session_start_EL_time, list(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time)
			for i in range(tps[0].shape[0]):
				pl.axvline(x = tps[0][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'r')
				pl.axvline(x = tps[1][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'k')
			s.set_ylim(ymin = pup_int_dec.min()-100, ymax = pup_int_dec.max()+100)
			s.set_xlim(xmin = tps[0][0] / float(sample_rate), xmax = tps[1][-1] / float(sample_rate))

			s = fig.add_subplot(312)
			pl.plot(x, pupil_bp_dec, 'b'); pl.plot(x, pupil_lp_dec, 'g');
			pl.ylabel('pupil size'); pl.xlabel('time (s)')
			pl.legend(['band_passed', 'lowpass'])
			s.set_title(self.subject.initials)

			ymin = pupil_raw.min(); ymax = pupil_raw.max()
			tps = (list(trial_phase_times[trial_phase_times['trial_phase_index'] == 2]['trial_phase_EL_timestamp']) - session_start_EL_time, list(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time)
			for i in range(tps[0].shape[0]):
				pl.axvline(x = tps[0][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'r')
				pl.axvline(x = tps[1][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'k')
			# s.set_ylim(ymin = pup_int_dec.min()-100, ymax = pup_int_dec.max()+100)
			s.set_xlim(xmin = tps[0][0] / float(sample_rate), xmax = tps[1][-1] / float(sample_rate))

			s = fig.add_subplot(313)
			pl.plot(x, pupil_bp_dec, 'b'); pl.plot(x, pupil_hp_dec, 'b');
			pl.ylabel('pupil size'); pl.xlabel('time (s)')
			pl.legend(['band_passed', 'highpass'])
			s.set_title(self.subject.initials)

			ymin = pupil_raw.min(); ymax = pupil_raw.max()
			tps = (list(trial_phase_times[trial_phase_times['trial_phase_index'] == 2]['trial_phase_EL_timestamp']) - session_start_EL_time, list(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time)
			for i in range(tps[0].shape[0]):
				pl.axvline(x = tps[0][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'r')
				pl.axvline(x = tps[1][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'k')
			# s.set_ylim(ymin = pup_int_dec.min()-100, ymax = pup_int_dec.max()+100)
			s.set_xlim(xmin = tps[0][0] / float(sample_rate), xmax = tps[1][-1] / float(sample_rate))

			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/eye/'), 'figs', alias + '.pdf'))

	def deconvolve_pupil(self, analysis_sample_rate = 20, interval = [-0.5,7.5], data_type = 'pupil_bp'):
		"""raw deconvolution, to see what happens when the fixation colour changes, 
		and when the sound chimes."""

		self.events_and_signals_in_time(data_type = data_type )
		cond_labels = ['blinks', 'fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']

		input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))
		events = [self.blink_times + interval[0], self.event_data[0] + interval[0], self.event_data[1] + interval[0], self.event_data[2] + interval[0], self.event_data[3] + interval[0]]
		do = ArrayOperator.DeconvolutionOperator( inputObject = input_signal,
							eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, deconvolutionInterval = interval[1] - interval[0], run = True )
		time_points = np.linspace(interval[0], interval[1], np.squeeze(do.deconvolvedTimeCoursesPerEventType).shape[1])
		do.residuals()
		#shell()
		f = pl.figure()
		ax = f.add_subplot(111)
		for x in range(len(cond_labels)):
			pl.plot(time_points, np.squeeze(do.deconvolvedTimeCoursesPerEventType)[x], ['k','b','b','g','g'][x], alpha = [0.5,1.0,0.5,1.0,0.5][x])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_xlim(xmin=interval[0], xmax=interval[1])
		pl.legend(cond_labels)
		simpleaxis(ax)
		spine_shift(ax)
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'pupil_deconvolution.pdf'))
		
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%('deconvolve_pupil', 'residuals'), pd.Series(np.squeeze(np.array(do.residuals))))
			h5_file.put("/%s/%s"%('deconvolve_pupil', 'time_points'), pd.Series(time_points))
			h5_file.put("/%s/%s"%('deconvolve_pupil', 'dec_time_course'), pd.DataFrame(np.squeeze(do.deconvolvedTimeCoursesPerEventType).T))

	def pupil_interval_analysis(self, analysis_sample_rate = 20, interval = [-0.5,10.0], data_type = 'pupil_bp', data_interval = [-0.5, 7.5]):

		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs

		reward_h5file = self.hdf5_file('reward')


		self.events_and_signals_in_time(data_type = data_type )
		cond_labels = ['blinks', 'fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		all_conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		all_event_data, blink_events = [], []
		stimulus_itis_fix_trials, fix_itis_fix_trials = [], []
		stimulus_itis_stimulus_trials, fix_itis_stimulus_trials = [], []

		events_of_interest = [] # fix R and stim R trials

		pupil_data = []

		nr_runs = 0

		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times')
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters')
			
			onsets_fix_reward_trials, raw_itis_of_fix_reward_trials, all_reward_itis_of_fix_reward_trials, fixation_reward_itis_fix_reward_trials, stimulus_reward_itis_fix_reward_trials = self.calculate_event_history_fix_reward(trial_times, parameter_data)
			onsets_stim_reward_trials, raw_itis_of_stim_reward_trials, all_reward_itis_of_stim_reward_trials, fixation_reward_itis_stim_reward_trials, stimulus_reward_itis_stim_reward_trials = self.calculate_event_history_stim_reward(trial_times, parameter_data)

			stimulus_itis_fix_trials.extend(stimulus_reward_itis_fix_reward_trials)
			fix_itis_fix_trials.extend(fixation_reward_itis_fix_reward_trials)
			stimulus_itis_stimulus_trials.extend(stimulus_reward_itis_stim_reward_trials)
			fix_itis_stimulus_trials.extend(fixation_reward_itis_stim_reward_trials)

			events_of_interest.append([onsets_fix_reward_trials + nr_runs * run_duration, onsets_stim_reward_trials + nr_runs * run_duration])

			this_run_all_events = []
			for cond in all_conds:
				this_run_all_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_all_events = np.array(this_run_all_events) + nr_runs * run_duration
			all_event_data.append(this_run_all_events)
			
			nr_runs += 1
		event_data = [np.concatenate([e[i] for e in all_event_data]) for i in range(len(all_event_data[0]))]
		events_of_interest = [np.array(np.round(np.concatenate([e[i] for e in events_of_interest]) * analysis_sample_rate), dtype = int) for i in range(2)]

		stimulus_itis = [np.array(stimulus_itis_fix_trials), np.array(stimulus_itis_stimulus_trials)]
		fix_itis = [np.array(fix_itis_fix_trials), np.array(fix_itis_stimulus_trials)]

		# get residuals from earlier deconvolution
		with pd.get_store(self.ho.inputObject) as h5_file:
			residuals = np.array(h5_file.get("/%s/%s"%('deconvolve_pupil', 'residuals')))	

		correlation_indices = [np.array([np.arange(data_interval[0] * analysis_sample_rate, data_interval[1] * analysis_sample_rate) + t for t in e], dtype = int) for e in events_of_interest]
		correlation_timecourses = [residuals[ci] for ci in correlation_indices]
		correlation_timecourses_no_zero = [(ci.T - ci[:,:abs(data_interval[0]*analysis_sample_rate)].mean(axis = 1)).T for ci in correlation_timecourses]

		for ctnz, si, fi, lbl in zip(correlation_timecourses_no_zero, stimulus_itis, fix_itis, ['fix', 'stim']): # fix trial timecourses, stimulus trial timecourses
			fix, stim = np.array([spearmanr(tc, fi) for tc in ctnz.T]), np.array([spearmanr(tc, si) for tc in ctnz.T])

			f = pl.figure()
			s = f.add_subplot(121)
			s.plot(fix[:,0], 'b', label = 'corr with fix intervals')
			s.plot(stim[:,0], 'g', label = 'corr with stim intervals')
			pl.legend()
			s = f.add_subplot(122)
			s.plot(-np.log10(fix[:,1]), 'b', label = 'fix P')
			s.plot(-np.log10(stim[:,1]), 'g', label = 'stim P')
			pl.legend()
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'pupil_correlation_%s.pdf'%lbl))

			with pd.get_store(self.ho.inputObject) as h5_file:
				h5_file.put("/%s/%s"%('deconvolve_pupil', 'fixation_interval_correlation_%s'%lbl), pd.DataFrame(fix))
				h5_file.put("/%s/%s"%('deconvolve_pupil', 'stimulus_interval_correlation_%s'%lbl), pd.DataFrame(stim))


		# shell()

		# self.ho.
		reward_h5file.close()
	
	def roi_interval_analysis_as_pupil(self, analysis_sample_rate = 1.333, interval = [-0.5,12.0], roi = 'V1', contrast = 'center_Z_pos', data_type = 'residuals', data_interval = [-0.5, 7.5]):

		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs

		reward_h5file = self.hdf5_file('reward')


		self.events_and_signals_in_time(data_type = data_type )
		cond_labels = ['blinks', 'fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		all_conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		all_event_data, blink_events = [], []
		stimulus_itis_fix_trials, fix_itis_fix_trials = [], []
		stimulus_itis_stimulus_trials, fix_itis_stimulus_trials = [], []

		events_of_interest = [] # fix R and stim R trials

		pupil_data = []

		nr_runs = 0

		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times')
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters')
			
			onsets_fix_reward_trials, raw_itis_of_fix_reward_trials, all_reward_itis_of_fix_reward_trials, fixation_reward_itis_fix_reward_trials, stimulus_reward_itis_fix_reward_trials = self.calculate_event_history_fix_reward(trial_times, parameter_data)
			onsets_stim_reward_trials, raw_itis_of_stim_reward_trials, all_reward_itis_of_stim_reward_trials, fixation_reward_itis_stim_reward_trials, stimulus_reward_itis_stim_reward_trials = self.calculate_event_history_stim_reward(trial_times, parameter_data)

			stimulus_itis_fix_trials.extend(stimulus_reward_itis_fix_reward_trials)
			fix_itis_fix_trials.extend(fixation_reward_itis_fix_reward_trials)
			stimulus_itis_stimulus_trials.extend(stimulus_reward_itis_stim_reward_trials)
			fix_itis_stimulus_trials.extend(fixation_reward_itis_stim_reward_trials)

			events_of_interest.append([onsets_fix_reward_trials + nr_runs * run_duration, onsets_stim_reward_trials + nr_runs * run_duration])

			this_run_all_events = []
			for cond in all_conds:
				this_run_all_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_all_events = np.array(this_run_all_events) + nr_runs * run_duration
			all_event_data.append(this_run_all_events)
			
			nr_runs += 1
		event_data = [np.concatenate([e[i] for e in all_event_data]) for i in range(len(all_event_data[0]))]
		events_of_interest = [np.array(np.round(np.concatenate([e[i] for e in events_of_interest]) * analysis_sample_rate), dtype = int) for i in range(2)]

		stimulus_itis = [np.array(stimulus_itis_fix_trials), np.array(stimulus_itis_stimulus_trials)]
		fix_itis = [np.array(fix_itis_fix_trials), np.array(fix_itis_stimulus_trials)]

		reward_h5file.close()
		# get residuals from earlier deconvolution
		with pd.get_store(self.hdf5_filename) as h5_file:
			residuals = np.array(h5_file.get("/%s/%s"%('deconvolution_results_mean_psc_hpf_data', roi + '_' + contrast + '_' + 'deconvolution_mean_residuals_psc_hpf_data')))


		correlation_indices = [np.array([np.arange(data_interval[0] * analysis_sample_rate, data_interval[1] * analysis_sample_rate) + t for t in e], dtype = int) for e in events_of_interest]
		correlation_timecourses = [residuals[ci] for ci in correlation_indices]
		correlation_timecourses_no_zero = [(ci.T - ci[:,:abs(data_interval[0]*analysis_sample_rate)].mean(axis = 1)).T for ci in correlation_timecourses]

		for ctnz, si, fi, lbl in zip(correlation_timecourses_no_zero, stimulus_itis, fix_itis, ['fix', 'stim']): # fix trial timecourses, stimulus trial timecourses
			fix, stim = np.array([spearmanr(tc, fi) for tc in ctnz.T]), np.array([spearmanr(tc, si) for tc in ctnz.T])

			f = pl.figure()
			s = f.add_subplot(121)
			s.plot(fix[:,0], 'b', label = 'corr with fix intervals')
			s.plot(stim[:,0], 'g', label = 'corr with stim intervals')
			pl.legend()
			s = f.add_subplot(122)
			s.plot(-np.log10(fix[:,1]), 'b', label = 'fix P')
			s.plot(-np.log10(stim[:,1]), 'g', label = 'stim P')
			pl.legend()
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + contrast + '_' + 'correlation_%s.pdf'%lbl))

			with pd.get_store(self.hdf5_filename) as h5_file:
				h5_file.put("/%s/%s"%('deconvolution_results_mean_psc_hpf_data', roi + '_' + contrast + '_' + 'fixation_interval_correlation_%s'%lbl), pd.DataFrame(fix))
				h5_file.put("/%s/%s"%('deconvolution_results_mean_psc_hpf_data', roi + '_' + contrast + '_' + 'stimulus_interval_correlation_%s'%lbl), pd.DataFrame(stim))


	def deconvolve(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution', signal_type = 'mean', data_type = 'psc_hpf_data'):
		results = []
		# neg_threshold = -1.0 * thres
		# neg_threshold = -neg_threshold
		# print threshold
		for roi in rois:
			results.append(self.deconvolve_roi(roi, threshold = threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type, data_type = data_type))
			results.append(self.deconvolve_roi(roi, threshold = -threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = signal_type, data_type = data_type))
			# results.append(self.deconvolve_roi(roi, threshold, mask_type = 'surround_center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type, data_type = data_type))
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_results' + '_' + signal_type + '_' + data_type
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_' + signal_type + '_' + data_type)
				reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_' + signal_type + '_per_run_' + data_type)
				reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_' + signal_type + '_residuals_' + data_type)
			except NoSuchNodeError:
				pass
			self.logger.info('deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type + '_' + data_type, r[-3], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			self.logger.info('per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type + '_per_run_' + data_type, r[-2], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			self.logger.info('deconvolution residuals for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type + '_residuals_' + data_type, np.array(r[-1]), 'deconvolution residuals for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def whole_brain_deconvolution(self, deco = True, average_intervals = [[3.5,12],[2,7]], to_surf = True):
		"""
		whole_brain_deconvolution takes all nii files from the reward condition and deconvolves the separate event types
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		nii_file_shape = list(niiFile.data.shape)
		
		nr_reward_runs = len(self.conditionDict['reward'])
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		time_signals = []
		interval = [0.0,16.0]
		
		if deco:
			event_data = []
			blink_events = []
			roi_data = []
			nr_runs = 0
			nii_data = np.zeros([nr_reward_runs] + nii_file_shape)
		
			for (j, r) in enumerate([self.runList[i] for i in self.conditionDict['reward']]):
				nii_data[j] = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','tf','psc'])).data
				this_run_events = []
				for cond in conds:
					this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
				this_run_events = np.array(this_run_events) + nr_runs * run_duration
				event_data.append(this_run_events)
				this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
				this_blink_events[:,0] += nr_runs * run_duration
				blink_events.append(this_blink_events)
				
				nr_runs += 1
		
			nii_data = nii_data.reshape((nr_reward_runs * nii_file_shape[0], -1))
			event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
			
			# nuisance_design = Design(nii_data.shape[0] * 2, tr/2.0 )
			# nuisance_design.configure(np.array([np.hstack(blink_events)]))
			deco = DeconvolutionOperator(inputObject = nii_data, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = True)
			# deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
			residuals = np.array(deco.residuals(), dtype = np.float32)
			residuals = residuals.reshape([residuals.shape[0]] + nii_file_shape[1:])[::2]
			
			try:
				os.system('rm -rf %s' % (os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'deco')))
				os.mkdir(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'deco'))
			except OSError:
				pass
			# save residuals
			outputFile = NiftiImage(residuals)
			outputFile.header = niiFile.header
			outputFile.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'residuals.nii.gz'))
			
		if to_surf:
			try:
				os.system('rm -rf %s' % (os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'surf')))
				os.mkdir(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'surf'))
			except OSError:
				pass
		for (i, c) in enumerate(cond_labels):
			if deco:
				# outputdata = deco.deconvolvedTimeCoursesPerEventTypeNuisance[i]
				outputdata = deco.deconvolvedTimeCoursesPerEventType[i]
				outputFile = NiftiImage(outputdata.reshape([outputdata.shape[0]]+nii_file_shape[1:]))
				outputFile.header = niiFile.header
				outputFile.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_deconv_' + c + '.nii.gz'))
			else:
				outputdata = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_deconv_' + c + '.nii.gz')).data
				# average over the interval [5,12] and [2,10] for reward and visual respectively. so, we'll just do [2,12]
			for (j, which_times) in enumerate(['reward', 'visual']):
				timepoints_for_averaging = (np.linspace(interval[0], interval[1], outputdata.shape[0]) < average_intervals[j][1]) * (np.linspace(interval[0], interval[1], outputdata.shape[0]) > average_intervals[j][0])
				meaned_data = outputdata[timepoints_for_averaging].mean(axis = 0)
				outputFile = NiftiImage(meaned_data.reshape(nii_file_shape[1:]))
				outputFile.header = niiFile.header
				ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_deconv_mean_' + c + '_' + which_times + '.nii.gz')
				outputFile.save(ofn)
			
				if to_surf:
					# vol to surf?
					# for (label, f) in zip(['left', 'right'], [left_file, right_file]):
					vsO = VolToSurfOperator(inputObject = ofn)
					sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
					vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute(wait = False)
				
					for hemi in ['lh','rh']:
						ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
						ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
						ssO.execute(wait = False)
		
		# now create the necessary difference images:
		# only possible if deco has already been run...
		for i in [0,2]:
			for (j, which_times) in enumerate(['reward', 'visual']):
				ipfs = [NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_deconv_mean_' + cond_labels[i] + '_' + which_times + '.nii.gz')), NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_deconv_mean_' + cond_labels[i+1] + '_' + which_times + '.nii.gz'))]
				diff_d = ipfs[0].data - ipfs[1].data
			
				ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), ['fix','','stimulus'][i] + '_reward_diff' + '_' + which_times + '.nii.gz')
				outputFile = NiftiImage(diff_d)
				outputFile.header = ipfs[0].header
				outputFile.save(ofn)
			
			
				if to_surf:
					vsO = VolToSurfOperator(inputObject = ofn)
					sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
					vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute(wait = False)
				
					for hemi in ['lh','rh']:
						ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
						ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
						ssO.execute(wait = False)
	
	def anova_stats_over_time(self, data_type = 'fmri', sample_rate = 2000, comparison_rate = 100):
		"""perform per-timepoint two-way anova on time-varying signals in four conditions. """
		import rpy2.robjects as robjects
		import rpy2.rlike.container as rlc
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		factor_names = ['visual', 'reward', 'interaction']
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		if data_type == 'fmri':
			# assuming this deconvolution period is now 16 s long.
			this_run_group_name = 'deconvolution_results'
			for deconv in reward_h5file.iterNodes(where = '/' + this_run_group_name, classname = 'Array'):
				# this runs through different deconvolution results
				if deconv._v_name.split('_')[-1] == 'run':	# these are per_run analyses, so we can do the anova on them.
					#get the data
					thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
					# do analysis. 
					these_data = eval('thisRunGroup.' + deconv._v_name + '.read()').transpose(2,1,0) # after transpose: timepoints by conditions by runs
					stat_results = np.zeros((these_data.shape[0], 2, 3))	# timepoints, by F/p values by 2 factors plus interaction
					timepoints = np.linspace(0, 16, these_data.shape[0])
					for (i, td) in enumerate(these_data):
						visual_conds = np.tile(np.array([0,0,1,1]), td.shape[1]).reshape(td.shape[1],4).T
						reward_conds = np.tile(np.array([0,1,0,1]), td.shape[1]).reshape(td.shape[1],4).T
					
						d = rlc.OrdDict([('visual', robjects.IntVector(list(visual_conds.ravel()))), ('reward', robjects.IntVector(list(reward_conds.ravel()))), ('values', robjects.FloatVector(list(td.ravel())))])
						robjects.r.assign('dataf', robjects.DataFrame(d))
						robjects.r('attach(dataf)')
						res = robjects.r('res = summary(aov(values ~ factor(visual)*factor(reward), dataf))')
						pvals =  res[0][4]
						fvals =  res[0][4]
						stat_results[i,0,:] = np.array([pvals[j] for j in range(3)])
						stat_results[i,1,:] = np.array([fvals[j] for j in range(3)])
						
					try:
						reward_h5file.remove_node(where = thisRunGroup, name = deconv._v_name + '_stats')
					except NoSuchNodeError:
						pass
					reward_h5file.create_array(thisRunGroup, deconv._v_name + '_stats', stat_results, 'ANOVA timecourses conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
					
					fig = pl.figure(figsize = (9, 4))
					s = fig.add_subplot(111)
					s.axhline(0, -10, 30, linewidth = 0.25)
					for i in range(3):
						s.plot(timepoints, -np.log10(stat_results[:,0,i]), ['r','r--','k'][i], alpha = [0.7, 0.7, 0.4][i], label = factor_names[i])
					s.set_xlabel('time [s]')
					s.set_ylabel('-log$_{10}$ p')
					s.set_xlim([-1.5, 16 + 1.5])
					leg = s.legend(fancybox = True)
					leg.get_frame().set_alpha(0.5)
					if leg:
						for t in leg.get_texts():
						    t.set_fontsize('small')    # the legend text fontsize
						for l in leg.get_lines():
						    l.set_linewidth(3.5)  # the legend line width
					s.set_title('reward signal stats for ' + deconv._v_name)
					pl.draw()
					pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), deconv._v_name+'_stats.pdf'))
		
		elif data_type == 'pupil':
			pupil_signals = reward_h5file.get_node(where = '/', name = 'all_pupil_scores', classname = 'Array').read().transpose(2,1,0)[::(sample_rate/comparison_rate)]
			# average per run, after which we need to transpose to conform to the bold data format. 
			pupil_signals = pupil_signals.reshape((pupil_signals.shape[0], 6, pupil_signals.shape[1]/6 , pupil_signals.shape[2])).mean(axis = 2).transpose(0,2,1)
			stat_results = np.zeros((pupil_signals.shape[0], 2, 3))	# timepoints, by F/p values by 2 factors plus interaction
			timepoints = np.linspace(0, 10, stat_results.shape[0])
			for (i, td) in enumerate(pupil_signals):
				visual_conds = np.tile(np.array([0,0,1,1]), td.shape[1]).reshape(td.shape[1],4).T
				reward_conds = np.tile(np.array([0,1,0,1]), td.shape[1]).reshape(td.shape[1],4).T
					
				d = rlc.OrdDict([('visual', robjects.IntVector(list(visual_conds.ravel()))), ('reward', robjects.IntVector(list(reward_conds.ravel()))), ('values', robjects.FloatVector(list(td.ravel())))])
				# shell()
				robjects.r.assign('dataf', robjects.DataFrame(d))
				robjects.r('attach(dataf)')
				res = robjects.r('res = summary(aov(values ~ factor(visual)*factor(reward), dataf))')
				pvals =  res[0][4]
				fvals =  res[0][3]
				stat_results[i,0,:] = np.array([pvals[j] for j in range(3)])
				stat_results[i,1,:] = np.array([fvals[j] for j in range(3)])
			
			fig = pl.figure(figsize = (9, 5))
			s = fig.add_subplot(211)
			s.axhline(0, -10, 30, linewidth = 0.25)
			for i in range(3):
				s.plot(timepoints, -np.log10(stat_results[:,0,i]), ['r','r--','k'][i], alpha = [0.7, 0.7, 0.4][i], label = factor_names[i])
			s.set_ylabel('-log$_{10}$ p')
			s.set_xlim([-1.5, 10 + 1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for l in leg.get_lines():
				    l.set_linewidth(3.5)  # the legend line width
			s.set_title('reward signal stats for pupil')
			s = fig.add_subplot(212)
			pm = pupil_signals.mean(axis = 2).transpose()
			ps = 1.96 * pupil_signals.std(axis = 2).transpose() / sqrt(pupil_signals.shape[1])
			for i in range(pm.shape[0]):
				pl.plot(timepoints, pm[i], ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = cond_labels[i])
				pl.fill_between(timepoints, pm[i]+ps[i], pm[i]-ps[i], color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])
			s.set_ylabel('$\Delta$ Z-scored Pupil Size')
			s.set_xlim([-1.5, 10 + 1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.75)
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for l in leg.get_lines():
				    l.set_linewidth(3.5)  # the legend line width
			s.set_xlabel('time [s]')
			pl.draw()
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'pupil_stats.pdf'))
			
			try:
				reward_h5file.remove_node(where = '/', name = 'all_pupil_stats')
			except NoSuchNodeError:
				pass
			reward_h5file.create_array('/', 'all_pupil_stats', stat_results, 'ANOVA timecourses on pupil data conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			
		
		reward_h5file.close()
		pl.show()
			# 		
			# elif data_type == 'pupil':
			# 	this_run_group_name = 'all_pupil_scores'
			
	
	def run_glm_on_hdf5(self, data_type = 'hpf_data', analysis_type = 'per_trial', post_fix_for_text_file = ['all_trials'], functionalPostFix = ['mcf'], which_conditions = ['mapper']): # 'reward','mapper'
		if 'reward' in which_conditions:
			reward_h5file = self.hdf5_file('reward', mode = 'r+')
			super(SingleRewardSession, self).run_glm_on_hdf5(run_list = [self.runList[i] for i in self.conditionDict['reward']], hdf5_file = reward_h5file, data_type = data_type, analysis_type = analysis_type, post_fix_for_text_file = post_fix_for_text_file, functionalPostFix = functionalPostFix)
			reward_h5file.close()
		if 'mapper' in which_conditions:
			mapper_h5file = self.hdf5_file('mapper', mode = 'r+')
			super(SingleRewardSession, self).run_glm_on_hdf5(run_list = [self.runList[i] for i in self.conditionDict['mapper']], hdf5_file = mapper_h5file, data_type = data_type, analysis_type = analysis_type, post_fix_for_text_file = post_fix_for_text_file, functionalPostFix = functionalPostFix)
			mapper_h5file.close()
		

	def mean_stats_for_roi(self, roi, threshold = 3.5, mask_type = 'center_surround_Z', stats_types = ['blank_silence', 'blank_sound', 'visual_silence', 'visual_sound'], mask_direction = 'pos'):
		"""docstring for mean_stats_for_roi"""
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		input_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, 'hpf_data')
		
		roi_data = np.zeros((len(stats_types), len(self.conditionDict['reward']), int(mapping_mask.sum())))
		for i, stat in enumerate(stats_types):
			for j, r in enumerate([self.runList[rew] for rew in self.conditionDict['reward']]):
				rd = self.roi_data_from_hdf(reward_h5file, r, roi, stat).ravel()
				sd = self.roi_data_from_hdf(reward_h5file, r, roi, 'hpf_data').mean(axis = 1).ravel()
				roi_data[i,j] = rd[mapping_mask] / sd[mapping_mask]
		
		reward_h5file.close()
		mapper_h5file.close()
		
		return roi_data
		
	def mean_stats(self, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], threshold = 2.3, mask_type = 'center_Z', stats_types = ['blank_silence', 'visual_sound', 'visual_silence', 'blank_sound'], mask_direction = 'pos' ):
		"""docstring for mean_stats"""
		res = []
		for roi in rois:
			res.append(self.mean_stats_for_roi(roi, threshold = threshold, mask_type = mask_type, stats_types = stats_types, mask_direction = mask_direction))
		# res = np.array(res)
		
		diff_res = []
		for d in res:
			# over rois
			dr = (d[1:,:,:] - d[0,:,:]).mean(axis = 2)
			diff_res.append([dr.mean(axis = 1), 1.96 * dr.std(axis = 1) / sqrt(dr.shape[1])])
		
		diff_res = np.array(diff_res)
		
		colors = ['g', 'g', 'b', 'k', 'y', 'm', 'c']
		alphas = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 05]
		
		fig = pl.figure(figsize = (12, 4))
		pl.subplots_adjust(left = 0.05, right = 0.97)
		s = fig.add_subplot(111)
		width = 1.0 / (diff_res.shape[1] + 2)
		pl.plot([-1, len(rois) + 1.0], [0,0], 'k', linewidth = 0.5)
		rects = []
		for i in np.arange(diff_res.shape[2]):
			rects.append(pl.bar(np.arange(diff_res.shape[0])+(i*+width), height = diff_res[:,0,i], width = width, yerr = diff_res[:,1,i], color=colors[i], alpha = alphas[i], edgecolor = (1.0,1.0,1.0), linewidth = 0.0, ecolor = (0.5, 0.5, 0.5), capsize = 0))
		pl.ylabel('beta [a.u.]')
		pl.xticks(np.arange(len(rois))+width, rois )
		s.set_xlim(-0.5, diff_res.shape[0]+.5)
		leg = pl.legend( tuple([r[0] for r in rects]), tuple([st.replace('sound', 'reward').replace('blank','fix').replace('silence','no_reward') for st in stats_types[1:]]), fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize(9)    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(2.5)  # the legend line width
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'betas.pdf'))
		
		return res

	def correlate_data_from_run(self, run, rois = ['V1', 'V2', 'V3', 'V4', 'V3AB'], data_pairs = [[['mapper', 'center_pe'], ['reward', 'visual_cope']], [['mapper', 'center_pe'], ['reward', 'reward_cope']]], plot = True, which_mapper_run = 0):
		"""
		correlates two types of data from regions of interest with one another, but more generally than the other function. 
		This function allows you to specify from what file and what type of stat you are going to correlate with one another.
		Specifically, the data_pairs argument is a list of two-item lists which specify the to be correlated stats
		"""
		from scipy import stats
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		corrs = np.zeros((len(rois), len(data_pairs)))
		colors = ['r', 'g', 'b', 'y', 'm', 'c']
		if reward_h5file != None:
			# there was a file and it has data in it
			if plot:	
				fig = pl.figure(figsize = (len(rois)*4, 3))
			for roi in rois:
				if plot: 
					s = fig.add_subplot(1, len(rois), rois.index(roi) + 1)
					s.set_title(roi, fontsize=9)
				for i in range(len(data_pairs)):
					if data_pairs[i][0][0] == 'mapper':
						cope1 = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][which_mapper_run]], roi, data_pairs[i][0][1])
					elif data_pairs[i][0][0] == 'reward':
						cope1 = self.roi_data_from_hdf(reward_h5file, run, roi, data_pairs[i][0][1])
					
					if data_pairs[i][1][0] == 'mapper':
						cope2 = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][which_mapper_run]], roi, data_pairs[i][1][1])
					elif data_pairs[i][1][0] == 'reward':
						cope2 = self.roi_data_from_hdf(reward_h5file, run, roi, data_pairs[i][1][1])
					
					if cope1 != None and cope2 != None:
						if plot:
							(ar,br)=polyfit(cope1[:,0], cope2[:,0], 1)
							xr=polyval([ar,br],cope1[:,0])
							pl.plot(cope1[:,0], xr, colors[i] + '-', alpha = 0.25, linewidth = 1.5)
							pl.plot(cope1[:,0], cope2[:,0], marker = 'o', ms = 3, mec = 'w', c = colors[i], mew = 0.5, alpha = 0.125, linewidth = 0) # , alpha = 0.25
							s.set_xlabel('-'.join(data_pairs[i][0]), fontsize=9)
							if rois.index(roi) == 0:
								s.set_ylabel('-'.join(data_pairs[i][1]), fontsize=9)
						srs = stats.spearmanr(cope1, cope2)
						corrs[rois.index(roi), i] = srs[0]
					else:
						self.logger.info('No data to correlate for ' + str(data_pairs[i]) + ' ' + str(roi))
		if plot:
			pl.draw()
			pdf_file_name = os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'data_scatter_' + str(run.ID) + '.pdf')
			pl.savefig(pdf_file_name)
		reward_h5file.close()
		mapper_h5file.close()
		return corrs
	
	def correlate_data(self, rois = ['V1', 'V2d', 'V2v', 'V3d', 'V3v', 'V4', 'V3AB'], data_pairs = [[['mapper', 'center_pe'], ['reward', 'visual_cope']], [['mapper', 'center_pe'], ['reward', 'reward_cope']]], scatter_plots = False, which_mapper_run = 0):
		"""
		correlate reward run cope values with one another from all reward runs separately.
		"""
		all_corrs = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			all_corrs.append(self.correlate_data_from_run(run = r, rois = rois, data_pairs = data_pairs, plot = scatter_plots, which_mapper_run = which_mapper_run))
			
		cs = np.array(all_corrs)
		colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
		comparison_names = ['-'.join([':'.join(d) for d in dp]) for dp in data_pairs]
		
		fig = pl.figure(figsize = (4, len(rois)*1.5))
		pl.subplots_adjust(hspace=0.4, wspace=0.4)
		for roi in rois:
			s = fig.add_subplot(len(rois), 1, rois.index(roi) + 1)
			s.grid()
			for i in range(cs.shape[-1]):
				pl.plot(np.arange(6), cs[:,rois.index(roi),i], colors[i], linewidth = 2.5, alpha = 0.75)
			s.set_title(roi, fontsize=9)
			if rois.index(roi) == len(rois)-1:
				s.set_xlabel('run number', fontsize=9)
			if rois.index(roi) == 2:
				s.set_ylabel('spearman correlation', fontsize=9)
			s.axis([-0.25,5.35,-1,1])
			# designate bad runs:
			s.axvspan(3.75, 4.25, facecolor='y', alpha=0.25, edgecolor = 'w')
			s.axvspan(1.75, 2.25, facecolor='y', alpha=0.25, edgecolor = 'w')
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'data_spearman_rho_over_runs.pdf'))
		
		# average across runs
		meancs = cs.mean(axis = 0)
		sdcs = 1.96 * cs.std(axis = 0) / sqrt(6) 
		
		fig = pl.figure(figsize = (12, 4))
		pl.subplots_adjust(left = 0.05, right = 0.97)
		s = fig.add_subplot(111)
		width = 1.0 / (meancs.shape[1] + 1)
		pl.plot([-1, len(rois) + 1.0], [0,0], 'k', linewidth = 0.5)
		rects = []
		for i in range(meancs.shape[1]):
			rects.append(pl.bar(np.arange(meancs.shape[0])+(i*+width), height = meancs[:,i], width = width, yerr = sdcs[:,i], color=colors[i], alpha = 0.7, edgecolor = (0.5, 0.5, 0.5), linewidth = 0.0, ecolor = (0.5, 0.5, 0.5), capsize = 0))
		pl.ylabel('Spearman correlation')
		pl.xticks(np.arange(len(rois))+width, rois )
		s.set_xlim(-0.5, meancs.shape[0]+2.5)
		leg = pl.legend( tuple([r[0] for r in rects]), tuple(comparison_names), fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize(9)    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(1.5)  # the legend line width
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'data_spearman_rho_bar_over_runs.pdf'))
		
		# average across runs - but take out runs with lower confidence
		meancs = cs[[0,1,3,5]].mean(axis = 0)
		sdcs = 1.96 * cs[[0,1,3,5]].std(axis = 0) / sqrt(4) 
		
		fig = pl.figure(figsize = (12, 4))
		pl.subplots_adjust(left = 0.05, right = 0.97)
		s = fig.add_subplot(111)
		width = 1.0 / (meancs.shape[1] + 1)
		pl.plot([-1, len(rois) + 1.0], [0,0], 'k', linewidth = 0.5)
		rects = []
		for i in range(meancs.shape[1]):
			rects.append(pl.bar(np.arange(meancs.shape[0])+(i*+width), height = meancs[:,i], width = width, yerr = sdcs[:,i], color=colors[i], alpha = 0.7, edgecolor = (0.5, 0.5, 0.5), linewidth = 0.0, ecolor = (0.5, 0.5, 0.5), capsize = 0))
		pl.ylabel('Spearman correlation')
		pl.xticks(np.arange(len(rois))+width, rois )
		s.set_xlim(-0.5, meancs.shape[0]+2.5)
		leg = pl.legend( tuple([r[0] for r in rects]), tuple(comparison_names), fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize(9)    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(1.5)  # the legend line width
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'data_spearman_rho_bar_over_runs_HC.pdf'))
		
		return all_corrs
	
	def histogram_data_from_roi(self, roi, threshold = 3.5, mask_type = 'center_surround_Z', stats_types = ['visual_reward_fix_reward','visual_silence_fix_silence'], mask_direction = 'pos'):
		"""docstring for mean_stats"""
		"""docstring for mean_stats_for_roi"""
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		input_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, 'hpf_data')
		
		roi_data = np.zeros((len(stats_types), len(self.conditionDict['reward']), int(mapping_mask.sum())))
		for i, stat in enumerate(stats_types):
			for j, r in enumerate([self.runList[rew] for rew in self.conditionDict['reward']]):
				rd = self.roi_data_from_hdf(reward_h5file, r, roi, stat).ravel()
				sd = self.roi_data_from_hdf(reward_h5file, r, roi, 'hpf_data').mean(axis = 1).ravel()
				roi_data[i,j] = rd[mapping_mask] / sd[mapping_mask]
		
		reward_h5file.close()
		mapper_h5file.close()
		
		return roi_data
	
	def histogram(self, rois = ['V1', 'V2', 'V3', 'V3A', 'V4'], threshold = 3.5, mask_type = 'center_surround_Z', stats_types = ['visual_reward_fix_reward','visual_silence_fix_silence'], mask_direction = 'pos'):
		"""docstring for mean_stats"""
		res = []
		for roi in rois:
			res.append(self.histogram_data_from_roi(roi, threshold = threshold, mask_type = mask_type, stats_types = stats_types, mask_direction = mask_direction))
		# res = np.array(res)
		
		diff_res = []
		for d in res:
			# over rois
			diff_res.append(d[0] - d[1])
		
		colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
		
		fig = pl.figure(figsize = (15, 8))
		pl.subplots_adjust(left = 0.05, right = 0.97, hspace=0.4, wspace=0.4)
		for i, roi_data in enumerate(diff_res):
			for j, run_roi_data in enumerate(roi_data):
				s = fig.add_subplot(len(rois), roi_data.shape[0], j+1 + roi_data.shape[0]*i)
				s.axvspan(-0.000001, 0.000001, facecolor='k', alpha=1.0, edgecolor = 'k')
				pl.hist(run_roi_data, bins = 20, alpha = 0.5, range = [-0.25, 0.25], normed = True, histtype = 'stepfilled', color = colors[i], linewidth = 2.0 )
				wilc = sp.stats.wilcoxon(run_roi_data)
				s.set_title(rois[i] + ' run ' + str(j+1))
				s.set_xlim([-0.25, 0.25])
				pl.text(-0.2,2,'p-value: ' + '%1.4f'%wilc[1])
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'beta_histograms_' + '-'.join(stats_types) + '.pdf'))
		pl.draw()
		return diff_res
	
	def correlate_pupil_and_BOLD_for_roi_per_time(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range_BOLD = [5.0, 9.0], time_range_pupil = [0.0, 10.0], stepsize = 0.25, area = ''):
		"""docstring for correlate_pupil_and_BOLD"""
		
		# take data 
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		pupil_data = []
		tr_timings = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			tr_timings.append(np.arange(0, run_duration, tr) + nr_runs * run_duration)
			# take pupil data
			try:
				thisRunGroup = reward_h5file.get_node(where = '/', name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))[1], classname='Group')
				# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			except NoSuchNodeError:
				self.logger.error('no such node.')
				pass
			this_run_pupil_data = thisRunGroup.filtered_pupil_zscore.read()
			this_run_pupil_data = this_run_pupil_data[(this_run_pupil_data[:,0] > thisRunGroup.trial_times.read()['trial_phase_timestamps'][0,0,0])][:run_duration * sample_rate]
			this_run_pupil_data[:,0] = ((this_run_pupil_data[:,0] - this_run_pupil_data[0,0]) / 1000.0) + nr_runs * run_duration
			pupil_data.append(this_run_pupil_data)
			
			nr_runs += 1
		reward_h5file.close()
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		pdc = np.concatenate(pupil_data)[::50]
		trts = np.concatenate(tr_timings)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		mapper_h5file.close()
		bold_timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		all_corrs = []
		fig = pl.figure(figsize = (9, 4))
		s = fig.add_subplot(1,1,1)
		for (i, e) in enumerate(event_data):
			bold_trials = np.array([bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts <= se + time_range_BOLD[1])].mean() for se in e])
			pupil_trials = []
			for time in np.arange(time_range_pupil[0],time_range_pupil[1]-stepsize, stepsize):
				pupil_trials.append(np.array([pdc[(pdc[:,0] > (se + time)) * (pdc[:,0] <= (se + time + stepsize)),1].mean() - pdc[(pdc[:,0] > se - 2.0) * (pdc[:,0] <= se),1].mean() for se in e]))
			shell()
			all_corrs.append(np.array([spearmanr(bold_trials, p) for p in pupil_trials]))
			
			pl.plot(np.arange(time_range_pupil[0],time_range_pupil[1]-stepsize, stepsize) + stepsize * 0.5, all_corrs[-1][:,0], color = ['b','b','g','g'][i], alpha = 0.7 * [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			
		s.set_title(self.subject.initials + ' ' + area)
		s.set_xlabel('time')
		s.set_ylabel('Rho')
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_pupil_corr.pdf'))
		return np.array(all_corrs)
		
	def correlate_pupil_and_BOLD_for_roi_variance(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range_BOLD = [0.0, 10.0], time_range_pupil = [0.0, 10.0], stepsize = 0.25, area = ''):
		"""docstring for correlate_pupil_and_BOLD"""
		
		# take data 
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		pupil_data = []
		tr_timings = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			tr_timings.append(np.arange(0, run_duration, tr) + nr_runs * run_duration)
			# take pupil data
			try:
				thisRunGroup = reward_h5file.get_node(where = '/', name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))[1], classname='Group')
				# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			except NoSuchNodeError:
				self.logger.error('no such node.')
				pass
			this_run_pupil_data = thisRunGroup.filtered_pupil_zscore.read()
			this_run_pupil_data = this_run_pupil_data[(this_run_pupil_data[:,0] > thisRunGroup.trial_times.read()['trial_phase_timestamps'][0,0,0])][:run_duration * sample_rate]
			this_run_pupil_data[:,0] = ((this_run_pupil_data[:,0] - this_run_pupil_data[0,0]) / 1000.0) + nr_runs * run_duration
			pupil_data.append(this_run_pupil_data)
			
			nr_runs += 1
		reward_h5file.close()
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		pdc = np.concatenate(pupil_data)[::50]
		trts = np.concatenate(tr_timings)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		mapper_h5file.close()
		bold_timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		from scipy.stats import *
		
		all_corrs = []
		fig = pl.figure(figsize = (9, 4))
		for (i, e) in enumerate(event_data):
			# standard version works on just the variance
			# var_bold_trials = np.array([bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts < se + time_range_BOLD[1])].std() for se in e])
			# non-standard version analyzes as a random walk 
			var_bold_trials = np.array([np.abs(np.diff(bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts < se + time_range_BOLD[1])])).mean() for se in e])
			pupil_trials = np.array([pdc[(pdc[:,0] > (se + time_range_pupil[0])) * (pdc[:,0] <= (se + time_range_pupil[1])),1].mean() for se in e])
			
			pupil_trials[pupil_trials < np.median(pupil_trials)].mean(), pupil_trials[pupil_trials > np.median(pupil_trials)].mean() 
			
			all_corrs.append(spearmanr(var_bold_trials, pupil_trials))
			s1 = fig.add_subplot(1,2,1)
			pl.plot(var_bold_trials, pupil_trials, 'o', color = ['b','b','g','g'][i], ms = 3.0, mew = 1.5, mec = 'None', alpha = 0.5 * [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			s2 = fig.add_subplot(1,2,2)
			s2.axhline(0, -1, 4, linewidth = 0.25)
			pl.bar(i, all_corrs[-1][0], width = 0.5, color = ['b','b','g','g'][i], alpha = 0.7 * [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			
		s1.set_title(self.subject.initials + ' ' + area)
		s1.set_xlabel('BOLD')
		s1.set_ylabel('Pupil')
		s2.set_xlabel('conditions')
		s2.set_ylabel('Spearman\'s Rho')
		
		leg = s1.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_pupil_corr.pdf'))
		return np.array(all_corrs)
	
	def calculate_BOLD_variance_for_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range_BOLD = [2.0, 8.0], time_range_pupil = [2.0, 8.0], stepsize = 0.25, area = '', data_type = 'psc_hpf_data'):
		"""docstring for correlate_pupil_and_BOLD"""
		
		# take data 
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		pupil_data = []
		tr_timings = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, data_type))
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			tr_timings.append(np.arange(0, run_duration, tr) + nr_runs * run_duration)
			# take pupil data
			try:
				thisRunGroup = reward_h5file.get_node(where = '/', name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))[1], classname='Group')
				# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			except NoSuchNodeError:
				self.logger.error('no such node.')
				pass
			this_run_pupil_data = thisRunGroup.filtered_pupil_zscore.read()
			this_run_pupil_data = this_run_pupil_data[(this_run_pupil_data[:,0] > thisRunGroup.trial_times.read()['trial_phase_timestamps'][0,0,0])][:run_duration * sample_rate]
			this_run_pupil_data[:,0] = ((this_run_pupil_data[:,0] - this_run_pupil_data[0,0]) / 1000.0) + nr_runs * run_duration
			pupil_data.append(this_run_pupil_data)
			
			nr_runs += 1
		reward_h5file.close()
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = roi_data
		
		roi_data = np.hstack(roi_data)
		pdc = np.concatenate(pupil_data)[::50]
		trts = np.concatenate(tr_timings)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		mapper_h5file.close()
		bold_timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		from scipy.stats import *
		
		all_corrs = []
		for (i, e) in enumerate(event_data):
			# standard version works on just the variance
			var_bold_trials = np.array([bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts < se + time_range_BOLD[1])].var() for se in e])
			mean_bold_trials = np.array([bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts < se + time_range_BOLD[1])].mean() for se in e])
			# non-standard version analyzes as a random walk 
			rw_bold_trials = np.array([np.abs(np.diff(bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts < se + time_range_BOLD[1])])).mean() for se in e])
			pupil_trials = np.array([pdc[(pdc[:,0] > (se + time_range_pupil[0])) * (pdc[:,0] <= (se + time_range_pupil[1])),1].mean() for se in e])
				
			all_corrs.append(np.array([mean_bold_trials, var_bold_trials, rw_bold_trials, pupil_trials]))
		return all_corrs
	
	def correlate_pupil_and_BOLD(self, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000):
		corrs = []
		areas = ['V1', 'V2', 'V3', 'V3AB', 'V4']
		for roi in areas:
			corrs.append(self.correlate_pupil_and_BOLD_for_roi_per_time(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi))
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'pupil-BOLD_correlation_results'
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'pupil/bold correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for (i, c) in enumerate(corrs):
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = areas[i])
			except NoSuchNodeError:
				pass
			reward_h5file.create_array(thisRunGroup, areas[i], np.array(corrs[i]), 'pupil-bold correlation timecourses conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
		
		# pl.show()
	
	
	def correlate_pupil_and_BOLD_variance(self, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000):
		corrs = []
		areas = ['V1', 'V2', 'V3', 'V3AB', 'V4']
		for roi in areas:
			corrs.append(self.correlate_pupil_and_BOLD_for_roi_variance(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi))
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'pupil-BOLD_variance_correlation_results'
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'pupil/bold correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for (i, c) in enumerate(corrs):
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = areas[i] + '_' +  mask_type + '_' + mask_direction)
			except NoSuchNodeError:
				pass
			reward_h5file.create_array(thisRunGroup, areas[i] + '_' +  mask_type + '_' + mask_direction, np.array(corrs[i]), 'pupil-bold correlation timecourses conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def calculate_BOLD_variance(self, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, data_type = 'psc_hpf_data'):
		corrs = []
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		areas = ['V1', 'V2', 'V3', 'V3AB', 'V4']
		for roi in areas:
			corrs.append(self.calculate_BOLD_variance_for_roi(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi, data_type = data_type))
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'BOLD_variance_results'
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'pupil/bold correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for (i, c) in enumerate(corrs):
			for j in range(len(cond_labels)):
				try:
					reward_h5file.remove_node(where = thisRunGroup, name = areas[i] + '_' +  mask_type + '_' + mask_direction + '_' + cond_labels[j])
				except NoSuchNodeError:
					pass
				reward_h5file.create_array(thisRunGroup, areas[i] + '_' +  mask_type + '_' + mask_direction + '_' + cond_labels[j], np.array(corrs[i][j]), 'bold signal level and variability conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def cross_correlate_pupil_and_BOLD_for_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range_BOLD = [5.0, 10.0], time_range_pupil = [0.5, 2.0], stepsize = 0.25, area = '', color = 1.0):
		"""docstring for correlate_pupil_and_BOLD"""
		
		# take data 
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		pupil_data = []
		tr_timings = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			tr_timings.append(np.arange(0, run_duration, tr) + nr_runs * run_duration)
			# take pupil data
			try:
				thisRunGroup = reward_h5file.get_node(where = '/', name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))[1], classname='Group')
				# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			except NoSuchNodeError:
				self.logger.error('no such node.')
				pass
			this_run_pupil_data = thisRunGroup.filtered_pupil_zscore.read()
			this_run_pupil_data = this_run_pupil_data[(this_run_pupil_data[:,0] > thisRunGroup.trial_times.read()['trial_phase_timestamps'][0,0,0])][:run_duration * sample_rate]
			this_run_pupil_data[:,0] = ((this_run_pupil_data[:,0] - this_run_pupil_data[0,0]) / 1000.0) + nr_runs * run_duration
			pupil_data.append(this_run_pupil_data[::int(sample_rate*tr)])
			
			nr_runs += 1
		reward_h5file.close()
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		pdc = np.concatenate(pupil_data)
		trts = np.concatenate(tr_timings)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		mapper_h5file.close()
		bold_timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		from scipy.signal import correlate
		from scipy.stats import spearmanr, linregress
		
		all_results = []
		all_spearman_results = []
		for (i, e) in enumerate(event_data):
			bold_trial_data = np.array([bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts <= se + time_range_BOLD[1])].mean() for se in e[:-1]])
			pupil_trial_data = np.array([pdc[(pdc[:,0] > se + time_range_pupil[0]) * (pdc[:,0] <= se + time_range_pupil[1]),1].mean() for se in e[:-1]])
		
			correlation = correlate(pupil_trial_data, bold_trial_data, 'same')
			scorr = spearmanr(pupil_trial_data, bold_trial_data)
			
			midpoint = correlation.shape[0] / 2
			plot_range = 20
			
			all_results.append(correlation[midpoint-plot_range/2:midpoint+plot_range/2])
			all_spearman_results.append(scorr)
			
			# pl.plot(np.linspace(-plot_range/2*tr, plot_range/2*tr, correlation[midpoint-plot_range/2:midpoint+plot_range/2].shape[0]), correlation[midpoint-plot_range/2:midpoint+plot_range/2], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			
			pl.plot(pupil_trial_data, bold_trial_data, ['ob','ob','og','og'][i], alpha = [0.5, 1.0, 0.5, 1.0][i] * 0.2, mec = 'w', mew = 1, ms = 6)
			
			# linear regression for regression lines
			slope, intercept, r_value, p_value, slope_std_error = stats.linregress(pupil_trial_data, bold_trial_data)
			predict_y = intercept + slope * pupil_trial_data
			pl.plot(pupil_trial_data, predict_y, ['--b','--b','--g','--g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], mec = 'w', mew = 1, ms = 6, label = cond_labels[i])
			
		#  all thingies across 
		# correlation = correlate(pdc[:,1], bold_timeseries, 'same')
		# midpoint = correlation.shape[0] / 2
		# plot_range = 20
		# 
		# pl.plot(np.linspace(-plot_range/2*tr, plot_range/2*tr, correlation[midpoint-plot_range/2:midpoint+plot_range/2].shape[0]), correlation[midpoint-plot_range/2:midpoint+plot_range/2], 'r', alpha = 0.5, label = 'across conditions')
		# shell()
		return (all_results, all_spearman_results)
	
	def cross_correlate_pupil_and_BOLD_for_roi_over_time(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range_BOLD = [-3.0, 16.0], time_range_pupil = [0.0, 2.0], stepsize = 0.25, area = '', color = 1.0):
		"""docstring for correlate_pupil_and_BOLD"""
		
		# take data 
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		pupil_data = []
		tr_timings = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			tr_timings.append(np.arange(0, run_duration, tr) + nr_runs * run_duration)
			# take pupil data
			try:
				thisRunGroup = reward_h5file.get_node(where = '/', name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))[1], classname='Group')
				# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			except NoSuchNodeError:
				self.logger.error('no such node.')
				pass
			this_run_pupil_data = thisRunGroup.filtered_pupil_zscore.read()
			this_run_pupil_data = this_run_pupil_data[(this_run_pupil_data[:,0] > thisRunGroup.trial_times.read()['trial_phase_timestamps'][0,0,0])][:run_duration * sample_rate]
			this_run_pupil_data[:,0] = ((this_run_pupil_data[:,0] - this_run_pupil_data[0,0]) / 1000.0) + nr_runs * run_duration
			pupil_data.append(this_run_pupil_data[::int(sample_rate*tr)])
			
			nr_runs += 1
		reward_h5file.close()
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		pdc = np.concatenate(pupil_data)
		trts = np.round(np.concatenate(tr_timings) * 4)/4
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		mapper_h5file.close()
		bold_timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		from scipy.signal import correlate
		from scipy.stats import spearmanr, linregress
		
		all_results = []
		all_spearman_results = []
		for (i, e) in enumerate(event_data):
			bold_trial_data = np.array([bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts <= se + time_range_BOLD[1])] for se in e[:-1]])
			pupil_trial_data = np.array([pdc[(pdc[:,0] > se + time_range_pupil[0]) * (pdc[:,0] <= se + time_range_pupil[1]),1].mean() for se in e[:-1]])
			time_trial_data = np.array([trts[(trts > se + time_range_BOLD[0]) * (trts <= se + time_range_BOLD[1])] - round(se*4.0)/4.0 for se in e[:-1]])
			all_spearman_results.append([])
			sample_time_points = np.arange(0.75, 16, 1.5)
			# shell()
			bold_trial_data = [b - b[0] for b in bold_trial_data]
			for stp in sample_time_points:
				indices = np.zeros((pupil_trial_data.shape[0]), dtype = bool)
				which_time_point_in_trial = []
				this_time_point_bold_data = []
				for j in range(pupil_trial_data.shape[0]):
					if (time_trial_data[j] == stp).sum() > 0:
						this_time_point_bold_data.append(bold_trial_data[j][time_trial_data[j] == stp])
						which_time_point_in_trial.append(j)
				all_spearman_results[-1].append(np.nan_to_num(np.array(spearmanr(pupil_trial_data[which_time_point_in_trial], np.array(this_time_point_bold_data).ravel()))))
			
			# for (j, b) in enumerate(bold_trial_data.T):
			# 	all_spearman_results[-1].append(spearmanr(pupil_trial_data, b))
				
				# linear regression for regression lines
				# slope, intercept, r_value, p_value, slope_std_error = stats.linregress(pupil_trial_data, bold_trial_data)
				# predict_y = intercept + slope * pupil_trial_data
			# shell()
			# print sample_time_points
			which_time = 0
			pl.plot(sample_time_points[which_time::], [asr[0] for asr in np.array(all_spearman_results[-1][which_time::])], ['--b','--b','--g','--g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], mec = 'w', mew = 1, ms = 6, label = cond_labels[i])
			significant_time_points = (np.array([asr[0] for asr in np.array(all_spearman_results[-1][which_time::])]) < 0.01)
			pl.plot(sample_time_points[which_time::][significant_time_points], np.array([asr[0] for asr in np.array(all_spearman_results[-1][which_time::])])[significant_time_points], ['ob','ob','og','og'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], mec = ['b','b','g','g'][i], mew = 2, ms = 6)
		# shell()
			
		#  all thingies across 
		# correlation = correlate(pdc[:,1], bold_timeseries, 'same')
		# midpoint = correlation.shape[0] / 2
		# plot_range = 20
		# 
		# pl.plot(np.linspace(-plot_range/2*tr, plot_range/2*tr, correlation[midpoint-plot_range/2:midpoint+plot_range/2].shape[0]), correlation[midpoint-plot_range/2:midpoint+plot_range/2], 'r', alpha = 0.5, label = 'across conditions')
		# shell()
		return np.array(all_spearman_results)
	
	def cross_correlate_pupil_and_BOLD(self, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range = 'long'):
		corrs = []
		spearman_corrs = []
		areas = ['V1', 'V2', 'V3', 'V3AB', 'V4']
		f = pl.figure(figsize = (4,12))
		for (i, roi) in enumerate(areas):
			s = f.add_subplot(len(areas), 1, i+1)
			if time_range == 'short':
				crs = self.cross_correlate_pupil_and_BOLD_for_roi(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi, time_range_BOLD = [0.0, 16.0], time_range_pupil = [0.5, 2.0])
			elif time_range == 'long':
				crs = self.cross_correlate_pupil_and_BOLD_for_roi(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi, time_range_BOLD = [0.0, 16.0], time_range_pupil = [3.0, 9.0])
			corrs.append(crs[0])
			spearman_corrs.append(crs[1])
			print spearman_corrs[-1]
			s.set_title(self.subject.initials + ' ' + roi )
			# s.set_xlabel('time [TR]')
			# s.set_ylabel('cross-correlation')
			s.set_xlabel('pupil [Z]')
			s.set_ylabel('BOLD [% signal change]')
			if i == 0:
				leg = s.legend(fancybox = True)
				leg.get_frame().set_alpha(0.5)
				if leg:
					for t in leg.get_texts():
					    t.set_fontsize('small')    # the legend text fontsize
					for l in leg.get_lines():
					    l.set_linewidth(3.5)  # the legend line width
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), mask_type + '_' + mask_direction + '_' + time_range + '_pupil_corr.pdf'))
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'pupil-BOLD_cross_correlation_results'
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'pupil/bold correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for (i, c) in enumerate(corrs):
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = areas[i] + '_' + mask_type + '_' + mask_direction)
			except NoSuchNodeError:
				pass
			reward_h5file.create_array(thisRunGroup, areas[i] + '_' + mask_type + '_' + mask_direction, np.array(corrs[i]), 'pupil-bold cross correlation timecourses conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		
		for (i, c) in enumerate(spearman_corrs):
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = areas[i] + '_' + mask_type + '_' + mask_direction + '_spearman' + '_' + time_range)
			except NoSuchNodeError:
				pass
			reward_h5file.create_array(thisRunGroup, areas[i] + '_' + mask_type + '_' + mask_direction + '_spearman' + '_' + time_range, np.array(c), 'pupil-bold spearman correlation results conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
		
		# pl.show()
	
	def cross_correlate_pupil_and_BOLD_over_time(self, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range = 'long'):
		corrs = []
		spearman_corrs = []
		areas = ['V1', 'V2', 'V3', 'V3AB', 'V4']
		f = pl.figure(figsize = (4,12))
		for (i, roi) in enumerate(areas):
			s = f.add_subplot(len(areas), 1, i+1)
			if time_range == 'short':
				crs = self.cross_correlate_pupil_and_BOLD_for_roi_over_time(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi, time_range_BOLD = [-3.0, 16.0], time_range_pupil = [0.5, 2.0])
			elif time_range == 'long':
				crs = self.cross_correlate_pupil_and_BOLD_for_roi_over_time(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi, time_range_BOLD = [-3.0, 16.0], time_range_pupil = [3.0, 9.0])
			spearman_corrs.append(crs)
			# print spearman_corrs
			s.set_title(self.subject.initials + ' ' + roi )
			s.set_xlabel('time [s]')
			s.set_ylabel('spearman correlation')
			if i == 0:
				leg = s.legend(fancybox = True)
				leg.get_frame().set_alpha(0.5)
				if leg:
					for t in leg.get_texts():
					    t.set_fontsize('small')    # the legend text fontsize
					for l in leg.get_lines():
					    l.set_linewidth(3.5)  # the legend line width
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), mask_type + '_' + mask_direction + '_' + time_range + '_pupil_corr_over_time.pdf'))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'pupil-BOLD_cross_correlation_results'
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'pupil/bold correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for (i, c) in enumerate(spearman_corrs):
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = areas[i] + '_' + mask_type + '_' + mask_direction + '_' + time_range + '_spearman')
			except NoSuchNodeError:
				pass
			reward_h5file.create_array(thisRunGroup, areas[i] + '_' + mask_type + '_' + mask_direction + '_' + time_range + '_spearman', np.array(c), 'pupil-bold spearman correlation results conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
		
	def blinks_per_trial(self, blink_detection_range = [0,16], granularity = 0.01, smoothing_kernel_width = 0.5):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		from scipy import stats, signal
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		
		event_data = []
		blink_events = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks'])).T
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			nr_runs += 1
		
		event_data_per_run = event_data
		
		# shell()
		
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		blink_data = np.vstack(blink_events)
		
		within_trial_timerange = np.arange(blink_detection_range[0], blink_detection_range[1], granularity)
		
		gauss_pdf = stats.norm.pdf( np.linspace(-4, 4, smoothing_kernel_width / granularity) )
		gauss_pdf = gauss_pdf / gauss_pdf.sum()
		
		event_blink_list = []
		blink_density_array = np.zeros((len(conds), within_trial_timerange.shape[0]))
		smoothed_blink_density_array = np.zeros((len(conds), within_trial_timerange.shape[0]))
		for (eti, event_type) in enumerate(event_data):
			event_blink_list.append([])
			for e in event_type:
				this_event_blinks = blink_data[(blink_data[:,0] > (e + blink_detection_range[0])) * (blink_data[:,0] < (e + blink_detection_range[1]))]
				if this_event_blinks.shape[0] > 0:
					this_event_blinks[:,0] = this_event_blinks[:,0] - e
					event_blink_list[-1].append(this_event_blinks[:,[0,1]])
			event_blink_list[-1] = np.vstack(event_blink_list[-1])
			blink_occurrence_array = np.zeros((event_blink_list[-1].shape[0], (blink_detection_range[1] - blink_detection_range[0]) / granularity))
			for (i, trial) in enumerate(event_blink_list[-1]):
				blink_occurrence_array[i, (within_trial_timerange > trial[0]) * (within_trial_timerange < (trial[0] + trial[1]))] = 1
			blink_density_array[eti] = blink_occurrence_array.sum(axis = 0) / blink_occurrence_array.shape[0]
			
			smoothed_dtc = np.zeros((within_trial_timerange.shape[0] + gauss_pdf.shape[0]))
			smoothed_dtc[gauss_pdf.shape[0]/2:-gauss_pdf.shape[0]/2] = signal.fftconvolve(blink_density_array[eti], gauss_pdf, 'same')
			smoothed_blink_density_array[eti] = smoothed_dtc[gauss_pdf.shape[0]/2:-gauss_pdf.shape[0]/2]
		
		fig = pl.figure(figsize = (9, 5))
		s = fig.add_subplot(211)
		s.axhline(0, blink_detection_range[0], blink_detection_range[1], linewidth = 0.25)
		
		for i in range(0, smoothed_blink_density_array.shape[0]):
			pl.plot(np.linspace(blink_detection_range[0],blink_detection_range[1],smoothed_blink_density_array.shape[1]), smoothed_blink_density_array[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([blink_detection_range[0]-1.5, blink_detection_range[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		
		s = fig.add_subplot(212)
		s.axhline(0, blink_detection_range[0], blink_detection_range[1], linewidth = 0.25)
		
		for i in range(0, len(event_data), 2):
			ts_diff = -(smoothed_blink_density_array[i] - smoothed_blink_density_array[i+1])
			pl.plot(np.linspace(blink_detection_range[0],blink_detection_range[1],smoothed_blink_density_array.shape[1]), ts_diff, ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
			# s.set_title('reward signal ' + roi + ' ' + mask_type + ' ' + analysis_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('$\Delta$ % signal change')
		s.set_xlim([blink_detection_range[0]-1.5, blink_detection_range[1] + 1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		pl.show()
	
	def calculate_event_history_fix_reward(self, times, parameters):
		"""
		calculate for each trial, the intertrial interval preceding that trial, based on:
		the raw last trial
		the last reward signal in the line
		return the fixation reward trial onsets, with their itis depending on iti, fixation reward and general reward itis.
		"""
		
		sound_trials, visual_trials = np.array((self.which_reward + parameters['sound']) % 2, dtype = 'bool'), np.array(parameters['contrast'], dtype = 'bool')
		
		# stolen this from the feat event file generator function:
		# conditions are made of boolean combinations
		visual_sound_trials = sound_trials * visual_trials
		visual_silence_trials = visual_trials * (-sound_trials)
		blank_silence_trials = -(visual_trials + sound_trials)
		blank_sound_trials = (-visual_trials) * sound_trials
		
		experiment_start_time = (times['trial_phase_timestamps'][0,0,0])
		stim_onsets = (times['trial_phase_timestamps'][:,1,0] - experiment_start_time ) / 1000.0
		
		delays = np.zeros((stim_onsets.shape[0], 4))
		last_reward_time = 0.0
		last_fix_reward_time = 0.0
		last_visual_reward_time = 0.0
		last_trial_time = 0.0
		for i in range(stim_onsets.shape[0]):
			delays[i,:] = [last_reward_time, last_fix_reward_time, last_visual_reward_time, last_trial_time]
			last_trial_time = stim_onsets[i]
			if i in np.arange(stim_onsets.shape[0])[sound_trials]:
				last_reward_time = stim_onsets[i]
			if i in np.arange(stim_onsets.shape[0])[blank_sound_trials]:
				last_fix_reward_time = stim_onsets[i]
			if i in np.arange(stim_onsets.shape[0])[visual_sound_trials]:
				last_visual_reward_time = stim_onsets[i]
			
		relative_delays = (delays.T - stim_onsets).T
		what_trials_are_sensible = delays.min(axis = 1)!=0.0
		
		raw_itis_of_fix_reward_trials = relative_delays[blank_sound_trials * what_trials_are_sensible, 3]
		onsets_fix_reward_trials = stim_onsets[blank_sound_trials * what_trials_are_sensible]
		all_reward_itis_of_fix_reward_trials = relative_delays[blank_sound_trials * what_trials_are_sensible, 0]
		fixation_reward_itis_fix_reward_trials = relative_delays[blank_sound_trials * what_trials_are_sensible, 1]
		stimulus_reward_itis_fix_reward_trials = relative_delays[blank_sound_trials * what_trials_are_sensible, 2]
		
		return onsets_fix_reward_trials, raw_itis_of_fix_reward_trials, all_reward_itis_of_fix_reward_trials, fixation_reward_itis_fix_reward_trials, stimulus_reward_itis_fix_reward_trials
	
	def calculate_event_history_stim_reward(self, times, parameters):
		"""
		calculate for each trial, the intertrial interval preceding that trial, based on:
		the raw last trial
		the last reward signal in the line
		return the fixation reward trial onsets, with their itis depending on iti, fixation reward and general reward itis.
		"""
		
		sound_trials, visual_trials = np.array((self.which_reward + parameters['sound']) % 2, dtype = 'bool'), np.array(parameters['contrast'], dtype = 'bool')
		
		# stolen this from the feat event file generator function:
		# conditions are made of boolean combinations
		visual_sound_trials = sound_trials * visual_trials
		visual_silence_trials = visual_trials * (-sound_trials)
		blank_silence_trials = -(visual_trials + sound_trials)
		blank_sound_trials = (-visual_trials) * sound_trials
		
		experiment_start_time = (times['trial_phase_timestamps'][0,0,0])
		stim_onsets = (times['trial_phase_timestamps'][:,1,0] - experiment_start_time ) / 1000.0
		
		delays = np.zeros((stim_onsets.shape[0], 4))
		last_reward_time = 0.0
		last_fix_reward_time = 0.0
		last_visual_reward_time = 0.0
		last_trial_time = 0.0
		for i in range(stim_onsets.shape[0]):
			delays[i,:] = [last_reward_time, last_fix_reward_time, last_visual_reward_time, last_trial_time]
			last_trial_time = stim_onsets[i]
			if i in np.arange(stim_onsets.shape[0])[sound_trials]:
				last_reward_time = stim_onsets[i]
			if i in np.arange(stim_onsets.shape[0])[blank_sound_trials]:
				last_fix_reward_time = stim_onsets[i]
			if i in np.arange(stim_onsets.shape[0])[visual_sound_trials]:
				last_visual_reward_time = stim_onsets[i]
			
		relative_delays = (delays.T - stim_onsets).T
		what_trials_are_sensible = delays.min(axis = 1)!=0.0
		
		raw_itis_of_stim_reward_trials = relative_delays[visual_sound_trials * what_trials_are_sensible, 3]
		onsets_stim_reward_trials = stim_onsets[visual_sound_trials * what_trials_are_sensible]
		all_reward_itis_of_stim_reward_trials = relative_delays[visual_sound_trials * what_trials_are_sensible, 0]
		fixation_reward_itis_stim_reward_trials = relative_delays[visual_sound_trials * what_trials_are_sensible, 1]
		stimulus_reward_itis_stim_reward_trials = relative_delays[visual_sound_trials * what_trials_are_sensible, 2]
		
		return onsets_stim_reward_trials, raw_itis_of_stim_reward_trials, all_reward_itis_of_stim_reward_trials, fixation_reward_itis_stim_reward_trials, stimulus_reward_itis_stim_reward_trials


	def deconvolve_interval_roi(self, roi, threshold = 3.5, mask_type = 'center_surround_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'mean', nr_bins = 4, iti_type = 'all_reward', binning_grain = 'session', zero_time_offset = -3.0, add_other_conditions = 'full_design'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		other_conds = ['blank_silence','visual_silence','visual_sound']
		other_cond_labels = ['fix_no_reward','stimulus_no_reward','stimulus_reward']
		
		iti_data = []
		event_data = []
		roi_data = []
		blink_events = []
		other_conditions_event_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times')
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters')
			
			onsets_fix_reward_trials, raw_itis_of_fix_reward_trials, all_reward_itis_of_fix_reward_trials, fixation_reward_itis_fix_reward_trials, stimulus_reward_itis_fix_reward_trials = self.calculate_event_history_fix_reward(trial_times, parameter_data)
			
			events_of_interest = onsets_fix_reward_trials + nr_runs * run_duration
			if iti_type == 'all_reward':
				itis = all_reward_itis_of_fix_reward_trials
			elif iti_type == 'fix_reward':
				itis = fixation_reward_itis_fix_reward_trials
			elif iti_type == 'stim_reward':
				itis = stimulus_reward_itis_fix_reward_trials
			elif iti_type == 'all_trials':
				itis = raw_itis_of_fix_reward_trials
			
			iti_order = np.argsort(itis)
			stepsize = floor(itis.shape[0]/float(nr_bins))
			if binning_grain == 'run':
				event_data.append([events_of_interest[iti_order[x*stepsize:(x+1)*stepsize]] for x in range(nr_bins)])
				iti_data.append([itis[iti_order[x*stepsize:(x+1)*stepsize]] for x in range(nr_bins)])
			else:
				iti_data.append([itis, events_of_interest])
			
			this_run_events = []
			for cond in other_conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			other_conditions_event_data.append(this_run_events)
			# do median split here
			# event_data.append([events_of_interest[itis < np.median(itis)], events_of_interest[itis > np.median(itis)]])
			
			nr_runs += 1
		
		if binning_grain == 'run':
			# event_data_per_run = event_data
			event_data = [np.concatenate([e[i] for e in event_data]) + zero_time_offset for i in range(nr_bins)]
			iti_data = [np.concatenate([e[i] for e in iti_data]) for i in range(nr_bins)]
		elif binning_grain == 'session':
			itis = np.concatenate([it[0] for it in iti_data])
			event_times = np.concatenate([it[1] for it in iti_data])
			iti_order = np.argsort(itis)
			stepsize = floor(itis.shape[0]/float(nr_bins))
			event_data = [event_times[iti_order[x*stepsize:(x+1)*stepsize]] + zero_time_offset for x in range(nr_bins)]
			iti_data = [itis[iti_order[x*stepsize:(x+1)*stepsize]] for x in range(nr_bins)]
			self.logger.info(self.subject.initials + ' ' + iti_type + ' bin means for itis: ' + str([i.mean() for i in iti_data]))
		# shell()
		
		other_conditions_event_data = [np.concatenate([e[i] for e in other_conditions_event_data]) + zero_time_offset for i in range(len(other_conditions_event_data[0]))]
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		roi_data_per_run = demeaned_roi_data
		roi_data = np.hstack(demeaned_roi_data)
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		
		fig = pl.figure(figsize = (6, 5))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		colors = [(c, 0, 1-c) for c in np.linspace(0.1,0.9,nr_bins)]
		time_signals = []
		interval = [0.0,16.0]
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.hstack(blink_events)]))
		
		if add_other_conditions == 'full_design':
			# this next line adds other conditions to the design
			event_data.extend(other_conditions_event_data)
		
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix.T)
		# shell()
		# for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
		for i in range(0, nr_bins):
			if add_other_conditions == 'full_design':
				time_signals.append((deco.deconvolvedTimeCoursesPerEventTypeNuisance[i] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[nr_bins]).squeeze())
			else:
				time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze())
			# shell()
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze()), color = colors[i], alpha = 0.7, label = '%2.1f'%iti_data[i].mean())
			
		s.set_title('deconvolution' + roi + ' ' + mask_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width

		reward_h5file.close()
		mapper_h5file.close()
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'interval_' + roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + iti_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + iti_type + '_' + add_other_conditions, event_data, timeseries, np.array(time_signals)]
	
	def deconvolve_intervals(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB'], analysis_type = 'deconvolution', signal_type = 'mean', zero_time_offset = 0.0, mask_direction = 'pos', add_other_conditions = 'full_design', nr_bins = 4 ):
		results = []
		for roi in rois:
			results.append([])
			for itit in ['all_reward', 'fix_reward', 'all_trials', 'stim_reward']:
				results[-1].append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = mask_direction, nr_bins = nr_bins, signal_type = signal_type, iti_type = itit, binning_grain = 'session', zero_time_offset = zero_time_offset, add_other_conditions = add_other_conditions))
			# results.append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = signal_type))
			# results.append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'surround_center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type))
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_interval_results' + '_' + signal_type
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for res in results:
			for r in res:
				try:
					reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_' + signal_type)
					# reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_' + signal_type + '_per_run')
				except NoSuchNodeError:
					pass
				reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type, r[-1], 'interval deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
				# reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type + '_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def calculate_discrete_event_history(self, times, parameters):
		"""
		calculate for each trial, the intertrial interval preceding that trial, based on:
		the raw last trial
		the last reward signal in the line
		return the fixation reward trial onsets, with their itis depending on iti, fixation reward and general reward itis.
		"""
		
		sound_trials, visual_trials = np.array((self.which_reward + parameters['sound']) % 2, dtype = 'bool'), np.array(parameters['contrast'], dtype = 'bool')
		
		# stolen this from the feat event file generator function:
		# conditions are made of boolean combinations
		visual_sound_trials = sound_trials * visual_trials
		visual_silence_trials = visual_trials * (-sound_trials)
		blank_silence_trials = -(visual_trials + sound_trials)
		blank_sound_trials = (-visual_trials) * sound_trials
		
		experiment_start_time = (times['trial_phase_timestamps'][0,0,0])
		stim_onsets = (times['trial_phase_timestamps'][:,1,0] - experiment_start_time ) / 1000.0
		
		itis = r_[0, np.diff(stim_onsets)]
		
		raw_consecutives = np.array([r_[np.zeros(j),[np.array(sound_trials, dtype = int)[i-j:i].sum() for i in range(j,sound_trials.shape[0])]] for j in [1,2,3]])
		for j in [1,2,3]:
			raw_consecutives[j-1,raw_consecutives[j-1]==0] = -j
		consecutives = raw_consecutives[2]
		consecutives[(consecutives == 1) + (consecutives == 2)] = raw_consecutives[1,(consecutives == 1) + (consecutives == 2)]
		consecutives[(consecutives == 1)] = raw_consecutives[0,(consecutives == 1)]
		
		binned_onset_times = []
		for i in [-3,-2,-1,1,2,3]:
			binned_onset_times.append(stim_onsets[3:][consecutives[3:] == i])
		
		return consecutives, itis, stim_onsets, blank_sound_trials, binned_onset_times
	
	
	def deconvolve_discrete_interval_roi(self, roi, threshold = 3.5, mask_type = 'center_surround_Z', mask_direction = 'pos', signal_type = 'mean', nr_bins = 6):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		other_conds = ['blank_silence','visual_silence','visual_sound']
		other_cond_labels = ['fix_no_reward','stimulus_no_reward','stimulus_reward']
		
		iti_data = []
		event_data = []
		roi_data = []
		blink_events = []
		other_conditions_event_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times')
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters')
			
			consecutives, itis, stim_onsets, blank_sound_trials, binned_onset_times = self.calculate_discrete_event_history(trial_times, parameter_data)
			
			event_data.append([ot + nr_runs * run_duration for ot in binned_onset_times])
			
			this_run_events = []
			for cond in other_conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			other_conditions_event_data.append(this_run_events)
			# do median split here
			# event_data.append([events_of_interest[itis < np.median(itis)], events_of_interest[itis > np.median(itis)]])
			
			nr_runs += 1
		
		
		other_conditions_event_data = [np.concatenate([e[i] for e in other_conditions_event_data]) for i in range(len(other_conditions_event_data[0]))]
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		roi_data_per_run = demeaned_roi_data
		roi_data = np.hstack(demeaned_roi_data)
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		
		fig = pl.figure(figsize = (6, 5))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		colors = [(c, 0, 1-c) for c in np.linspace(0.1,0.9,nr_bins)]
		time_signals = []
		interval = [0.0,16.0]
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.hstack(blink_events)]))
		
		# this next line adds other conditions to the design
		event_data.extend(other_conditions_event_data)
		
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix.T)
		for i in range(0, nr_bins):
			time_signals.append((deco.deconvolvedTimeCoursesPerEventTypeNuisance[i] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[nr_bins]).squeeze())
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze()), color = colors[i], alpha = 0.7, label = str([-3,-2,-1,1,2,3][i]))
			
		s.set_title('deconvolution' + roi + ' ' + mask_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width

		reward_h5file.close()
		mapper_h5file.close()
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'discrete_interval_' + roi + '_' + mask_type + '_' + mask_direction + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_discrete', event_data, timeseries, np.array(time_signals)]
	
	def deconvolve_discrete_intervals(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], signal_type = 'mean', mask_direction = 'pos' ):
		results = []
		for roi in rois:
			results.append(self.deconvolve_discrete_interval_roi(roi, threshold, mask_type = 'center_Z', mask_direction = mask_direction, signal_type = signal_type))
			# results.append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = signal_type))
			# results.append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'surround_center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type))
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_discrete_interval_results' + '_' + signal_type
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_' + signal_type)
			except NoSuchNodeError:
				pass
			reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type, r[-1], 'discrete interval deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			# reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type + '_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def correlate_patterns(self, template, test):
		"""
		correlate_patterns correlates template and test patterns. 
		returns a 3 array, with spearman's correlation rho, its p-value and the scaled norm of the linear projection of the test and template as defined in Ress and Heeger, 2001.
		"""
		return np.squeeze(np.concatenate((list(spearmanr(template, test)), [np.dot(template, test)/(np.linalg.norm(template)**2)])))

	def variance_from_whole_brain_residuals(self, time_range_BOLD = [3.0, 9.0], var = True, to_surf = True):
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		if var:
			# take data 
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
			tr, nr_trs = niiFile.rtime, niiFile.timepoints
			run_duration = tr * nr_trs
		
			residuals = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'residuals.nii.gz')).data
		
			event_data = []
			tr_timings = []
			nr_runs = 0
			for r in [self.runList[i] for i in self.conditionDict['reward']]:
				this_run_events = []
				for cond in conds:
					this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
				this_run_events = np.array(this_run_events) + nr_runs * run_duration
				event_data.append(this_run_events)
				tr_timings.append(np.arange(0, run_duration, tr) + nr_runs * run_duration)
			trts = np.concatenate(tr_timings)
			# event_data = np.hstack(event_data)
			event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
			# shell()
			all_vars = []
			for (i, e) in enumerate(event_data):
				# standard version works on just the variance
				all_vars.append(np.array([residuals[(trts > se + time_range_BOLD[0]) * (trts < se + time_range_BOLD[1])].var(axis = 0) for se in e]).mean(axis = 0))
			all_vars = np.array(all_vars)
			
			opf = NiftiImage(all_vars)
			opf.header = niiFile.header
			opf.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'variance_residuals.nii.gz'))
			
			opf = NiftiImage(all_vars[1] - all_vars[0])
			opf.header = niiFile.header
			opf.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'variance_residuals_fix_diff.nii.gz'))
			
			opf = NiftiImage(all_vars[3] - all_vars[2])
			opf.header = niiFile.header
			opf.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'variance_residuals_stim_diff.nii.gz'))
			
		
		if to_surf:
			for file_name in ['variance_residuals.nii.gz', 'variance_residuals_fix_diff.nii.gz', 'variance_residuals_stim_diff.nii.gz']:
				# vol to surf?
				# for (label, f) in zip(['left', 'right'], [left_file, right_file]):
				if file_name == 'variance_residuals.nii.gz':
					frames = dict(zip(['_'+c for c in cond_labels], range(4)))
				else:
					frames = {'':0}
				ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), file_name)
				vsO = VolToSurfOperator(inputObject = ofn)
				sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
				vsO.configure(frames = frames, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
				vsO.execute()
				
				
				if file_name == 'variance_residuals.nii.gz':
					for cond in ['_'+c for c in cond_labels]:
						for hemi in ['lh','rh']:
							ssO = SurfToSurfOperator(vsO.outputFileName + cond + '-' + hemi + '.mgh')
							ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
							ssO.execute()
				else:
					cond = ''
					for hemi in ['lh','rh']:
						ssO = SurfToSurfOperator(vsO.outputFileName + cond + '-' + hemi + '.mgh')
						ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
						ssO.execute()
					
	
	def mask_residual_variance_to_hdf(self, run_type = 'reward'):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
		
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		h5file = open_file(self.hdf5_filename, mode = "r+", title = run_type + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = open_file(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		this_run_group_name = 'residuals_variance'
		try:
			thisRunGroup = h5file.remove_node(where = '/', name = this_run_group_name, recursive = True)
			# self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			# self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			pass
		thisRunGroup = h5file.createGroup("/", this_run_group_name, 'residual variance')
			
		"""
		Now, take different stat masks based on the run_type
		"""
		if run_type == 'reward':
			stat_files = {
							'residual_variance': os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'variance_residuals.nii.gz'),
							}
				
		stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
		for (roi, roi_name) in zip(rois, roinames):
			for (i, sf) in enumerate(stat_files.keys()):
				# loop over stat_files and rois
				# to mask the stat_files with the rois:
				imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
				these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
				h5file.create_array(thisRunGroup, roi_name.replace('.','_') + '_' + sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
		h5file.close()
		
	
	def residual_variance_per_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos'):
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
			
		this_run_group_name = 'residuals_variance'
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name)
			# self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			self.logger.info('No group ' + this_run_group_name + ' in this file')
			pass
		
		these_data = []
		for hemi in ['lh', 'rh']:
			these_data.append(eval('thisRunGroup.' + hemi + '_' + roi + '_residual_variance.read()'))
		these_data = np.hstack(these_data)
		
		masked_data = these_data[:,mapping_mask]
		mean_masked_data = masked_data.mean(axis = 1)
		
		# shell()
			
		reward_h5file.close()
		mapper_h5file.close()
		
		return mean_masked_data
	
	def residual_variance_analysis(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB']):
		
		d = []
		for roi in rois:
			d.append(self.residual_variance_per_roi(roi, threshold = threshold))
		
		return d
		
	
	def fsl_results_to_deco_folder(self, run_type = 'reward', postFix = ['mcf']):
		for j,r in enumerate([self.runList[i] for i in self.conditionDict['reward']]):
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			stat_files = {
							'visual_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
							'visual_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
							'visual_cope': os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
								
							'reward_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
							'reward_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
							'reward_cope': os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
								
							'blinks': os.path.join(this_feat, 'stats', 'pe1.nii.gz'),
							'blank_silence': os.path.join(this_feat, 'stats', 'cope3.nii.gz'),
							'blank_sound': os.path.join(this_feat, 'stats', 'cope4.nii.gz'),
							'blank_sound_Z': os.path.join(this_feat, 'stats', 'zstat4.nii.gz'),
							'visual_silence': os.path.join(this_feat, 'stats', 'cope5.nii.gz'),
							'visual_sound': os.path.join(this_feat, 'stats', 'cope6.nii.gz'),
								
							'fix_reward_silence': os.path.join(this_feat, 'stats', 'cope7.nii.gz'),
							'visual_reward_silence': os.path.join(this_feat, 'stats', 'cope8.nii.gz'),
								
							'visual_silence_fix_silence': os.path.join(this_feat, 'stats', 'cope9.nii.gz'),
							'visual_reward_fix_reward': os.path.join(this_feat, 'stats', 'cope10.nii.gz'),
							
							}
			for sf in stat_files.keys():
				os.system( 'cp ' + stat_files[sf] + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'FSL_' + sf + '_' + str(j) + '.nii.gz'))
		for j,r in enumerate([self.runList[i] for i in self.conditionDict['mapper']]):
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			stat_files = {
									'center_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
									'center_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
									'center_cope': os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
									'center_pe': os.path.join(this_feat, 'stats', 'pe1.nii.gz'),
								
									'surround_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
									'surround_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
									'surround_cope': os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
									'surround_pe': os.path.join(this_feat, 'stats', 'pe3.nii.gz'),
								
									'center>surround_T': os.path.join(this_feat, 'stats', 'tstat3.nii.gz'),
									'center>surround_Z': os.path.join(this_feat, 'stats', 'zstat3.nii.gz'),
									'center>surround_cope': os.path.join(this_feat, 'stats', 'cope3.nii.gz'),
								
									'surround>center_T': os.path.join(this_feat, 'stats', 'tstat4.nii.gz'),
									'surround>center_Z': os.path.join(this_feat, 'stats', 'zstat4.nii.gz'),
									'surround>center_cope': os.path.join(this_feat, 'stats', 'cope4.nii.gz'),
							
							}
			for sf in stat_files.keys():
				os.system( 'cp ' + stat_files[sf] + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'FSL_' + sf + '_' + str(j) + '.nii.gz'))
	
	def SNR_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', signal_type = 'mean', data_type = 'psc_hpf_data'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		print self.base_dir()
		print self.stageFolder(stage = 'processed/mri')
		print self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]], postFix = ['mcf'])
		print self.runFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]])
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]], postFix = ['mcf']))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		blink_events = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, data_type))
			if 'residuals' in data_type:
				roi_data[-1] = roi_data[-1] ** 2
			
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			nr_runs += 1
		
		reward_h5file.close()
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data_all_types = [np.round(np.concatenate([e[i] for e in event_data]) / 0.75) * 0.75 for i in range(len(event_data[0]))]
		# event_data = [np.round(np.sort(np.concatenate([event_data[0],event_data[1]])) * 4.0)/4.0, np.round(np.sort(np.concatenate([event_data[2],event_data[3]])) * 4.0)/4.0]
		event_data_grouped = [np.sort(np.concatenate([event_data_all_types[0],event_data_all_types[1]])), np.sort(np.concatenate([event_data_all_types[2],event_data_all_types[3]]))]
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		mapper_h5file.close()
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
				
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		if signal_type in ['std', 'var']:
			timeseries = (timeseries - timeseries.mean() ) / timeseries.std()
		
		fig = pl.figure(figsize = (9, 7))
		s = fig.add_subplot(311)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		interval = [0.0,15.0]
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.hstack(blink_events)]))
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data_grouped[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix.T)
		for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
			time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze())
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze()), ['k','k--'][i], alpha = [0.25,1.0][i], label = ['fix', 'visual'][i], linewidth = 3.0)
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		
		# after deconvolution, correlate with all different event types. 
		fit_results = [[]]
		timepoints_of_samples = np.arange(0,deco.workingDataArray.shape[0]*0.75,0.75)
		projection_data = []
		s = fig.add_subplot(323)
		minimal_amount_of_events = 10000
		for k, event_type in enumerate(event_data_all_types):
			projection_data.append([])
			for i, e in enumerate(event_type):
				which_start_time_index = np.arange(timepoints_of_samples.shape[0])[timepoints_of_samples == e]
				which_end_time_index = which_start_time_index + time_signals[0].shape[0]
				this_event_data = deco.workingDataArray[which_start_time_index:which_end_time_index]
				projection_data[-1].append([spearmanr(time_signals[[0, 0, 1, 1][k]], this_event_data)[0], np.dot(time_signals[[0, 0, 1, 1][k]], this_event_data)/(np.dot(time_signals[[0, 0, 1, 1][k]],time_signals[[0, 0, 1, 1][k]]))])
			# projection_data[-1] = np.array(projection_data[-1])
			if minimal_amount_of_events > i:
				minimal_amount_of_events = i
			x_values = np.array(projection_data[-1])[:,0]
			y_values = np.linspace(0,1,x_values.shape[0])
			order = np.argsort(x_values)
			fit_results[0].append(fitGaussian(x_values))
			pl.plot(x_values[order], y_values, color = ['b','b','g','g'][k], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][k], label = str(fit_results[0][-1]) + ' ' + cond_labels[k] )
		
		for k, event_type in enumerate(event_data_all_types):
			 projection_data[k] = projection_data[k][:minimal_amount_of_events]
		
		
		s.set_xlabel('spearman correlation between template and condition')
		leg = s.legend(fancybox = True, loc = 'upper left')
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		s = fig.add_subplot(324)
		for k, event_type in enumerate(event_data_all_types):
			y_values = np.array(projection_data[k])[:,0]
			pl.plot(np.linspace(0,1,y_values.shape[0]), y_values, ['bo','bo','go','go'][k], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][k], label = cond_labels[k] )
			
		
		fit_results.append([])
		s = fig.add_subplot(325)
		for k, event_type in enumerate(event_data_all_types):
			x_values = np.array(projection_data[k])[:,1]
			y_values = np.linspace(0,1,x_values.shape[0])
			order = np.argsort(x_values)
			fit_results[1].append(fitGaussian(x_values))
			pl.plot(x_values[order], y_values, color = ['b','b','g','g'][k], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][k], label = str(fit_results[1][-1]) + ' ' + cond_labels[k] )
			
		s.set_xlabel('linear projection between template and condition')
		leg = s.legend(fancybox = True, loc = 'upper left')
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		
		s = fig.add_subplot(326)
		for k, event_type in enumerate(event_data_all_types):
			y_values = np.array(projection_data[k])[:,1]
			pl.plot(np.linspace(0,1,y_values.shape[0]), y_values, ['bo','bo','go','go'][k], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][k], label = cond_labels[k] )
		
		projection_data = np.array(projection_data)
		fit_results = np.array(fit_results)
		print fit_results, projection_data.shape
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + signal_type + '_' + data_type + '_SNR.pdf'))
		return [roi + '_' + mask_type + '_' + mask_direction, fit_results, projection_data]
		
		
	def SNR(self, rois = ['V1', 'V2', 'V3', 'V3AB'], signal_type = 'mean', data_type = 'psc_hpf_data'):
		"""docstring for SNR"""
		results = []
		for roi in rois:
			results.append(self.SNR_roi(roi, threshold = 2.5, mask_type = 'center_Z', mask_direction = 'pos', signal_type = signal_type, data_type = data_type))
			results.append(self.SNR_roi(roi, threshold = -2.5, mask_type = 'center_Z', mask_direction = 'neg', signal_type = signal_type, data_type = data_type))
			# results.append(self.deconvolve_roi(roi, threshold, mask_type = 'surround_center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type, data_type = data_type))
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
		
		# exit()
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'SNR_results' + '_' + signal_type + '_' + data_type
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_projection_data_' + signal_type + '_' + data_type)
			except NoSuchNodeError:
				pass
			try:
				reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_fit_results_' + signal_type + '_' + data_type)
			except NoSuchNodeError:
				pass
			
			reward_h5file.create_array(thisRunGroup, r[0] + '_fit_results_' + signal_type + '_' + data_type, r[1], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.create_array(thisRunGroup, r[0] + '_projection_data_' + signal_type + '_' + data_type, r[2], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			
		reward_h5file.close()
		 
	def whole_brain_SNR(self, deco = True, to_surf = True):
		"""
		whole_brain_deconvolution takes all nii files from the reward condition and deconvolves the separate event types
		"""
		mask = np.array(NiftiImage(self.runFile(stage = 'processed/mri/reg', base = 'BET_mask', postFix = [self.ID] )).data, dtype = bool)
		nr_voxels = np.sum(mask)
		# check out the duration of these runs, assuming they're all the same length.		
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		nii_file_shape = list(niiFile.data.shape)
		
		nr_reward_runs = len(self.conditionDict['reward'])
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		time_signals = []
		interval = [0.0,16.0]
		
		event_data = []
		blink_events = []
		roi_data = []
		nr_runs = 0
		# nii_data = np.zeros([nr_reward_runs] + nii_file_shape)
		nii_data = np.zeros([nr_reward_runs, nr_trs, nr_voxels] )
	
		for (j, r) in enumerate([self.runList[i] for i in self.conditionDict['reward']]):
			nii_data[j] = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','tf','psc'])).data[:,mask]
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			nr_runs += 1
	
		# nii_data = nii_data.reshape((nr_reward_runs * nii_file_shape[0], -1))
		# event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		nii_data = nii_data.reshape((nr_reward_runs * nr_trs, nr_voxels))
		
		event_data_all_types = [np.round(np.concatenate([e[i] for e in event_data]) / 0.75) * 0.75 for i in range(len(event_data[0]))]
		# event_data = [np.round(np.sort(np.concatenate([event_data[0],event_data[1]])) * 4.0)/4.0, np.round(np.sort(np.concatenate([event_data[2],event_data[3]])) * 4.0)/4.0]
		event_data_grouped = [np.sort(np.concatenate([event_data_all_types[0],event_data_all_types[1]])), np.sort(np.concatenate([event_data_all_types[2],event_data_all_types[3]]))]
		
		nuisance_design = Design(nii_data.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.hstack(blink_events)]))
		deco = DeconvolutionOperator(inputObject = nii_data, eventObject = event_data_grouped[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix.T)
		
		# after deconvolution, correlate with all different event types. 
		timepoints_of_samples = np.arange(0,deco.workingDataArray.shape[0]*0.75,0.75)
		minimal_amount_of_events = np.min([len(e) for e in event_data_all_types])
		
		projection_data = np.zeros((len(event_data_all_types), nii_data.shape[1], minimal_amount_of_events))
		for k, event_type in enumerate(event_data_all_types):
			for j, e in enumerate(event_type[:minimal_amount_of_events]):
				which_start_time_index = np.arange(timepoints_of_samples.shape[0])[timepoints_of_samples == e]
				which_end_time_index = which_start_time_index + deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]
				for i in range(nii_data.shape[1]):
					projection_data[k,i,j] = spearmanr(deco.deconvolvedTimeCoursesPerEventTypeNuisance[[0, 0, 1, 1][k],:,i].squeeze(), deco.workingDataArray[which_start_time_index:which_end_time_index,i])[0]
		
		var_proj_data = projection_data.var(axis = -1)
		mean_proj_data = projection_data.mean(axis = -1)
		
		for (i, d) in enumerate([mean_proj_data, var_proj_data]):
			# outputdata = deco.deconvolvedTimeCoursesPerEventTypeNuisance[i]
			outputdata = np.zeros((var_proj_data.shape[0], nii_file_shape[1], nii_file_shape[2], nii_file_shape[3]))
			outputdata[:,mask] = d
			outputFile = NiftiImage(outputdata.reshape([outputdata.shape[0]]+nii_file_shape[1:]))
			outputFile.header = niiFile.header
			ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_SNR_' + ['mean_proj_data', 'var_proj_data'][i] + '.nii.gz')
			outputFile.save(ofn)
			# average over the interval [5,12] and [2,10] for reward and visual respectively. so, we'll just do [2,12]
			
			if to_surf:
				# vol to surf?
				vsO = VolToSurfOperator(inputObject = ofn)
				sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
				# vsO.configure(frames = dict(zip(cond_labels, range(len(cond_labels)))), hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
				vsO.configure(frames = dict(zip(cond_labels, range(len(cond_labels)))), hemispheres = None, register = '/Volumes/HDD/research/projects/reward/man/data/first/AV/AV_150312/processed/mri/reg/register_VisRewAV.dat', outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
				vsO.execute(wait = False)
		
				for hemi in ['lh','rh']:
					ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
					# ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
					ssO.configure(fsSourceSubject = 'AV_270411', fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
					ssO.execute(wait = False)
	
	def snr_to_hdf(self, run_type = 'reward', postFix = ['mcf']):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
		
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		h5file = open_file(self.hdf5_filename, mode = "r+", title = run_type + " file")
			
		if run_type == 'reward':
			stat_files = {
							'snr_mean_corr': os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_SNR_mean_proj_data.nii.gz'),
							'snr_diff': os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_SNR_proj_diff.nii.gz'),
							'snr_var_corr': os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_SNR_var_proj_data.nii.gz'),
							}
				
		stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
		for r in [self.runList[i] for i in self.conditionDict[run_type]]:
			"""loop over runs, and try to open a group for this run's data"""
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			try:
				thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
		
			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
			
				for (i, sf) in enumerate(stat_files.keys()):
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
					these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
					try:
						h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
					except NodeError:
						self.logger.info('Array ' + sf.replace('>', '_') + ' existed in ' + this_run_group_name)
		
		try:
			thisRunGroup = h5file.remove_node(where = '/', name = 'snr', recursive = True)
			self.logger.info('snr data ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			pass
		thisRunGroup = h5file.createGroup("/", 'snr', 'SNR results for whole visual areas')
		self.logger.info('snr data ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' started in ' + self.hdf5_filename)
		
		for roi in ['V1', 'V2', 'V3', 'V4', 'LO1']:
			for i, dt in enumerate(['snr_diff', 'snr_mean_corr', 'snr_var_corr', 'fis_reward_silence', '']):
				dd = self.roi_data_from_hdf(h5file, self.runList[self.conditionDict[run_type][0]], roi_wildcard = roi, data_type = dt, postFix = ['mcf'])
				h5file.create_array(thisRunGroup, roi + '_' + dt, dd.astype(np.float32), roi + ' data from ' + stat_files[dt])
				
		h5file.close()
	
	def snr_pattern_correlations(self, postFix = ['mcf']):
		"""docstring for snr_pattern_correlations"""
		self.reward_hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]), 'reward' + '.hdf5')
		reward_h5file = open_file(self.reward_hdf5_filename, mode = "r", title = 'reward' + " file")
		self.mapper_hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['mapper'][0]]), 'mapper' + '.hdf5')
		mapper_h5file = open_file(self.mapper_hdf5_filename, mode = "r", title = 'mapper' + " file")
		
		
		for roi in ['V1', 'V2', 'V3', 'V4', 'LO1']:
			roi_data = {}
			for i, dt in enumerate(['snr_diff', 'snr_mean_corr', 'snr_var_corr', 'fix_reward_silence', 'blank_sound', 'blank_silence', 'visual_Z', 'reward_Z']):
				roi_data[dt] = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi_wildcard = roi, data_type = dt, postFix = ['mcf'])
			roi_data['center_Z'] = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi_wildcard = roi, data_type = 'center_Z', postFix = ['mcf'])
			# correlate these with snr_diff
			
			these_roi_pattern_corrs = dict([(p, spearmanr(roi_data['snr_diff'][:,1], roi_data[p][:,0])[0]) for i,p in enumerate(['fix_reward_silence', 'blank_sound', 'blank_silence', 'visual_Z', 'reward_Z', 'center_Z'])])
			print these_roi_pattern_corrs
		
		reward_h5file.close()
		mapper_h5file.close()
		
	
	def snr_to_surf(self):
		"""docstring for snr_to_surf"""
		executable = 'mri_concat %s %s --o %s'
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_SNR_' + 'var_proj_data' + '.nii.gz')
		proj_data_file = NiftiImage(ofn)
		proj_data = np.array([proj_data_file.data[0] - proj_data_file.data[1], proj_data_file.data[2] - proj_data_file.data[3]])
		
		diff_f = NiftiImage(proj_data)
		diff_f.header = proj_data_file.header
		diff_fn = os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_SNR_' + 'proj_diff' + '.nii.gz')
		diff_f.save(diff_fn)
		
		
		# vol to surf?
		vsO = VolToSurfOperator(inputObject = ofn)
		sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
		vsO.configure(frames = dict(zip(cond_labels, range(len(cond_labels)))), hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
		vsO.execute(wait = False)

		for hemi in ['lh','rh']:
			for c in cond_labels:
				ssO = SurfToSurfOperator(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco/surf'), 'reward_SNR_' + 'var_proj_data' + c + '.mgh'))
				ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
				ssO.execute(wait = False)
		
		# vol to surf?
		vsO = VolToSurfOperator(inputObject = diff_fn)
		sofn = os.path.join(os.path.split(diff_fn)[0], 'surf/', os.path.split(diff_fn)[1])
		vsO.configure(frames = dict(zip(['fix','stim'], range(2))), hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
		vsO.execute(wait = False)

		for hemi in ['lh','rh']:
			for c in ['fix','stim']:
				ssO = SurfToSurfOperator(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco/surf'), 'reward_SNR_' + 'var_proj_data' + c + '.mgh'))
				ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
				ssO.execute(wait = False)
		
		
		# # now create the necessary difference images:
		# # only possible if deco has already been run...
		# for i in [0,2]:
		# 	for (j, which_times) in enumerate(['reward', 'visual']):
		# 		ipfs = [NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_deconv_mean_' + cond_labels[i] + '_' + which_times + '.nii.gz')), NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'reward_deconv_mean_' + cond_labels[i+1] + '_' + which_times + '.nii.gz'))]
		# 		diff_d = ipfs[0].data - ipfs[1].data
		# 	
		# 		ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), ['fix','','stimulus'][i] + '_reward_diff' + '_' + which_times + '.nii.gz')
		# 		outputFile = NiftiImage(diff_d)
		# 		outputFile.header = ipfs[0].header
		# 		outputFile.save(ofn)
		# 	
		# 	
		# 		if to_surf:
		# 			vsO = VolToSurfOperator(inputObject = ofn)
		# 			sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
		# 			vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
		# 			vsO.execute(wait = False)
		# 		
		# 			for hemi in ['lh','rh']:
		# 				ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
		# 				ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
		# 				ssO.execute(wait = False)
	
	def deconvolve_and_regress_trials_roi(self, roi, threshold = 3.5, mask_type = 'center_surround_Z', mask_direction = 'pos', signal_type = 'mean', data_type = 'psc_hpf_data'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		mapper_h5file = self.hdf5_file('mapper')
		reward_h5file = self.hdf5_file('reward')
		
		event_data = []
		roi_data = []
		blink_events = []
		mocos = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, data_type))
			if 'residuals' in data_type:
				roi_data[-1] = roi_data[-1] ** 2
			
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			mocos.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.nii.gz.par', postFix = ['mcf'])))
			
			nr_runs += 1
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		mocos = np.vstack(mocos)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		if signal_type in ['std', 'var']:
			timeseries = (timeseries - timeseries.mean() ) / timeseries.std()
		
		
		time_signals = []
		interval = [0.0,16.0]
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.hstack(blink_events)]))
		full_nuisance_design = r_[nuisance_design.designMatrix, np.repeat(mocos,2, axis = 0).T].T
		
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(full_nuisance_design)
		
		# mean stimulus response:
		stim_resp = (((deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visual_sound')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('blank_sound')]) + (deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visual_silence')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('blank_silence')])) / 2.0).squeeze()
		# mean reward response:
		rew_resp = (((deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visual_sound')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visual_silence')]) + (deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('blank_sound')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('blank_silence')])) / 2.0).squeeze()
		
		if True:
			f = pl.figure(figsize = (6,3))
			s = f.add_subplot(1,1,1)
			s.set_title(roi + ' ' + 'reward')
			pl.plot(np.linspace(interval[0], interval[1], stim_resp.shape[0]), stim_resp, 'k', label = 'stimulus')
			pl.plot(np.linspace(interval[0], interval[1], rew_resp.shape[0]), rew_resp, 'r', label = 'reward')
			s.set_xlabel('time [s]')
			s.set_ylabel('% signal change')
			# s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for l in leg.get_lines():
				    l.set_linewidth(3.5)  # the legend line width
			simpleaxis(s)
			spine_shift(s)
			# s.set_ylim([-2,2])
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_template_deconvolutions.pdf'))
			
		
		rounded_event_array = np.array([np.array(((ev / 1.5) * 2.0), dtype = int) for ev in event_data])
		rounded_event_types = np.array([np.ones(ev.shape) * i for i, ev in enumerate(event_data)])
		
		nr_trials = np.concatenate(rounded_event_array).shape[0]
		per_trial_design_matrix = np.zeros((nr_trials * 2, timeseries.shape[0] * 2))
		
		for i in range(nr_trials):
			# stimulus regressors:
			per_trial_design_matrix[i][np.concatenate(rounded_event_array)[i]] = 1.0
			per_trial_design_matrix[i] = np.correlate(per_trial_design_matrix[i], stim_resp, 'same')
			# reward regressors:
			per_trial_design_matrix[i + nr_trials][np.concatenate(rounded_event_array)[i]] = 1.0
			per_trial_design_matrix[i + nr_trials] = np.correlate(per_trial_design_matrix[i], rew_resp, 'same')
		
		full_per_trial_design_matrix = np.mat(np.vstack((per_trial_design_matrix, full_nuisance_design.T))).T
		full_per_trial_betas = ((full_per_trial_design_matrix.T * full_per_trial_design_matrix).I * full_per_trial_design_matrix.T) * np.mat(deco.workingDataArray.T).T
		full_per_trial_betas_no_nuisance = np.array(full_per_trial_betas[:nr_trials*2].reshape(2,-1).T).squeeze()
		
		# shell()
		
		trial_info = pd.DataFrame({'stim_betas': full_per_trial_betas_no_nuisance[:,0], 'reward_betas': full_per_trial_betas_no_nuisance[:,1], 'event_times': np.concatenate(rounded_event_array), 'event_types': np.concatenate(rounded_event_types)})
		
		reward_h5file.close()
		mapper_h5file.close()
		with pd.get_store(self.hdf5_filename) as h5_file: # hdf5_filename is now the reward file as that was opened last
			h5_file.put("/per_trial_glm_results/%s"% roi + '_' + mask_type + '_' + mask_direction + '_' + data_type, trial_info)


	def deconvolve_and_regress_trials(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], signal_type = 'mean', data_type = 'psc_hpf_data'):
		"""docstring for deconvolve_and_regress_trials_roi"""
		for roi in rois:
			self.deconvolve_and_regress_trials_roi(roi, threshold = threshold, mask_type = 'center_Z', mask_direction = 'pos', signal_type = signal_type, data_type = data_type)
			self.deconvolve_and_regress_trials_roi(roi, threshold = -threshold, mask_type = 'center_Z', mask_direction = 'neg', signal_type = signal_type, data_type = data_type)
	
	
	def trial_history_from_per_trial_glm_results_roi(self, roi = 'V1', mask_type = 'center_Z', mask_direction = 'pos', plots = True):
		"""docstring for trial_history_from_per_trial_glm_results"""
		# set up the right reward file
		reward_h5file = self.hdf5_file('reward')
		reward_h5file.close()
		with pd.get_store(self.hdf5_filename) as h5_file:
			trials = h5_file["/per_trial_glm_results/%s"% roi + '_' + mask_type + '_' + mask_direction + '_' + 'psc_hpf_data']
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		time_order = np.argsort(trials.event_times)
		time_ordered_trials = trials.irow(time_order)
		
		time_ordered_trials['intertrial_intervals'] = np.r_[0,np.diff(np.array(time_ordered_trials.event_times))]
		
		fix_rewards = np.array(time_ordered_trials.event_types) == 1
		stim_rewards = np.array(time_ordered_trials.event_types) == 3
		stim_norewards = np.array(time_ordered_trials.event_types) == 2
		reward_times = np.array(time_ordered_trials[fix_rewards + stim_rewards].event_times)
		
		time_ordered_trials['reward_intervals'] = np.array([np.abs(np.max((time_ordered_trials[fix_rewards + stim_rewards].event_times - t)[(time_ordered_trials[fix_rewards + stim_rewards].event_times - t) < 0])) for t in np.array(time_ordered_trials.event_times)])
		time_ordered_trials['fix_reward_intervals'] = np.array([np.abs(np.max((time_ordered_trials[fix_rewards].event_times - t)[(time_ordered_trials[fix_rewards].event_times - t) < 0])) for t in np.array(time_ordered_trials.event_times)])
		time_ordered_trials['stim_reward_intervals'] = np.array([np.abs(np.max((time_ordered_trials[stim_rewards].event_times - t)[(time_ordered_trials[stim_rewards].event_times - t) < 0])) for t in np.array(time_ordered_trials.event_times)])
		time_ordered_trials['stim_noreward_intervals'] = np.array([np.abs(np.max((time_ordered_trials[stim_norewards].event_times - t)[(time_ordered_trials[stim_norewards].event_times - t) < 0])) for t in np.array(time_ordered_trials.event_times)])
		
		# time_ordered_trials['reward_intervals'] = np.array([np.abs(np.max((time_ordered_trials[fix_rewards + stim_rewards].event_times - t))) for t in np.array(time_ordered_trials.event_times)])
		# time_ordered_trials['fix_reward_intervals'] = np.array([np.abs(np.max((time_ordered_trials[fix_rewards].event_times - t))) for t in np.array(time_ordered_trials.event_times)])
		# time_ordered_trials['stim_reward_intervals'] = np.array([np.abs(np.max((time_ordered_trials[stim_rewards].event_times - t))) for t in np.array(time_ordered_trials.event_times)])		
		
		with pd.get_store(self.hdf5_filename) as h5_file:
			h5_file.put("/per_trial_glm_results_ordered/%s"% roi + '_' + mask_type + '_' + mask_direction, time_ordered_trials)
		
		if plots:
			smooth_width = 100
			
			def smooth(values, indices):
				valids = -np.isnan(values) * -np.isnan(indices)
				sm_values = np.array(values[valids], dtype = float)[np.array(indices[valids], dtype = int)]
				return np.convolve( sm_values, np.ones((smooth_width))/float(smooth_width), 'valid' )
			
			f = pl.figure(figsize = (9,4))
			s = f.add_subplot(1,2,1)
			s.set_title(roi + ' ' + 'reward')
			
			pl.plot(np.log(time_ordered_trials[fix_rewards].fix_reward_intervals), time_ordered_trials[fix_rewards].reward_betas, 'ro', label = 'fix_reward_int_fix_reward_reward', alpha = 0.125, ms = 2.0)
			pl.plot(np.log(time_ordered_trials[fix_rewards].stim_reward_intervals), time_ordered_trials[fix_rewards].reward_betas, 'bo', label = 'stim_reward_int_fix_reward_reward', alpha = 0.125, ms = 2.0)
			# pl.plot(np.log(time_ordered_trials[fix_rewards].reward_intervals), time_ordered_trials[fix_rewards].reward_betas, 'go', label = 'reward_int_fix_reward_reward', alpha = 0.125, ms = 2.0)
			
			pl.plot(np.log(time_ordered_trials[stim_rewards].fix_reward_intervals), time_ordered_trials[stim_rewards].reward_betas, 'rv', label = 'fix_reward_int_stim_reward_reward', alpha = 0.125, ms = 2.0)
			pl.plot(np.log(time_ordered_trials[stim_rewards].stim_reward_intervals), time_ordered_trials[stim_rewards].reward_betas, 'bv', label = 'stim_reward_int_stim_reward_reward', alpha = 0.125, ms = 2.0)
			# pl.plot(np.log(time_ordered_trials[stim_rewards].reward_intervals), time_ordered_trials[stim_rewards].reward_betas, 'gv', label = 'reward_int_stim_reward_reward', alpha = 0.125, ms = 2.0)
			
			
			pl.plot(smooth(np.log(time_ordered_trials[fix_rewards+stim_rewards].fix_reward_intervals), np.argsort(time_ordered_trials[fix_rewards+stim_rewards].fix_reward_intervals)), smooth(time_ordered_trials[fix_rewards+stim_rewards].reward_betas, np.argsort(time_ordered_trials[fix_rewards+stim_rewards].fix_reward_intervals)), 'r', label = 'fix_reward_int_fix_reward_reward', alpha = 0.75)
			pl.plot(smooth(np.log(time_ordered_trials[fix_rewards+stim_rewards].stim_reward_intervals), np.argsort(time_ordered_trials[fix_rewards+stim_rewards].stim_reward_intervals)), smooth(time_ordered_trials[fix_rewards+stim_rewards].reward_betas, np.argsort(time_ordered_trials[fix_rewards+stim_rewards].stim_reward_intervals)), 'b', label = 'stim_reward_int_fix_reward_reward', alpha = 0.75)
			# pl.plot(smooth(np.log(time_ordered_trials[fix_rewards+stim_rewards].reward_intervals), np.argsort(time_ordered_trials[fix_rewards+stim_rewards].reward_intervals)), smooth(time_ordered_trials[fix_rewards+stim_rewards].reward_betas, np.argsort(time_ordered_trials[fix_rewards+stim_rewards].reward_intervals)), 'g', label = 'reward_int_fix_reward_reward', alpha = 0.75)
			
			# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].fix_reward_intervals), np.argsort(time_ordered_trials[stim_rewards].fix_reward_intervals)), smooth(time_ordered_trials[stim_rewards].reward_betas, np.argsort(time_ordered_trials[stim_rewards].fix_reward_intervals)), 'r--', label = 'fix_reward_int_stim_reward_reward', alpha = 0.75)
			# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].stim_reward_intervals), np.argsort(time_ordered_trials[stim_rewards].stim_reward_intervals)), smooth(time_ordered_trials[stim_rewards].reward_betas, np.argsort(time_ordered_trials[stim_rewards].stim_reward_intervals)), 'b--', label = 'stim_reward_int_stim_reward_reward', alpha = 0.75)
			# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].reward_intervals), np.argsort(time_ordered_trials[stim_rewards].reward_intervals)), smooth(time_ordered_trials[stim_rewards].reward_betas, np.argsort(time_ordered_trials[stim_rewards].reward_intervals)), 'g--', label = 'reward_int_stim_reward_reward', alpha = 0.75)
			
			# shell()
			
			s.set_xlabel('interval duration')
			s.set_ylabel('betas')
			# s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for l in leg.get_lines():
				    l.set_linewidth(3.5)  # the legend line width
			simpleaxis(s)
			spine_shift(s)
			s.set_ylim([-2,2])
			
			s = f.add_subplot(1,2,2)
			s.set_title(roi + ' ' + 'stimulus')
			
			# shell() 
			pl.plot(np.log(time_ordered_trials[stim_norewards].fix_reward_intervals), time_ordered_trials[stim_norewards].stim_betas, 'ro', label = 'fix_reward_int_stim_noreward_stim', alpha = 0.125, ms = 2.0)
			pl.plot(np.log(time_ordered_trials[stim_norewards].stim_reward_intervals), time_ordered_trials[stim_norewards].stim_betas, 'bo', label = 'stim_reward_int_stim_noreward_stim', alpha = 0.125, ms = 2.0)
			# pl.plot(np.log(time_ordered_trials[stim_norewards].reward_intervals), time_ordered_trials[fix_rewards].stim_betas, 'go', label = 'reward_int_stim_noreward_stim', alpha = 0.125, ms = 2.0)
			
			pl.plot(np.log(time_ordered_trials[stim_rewards].fix_reward_intervals), time_ordered_trials[stim_rewards].stim_betas, 'rv', label = 'fix_reward_int_stim_reward_stim', alpha = 0.125, ms = 2.0)
			pl.plot(np.log(time_ordered_trials[stim_rewards].stim_reward_intervals), time_ordered_trials[stim_rewards].stim_betas, 'bv', label = 'stim_reward_int_stim_reward_stim', alpha = 0.125, ms = 2.0)
			# pl.plot(np.log(time_ordered_trials[stim_rewards].reward_intervals), time_ordered_trials[stim_rewards].stim_betas, 'gv', label = 'reward_int_stim_reward_stim', alpha = 0.125, ms = 2.0)
			
			pl.plot(smooth(np.log(time_ordered_trials[stim_norewards+stim_rewards].fix_reward_intervals), np.argsort(time_ordered_trials[stim_norewards+stim_rewards].fix_reward_intervals)), smooth(time_ordered_trials[stim_norewards+stim_rewards].stim_betas, np.argsort(time_ordered_trials[stim_norewards+stim_rewards].fix_reward_intervals)), 'r', label = 'fix_reward_int_stim_noreward_stim', alpha = 0.75)
			pl.plot(smooth(np.log(time_ordered_trials[stim_norewards+stim_rewards].stim_reward_intervals), np.argsort(time_ordered_trials[stim_norewards+stim_rewards].stim_reward_intervals)), smooth(time_ordered_trials[stim_norewards+stim_rewards].stim_betas, np.argsort(time_ordered_trials[stim_norewards+stim_rewards].stim_reward_intervals)), 'b', label = 'stim_reward_int_stim_noreward_stim', alpha = 0.75)
			# pl.plot(smooth(np.log(time_ordered_trials[stim_norewards+stim_rewards].reward_intervals), np.argsort(time_ordered_trials[stim_norewards+stim_rewards].reward_intervals)), smooth(time_ordered_trials[stim_norewards+stim_rewards].stim_betas, np.argsort(time_ordered_trials[stim_norewards+stim_rewards].reward_intervals)), 'g', label = 'reward_int_stim_noreward_stim', alpha = 0.75)
			
			# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].fix_reward_intervals), np.argsort(time_ordered_trials[stim_rewards].fix_reward_intervals)), smooth(time_ordered_trials[stim_rewards].stim_betas, np.argsort(time_ordered_trials[stim_rewards].fix_reward_intervals)), 'r--', label = 'fix_reward_int_stim_reward_stim', alpha = 0.75)
			# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].stim_reward_intervals), np.argsort(time_ordered_trials[stim_rewards].stim_reward_intervals)), smooth(time_ordered_trials[stim_rewards].stim_betas, np.argsort(time_ordered_trials[stim_rewards].stim_reward_intervals)), 'b--', label = 'stim_reward_int_stim_reward_stim', alpha = 0.75)
			# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].reward_intervals), np.argsort(time_ordered_trials[stim_rewards].reward_intervals)), smooth(time_ordered_trials[stim_rewards].stim_betas, np.argsort(time_ordered_trials[stim_rewards].reward_intervals)), 'g--', label = 'reward_int_stim_reward_stim', alpha = 0.75)
			
			s.set_xlabel('interval duration')
			s.set_ylabel('betas')
			# s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for l in leg.get_lines():
				    l.set_linewidth(3.5)  # the legend line width
			simpleaxis(s)
			spine_shift(s)
			s.set_ylim([-2,2])
			
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_interval_beta_correlations.pdf'))
			
			# def simulate_run(alphas, beta, stimuli, reward_history, V0 = 0.0):
			# 	V = alphas * V0
			# 	Vs = np.zeros((reward_history.shape[0], alphas.shape[0] * 2 + 1))
			# 	for i in range(reward_history.shape[0]):
			# 		dV = [alpha * beta * (stimuli[i,j]*reward_history[i] - V[j]) for j, alpha in enumerate(alphas)]
			# 		V = V + dV
			# 		Vs[i] = r_[V[0]-V[1], V, dV]
			# 	return Vs
			
			def simulate_run(alphas, beta, stimuli, reward_history, ss = 2.0, V0 = 0.0):
				V = alphas * V0
				Vs = np.zeros((reward_history.shape[0], alphas.shape[0] * 2 + 1))
				for i in range(reward_history.shape[0]):
					dV = [
						alphas[0] * beta * ((ss*stimuli[i,0] - 1.0)*reward_history[i] - V[0]),
						alphas[1] * beta * (stimuli[i,1]*reward_history[i] - V[1])
					]
					V = V + dV
					Vs[i] = r_[V[0]-V[1], V, dV]
				return Vs
			
			lrs = np.array([0.0625, 0.0625])
			
			# experiment 1
			stim_trials = np.array(stim_rewards + stim_norewards, dtype = int)
			fix_trials = np.array(-np.array(stim_trials, dtype = bool), dtype = int)
			reward_trials = np.array(fix_rewards + stim_rewards, dtype = int)
			
			lrs = np.linspace(0.15,0.6,20)
			sss = np.linspace(0.5,2.5,12)
			beta = 1.0
			corr_res = np.zeros((20,12,5))
			llmax, ssmax = 0, 0
			for k, ll in enumerate(lrs):
				for j, ss in enumerate(sss):
					rwtc = simulate_run(np.array([ll, ll]), beta, np.vstack((stim_trials, fix_trials)).T, reward_trials, ss = ss )
					corr_res[k,j,:] = np.array([spearmanr(rwtc[fix_rewards+stim_rewards,i], time_ordered_trials[fix_rewards+stim_rewards].reward_betas)[0] for i in range(5)])
					if corr_res[k,j,0] == np.max(corr_res[:,:,0]):
						llmax, ssmax = ll, ss
			print llmax, ssmax
			
			a1 = simulate_run(np.array([llmax, llmax]), beta, np.vstack((stim_trials, fix_trials)).T, reward_trials, ss = ssmax )
			
			f = pl.figure(figsize = (9,9))
			s = f.add_subplot(2,1,1)
			for i in range(5):
				s.set_title('RW timecourse %1.2f, %1.2f'%(llmax, ssmax))
				pl.plot(a1[:,i], ['r','g','b','m','c'][i], label = '', alpha = 0.825, linewidth = 2.0)
			simpleaxis(s)
			spine_shift(s)
			s.set_xlabel('trials')
			s.set_ylabel('RW values')
			
			s = f.add_subplot(2,4,8)
			s.set_title('correlations')
			for j, ss in enumerate(np.linspace(0.5,6,12)+1):
				pl.plot(lrs, corr_res[:,j,0], 'r',  label = '', alpha = 0.25, lw = ss)
				pl.plot(lrs, corr_res[:,j,1], 'g',  label = '', alpha = 0.25, lw = ss)
				pl.plot(lrs, corr_res[:,j,2], 'b',  label = '', alpha = 0.25, lw = ss)
				pl.plot(lrs, corr_res[:,j,3], 'c',  label = '', alpha = 0.25, lw = ss)
				pl.plot(lrs, corr_res[:,j,4], 'm',  label = '', alpha = 0.25, lw = ss)
			
			simpleaxis(s)
			spine_shift(s)
			# s.set_ylim([-0.2,0.2])
	
			s.set_ylabel('correlation')
			s.set_xlabel('Learning rate')
			
			for i, tr in enumerate([fix_rewards, stim_rewards, fix_rewards+stim_rewards]):
				s = f.add_subplot(2,4,i+5)
				s.set_title(roi + ' ' + ['fix', 'stim', 'fix+stim'][i] + ' ' + 'reward')
			
				pl.plot(a1[tr,0], time_ordered_trials[tr].reward_betas, 'ro', label = '', alpha = 0.125, ms = 4.0)
				pl.plot(a1[tr,1], time_ordered_trials[tr].reward_betas, 'go', label = '', alpha = 0.125, ms = 4.0)
				pl.plot(a1[tr,2], time_ordered_trials[tr].reward_betas, 'bo', label = '', alpha = 0.125, ms = 4.0)
				pl.plot(a1[tr,3], time_ordered_trials[tr].reward_betas, 'co', label = '', alpha = 0.125, ms = 4.0)
				pl.plot(a1[tr,4], time_ordered_trials[tr].reward_betas, 'mo', label = '', alpha = 0.125, ms = 4.0)
			
				smooth_width = 20
				pl.plot(smooth(a1[tr,0], np.argsort(a1[tr,0])), smooth(time_ordered_trials[tr].reward_betas, np.argsort(a1[tr,0])) , 'r', label = 'diff_V', alpha = 0.825, linewidth = 2.0)
				pl.plot(smooth(a1[tr,1], np.argsort(a1[tr,1])), smooth(time_ordered_trials[tr].reward_betas, np.argsort(a1[tr,1])) , 'g', label = 'V0', alpha = 0.825, linewidth = 2.0)
				pl.plot(smooth(a1[tr,2], np.argsort(a1[tr,2])), smooth(time_ordered_trials[tr].reward_betas, np.argsort(a1[tr,2])) , 'b', label = 'V1', alpha = 0.825, linewidth = 2.0)
				pl.plot(smooth(a1[tr,3], np.argsort(a1[tr,3])), smooth(time_ordered_trials[tr].reward_betas, np.argsort(a1[tr,3])) , 'c', label = 'dV0', alpha = 0.825, linewidth = 2.0)
				pl.plot(smooth(a1[tr,4], np.argsort(a1[tr,4])), smooth(time_ordered_trials[tr].reward_betas, np.argsort(a1[tr,4])) , 'm', label = 'dV1', alpha = 0.825, linewidth = 2.0)
			
			
				# pl.plot(smooth(np.log(time_ordered_trials[fix_rewards+stim_rewards].fix_reward_intervals), np.argsort(time_ordered_trials[fix_rewards+stim_rewards].fix_reward_intervals)), smooth(time_ordered_trials[fix_rewards+stim_rewards].reward_betas, np.argsort(time_ordered_trials[fix_rewards+stim_rewards].fix_reward_intervals)), 'r', label = 'fix_reward_int_fix_reward_reward', alpha = 0.75)
				# pl.plot(smooth(np.log(time_ordered_trials[fix_rewards+stim_rewards].stim_reward_intervals), np.argsort(time_ordered_trials[fix_rewards+stim_rewards].stim_reward_intervals)), smooth(time_ordered_trials[fix_rewards+stim_rewards].reward_betas, np.argsort(time_ordered_trials[fix_rewards+stim_rewards].stim_reward_intervals)), 'b', label = 'stim_reward_int_fix_reward_reward', alpha = 0.75)
				# pl.plot(smooth(np.log(time_ordered_trials[fix_rewards+stim_rewards].reward_intervals), np.argsort(time_ordered_trials[fix_rewards+stim_rewards].reward_intervals)), smooth(time_ordered_trials[fix_rewards+stim_rewards].reward_betas, np.argsort(time_ordered_trials[fix_rewards+stim_rewards].reward_intervals)), 'g', label = 'reward_int_fix_reward_reward', alpha = 0.75)
			
				# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].fix_reward_intervals), np.argsort(time_ordered_trials[stim_rewards].fix_reward_intervals)), smooth(time_ordered_trials[stim_rewards].reward_betas, np.argsort(time_ordered_trials[stim_rewards].fix_reward_intervals)), 'r--', label = 'fix_reward_int_stim_reward_reward', alpha = 0.75)
				# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].stim_reward_intervals), np.argsort(time_ordered_trials[stim_rewards].stim_reward_intervals)), smooth(time_ordered_trials[stim_rewards].reward_betas, np.argsort(time_ordered_trials[stim_rewards].stim_reward_intervals)), 'b--', label = 'stim_reward_int_stim_reward_reward', alpha = 0.75)
				# pl.plot(smooth(np.log(time_ordered_trials[stim_rewards].reward_intervals), np.argsort(time_ordered_trials[stim_rewards].reward_intervals)), smooth(time_ordered_trials[stim_rewards].reward_betas, np.argsort(time_ordered_trials[stim_rewards].reward_intervals)), 'g--', label = 'reward_int_stim_reward_reward', alpha = 0.75)
			
				# shell()
				simpleaxis(s)
				spine_shift(s)
				# s.set_ylim([-0.2,0.2])
		
				s.set_ylabel('betas')
				s.set_xlabel('RW_a')
			
				if i == 1:
				# s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			
					leg = s.legend(fancybox = True)
					leg.get_frame().set_alpha(0.5)
					if leg:
						for t in leg.get_texts():
						    t.set_fontsize('small')    # the legend text fontsize
						for l in leg.get_lines():
						    l.set_linewidth(3.5)  # the legend line width
			
			
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_interval_beta_correlations_RW.pdf'))
	
	def trial_history_from_per_trial_glm_results(self, rois = ['V1',], mask_type = 'center_Z'): #   'V2', 'V3', 'V3AB', 'V4'
		"""docstring for trial_history_from_per_trial_glm_results_roi"""
		for roi in rois:
			self.trial_history_from_per_trial_glm_results_roi(roi, mask_type = mask_type, mask_direction = 'pos')
			self.trial_history_from_per_trial_glm_results_roi(roi, mask_type = mask_type, mask_direction = 'neg')

	def deconvolve_and_regress_trials_roi_no_stim(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', signal_type = 'mean', data_type = 'psc_hpf_data'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		mapper_h5file = self.hdf5_file('mapper')
		reward_h5file = self.hdf5_file('reward')
		
		event_data = []
		roi_data = []
		blink_events = []
		mocos = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, data_type))
			if 'residuals' in data_type:
				roi_data[-1] = roi_data[-1] ** 2
			
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			mocos.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.nii.gz.par', postFix = ['mcf'])))
			
			nr_runs += 1
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		mocos = np.vstack(mocos)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		if signal_type in ['std', 'var']:
			timeseries = (timeseries - timeseries.mean() ) / timeseries.std()
		
		time_signals = []
		interval = [0.0,12.75]
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.hstack(blink_events)]))
		full_nuisance_design = r_[nuisance_design.designMatrix, np.repeat(mocos,2, axis = 0).T].T
		
		# split up for stimulus/no stimulus
		stim_ev_data = [np.concatenate((event_data[0], event_data[1])), np.concatenate((event_data[2], event_data[3]))]

		stim_deco = DeconvolutionOperator(inputObject = timeseries, eventObject = stim_ev_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		stim_deco.runWithConvolvedNuisanceVectors(full_nuisance_design)
		stim_deco.residuals()

		# split up for stimulus/no stimulus
		rew_ev_data = [np.concatenate((event_data[0], event_data[2])), np.concatenate((event_data[1], event_data[3]))]

		rew_deco = DeconvolutionOperator(inputObject = timeseries, eventObject = rew_ev_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = True)

		np.array(((rew_deco.designMatrix.T * rew_deco.designMatrix).I * rew_deco.designMatrix.T) * np.mat(np.squeeze(stim_deco.residuals)).T).reshape((2,-1))
		# rew_deco = DeconvolutionOperator(inputObject = timeseries, eventObject = rew_ev_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		# rew_deco.runWithConvolvedNuisanceVectors(full_nuisance_design)

		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(full_nuisance_design)

		# mean stimulus response:
		stim_resp = (((deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visual_sound')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('blank_sound')]) + (deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visual_silence')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('blank_silence')])) / 2.0).squeeze()
		# mean reward response:
		rew_resp = (((deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visual_sound')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visual_silence')]) + (deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('blank_sound')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('blank_silence')])) / 2.0).squeeze()
		
		if True:
			f = pl.figure(figsize = (6,3))
			s = f.add_subplot(1,1,1)
			s.set_title(roi + ' ' + 'reward')
			pl.plot(np.linspace(interval[0], interval[1], stim_resp.shape[0]), stim_resp, 'k', label = 'stimulus')
			pl.plot(np.linspace(interval[0], interval[1], rew_resp.shape[0]), rew_resp, 'r', label = 'reward')
			s.set_xlabel('time [s]')
			s.set_ylabel('% signal change')
			# s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for l in leg.get_lines():
				    l.set_linewidth(3.5)  # the legend line width
			simpleaxis(s)
			spine_shift(s)
			# s.set_ylim([-2,2])
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_template_deconvolutions.pdf'))
			
		
		rounded_event_array = np.array([np.array(((ev / 1.5) * 2.0), dtype = int) for ev in event_data])
		rounded_event_types = np.array([np.ones(ev.shape) * i for i, ev in enumerate(event_data)])
		
		nr_trials = np.concatenate(rounded_event_array).shape[0]
		per_trial_design_matrix = np.zeros((nr_trials * 2, timeseries.shape[0] * 2))
		
		for i in range(nr_trials):
			# stimulus regressors:
			per_trial_design_matrix[i][np.concatenate(rounded_event_array)[i]] = 1.0
			per_trial_design_matrix[i] = np.correlate(per_trial_design_matrix[i], stim_resp, 'same')
			# reward regressors:
			per_trial_design_matrix[i + nr_trials][np.concatenate(rounded_event_array)[i]] = 1.0
			per_trial_design_matrix[i + nr_trials] = np.correlate(per_trial_design_matrix[i], rew_resp, 'same')
		
		full_per_trial_design_matrix = np.mat(np.vstack((per_trial_design_matrix, full_nuisance_design.T))).T
		full_per_trial_betas = ((full_per_trial_design_matrix.T * full_per_trial_design_matrix).I * full_per_trial_design_matrix.T) * np.mat(deco.workingDataArray.T).T
		full_per_trial_betas_no_nuisance = np.array(full_per_trial_betas[:nr_trials*2].reshape(2,-1).T).squeeze()
		
		shell()
		
		trial_info = pd.DataFrame({'stim_betas': full_per_trial_betas_no_nuisance[:,0], 'reward_betas': full_per_trial_betas_no_nuisance[:,1], 'event_times': np.concatenate(rounded_event_array), 'event_types': np.concatenate(rounded_event_types)})
		
		reward_h5file.close()
		mapper_h5file.close()
		with pd.get_store(self.hdf5_filename) as h5_file: # hdf5_filename is now the reward file as that was opened last
			h5_file.put("/per_trial_glm_results/%s"% roi + '_' + mask_type + '_' + mask_direction + '_' + data_type, trial_info)


	def deconvolve_interval_roi_no_stim_response(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', response_type = 'fix', iti_type = 'all_reward', binning_grain = 'session', zero_time_offset = -3.0, add_other_conditions = 'full_design'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""

		signal_type = 'mean'
		analysis_type = 'deconvolution'
		nr_bins = 2

		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		if response_type == 'fix':
			other_conds = ['blank_silence','visual_silence','visual_sound']
			other_cond_labels = ['fix_no_reward','stimulus_no_reward','stimulus_reward']
		elif response_type == 'stim':
			other_conds = ['blank_silence','blank_sound','visual_silence']
			other_cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward']
	
		all_conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		all_cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']


		iti_data = []
		event_data = []
		roi_data = []
		blink_events = []
		other_conditions_event_data = []
		nr_runs = 0
		all_event_data = []

		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times')
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters')
			
			onsets_fix_reward_trials, raw_itis_of_fix_reward_trials, all_reward_itis_of_fix_reward_trials, fixation_reward_itis_fix_reward_trials, stimulus_reward_itis_fix_reward_trials = self.calculate_event_history_fix_reward(trial_times, parameter_data)
			onsets_stim_reward_trials, raw_itis_of_stim_reward_trials, all_reward_itis_of_stim_reward_trials, fixation_reward_itis_stim_reward_trials, stimulus_reward_itis_stim_reward_trials = self.calculate_event_history_stim_reward(trial_times, parameter_data)
			
			if response_type == 'fix':
				events_of_interest = onsets_fix_reward_trials + nr_runs * run_duration
				if iti_type == 'all_reward':
					itis = all_reward_itis_of_fix_reward_trials
				elif iti_type == 'fix_reward':
					itis = fixation_reward_itis_fix_reward_trials
				elif iti_type == 'stim_reward':
					itis = stimulus_reward_itis_fix_reward_trials
				elif iti_type == 'all_trials':
					itis = raw_itis_of_fix_reward_trials
			elif response_type == 'stim':
				events_of_interest = onsets_stim_reward_trials + nr_runs * run_duration
				if iti_type == 'all_reward':
					itis = all_reward_itis_of_stim_reward_trials
				elif iti_type == 'fix_reward':
					itis = fixation_reward_itis_stim_reward_trials
				elif iti_type == 'stim_reward':
					itis = stimulus_reward_itis_stim_reward_trials
				elif iti_type == 'all_trials':
					itis = raw_itis_of_stim_reward_trials

			iti_data.append([itis, events_of_interest])

			this_run_events = []
			for cond in other_conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			other_conditions_event_data.append(this_run_events)
			this_run_all_events = []
			for cond in all_conds:
				this_run_all_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_all_events = np.array(this_run_all_events) + nr_runs * run_duration
			all_event_data.append(this_run_all_events)
			
			nr_runs += 1
		
		# shell()
		itis = np.concatenate([it[0] for it in iti_data])
		event_times = np.concatenate([it[1] for it in iti_data])
		iti_order = np.argsort(itis)
		stepsize = floor(itis.shape[0]/float(nr_bins))
		event_data = [event_times[iti_order[x*stepsize:(x+1)*stepsize]] + zero_time_offset for x in range(nr_bins)]
		iti_data = [itis[iti_order[x*stepsize:(x+1)*stepsize]] for x in range(nr_bins)]
		self.logger.info(self.subject.initials + ' ' + iti_type + ' bin means for itis: ' + str([i.mean() for i in iti_data]))
		
		other_conditions_event_data = [np.concatenate([e[i] for e in other_conditions_event_data]) + zero_time_offset for i in range(len(other_conditions_event_data[0]))]
		
		all_event_data = [np.concatenate([e[i] for e in all_event_data]) + zero_time_offset for i in range(len(all_event_data[0]))]

		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		roi_data_per_run = demeaned_roi_data
		roi_data = np.hstack(demeaned_roi_data)
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		interval = [0.0,16.0]

		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		
		# design 
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.hstack(blink_events)]))
		
		# split up for stimulus/no stimulus
		stim_ev_data = [np.concatenate((all_event_data[0], all_event_data[1])), np.concatenate((all_event_data[2], all_event_data[3]))]

		stim_deco = DeconvolutionOperator(inputObject = timeseries, eventObject = stim_ev_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		stim_deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix.T)
		stim_deco.residuals()

		fig = pl.figure(figsize = (6, 5))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		colors = [(c, 0, 1-c) for c in np.linspace(0.1,0.9,stim_deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0])]
		for i in range(0, stim_deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
			pl.plot(np.linspace(interval[0],interval[1],stim_deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(stim_deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze()), alpha = 0.7)

		s.set_title('deconvolution' + roi + ' ' + mask_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'stim_response_interval_' + roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + iti_type + '_' + response_type +  '.pdf'))


		if add_other_conditions == 'full_design':
			# this next line adds other conditions to the design
			event_data.extend(other_conditions_event_data)

		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.workingDataArray[:] = stim_deco.residuals[0,:]
		deco.run()

		time_signals = []
		for i in range(0, nr_bins):
			if add_other_conditions == 'full_design':
				time_signals.append((deco.deconvolvedTimeCoursesPerEventType[i] - deco.deconvolvedTimeCoursesPerEventType[nr_bins]).squeeze())
			else:
				time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i].squeeze())

		fig = pl.figure(figsize = (6, 5))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		colors = [(c, 0, 1-c) for c in np.linspace(0.1,0.9,nr_bins)]
		for i in range(0, nr_bins):
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), np.array(deco.deconvolvedTimeCoursesPerEventType[i].squeeze()), color = colors[i], alpha = 0.7, label = '%2.1f'%iti_data[i].mean())

		s.set_title('deconvolution' + roi + ' ' + mask_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'interval_' + roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + iti_type + '_' + response_type +  '.pdf'))

		reward_h5file.close()
		mapper_h5file.close()
		
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + iti_type + '_' + response_type + '_' + add_other_conditions, event_data, timeseries, np.array(time_signals)]
	
	def deconvolve_intervals_no_stim(self, threshold = 3.0, rois = ['V1', 'V2', 'V3'], analysis_type = 'deconvolution', signal_type = 'mean', zero_time_offset = 0.0, mask_direction = 'pos', add_other_conditions = 'full_design' ):
		results = []
		for roi in rois:
			results.append([])
			for itit in ['all_reward', 'fix_reward', 'all_trials', 'stim_reward']:
				results[-1].append(self.deconvolve_interval_roi_no_stim_response(roi, threshold, mask_type = 'center_Z', mask_direction = mask_direction, response_type = 'fix', iti_type = itit, binning_grain = 'session', zero_time_offset = zero_time_offset, add_other_conditions = add_other_conditions))
				results[-1].append(self.deconvolve_interval_roi_no_stim_response(roi, threshold, mask_type = 'center_Z', mask_direction = mask_direction, response_type = 'stim', iti_type = itit, binning_grain = 'session', zero_time_offset = zero_time_offset, add_other_conditions = add_other_conditions))
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_interval_results' + '_' + signal_type
		try:
			thisRunGroup = reward_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for res in results:
			for r in res:
				try:
					reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_' + signal_type)
					# reward_h5file.remove_node(where = thisRunGroup, name = r[0] + '_' + signal_type + '_per_run')
				except NoSuchNodeError:
					pass
				reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type, r[-1], 'interval deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
				# reward_h5file.create_array(thisRunGroup, r[0] + '_' + signal_type + '_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()



	def whole_brain_deconvolve_interval_roi_no_stim_response(self, response_type = 'fix', iti_type = 'all_reward', binning_grain = 'session', zero_time_offset = -3.0, add_other_conditions = 'full_design'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""

		signal_type = 'mean'
		analysis_type = 'deconvolution'
		nr_bins = 2

		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		nii_file_shape = list(niiFile.data.shape)

		reward_h5file = self.hdf5_file('reward')
		
		if response_type == 'fix':
			other_conds = ['blank_silence','visual_silence','visual_sound']
			other_cond_labels = ['fix_no_reward','stimulus_no_reward','stimulus_reward']
		elif response_type == 'stim':
			other_conds = ['blank_silence','blank_sound','visual_silence']
			other_cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward']
	
		all_conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		all_cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']

		nr_reward_runs = len(self.conditionDict['reward'])
		iti_data = []
		event_data = []
		nii_data = np.zeros([nr_reward_runs] + nii_file_shape)
		blink_events = []
		other_conditions_event_data = []
		nr_runs = 0
		all_event_data = []

		for j, r in enumerate([self.runList[i] for i in self.conditionDict['reward']]):
			nii_data[j] = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','tf','psc'])).data
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times')
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters')
			
			onsets_fix_reward_trials, raw_itis_of_fix_reward_trials, all_reward_itis_of_fix_reward_trials, fixation_reward_itis_fix_reward_trials, stimulus_reward_itis_fix_reward_trials = self.calculate_event_history_fix_reward(trial_times, parameter_data)
			onsets_stim_reward_trials, raw_itis_of_stim_reward_trials, all_reward_itis_of_stim_reward_trials, fixation_reward_itis_stim_reward_trials, stimulus_reward_itis_stim_reward_trials = self.calculate_event_history_stim_reward(trial_times, parameter_data)
			
			if response_type == 'fix':
				events_of_interest = onsets_fix_reward_trials + nr_runs * run_duration
				if iti_type == 'all_reward':
					itis = all_reward_itis_of_fix_reward_trials
				elif iti_type == 'fix_reward':
					itis = fixation_reward_itis_fix_reward_trials
				elif iti_type == 'stim_reward':
					itis = stimulus_reward_itis_fix_reward_trials
				elif iti_type == 'all_trials':
					itis = raw_itis_of_fix_reward_trials
			elif response_type == 'stim':
				events_of_interest = onsets_stim_reward_trials + nr_runs * run_duration
				if iti_type == 'all_reward':
					itis = all_reward_itis_of_stim_reward_trials
				elif iti_type == 'fix_reward':
					itis = fixation_reward_itis_stim_reward_trials
				elif iti_type == 'stim_reward':
					itis = stimulus_reward_itis_stim_reward_trials
				elif iti_type == 'all_trials':
					itis = raw_itis_of_stim_reward_trials

			iti_data.append([itis, events_of_interest])

			this_run_events = []
			for cond in other_conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			other_conditions_event_data.append(this_run_events)
			this_run_all_events = []
			for cond in all_conds:
				this_run_all_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_all_events = np.array(this_run_all_events) + nr_runs * run_duration
			all_event_data.append(this_run_all_events)
			
			nr_runs += 1
		
		
		reward_h5file.close()
		nii_data = nii_data.reshape((nr_reward_runs * nii_file_shape[0], -1))
		# shell()
		itis = np.concatenate([it[0] for it in iti_data])
		event_times = np.concatenate([it[1] for it in iti_data])
		iti_order = np.argsort(itis)
		stepsize = floor(itis.shape[0]/float(nr_bins))
		event_data = [event_times[iti_order[x*stepsize:(x+1)*stepsize]] + zero_time_offset for x in range(nr_bins)]
		iti_data = [itis[iti_order[x*stepsize:(x+1)*stepsize]] for x in range(nr_bins)]
		self.logger.info(self.subject.initials + ' ' + iti_type + ' bin means for itis: ' + str([i.mean() for i in iti_data]))
		
		other_conditions_event_data = [np.concatenate([e[i] for e in other_conditions_event_data]) + zero_time_offset for i in range(len(other_conditions_event_data[0]))]
		
		all_event_data = [np.concatenate([e[i] for e in all_event_data]) + zero_time_offset for i in range(len(all_event_data[0]))]		
		
		interval = [0.0,16.0]
		
		# shell()

		# design 
		# nuisance version?
		nuisance_design = Design(nii_data.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.hstack(blink_events)]))
		
		# split up for stimulus/no stimulus
		stim_ev_data = [np.concatenate((all_event_data[0], all_event_data[1])), np.concatenate((all_event_data[2], all_event_data[3]))]

		stim_deco = DeconvolutionOperator(inputObject = nii_data, eventObject = stim_ev_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		stim_deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix.T)
		stim_deco.residuals()

		# output IR curves and residuals after deconvolution
		
		res = NiftiImage(np.array(stim_deco.residuals).reshape((nr_reward_runs * nii_file_shape[0] * 2, nii_file_shape[1], nii_file_shape[2], nii_file_shape[3])))
		res.header = niiFile.header
		res.rtime = tr/2.0
		res.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + 'residuals_stim_deconvolution.nii.gz'))

		stim_response = NiftiImage(stim_deco.deconvolvedTimeCoursesPerEventTypeNuisance[0].reshape((stim_deco.deconvolvedTimeCoursesPerEventTypeNuisance[0].shape[0], nii_file_shape[1], nii_file_shape[2], nii_file_shape[3])))
		stim_response.header = niiFile.header
		stim_response.rtime = tr/2.0
		stim_response.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + 'stim_response_stim_deconvolution.nii.gz'))

		fix_response = NiftiImage(stim_deco.deconvolvedTimeCoursesPerEventTypeNuisance[1].reshape((stim_deco.deconvolvedTimeCoursesPerEventTypeNuisance[1].shape[0], nii_file_shape[1], nii_file_shape[2], nii_file_shape[3])))
		fix_response.header = niiFile.header
		stim_response.rtime = tr/2.0
		fix_response.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + 'fix_response_stim_deconvolution.nii.gz'))

		if add_other_conditions == 'full_design':
			# this next line adds other conditions to the design
			event_data.extend(other_conditions_event_data)

		deco = DeconvolutionOperator(inputObject = nii_data, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.workingDataArray[:] = stim_deco.residuals[:]
		deco.run()

		frame_names = ['long', 'short'] + other_conds
		for frame in range(len(deco.deconvolvedTimeCoursesPerEventType)):
			res = NiftiImage(deco.deconvolvedTimeCoursesPerEventType[frame].reshape((deco.deconvolvedTimeCoursesPerEventType.shape[1], nii_file_shape[1], nii_file_shape[2], nii_file_shape[3])))
			res.header = niiFile.header
			res.rtime = tr/2.0
			res.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + frame_names[frame] + '_' + 'residuals_stim_deconvolution.nii.gz'))

		# reference against the fix no reward condition and then project
		thisFolder = '/home/knapen/projects/reward/man/'
		mean_reward_response_across = np.loadtxt(os.path.join(thisFolder, 'data', 'first', 'group_level', 'data', 'V1_mean_reward_response_across.txt' ))

		for frame in [0,1]:
			these_data = (deco.deconvolvedTimeCoursesPerEventType[frame]-deco.deconvolvedTimeCoursesPerEventType[nr_bins])
			res = NiftiImage(these_data.reshape((deco.deconvolvedTimeCoursesPerEventType.shape[1], nii_file_shape[1], nii_file_shape[2], nii_file_shape[3])))
			res.header = niiFile.header
			res.rtime = tr/2.0
			res.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + frame_names[frame] + '_' + 'reference_subtracted_residuals_stim_deconvolution.nii.gz'))


			res = NiftiImage(np.dot(mean_reward_response_across, these_data).reshape((nii_file_shape[1], nii_file_shape[2], nii_file_shape[3])))
			res.header = niiFile.header
			# res.rtime = tr/2.0
			res.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + frame_names[frame] + '_' + 'reference_subtracted_projected_residuals_stim_deconvolution.nii.gz'))

			res = NiftiImage(np.dot(mean_reward_response_across, deco.deconvolvedTimeCoursesPerEventType[frame]).reshape((nii_file_shape[1], nii_file_shape[2], nii_file_shape[3])))
			res.header = niiFile.header
			# res.rtime = tr/2.0
			res.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + frame_names[frame] + '_' + 'projected_residuals_stim_deconvolution.nii.gz'))

		res = NiftiImage(np.dot(mean_reward_response_across, deco.deconvolvedTimeCoursesPerEventType[0]-deco.deconvolvedTimeCoursesPerEventType[1]).reshape((nii_file_shape[1], nii_file_shape[2], nii_file_shape[3])))
		res.header = niiFile.header
		res.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + '_LS_diff_projected_residuals_stim_deconvolution.nii.gz'))

		
	def whole_brian_deconvolution_interval_no_stim(self):
		for itit in ['all_reward', 'fix_reward', 'all_trials', 'stim_reward']:
			self.whole_brain_deconvolve_interval_roi_no_stim_response(response_type = 'fix', iti_type = itit )
			self.whole_brain_deconvolve_interval_roi_no_stim_response(response_type = 'stim', iti_type = itit )


	def all_whole_brain_interval_decos_to_surface(self):
		nii_files = subprocess.Popen('ls ' + self.stageFolder(stage = 'processed/mri/reward/deco/') + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		nii_files_LS = [nii for nii in nii_files if '_LS_diff_' in nii]

		for f in nii_files_LS:
			vsO = VolToSurfOperator(inputObject = f)
			ofn = os.path.join(os.path.split(f)[0], 'surf/', os.path.split(f)[-1])
			vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
			vsO.execute()

			for hemi in ['lh','rh']:
				ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
				ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
				ssO.execute(wait = False)

		# for itit in ['all_reward', 'fix_reward', 'all_trials', 'stim_reward']:
		# 	for rt in ['fix', 'stim']:
		# 		for files in ['stim_response_stim_deconvolution', 'fix_response_stim_deconvolution', 'reference_subtracted_residuals_stim_deconvolution', '']

		# 			os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), response_type + '_' + iti_type + '_' + '_LS_diff_projected_residuals_stim_deconvolution.nii.gz')


