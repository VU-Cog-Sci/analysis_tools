#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from Session import * 
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from ..circularTools import *
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle

class VisualRewardSession(Session):
	"""
	Analyses for visual reward sessions
	"""
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
				np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']), a.T, fmt = '%3.2f', delimiter = '\t')
			
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
			
		
	
	
	def feat_reward_analysis(self, version = '', postFix = ['mcf'], run_feat = True):
		"""
		Runs feat analysis for all reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		if not hasattr(self, 'dual_pilot'):
			for r in [self.runList[i] for i in self.conditionDict['reward']]:
				self.create_feat_event_files_one_run(r)
			
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
						thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_more_contrasts.fsf'
					else:
						thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward_more_contrasts.fsf'
				
					REDict = {
					'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
					'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
					'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']), 	
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
						thisFeatFile = '/home/knapen/projects/reward/man/analysis/mapper.fsf'
					else:
						thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/mapper.fsf'
					REDict = {
					'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
					'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
					}
					featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.fsf')
					featOp = FEATOperator(inputObject = thisFeatFile)
					if r == [self.runList[i] for i in self.conditionDict['mapper']][-1]:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
					else:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
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
						thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_dual_pilot_orientation_reward.fsf'
					else:
						thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward_dual_pilot_orientation_reward.fsf'
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
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
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
							thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward_dual_pilot_orientation_mapper.fsf'
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
							featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
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
							thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward_dual_pilot_location_mapper.fsf'
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
							featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
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
		h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		if not hasattr(self, 'dual_pilot'):
			for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
				"""loop over runs, and try to open a group for this run's data"""
				this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
				try:
					thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
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
									'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'psc', 'tf']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									# for these final two, we need to pre-setup the retinotopic mapping data
									'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
									'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz')
				})
				
				stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
				for (roi, roi_name) in zip(rois, roinames):
					try:
						thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
					except NoSuchNodeError:
						# import actual data
						self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
						thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
					for (i, sf) in enumerate(stat_files.keys()):
						# loop over stat_files and rois
						# to mask the stat_files with the rois:
						imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
						these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
						h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
		else:
			version_postFix = postFix + ['orientation']
			for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
				"""loop over runs, and try to open a group for this run's data"""
				this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
				try:
					thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
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
						thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
					except NoSuchNodeError:
						# import actual data
						self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
						thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
					for (i, sf) in enumerate(stat_files.keys()):
						# loop over stat_files and rois
						# to mask the stat_files with the rois:
						imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
						these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
						h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
			
		h5file.close()
	
	def hdf5_file(self, run_type, mode = 'r'):
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('no table file ' + self.hdf5_filename + 'found for stat mask')
			return None
		else:
			# self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = mode, title = run_type + " file")
		return h5file
	
	
	def pupil_responses_one_run(self, run, frequency, sample_rate = 2000, postFix = ['mcf'], analysis_duration = 10):
		if run.condition == 'reward':
			# get EL Data
			
			h5f = openFile(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'), mode = "r" )
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
				thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix))
			
			# save parameter data to joint file
			import numpy.lib.recfunctions as rfn
			try: 
				h5file.removeNode(where = thisRunGroup, name = 'trial_parameters')
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
				h5file.removeNode(where = thisRunGroup, name = 'trial_times')
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
				h5file.removeNode(where = thisRunGroup, name = 'filtered_pupil_zscore')
				h5file.removeNode(where = thisRunGroup, name = 'per_trial_filtered_pupil_zscore')
				h5file.removeNode(where = thisRunGroup, name = 'per_condition_filtered_pupil_zscore')
			except NoSuchNodeError:
				pass
			h5file.createArray(thisRunGroup, 'filtered_pupil_zscore', np.vstack((gaze_timestamps, pupil_zscore)).T, 'filtered_pupil_zscore conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			h5file.createArray(thisRunGroup, 'per_trial_filtered_pupil_zscore', np.array(tr_data_timed), 'per_trial_filtered_pupil_zscore conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			h5file.createArray(thisRunGroup, 'per_condition_filtered_pupil_zscore', np.array([np.array(tr).mean(axis = 0) for tr in tr_data]), 'per_condition_filtered_pupil_zscore conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
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
			
	def pupil_responses(self, sample_rate = 2000, save_all = False):
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
				h5file.removeNode(where = '/', name = 'all_pupil_scores')
			except NoSuchNodeError:
				pass
			h5file.createArray('/', 'all_pupil_scores', np.array(all_data), '_'.join(cond_labels) + ' conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
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
	
	def deconvolve_roi(self, roi, threshold = 3.5, mask_type = 'center_surround_Z', analysis_type = 'deconvolution', mask_direction = 'pos'):
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
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
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
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
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
			deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
			for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
				time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
				# shell()
				pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])[0], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			
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
				pl.plot(np.linspace(0,interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), np.array(ts_diff)[0], ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
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
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals), np.array(deco_per_run)]
	
	def deconvolve_pupil(self, sample_rate = 2000, postFix = ['mcf'], subsampled_sample_frequency = 5):
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		event_data = []
		pupil_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			try:
				thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.error('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' does not contain ' + this_run_group_name + '. Exiting.')
				return
			
			timings = thisRunGroup.trial_times.read()
			experiment_start_time = (timings['trial_phase_timestamps'][0,0,0] / 1000.0)
			pupil_data_this_run = thisRunGroup.filtered_pupil_zscore.read()
			subsampled_pupil_data_this_run = pupil_data_this_run[pupil_data_this_run[:,0]>experiment_start_time, 1][0:(run_duration*sample_rate):(sample_rate/subsampled_sample_frequency)]
			pupil_data.append(subsampled_pupil_data_this_run)
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			nr_runs += 1
		
		event_data_per_run = event_data
		
		pupil_data_per_run = pupil_data
		pupil_data = np.hstack(pupil_data)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		fig = pl.figure(figsize = (9, 5))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		interval = [0.0,16.0]
			
		deco = DeconvolutionOperator(inputObject = pupil_data, eventObject = event_data[:], TR = 1.0/subsampled_sample_frequency, deconvolutionSampleDuration = 1.0/subsampled_sample_frequency, deconvolutionInterval = interval[1])
		for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		shell()
		s.set_title('deconvolution pupil')
		deco_per_run = []
		for i, pd in enumerate(pupil_data_per_run):
			event_data_this_run = event_data_per_run[i] - i * run_duration
			deco = DeconvolutionOperator(inputObject = pd, eventObject = event_data_per_run[i], TR = 1.0/subsampled_sample_frequency, deconvolutionSampleDuration = 1.0/subsampled_sample_frequency, deconvolutionInterval = interval[1])
			deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
		deco_per_run = np.array(deco_per_run)
		mean_deco = deco_per_run.mean(axis = 0)
		std_deco = 1.96 * deco_per_run.std(axis = 0) / sqrt(len(pupil_data_per_run))
		for i in range(0, mean_deco.shape[0]):
			# pl.plot(np.linspace(interval[0],interval[1],mean_deco.shape[1]), mean_deco[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), time_signals[i] + std_deco[i], time_signals[i] - std_deco[i], color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])
		
		s.set_xlabel('time [s]')
		s.set_ylabel('Z')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'pupil_deconvolution.pdf'))
		reward_h5file.close()
	
	def deconvolve(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution'):
		results = []
		for roi in rois:
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'surround_center_Z', analysis_type = analysis_type, mask_direction = 'pos'))
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_results'
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = r[0])
				reward_h5file.removeNode(where = thisRunGroup, name = r[0]+'_per_run')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, r[0], r[-2], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.createArray(thisRunGroup, r[0]+'_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def whole_brain_deconvolution(self, deco = True, average_interval = [2,12], to_surf = True):
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
			roi_data = []
			nr_runs = 0
			nii_data = np.zeros([nr_reward_runs] + nii_file_shape)
		
			for (j, r) in enumerate([self.runList[i] for i in self.conditionDict['reward']]):
				nii_data[j] = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','psc','tf'])).data
				this_run_events = []
				for cond in conds:
					this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
				this_run_events = np.array(this_run_events) + nr_runs * run_duration
				event_data.append(this_run_events)
				nr_runs += 1
		
			nii_data = nii_data.reshape((nr_reward_runs * nii_file_shape[0], -1))
			event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
			deco = DeconvolutionOperator(inputObject = nii_data, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		
		for (i, c) in enumerate(cond_labels):
			ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_mean_' + c + '.nii.gz')
			if deco:
				outputdata = deco.deconvolvedTimeCoursesPerEventType[i]
				outputFile = NiftiImage(outputdata.reshape([outputdata.shape[0]]+nii_file_shape[1:]))
				outputFile.header = niiFile.header
				outputFile.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_' + c + '.nii.gz'))
			
				# average over the interval [5,12] and [2,10] for reward and visual respectively. so, we'll just do [2,12]
				timepoints_for_averaging = (np.linspace(interval[0], interval[1], outputdata.shape[0]) < average_interval[1]) * (np.linspace(interval[0], interval[1], outputdata.shape[0]) > average_interval[0])
				meaned_data = outputdata[timepoints_for_averaging].mean(axis = 0)
				outputFile = NiftiImage(meaned_data.reshape(nii_file_shape[1:]))
				outputFile.header = niiFile.header
				outputFile.save(ofn)
			
			if to_surf:
				# vol to surf?
				# for (label, f) in zip(['left', 'right'], [left_file, right_file]):
				vsO = VolToSurfOperator(inputObject = ofn)
				sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
				vsO.configure(frames = {'ss':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
				vsO.execute()
				
				for hemi in ['lh','rh']:
					ssO = SurfToSurfOperator(vsO.outputFileName + c + '-' + hemi + '.w')
					ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_average', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
					ssO.execute()
	
	
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
					thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
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
						reward_h5file.removeNode(where = thisRunGroup, name = deconv._v_name + '_stats')
					except NoSuchNodeError:
						pass
					reward_h5file.createArray(thisRunGroup, deconv._v_name + '_stats', stat_results, 'ANOVA timecourses conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
					
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
			pupil_signals = reward_h5file.getNode(where = '/', name = 'all_pupil_scores', classname = 'Array').read().transpose(2,1,0)[::(sample_rate/comparison_rate)]
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
				reward_h5file.removeNode(where = '/', name = 'all_pupil_stats')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray('/', 'all_pupil_stats', stat_results, 'ANOVA timecourses on pupil data conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			
		
		reward_h5file.close()
		pl.show()
			# 		
			# elif data_type == 'pupil':
			# 	this_run_group_name = 'all_pupil_scores'
			
	
	def run_glm_on_hdf5(self, data_type = 'hpf_data', analysis_type = 'per_trial', post_fix_for_text_file = ['all_trials'], functionalPostFix = ['mcf'], which_conditions = ['reward','mapper']):
		if 'reward' in which_conditions:
			reward_h5file = self.hdf5_file('reward', mode = 'r+')
			super(VisualRewardSession, self).run_glm_on_hdf5(run_list = [self.runList[i] for i in self.conditionDict['reward']], hdf5_file = reward_h5file, data_type = data_type, analysis_type = analysis_type, post_fix_for_text_file = post_fix_for_text_file, functionalPostFix = functionalPostFix)
			reward_h5file.close()
		if 'mapper' in which_conditions:
			mapper_h5file = self.hdf5_file('mapper', mode = 'r+')
			super(VisualRewardSession, self).run_glm_on_hdf5(run_list = [self.runList[i] for i in self.conditionDict['mapper']], hdf5_file = mapper_h5file, data_type = data_type, analysis_type = analysis_type, post_fix_for_text_file = post_fix_for_text_file, functionalPostFix = functionalPostFix)
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
				thisRunGroup = reward_h5file.getNode(where = '/', name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))[1], classname='Group')
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
		s = fig.add_subplot(1,1,1)
		for (i, e) in enumerate(event_data):
			bold_trials = np.array([bold_timeseries[(trts > se + time_range_BOLD[0]) * (trts <= se + time_range_BOLD[1])].mean() for se in e])
			pupil_trials = []
			for time in np.arange(time_range_pupil[0],time_range_pupil[1]-stepsize, stepsize):
				pupil_trials.append(np.array([pdc[(pdc[:,0] > (se + time)) * (pdc[:,0] <= (se + time + stepsize)),1].mean() - pdc[(pdc[:,0] > se - 2.0) * (pdc[:,0] <= se),1].mean() for se in e]))
			all_corrs.append(np.array([spearmanr(bold_trials, p) for p in pupil_trials]))
			
			pl.plot(np.arange(time_range_pupil[0],time_range_pupil[1]-stepsize, stepsize) + stepsize * 0.5, all_corrs[-1][:,0], color = ['b','b','g','g'][i], alpha = 0.7 * [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			
			# print roi, cond_labels[i], spearmanr(bold_trials, pupil_trials)
			
			# s.scatter(bold_trials,  pupil_trials, color = ['b','b','g','g'][i], marker = 'o', s = 27, edgecolor = 'w', alpha = 0.7 * [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			# pl.plot(bold_trials,  pupil_trials, marker = 'o', ms = 7, mec = 'w', c = ['b','b','g','g'][i], mew = 1.0, alpha = 0.7 * [0.5, 1.0, 0.5, 1.0][i], linewidth = 0, label = cond_labels[i]) # , alpha = 0.25
			
			# split into quartiles and t-test the most peripheral bins against one another.
			# plot the quartile means into the correlation plot. 
			# bold_order = np.argsort(bold_trials)
			# pupil_order = np.argsort(pupil_trials)
			# bin_edge = bold_trials.shape[0] / 4.0
			# print ttest_rel(bold_trials[pupil_order[:floor(bin_edge)]], bold_trials[pupil_order[ceil(3*bin_edge):]]), ttest_rel(pupil_trials[bold_order[:floor(bin_edge)]], pupil_trials[bold_order[ceil(3*bin_edge):]])
			
			
			# s.set_xlabel('BOLD % signal change')
			# s.set_ylabel('Z-scored pupil size')
			# if i == 0:
			# 	leg = s.legend(fancybox = True)
			# 	leg.get_frame().set_alpha(0.5)
			# 	if leg:
			# 		for t in leg.get_texts():
			# 		    t.set_fontsize('small')    # the legend text fontsize
			# 		for l in leg.get_lines():
			# 		    l.set_linewidth(3.5)  # the legend line width
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
		
		
		
	def correlate_pupil_and_BOLD(self, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000):
		corrs = []
		areas = ['V1', 'V2', 'V3', 'V3AB', 'V4']
		for roi in areas:
			corrs.append(self.correlate_pupil_and_BOLD_for_roi(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi))
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'pupil-BOLD_correlation_results'
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'pupil/bold correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for (i, c) in enumerate(corrs):
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = areas[i])
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, areas[i], np.array(corrs[i]), 'pupil-bold correlation timecourses conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
		
		# pl.show()
	
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
				thisRunGroup = reward_h5file.getNode(where = '/', name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))[1], classname='Group')
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
	
	def cross_correlate_pupil_and_BOLD(self, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range = 'long'):
		corrs = []
		spearman_corrs = []
		areas = ['V1', 'V2', 'V3', 'V3AB', 'V4']
		f = pl.figure(figsize = (4,12))
		for (i, roi) in enumerate(areas):
			s = f.add_subplot(len(areas), 1, i+1)
			if time_range == 'short':
				crs = self.cross_correlate_pupil_and_BOLD_for_roi(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi, time_range_BOLD = [1.5, 7.0], time_range_pupil = [0.5, 2.0])
			elif time_range == 'long':
				crs = self.cross_correlate_pupil_and_BOLD_for_roi(roi = roi, threshold = threshold, mask_type = mask_type, mask_direction = mask_direction, sample_rate = sample_rate, area = roi, time_range_BOLD = [5.0, 10.0], time_range_pupil = [3.0, 9.0])
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
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'pupil/bold correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for (i, c) in enumerate(corrs):
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = areas[i] + '_' + mask_type + '_' + mask_direction)
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, areas[i] + '_' + mask_type + '_' + mask_direction, np.array(corrs[i]), 'pupil-bold cross correlation timecourses conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		
		for (i, c) in enumerate(spearman_corrs):
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = areas[i] + '_' + mask_type + '_' + mask_direction + '_spearman' + '_' + time_range)
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, areas[i] + '_' + mask_type + '_' + mask_direction + '_spearman' + '_' + time_range, np.array(c), 'pupil-bold spearman correlation results conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
		
		# pl.show()
	

class VisualRewardDualSession(VisualRewardSession):
	
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
		h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		version_postFix = postFix + ['orientation']
		for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
			"""loop over runs, and try to open a group for this run's data"""
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			try:
				thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
			
			# add parameters and the like 
			eye_h5file = openFile(self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'), mode = "r")
			eyeGroup = eye_h5file.getNode(where = '/', name = 'bla', classname='Group')
			eyeGroup._f_copyChildren(thisRunGroup) 
			eye_h5file.close()
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
								
								'center_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat1.nii.gz'),
								'center_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat1.nii.gz'),
								'center_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope1.nii.gz'),
								'center_pe': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'pe1.nii.gz'),
								
								'surround_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat2.nii.gz'),
								'surround_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat2.nii.gz'),
								'surround_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope2.nii.gz'),
								'surround_pe': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'pe3.nii.gz'),
								
								'center>surround_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat3.nii.gz'),
								'center>surround_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat3.nii.gz'),
								'center>surround_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope3.nii.gz'),
								
								'surround>center_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat4.nii.gz'),
								'surround>center_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat4.nii.gz'),
								'surround>center_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope4.nii.gz'),
								
			})
				
			stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
				for (i, sf) in enumerate(stat_files.keys()):
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
					these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
					h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
			
		h5file.close()
		
	
	
	def deconvolve_roi(self, roi, threshold = 2.5, mask_type = 'left_Z', analysis_type = 'deconvolution', mask_direction = 'pos'):
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
		
		conds = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW', 'blank_silence', 'blank_rewarded']
		colors = ['r','r','g','g','k','k']
		alphas = [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
		
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data', postFix = ['mcf','tf']))
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
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
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type, postFix = ['mcf','tf'])
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		fig = pl.figure(figsize = (7, 3))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		interval = [0.0,16.0]
			
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], colors[i], alpha = alphas[i], label = conds[i])
			time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		s.set_title('deconvolution' + roi + ' ' + mask_type)
		deco_per_run = []
		for i, rd in enumerate(roi_data_per_run):
			event_data_this_run = event_data_per_run[i] - i * run_duration
			deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
			deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
		deco_per_run = np.array(deco_per_run)
		mean_deco = deco_per_run.mean(axis = 0)
		std_deco = 1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))
		for i in range(0, mean_deco.shape[0]):
			s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), time_signals[i] + std_deco[i], time_signals[i] - std_deco[i], color = colors[i], alpha = 0.3 * alphas[i])
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				if i == self.which_stimulus_rewarded:
					l.set_linewidth(3.5)  # the legend line width
				else:
					l.set_linewidth(2.0)  # the legend line width
		reward_h5file.close()
		mapper_h5file.close()
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals), deco_per_run]
	
	def deconvolve(self, threshold = 2.5, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution'):
		results = []
		for roi in rois:
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'left_Z', analysis_type = analysis_type, mask_direction = 'pos'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'right_Z', analysis_type = analysis_type, mask_direction = 'pos'))
			results.append(self.deconvolve_roi(roi, 4.0, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos'))
			results.append(self.deconvolve_roi(roi, 4.0, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg'))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_results'
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = r[0])
				reward_h5file.removeNode(where = thisRunGroup, name = r[0]+'_per_run')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, r[0], r[-2], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.createArray(thisRunGroup, r[0]+'_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def project_stats(self, which_file = 'zstat', postFix = ['mcf','tf']):
		
		for r in [self.runList[i] for i in self.conditionDict['mapper']]:
			this_location_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['location'], extension = '.feat')
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat') # to look at the locations, which is what we're doing here, add  + 'tf' + 'location' to postfix when calling this method.
			left_file = os.path.join(this_location_feat, 'stats', which_file + '1.nii.gz')
			right_file = os.path.join(this_location_feat, 'stats', which_file + '2.nii.gz')
			for (label, f) in zip(['left', 'right'], [left_file, right_file]):
				vsO = VolToSurfOperator(inputObject = f)
				ofn = self.runFile(stage = 'processed/mri/', run = r, base = which_file, postFix = [label] )
				ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
				vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
				vsO.execute()
	
	def rewarded_stimulus_run(self, run, postFix = ['mcf','tf']):
		reward_h5file = self.hdf5_file('reward')
		this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('couldn\'t find the data file ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' in ' + self.hdf5_filename)
			pass
		parameter_data = thisRunGroup.trial_parameters.read()
		reward_h5file.close()
		
		left_none_right = np.sign(parameter_data['x_position'])
		CW_none_CCW = np.sign(parameter_data['orientation'])
			
		condition_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW']
		left_CW_trials = (left_none_right == -1) * (CW_none_CCW == -1)
		left_CCW_trials = (left_none_right == -1) * (CW_none_CCW == 1)
		right_CW_trials = (left_none_right == 1) * (CW_none_CCW == -1)
		right_CCW_trials = (left_none_right == 1) * (CW_none_CCW == 1)
				
		run.all_stimulus_trials = [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials]
		all_reward_trials = np.array(parameter_data['sound'], dtype = bool)
				
		which_trials_rewarded = np.array([(trials * all_reward_trials).sum() > 0 for trials in [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials]])
		which_stimulus_rewarded = np.arange(4)[which_trials_rewarded]
		stim_trials_rewarded = np.squeeze(np.array(run.all_stimulus_trials)[which_trials_rewarded])
				
		blank_trials_rewarded = all_reward_trials - stim_trials_rewarded
		blank_trials_silence = -np.array(np.abs(left_none_right) + blank_trials_rewarded, dtype = bool)
				
		# identify rewarded condition by label
		# condition_labels[which_stimulus_rewarded] += '_rewarded'
		run.which_stimulus_rewarded = which_stimulus_rewarded
		run.parameter_data = parameter_data
		# try to make this permanent.
		self.which_stimulus_rewarded = which_stimulus_rewarded
		
	
	def correlate_patterns_over_time_for_roi(self, roi, classification_data_type = 'per_trial_hpf_data_zscore', data_type_mask = 'Z', mask_threshold = 3.5, mask_direction = 'pos', postFix = ['mcf','tf']):
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		conditions_data_types = ['left_CW_Z', 'left_CCW_Z', 'right_CW_Z', 'right_CCW_Z']
		condition_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW']
		
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, classification_data_type, postFix = ['mcf','tf']))
			this_run_events = []
			for cond in condition_labels:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			self.rewarded_stimulus_run(r, postFix = postFix)
			nr_runs += 1
		
		event_data_per_run = event_data		
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data_L = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'left_' + data_type_mask, postFix = ['mcf','tf'])
		mapping_data_R = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'right_' + data_type_mask, postFix = ['mcf','tf'])
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask_L = mapping_data_L[:,0] > mask_threshold
			mapping_mask_R = mapping_data_R[:,0] > mask_threshold
		else:
			mapping_mask_L = mapping_data_L[:,0] < mask_threshold
			mapping_mask_R = mapping_data_R[:,0] < mask_threshold
		
		# if self.which_stimulus_rewarded < 2:
		rewarded_mask = mapping_mask_L
		non_rewarded_mask = mapping_mask_R
		# else:
		# 	rewarded_mask = mapping_mask_R
		# 	non_rewarded_mask = mapping_mask_L
			
		which_orientation_rewarded = self.which_stimulus_rewarded % 2
		reward_run_list = [self.runList[j] for j in self.conditionDict['reward']]
		L_data = np.hstack([[r[:,k].T for k in reward_run_list[i].all_stimulus_trials] for (i, r) in enumerate(roi_data)])[...,mapping_mask_L]
		R_data = np.hstack([[r[:,k].T for k in reward_run_list[i].all_stimulus_trials] for (i, r) in enumerate(roi_data)])[...,mapping_mask_R]
		
		orientation_mapper_data = [self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi_wildcard = roi, data_type = dt, postFix = ['mcf','tf']) for dt in conditions_data_types]
		# rew_ori_data = np.squeeze(np.array([orid[rewarded_mask] for orid in orientation_mapper_data]))
		# non_rew_ori_data = np.squeeze(np.array([orid[non_rewarded_mask] for orid in orientation_mapper_data]))
		
		L_ori_data = np.squeeze(np.array([orientation_mapper_data[i][mapping_mask_L] - orientation_mapper_data[i+1][mapping_mask_L] for i in [0,2]]))
		R_ori_data = np.squeeze(np.array([orientation_mapper_data[i][mapping_mask_R] - orientation_mapper_data[i+1][mapping_mask_R] for i in [0,2]]))
		# L_ori_data = np.squeeze(np.array([(orientation_mapper_data[i][mapping_mask_L] - orientation_mapper_data[i][mapping_mask_L].mean()) - (orientation_mapper_data[i+1][mapping_mask_L] - orientation_mapper_data[i+1][mapping_mask_L].mean()) for i in [0,2]]))
		# R_ori_data = np.squeeze(np.array([(orientation_mapper_data[i][mapping_mask_R] - orientation_mapper_data[i][mapping_mask_R].mean()) - (orientation_mapper_data[i+1][mapping_mask_R] - orientation_mapper_data[i+1][mapping_mask_R].mean()) for i in [0,2]]))
		L_ori_data = np.array([(L_ori_data[i]/L_ori_data[i].mean())/L_ori_data[i].std() for i in [0,1]])
		R_ori_data = np.array([(R_ori_data[i]/R_ori_data[i].mean())/R_ori_data[i].std() for i in [0,1]])
		
		from scipy import stats
		L_corr = np.zeros((4, L_data.shape[1]))
		R_corr = np.zeros((4, R_data.shape[1]))
		for i in range(4):
			# rew_corr[i] = np.array([stats.spearmanr(rew_ori_data[i], d) for d in rew_data[i]])
			# non_rew_corr[i] = np.array([stats.spearmanr(non_rew_ori_data[i], d) for d in non_rew_data[i]])
			L_corr[i] = np.array([np.dot(L_ori_data[int(floor(i/2))], (d - d.mean())/d.std()) / d.shape[0] for d in L_data[i]])
			R_corr[i] = np.array([np.dot(R_ori_data[int(floor(i/2))], (d - d.mean())/d.std()) / d.shape[0] for d in R_data[i]])
		
		L_corr_diffs = [L_corr[i] - L_corr[i+1] for i in [0,2]]
		R_corr_diffs = [R_corr[i] - R_corr[i+1] for i in [0,2]]
		
		
		alphas = np.ones(4) * 0.45
		alphas[self.which_stimulus_rewarded] = 1.0
		colors = ['r', 'r--', 'b', 'b--']
		if self.which_stimulus_rewarded % 2 == 0:
			diff_color = 'k'
		else:
			diff_color = 'k--'
		if self.which_stimulus_rewarded < 2:
			diff_alpha = [0.75, 0, 0.25]
			hist_alphas = [0.75, 0.25]
		else:
			diff_alpha = [0.25, 0, 0.75]
			hist_alphas = [0.25, 0.75]
		
		f = pl.figure(figsize = (12,6))
		s = f.add_subplot(2,2,1)
		s.set_title('left stimulus ROI in ' + roi)
		s.set_xlabel('time [trials]')
		s.set_ylabel('pattern distance to mapper')
		s.set_xlim([0,L_corr[i].shape[0]])
		s.set_ylim([-2, 2])
		[s.axvspan(i * 12, (i+1) * 12, facecolor='k', alpha=0.05, edgecolor = 'w') for i in [0,2,4]]
		for i in range(4):
			plot(L_corr[i], colors[i], alpha = alphas[i], label = condition_labels[i], linewidth = alphas[i] * 2.0)
		for i in [0,1]:
			plot(L_corr_diffs[i], diff_color, alpha = hist_alphas[i], label = ['left','right'][i], linewidth = diff_alpha[i] * 2.0)
		# s.axis([0,rew_corr[i,:,0].shape[0],-1.0,1.0])
		s = f.add_subplot(2,2,3)
		s.set_title('right stimulus ROI in ' + roi)
		[s.axvspan(i * 12, (i+1) * 12, facecolor='k', alpha=0.05, edgecolor = 'w') for i in [0,2,4]]
		for i in range(4):
			plot(R_corr[i], colors[i], alpha = alphas[i], label = condition_labels[i], linewidth = alphas[i] * 2.0)
		for i in [0,1]:
			plot(R_corr_diffs[i], diff_color, alpha = hist_alphas[i], label = ['left','right'][i], linewidth = diff_alpha[i] * 2.0)
		s.set_xlabel('time [trials]')
		s.set_ylabel('pattern distance to mapper')
		s.set_xlim([0,L_corr[i].shape[0]])
		s.set_ylim([-2, 2])
		# s.axis([0,rew_corr[i,:,0].shape[0],-1.0,1.0])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		s = f.add_subplot(2,2,2)
		s.set_title('histogram left ROI in ' + roi)
		s.axvline(x = 0, c = 'k', linewidth = 0.5)
		for i in [0,1]:
			s.axhline(y = L_corr_diffs[i].mean(), c = 'k', linewidth = 2.5, linestyle = '--', alpha = hist_alphas[i])
			pl.hist(L_corr_diffs[i], color='k', alpha = hist_alphas[i], normed = True, bins = 20, rwidth = 0.5, histtype = 'step', linewidth = 2.5, orientation = 'horizontal' )
			pl.text(0.1, 0.75 - i/2.0, ['left stimulus orientation difference p-value: ','right stimulus orientation difference p-value: '][i] + str(stats.ttest_rel(L_corr[i*2], L_corr[i*2+1])[1]), transform = s.transAxes)
		s.set_ylim([-2, 2])
		s = f.add_subplot(2,2,4)
		s.set_title('histogram right ROI in ' + roi)
		s.axvline(x = 0, c = 'k', linewidth = 0.5)
		for i in [0,1]:
			s.axhline(y = R_corr_diffs[i].mean(), c = 'k', linewidth = 2.5, linestyle = '--', alpha = hist_alphas[i])
			pl.hist(R_corr_diffs[i], color='k', alpha = hist_alphas[i], normed = True, bins = 20, rwidth = 0.5, histtype = 'stepfilled', orientation = 'horizontal' )
			pl.text(0.1, 0.75 - i/2.0, ['left stimulus orientation difference p-value: ','right stimulus orientation difference p-value: '][i] + str(stats.ttest_rel(R_corr[i*2], R_corr[i*2+1])[1]), transform = s.transAxes)
		s.set_ylim([-2, 2])
		return [L_corr, R_corr]
		
	def correlate_patterns_over_time(self, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], classification_data_type = 'per_trial_hpf_data_zscore', data_type_mask = 'Z', mask_threshold = 2.3, mask_direction = 'pos', postFix = ['mcf','tf']):
		from matplotlib.backends.backend_pdf import PdfPages
		pp = PdfPages(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'per_trial_pattern_correlation_' + mask_direction + '_' + classification_data_type + '_' + data_type_mask + '.pdf'))
		
		pattern_corr_results = []
		for roi in rois:
			pattern_corr_results.append(self.correlate_patterns_over_time_for_roi(roi = roi, classification_data_type = classification_data_type, data_type_mask = data_type_mask, mask_threshold = mask_threshold, mask_direction = mask_direction, postFix = postFix))
			pp.savefig()
		pp.close()
		pl.show()
	
	def decode_patterns_per_trial(self, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], classification_data_type = 'per_trial_hpf_data_zscore', data_type_mask = 'Z', mask_threshold = 3.5, mask_direction = 'pos', postFix = ['mcf','tf']):
		from matplotlib.backends.backend_pdf import PdfPages
		pp = PdfPages(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'per_trial_pattern_decoding_' + mask_direction + '_' + classification_data_type + '_' + data_type_mask + '.pdf'))
		
		pattern_decoding_results = []
		for roi in rois:
			pattern_decoding_results.append(self.decode_patterns_per_trial_for_roi(roi = roi, classification_data_type = classification_data_type, data_type_mask = data_type_mask, mask_threshold = mask_threshold, mask_direction = mask_direction, postFix = postFix))
			pp.savefig()
		pp.close()
		pl.show()
		
		
	def decode_patterns_per_trial_for_roi(self, roi, classification_data_type = 'per_trial_hpf_data_zscore', data_type_mask = 'Z', mask_threshold = 3.5, mask_direction = 'pos', postFix = ['mcf','tf']):
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		conditions_data_types = ['left_CW_Z', 'left_CCW_Z', 'right_CW_Z', 'right_CCW_Z']
		condition_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW']
		
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, classification_data_type, postFix = ['mcf','tf']))
			this_run_events = []
			for cond in condition_labels:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			self.rewarded_stimulus_run(r, postFix = postFix)
			nr_runs += 1
		
		# zscore the functional data
		for roid in roi_data:
			roid = ((roid.T - roid.mean(axis = 1)) / roid.std(axis = 1)).T
			roid = (roid - roid.mean(axis = 0)) / roid.std(axis = 0)
		
		event_data_per_run = event_data
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data_L = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'left_' + data_type_mask, postFix = ['mcf','tf'])
		mapping_data_R = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'right_' + data_type_mask, postFix = ['mcf','tf'])
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask_L = mapping_data_L[:,0] > mask_threshold
			mapping_mask_R = mapping_data_R[:,0] > mask_threshold
		else:
			mapping_mask_L = mapping_data_L[:,0] < mask_threshold
			mapping_mask_R = mapping_data_R[:,0] < mask_threshold
		
		which_orientation_rewarded = self.which_stimulus_rewarded % 2
		reward_run_list = [self.runList[j] for j in self.conditionDict['reward']]
		L_data = np.hstack([[r[:,k].T for k in reward_run_list[i].all_stimulus_trials] for (i, r) in enumerate(roi_data)])[...,mapping_mask_L]
		R_data = np.hstack([[r[:,k].T for k in reward_run_list[i].all_stimulus_trials] for (i, r) in enumerate(roi_data)])[...,mapping_mask_R]
		
		ormd = []
		mcd = []
		for mf in [0,-1]:
			mapper_run_events = []
			for cond in condition_labels:
				mapper_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['mapper'][mf]], extension = '.txt', postFix = [cond]))[:,0])	
			mapper_condition_order = np.array([np.array([np.ones(m.shape) * i, m]).T for (i, m) in enumerate(mapper_run_events)]).reshape(-1,2)
			mapper_condition_order = mapper_condition_order[np.argsort(mapper_condition_order[:,1]), 0]
			mapper_condition_order_indices = np.array([mapper_condition_order == i for i in range(len(condition_labels))], dtype = bool)
			mcd.append(mapper_condition_order_indices)
			orientation_mapper_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][mf]], roi_wildcard = roi, data_type = classification_data_type, postFix = ['mcf','tf'])
			orientation_mapper_data = ((orientation_mapper_data.T - orientation_mapper_data.mean(axis = 1)) / orientation_mapper_data.std(axis = 1)).T
			orientation_mapper_data = (orientation_mapper_data - orientation_mapper_data.mean(axis = 0)) / orientation_mapper_data.std(axis = 0)
			ormd.append(orientation_mapper_data)
		
		mapper_condition_order_indices = np.hstack(mcd)
		orientation_mapper_data = np.hstack(ormd)
		
		all_ori_data = np.array([orientation_mapper_data[:,moi].T for moi in mapper_condition_order_indices])
		L_ori_data = all_ori_data[...,mapping_mask_L]
		R_ori_data = all_ori_data[...,mapping_mask_R]
		
		train_data_L, train_labels_L = [np.vstack((L_ori_data[i],L_ori_data[i+1])) for i in [0,2]], [np.concatenate((-np.ones(L_ori_data.shape[1]),np.ones(L_ori_data.shape[1])))  for i in [0,2]]
		train_data_R, train_labels_R = [np.vstack((R_ori_data[i],R_ori_data[i+1])) for i in [0,2]], [np.concatenate((-np.ones(R_ori_data.shape[1]),np.ones(R_ori_data.shape[1])))  for i in [0,2]]
		
		test_data_L, test_labels_L = [np.vstack((L_data[i],L_data[i+1])) for i in [0,2]], [np.concatenate((-np.ones(L_data.shape[1]),np.ones(L_data.shape[1])))  for i in [0,2]]
		test_data_R, test_labels_R = [np.vstack((R_data[i],R_data[i+1])) for i in [0,2]], [np.concatenate((-np.ones(R_data.shape[1]),np.ones(R_data.shape[1])))  for i in [0,2]]
		
		# shell()
		from sklearn import neighbors, datasets, linear_model, svm, lda, qda
		# kern = svm.SVC(probability=True, kernel = 'linear', C=1e4) # , C=1e3), NuSVC , C = 1.0
		# kern = svm.LinearSVC(C=1e5, loss='l1') # , C=1e3), NuSVC , C = 1.0
		# kern = svm.SVC(probability=True, kernel='rbf', degree=2) # , C=1e3), NuSVC , C = 1.0
		kern = lda.LDA()
		# kern = qda.QDA()
		# kern = neighbors.KNeighborsClassifier()
		# kern = linear_model.LogisticRegression(C=1e5)
		
		
		corrects_L = [kern.fit(train_data_L[i], train_labels_L[i]).predict(test_data_L[i]) * test_labels_L[i] for i in [0,1]]
		corrects_R = [kern.fit(train_data_R[i], train_labels_R[i]).predict(test_data_R[i]) * test_labels_R[i] for i in [0,1]]
		
		corrects_per_cond_L = np.array([[cl[:cl.shape[0]/2], cl[cl.shape[0]/2:]] for cl in corrects_L]).reshape(4,-1)
		corrects_per_cond_R = np.array([[cr[:cr.shape[0]/2], cr[cr.shape[0]/2:]] for cr in corrects_R]).reshape(4,-1)
		
		probs_L = [kern.fit(train_data_L[i], train_labels_L[i]).predict_proba(test_data_L[i])[:,0] for i in [0,1]]
		probs_R = [kern.fit(train_data_R[i], train_labels_R[i]).predict_proba(test_data_R[i])[:,0] for i in [0,1]]
		
		probs_per_cond_L = np.array([[cl[:cl.shape[0]/2], cl[cl.shape[0]/2:]] for cl in probs_L]).reshape(4,-1)
		probs_per_cond_R = np.array([[cr[:cr.shape[0]/2], cr[cr.shape[0]/2:]] for cr in probs_R]).reshape(4,-1)
		
		print roi
		print 'left: ' + str(((corrects_per_cond_L + 1) / 2).mean(axis = 1))
		print 'right: ' + str(((corrects_per_cond_R + 1) / 2).mean(axis = 1))
		
		# now, plotting
		alphas = np.ones(4) * 0.45
		alphas[self.which_stimulus_rewarded] = 1.0
		colors = ['r', 'r--', 'k', 'k--']
		if self.which_stimulus_rewarded % 2 == 0:
			diff_color = 'b'
		else:
			diff_color = 'b--'
		if self.which_stimulus_rewarded < 2:
			diff_alpha = [0.75, 0, 0.25]
		else:
			diff_alpha = [0.25, 0, 0.75]
		
		f = pl.figure(figsize = (12,6))
		s = f.add_subplot(2,2,1)
		s.axhline(y = 0.5, c = 'k', linewidth = 0.5)
		s.set_title('left stimulus ROI in ' + roi)
		s.set_xlabel('time [trials]')
		s.set_ylabel('percentage CW')
		s.set_xlim([0,probs_per_cond_L[0].shape[0]])
		s.set_ylim([0,1])
		[s.axvspan(i * 12, (i+1) * 12, facecolor='k', alpha=0.05, edgecolor = 'w') for i in [0,2,4]]
		for i in range(4):
			plot(probs_per_cond_L[i], colors[i], alpha = alphas[i], label = condition_labels[i], linewidth = alphas[i] * 2.0)
		# s.axis([0,rew_corr[i,:,0].shape[0],-1.0,1.0])
		s = f.add_subplot(2,2,3)
		s.axhline(y = 0.5, c = 'k', linewidth = 0.5)
		s.set_title('right stimulus ROI in ' + roi)
		[s.axvspan(i * 12, (i+1) * 12, facecolor='k', alpha=0.05, edgecolor = 'w') for i in [0,2,4]]
		for i in range(4):
			plot(probs_per_cond_R[i], colors[i], alpha = alphas[i], label = condition_labels[i], linewidth = alphas[i] * 2.0)
		s.set_xlabel('time [trials]')
		s.set_ylabel('percentage CW')
		s.set_xlim([0,probs_per_cond_L[0].shape[0]])
		s.set_ylim([0,1])
		# s.axis([0,rew_corr[i,:,0].shape[0],-1.0,1.0])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		s = f.add_subplot(2,2,2)
		s.set_title('histogram left ROI in ' + roi)
		s.axhline(y = 0.5, c = 'k', linewidth = 0.5)
		for i in range(4):
			s.axhline(y = probs_per_cond_L[i].mean(), c = colors[i][0], linewidth = 2.5, linestyle = '--')
			pl.hist(probs_per_cond_L[i], color=colors[i][0], alpha = alphas[i], normed = True, bins = 20, rwidth = 0.5, histtype = 'step', linewidth = 2.5, orientation = 'horizontal' )
		pl.text(0.5, 0.5, str(((corrects_per_cond_L + 1) / 2).mean(axis = 1)))
		s.set_ylim([0,1])
		s = f.add_subplot(2,2,4)
		s.set_title('histogram right ROI in ' + roi)
		s.axhline(y = 0.5, c = 'k', linewidth = 0.5)
		for i in range(4):
			s.axhline(y = probs_per_cond_R[i].mean(), c = colors[i][0], linewidth = 2.5, linestyle = '--')
			pl.hist(probs_per_cond_R[i], color=colors[i][0], alpha = alphas[i], normed = True, bins = 20, rwidth = 0.5, histtype = 'stepfilled', orientation = 'horizontal' )
		pl.text(0.5, 0.5, str(((corrects_per_cond_R + 1) / 2).mean(axis = 1)))
		s.set_ylim([0,1])
		
		# shell()
		
		return [((corrects_per_cond_L + 1) / 2).mean(axis = 1), ((corrects_per_cond_R + 1) / 2).mean(axis = 1)]

class VisualRewardVarSession(VisualRewardSession):
	
	def mask_stats_to_hdf(self, run_type = 'reward', postFix = ['mcf']):
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
		h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
			"""loop over runs, and try to open a group for this run's data"""
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			try:
				thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
			
			# add parameters and the like 
			eye_h5file = openFile(self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'), mode = "r")
			eyeGroup = eye_h5file.getNode(where = '/', name = 'bla', classname='Group')
			eyeGroup._f_copyChildren(thisRunGroup) 
			eye_h5file.close()
			"""
			Now, take different stat masks based on the run_type
			"""
			stat_files = {}
			
			# general info we want in all hdf files
			stat_files.update({
								# 'residuals': os.path.join(this_orientation_feat, 'stats', 'res4d.nii.gz'),
								'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf', 'psc']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf']), # os.path.join(this_orientation_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								# for these final two, we need to pre-setup the retinotopic mapping data
								'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
								'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz'),
								
								'center_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat1.nii.gz'),
								'center_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat1.nii.gz'),
								'center_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope1.nii.gz'),
								'center_pe': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'pe1.nii.gz'),
								
								'surround_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat2.nii.gz'),
								'surround_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat2.nii.gz'),
								'surround_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope2.nii.gz'),
								'surround_pe': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'pe3.nii.gz'),
								
								'center>surround_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat3.nii.gz'),
								'center>surround_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat3.nii.gz'),
								'center>surround_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope3.nii.gz'),
								
								'surround>center_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat4.nii.gz'),
								'surround>center_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat4.nii.gz'),
								'surround>center_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope4.nii.gz'),
								
			})
				
			stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
				for (i, sf) in enumerate(stat_files.keys()):
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
					these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
					h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
			
		h5file.close()
	
	def deconvolve_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'reward'):
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
		
		self.condition_labels = ['+_all_stim', '-high', '+high', '-low', '+low', '-none', '+all_nostim']
		import itertools
		self.deconvolution_labels = list(itertools.chain.from_iterable([[cl + '_stim' for cl in self.condition_labels], [cl + '_reward' for cl in self.condition_labels]]))
		print len(self.deconvolution_labels)
		# there is no stimulus in the '+all_nostim' condition
		self.deconvolution_labels.pop(self.deconvolution_labels.index('+all_nostim_stim'))
		print len(self.deconvolution_labels)
		colors = ['b','r','r','g','g','k', 'b--','r--','r--','g--','g--','k--','k--']
		alphas = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
		
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data', postFix = ['mcf']))
			this_run_events = []
			for cond in self.deconvolution_labels:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
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
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, mask_type, postFix = ['mcf'])
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		fig = pl.figure(figsize = (7, 3))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		interval = [0.0,16.0]
			
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
			pl.plot(np.linspace(interval[0], interval[1], deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], colors[i], alpha = alphas[i], label = self.deconvolution_labels[i])
			time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		s.set_title('deconvolution' + roi + ' ' + mask_type)
		# deco_per_run = []
		# for i, rd in enumerate(roi_data_per_run):
		# 	event_data_this_run = event_data_per_run[i] - i * run_duration
		# 	deco2 = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# 	deco_per_run.append(deco2.deconvolvedTimeCoursesPerEventType)
		# deco_per_run = np.array(deco_per_run)
		# mean_deco = deco_per_run.mean(axis = 0).squeeze()
		# std_deco = (1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))).squeeze()
		# for i in range(0, mean_deco.shape[0]):
		# 	s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), mean_deco[i] + std_deco[i], mean_deco[i] - std_deco[i], color = colors[i], alpha = 0.3 * alphas[i])
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				# if i == self.which_stimulus_rewarded:
				l.set_linewidth(3.5)  # the legend line width
				# else:
					# l.set_linewidth(2.0)  # the legend line width
		reward_h5file.close()
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals)] #, deco_per_run]
	
	def deconvolve(self, threshold = 2.5, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution'):
		results = []
		for roi in rois:
			results.append(self.deconvolve_roi(roi, 3.0, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = 'reward'))
			results.append(self.deconvolve_roi(roi, 3.0, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = 'reward'))
			results.append(self.deconvolve_roi(roi, 3.0, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = 'stim'))
			results.append(self.deconvolve_roi(roi, 3.0, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = 'stim'))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		# reward_h5file = self.hdf5_file('reward', mode = 'r+')
		# this_run_group_name = 'deconvolution_results'
		# try:
		# 	thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
		# 	self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		# except NoSuchNodeError:
		# 	# import actual data
		# 	self.logger.info('Adding group ' + this_run_group_name + ' to this file')
		# 	thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		# 
		# for r in results:
		# 	try:
		# 		reward_h5file.removeNode(where = thisRunGroup, name = r[0])
		# 		reward_h5file.removeNode(where = thisRunGroup, name = r[0]+'_per_run')
		# 	except NoSuchNodeError:
		# 		pass
		# 	reward_h5file.createArray(thisRunGroup, r[0], r[-2], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		# 	reward_h5file.createArray(thisRunGroup, r[0]+'_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		# reward_h5file.close()
	
	def create_feat_event_files_one_run(self, run, minimum_blink_duration = 0.01):
		"""
		creates feat analysis event files for reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
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
		reward_onset_times = (elO.timings['trial_phase_timestamps'][:,2,0] / 1000) - experiment_start_time + elO.parameter_data['reward_delay']
		
		self.condition_labels = ['+_all_stim', '-high', '+high', '-low', '+low', '-none', '+all_nostim']
		rewarded_trials = elO.parameter_data['sound'] == self.do_i_play_sound[1]
		orientation_trials = [elO.parameter_data['stim_orientation'] == ori for ori in self.orientations_in_order]
		
		# these trials were always rewarded
		all_stim_plus_trials = orientation_trials[0]
		no_stim_plus_trials = orientation_trials[-1]
		# these trials were never rewarded
		none_stim_min_trials = orientation_trials[-2]
		# high probability trials
		high_stim_plus_trials = orientation_trials[1] * rewarded_trials
		high_stim_min_trials = orientation_trials[1] * -rewarded_trials
		# low probability trials
		low_stim_plus_trials = orientation_trials[2] * rewarded_trials
		low_stim_min_trials = orientation_trials[2] * -rewarded_trials
		
		run.condition_trials = [all_stim_plus_trials, high_stim_min_trials, high_stim_plus_trials, low_stim_min_trials, low_stim_plus_trials, none_stim_min_trials, no_stim_plus_trials]
				
		for (cond, label) in zip(run.condition_trials, self.condition_labels):
			try:
				os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label, 'stim']))
			except OSError:
				pass
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label, 'stim']), np.array([stimulus_onset_times[cond], np.ones((cond.sum())), np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
			try:
				os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label, 'reward']))
			except OSError:
				pass
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label, 'reward']), np.array([reward_onset_times[cond], np.ones((cond.sum())), np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
			
		# make an all_trials_stimulus txt file
		all_stimulus_trials_sum = np.array([run.condition_trials[i] for i in range(6)], dtype = bool).sum(axis = 0, dtype = bool)
		try:
			os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials', 'stim']))
		except OSError:
			pass
		np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials', 'stim']), np.array([stimulus_onset_times[all_stimulus_trials_sum], np.ones((all_stimulus_trials_sum.sum())), np.ones((all_stimulus_trials_sum.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
		# make an all_trials_reward txt file
		all_reward_trials_sum = np.array([run.condition_trials[i] for i in [0,2,4,6]], dtype = bool).sum(axis = 0, dtype = bool)
		try:
			os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials', 'reward']))
		except OSError:
			pass
		np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all_trials', 'reward']), np.array([reward_onset_times[all_reward_trials_sum], np.ones((all_reward_trials_sum.sum())), np.ones((all_reward_trials_sum.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
	
	def feat_reward_analysis(self, version = '', postFix = ['mcf'], run_feat = False):
		"""
		Runs feat analysis for all reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			self.create_feat_event_files_one_run(r)
			
			# not running feats just yet
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
					thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_more_contrasts.fsf'
				else:
					thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward_more_contrasts.fsf'
				
				REDict = {
				'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
				'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
				'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']), 	
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


class VisualRewardVar2Session(VisualRewardVarSession):
	def deconvolve_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'reward'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = round(niiFile.rtime*100)/100.0, niiFile.timepoints
		run_duration = tr * nr_trs
		
		reward_h5file = self.hdf5_file('reward')
		
		self.deconvolution_labels = ['75%_yes', '75%_no', '75%_stim', '50%_yes', '50%_no', '50%_stim', '25%_yes', '25%_no', '25%_stim', 'blank_reward']
		decon_label_grouping = [[0,1,2],[3,4,5],[6,7,8],[-1]]
		colors = [['b--','b','b'],['g--','g','g'],['r--','r','r'], ['k--']]
		alphas = [[1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0]]
		lthns = [[2.0, 2.0, 2.0],[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0]]
		
		event_data = []
		roi_data = []
		nr_runs = 0
		blink_events = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data', postFix = ['mcf']))
			this_run_events = []
			for cond in self.deconvolution_labels:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			nr_runs += 1
			
		# join event data for stimulus events per probability bin - the stimulus responses cannot be different for rewarded or unrewarded trials
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, mask_type, postFix = ['mcf'])
		# and close the file
		reward_h5file.close()
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		time_signals = []
		interval = [0.0,16.0]
		
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.vstack(blink_events)]))
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
		for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
			time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
		
		# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
		# 	time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		time_signals = np.array(time_signals).squeeze()
		# shell()
		fig = pl.figure(figsize = (8, 16))
		s = fig.add_subplot(411)
		s.axhline(0, -10, 30, linewidth = 0.25)
		for i in range(3): # plot stimulus responses
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[decon_label_grouping[i][-1]], colors[i][-1], alpha = alphas[i][-1], linewidth = lthns[i][-1], label = self.deconvolution_labels[decon_label_grouping[i][-1]])
		s.set_title('deconvolution stimulus response' + roi + ' ' + mask_type)		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				# if i == self.which_stimulus_rewarded:
				l.set_linewidth(3.5)  # the legend line width
				# else:
					# l.set_linewidth(2.0)  # the legend line width
		for i in range(3): # plot stimulus responses
		
			s = fig.add_subplot(4,1,2+i)
			s.axhline(0, -10, 30, linewidth = 0.25)
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[decon_label_grouping[i][0]], colors[i][0], alpha = alphas[i][0], linewidth = lthns[i][0], label = self.deconvolution_labels[decon_label_grouping[i][0]])
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[decon_label_grouping[i][1]], colors[i][1], alpha = alphas[i][1], linewidth = lthns[i][1], label = self.deconvolution_labels[decon_label_grouping[i][1]])
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[decon_label_grouping[-1][0]], colors[-1][0], alpha = alphas[-1][0], linewidth = lthns[-1][0], label = self.deconvolution_labels[decon_label_grouping[-1][0]])
			s.set_title('deconvolution reward response' + roi + ' ' + mask_type)		
			s.set_xlabel('time [s]')
			s.set_ylabel('% signal change')
			s.set_xlim([interval[0]-1.5, interval[1]+1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for (i, l) in enumerate(leg.get_lines()):
					# if i == self.which_stimulus_rewarded:
					l.set_linewidth(3.5)  # the legend line width
					# else:
						# l.set_linewidth(2.0)  # the legend line width
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals)] #, deco_per_run]
	
	def deconvolve(self, threshold = 2.5, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution'):
		results = []
		for roi in rois:
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = 'reward'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = 'reward'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = 'stim'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = 'stim'))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		# reward_h5file = self.hdf5_file('reward', mode = 'r+')
		# this_run_group_name = 'deconvolution_results'
		# try:
		# 	thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
		# 	self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		# except NoSuchNodeError:
		# 	# import actual data
		# 	self.logger.info('Adding group ' + this_run_group_name + ' to this file')
		# 	thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		# 
		# for r in results:
		# 	try:
		# 		reward_h5file.removeNode(where = thisRunGroup, name = r[0])
		# 		reward_h5file.removeNode(where = thisRunGroup, name = r[0]+'_per_run')
		# 	except NoSuchNodeError:
		# 		pass
		# 	reward_h5file.createArray(thisRunGroup, r[0], r[-2], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		# 	reward_h5file.createArray(thisRunGroup, r[0]+'_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		# reward_h5file.close()
	
	def create_feat_event_files_one_run(self, run, minimum_blink_duration = 0.01):
		"""
		creates feat analysis event files for reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
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
		reward_onset_times = (elO.timings['trial_phase_timestamps'][:,2,0] / 1000) - experiment_start_time + elO.parameter_data['reward_delay']
		
		self.condition_labels = ['75%_yes', '75%_no', '75%_stim', '50%_yes', '50%_no', '50%_stim', '25%_yes', '25%_no', '25%_stim', 'blank_reward']
		rewarded_trials = elO.parameter_data['sound'] == self.do_i_play_sound[1]
		orientation_trials = [elO.parameter_data['stim_orientation'] == ori for ori in self.orientations_in_order]
		
		# these trials were never rewarded
		no_stim_plus_trials = orientation_trials[-1]
		# high probability trials
		high_stim_plus_trials = orientation_trials[0] * rewarded_trials
		high_stim_min_trials = orientation_trials[0] * -rewarded_trials
		# medium probability trials
		med_stim_plus_trials = orientation_trials[1] * rewarded_trials
		med_stim_min_trials = orientation_trials[1] * -rewarded_trials
		# low probability trials
		low_stim_plus_trials = orientation_trials[2] * rewarded_trials
		low_stim_min_trials = orientation_trials[2] * -rewarded_trials
		
		run.condition_trials = [high_stim_plus_trials, high_stim_min_trials, orientation_trials[0], med_stim_plus_trials, med_stim_min_trials, orientation_trials[1], low_stim_plus_trials, low_stim_min_trials, orientation_trials[2], no_stim_plus_trials]
				
		for (cond, label) in zip(run.condition_trials, self.condition_labels):
			try:
				os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]))
			except OSError:
				pass
			if label.split('_')[-1] == 'stim':
				times = stimulus_onset_times[cond]
			else:
				times = reward_onset_times[cond]
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]), np.array([times, np.ones((cond.sum())), np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
			
	
	def feat_reward_analysis(self, version = '', postFix = ['mcf'], run_feat = False):
		"""
		Runs feat analysis for all reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			self.create_feat_event_files_one_run(r)
			
			# not running feats just yet
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
					thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_more_contrasts.fsf'
				else:
					thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward_more_contrasts.fsf'
				
				REDict = {
				'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
				'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
				'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']), 	
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
	
	def import_stats_from_initial_session(self, example_func_to_highres_file, original_stat_folder, nr_stat_files = 4, stat_file_names = ['cope', 'tstat', 'pe', 'zstat']):
		"""
		"""
		# concatenate older session reg to newer session
		cfO = ConcatFlirtOperator(example_func_to_highres_file)
		cfO.configure(secondInputFile = os.path.join(self.stageFolder('processed/mri/reg/feat'), 'highres2example_func.mat'), 
					outputFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_older_stat_masks'], extension = '.mat' ))
		cfO.execute()
		
		for stat_name in stat_file_names:
			for i in np.arange(nr_stat_files)+1:
				# apply the transform
				flO = FlirtOperator(inputObject = os.path.join(original_stat_folder, stat_name+str(i)+'.nii.gz'), referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf']))
				flO.configureApply(self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_older_stat_masks'], extension = '.mat' ), 
										outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat'), stat_name+str(i)+'.nii.gz') )
				flO.execute()
		
	def glm_with_reward_convolution_and_nuisances_for_run(self, run, postFix = ['mcf','tf']):
		"""
		This function takes a run, opens its nifti file and runs a glm on it, that incorporates standard HRF responses for visual and trial structure events,
		and negative BOLD HRF responses for reward and non-reward events.
		"""
		# prepare the save location
		try:
			os.mkdir(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']))
		except OSError:
			pass
		self.logger.debug('running glm analysis on run %s' + self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']))
		# open an hdf5 file to save stuff?
		# self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]), 'reward.hdf5')
		# h5file = openFile(self.hdf5_filename, mode = "r+")
		# # the groups are named only after the mcf file.
		# this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = [postFix[0]]))[1]
		# try:
		# 	thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
		# 	self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = run, postFix = [postFix[0]]) + ' already in ' + self.hdf5_filename)
		# except NoSuchNodeError:
		# 	# import actual data
		# 	self.logger.error('data file ' + self.runFile(stage = 'processed/mri', run = run, postFix = [postFix[0]]) + ' does not contain ' + this_run_group_name + '. Exiting.')
		# 	return
		
		# input file opening
		niiFileName = self.runFile(stage = 'processed/mri', run = run, postFix = postFix)
		niiFile = NiftiImage(niiFileName)
		
		# shell()
		# visual and task timing responses:
		trial_times = [np.vstack([np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [pf])) for pf in ['%i%%_yes'%perc, '%i%%_no'%perc]]) for perc in [75, 50, 25]]
		stim_times = [np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [pf])) for pf in ['75%_stim', '50%_stim', '25%_stim']]
		blink_times = np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']))
		
		visual_task_names = ['75_tt', '50_tt', '25_tt', '75%_stim', '50%_stim', '25%_stim', 'blinks']
		
		norm_design = Design(nrTimePoints = niiFile.timepoints, rtime = niiFile.rtime)
		for condition_event_data in trial_times:
			# for i in range(len(condition_event_data)):
			norm_design.addRegressor(condition_event_data)
		for condition_event_data in stim_times:
			# for i in range(len(condition_event_data)):
			norm_design.addRegressor(condition_event_data)
		norm_design.addRegressor(blink_times)
		norm_design.convolveWithHRF()	# standard HRF
		
		# reward times
		reward_times = [np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [pf])) for pf in ['75%_yes', '50%_yes', '25%_yes', 'blank_reward']]
		no_reward_times = [np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [pf])) for pf in ['75%_no', '50%_no', '25%_no']]
		
		reward_names = ['75%_yes', '50%_yes', '25%_yes', 'blank_reward', '75%_no', '50%_no', '25%_no']
		
		# design matrix for reward with negative bold convolution kernel
		reward_design = Design(nrTimePoints = niiFile.timepoints, rtime = niiFile.rtime)
		for condition_event_data in reward_times:
			# for i in range(len(condition_event_data)):
			reward_design.addRegressor(condition_event_data)
		for condition_event_data in no_reward_times:
			# for i in range(len(condition_event_data)):
			reward_design.addRegressor(condition_event_data)
		# convolve with double-gamma hrf fitted from the first session's reward response.
		reward_design.convolveWithHRF(hrfType = 'double_gamma', hrfParameters = {'a1':-1.43231888, 'sh1':9.09749517, 'sc1':0.85289563, 'a2':0.14215637, 'sh2':103.37806306, 'sc2':0.11897103})
		
		# motion parameters are their own design matrix
		motion_pars = np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf']))
		
		motion_names = [str(i) + '_motion' for i in range(motion_pars.shape[1])]
		
		full_design = np.hstack((norm_design.designMatrix, reward_design.designMatrix, motion_pars))
		full_design_names = np.concatenate([visual_task_names, reward_names, motion_names])
		fd_name_string = ', '.join(full_design_names)
		
		# plot and save the design used.
		f = pl.figure(figsize = (10,8))
		s = f.add_subplot(111)
		pl.imshow(full_design)
		s.set_title(fd_name_string, fontsize = 5)
		pl.savefig(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'design.pdf'))
		np.savetxt(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'design.txt'), full_design, fmt = '%3.2f', delimiter = '\t')
		
		# GLM!
		my_glm = nipy.labs.glm.glm.glm()
		fit_data = (niiFile.data - niiFile.data.mean(axis = 0)).reshape((niiFile.timepoints,-1))	# demean first
		glm = my_glm.fit(fit_data, full_design, method="kalman", model="ar1")
		
		stat_matrix = []
		zscore_matrix = []
		beta_matrix = []
		# contrasts for each of the regressors separately
		for i in range(full_design.shape[-1]):
			this_contrast = np.zeros(full_design.shape[-1])
			this_contrast[i] = 1.0
			stat_matrix.append(my_glm.contrast(this_contrast).stat())
			zscore_matrix.append(my_glm.contrast(this_contrast).zscore())
			beta_matrix.append(my_glm.betas)
		
		# interesting contrast, the difference between reward and no reward per probability.
		diff_yes_no = np.zeros((3, full_design.shape[-1]))
		full_design_names = list(full_design_names)
		full_design_names.extend( ['75%_rew_diff', '50%_rew_diff', '25%_rew_diff'] )
		print full_design_names
		for (i, perc) in enumerate([75, 50, 25]):
			 diff_yes_no[i, full_design_names.index(str(perc)+'%_yes')] = 1
			 diff_yes_no[i, full_design_names.index(str(perc)+'%_no')] = -1
			 stat_matrix.append(my_glm.contrast(diff_yes_no[i]).stat())
			 zscore_matrix.append(my_glm.contrast(diff_yes_no[i]).zscore())
		
		# prepare stat images for saving.
		stat_matrix = np.array(stat_matrix).reshape((np.concatenate(([-1], niiFile.data.shape[1:]))))
		zscore_matrix = np.array(zscore_matrix).reshape((np.concatenate(([-1], niiFile.data.shape[1:]))))
		
		# save stat and zscore
		stat_nii = NiftiImage(stat_matrix)
		stat_nii.header = niiFile.header
		stat_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'stat.nii.gz'))
		z_nii = NiftiImage(zscore_matrix)
		z_nii.header = niiFile.header
		z_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'zscore.nii.gz'))
		
		# h5file.close()
		
	def glm_with_reward_convolution_and_nuisances(self, postFix = ['mcf','tf']):
		"""docstring for fname"""
		self.logger.debug('running glm analysis on all runs')
		for r in self.conditionDict['reward']:
			self.glm_with_reward_convolution_and_nuisances_for_run(run = self.runList[r], postFix = postFix)
		