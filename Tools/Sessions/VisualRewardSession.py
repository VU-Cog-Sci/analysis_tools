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
				condition_labels[which_stimulus_rewarded] += '_rewarded'
				
				# reorder the conditions
				all_stimulus_trials.pop(which_stimulus_rewarded)
				all_stimulus_trials.append(stim_trials_rewarded)
				# reorder the condition labels
				reward_label = condition_labels.pop(which_stimulus_rewarded)
				condition_labels.append(reward_label)
				
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
					if os.uname()[1].split('.')[-2] == 'sara':
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
					if os.uname()[1].split('.')[-2] == 'sara':
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
					try:
						self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.fsf'))
					except OSError:
						pass
				
					# now segment according to reward condition.
			
			
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
						if os.uname()[1].split('.')[-2] == 'sara':
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
						if os.uname()[1].split('.')[-2] == 'sara':
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
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
		
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
	
	
	def pupil_responses_one_run(self, run, frequency, sample_rate = 2000):
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
				if (ze[0] - new_ze[-1][-1])/sample_rate < 0.1:
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
			trial_phase_timestamps = [trial_times['trial_phase_timestamps'][:,1][cond,0] for cond in [blank_silence_trials, blank_sound_trials, visual_silence_trials, visual_sound_trials]]			
			tr_data = np.array([[pupil_zscore[(gaze_timestamps>tpt) * (gaze_timestamps<(tpt+500 + (10 * sample_rate)))][:(10 * sample_rate)] for tpt in trphts] for trphts in trial_phase_timestamps])
			
			h5f.close()
			# shell()
			pl.figure(figsize = (10,4))
			for i in range(tr_data.shape[0]):
				pl.plot(np.array(tr_data[0]).T, ['b','b','g','g'][i], linewidth = 1.75)
			pl.savefig(self.runFile(stage = 'processed/eye', run = run, extension = '.pdf', postFix = ['pupil']))
			return tr_data
			
	def pupil_responses(self, sample_rate = 2000):
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
		for i in range(4):
			all_data_this_condition = np.vstack([all_pupil_responses[j][i] for j in range(len(all_pupil_responses))])
			zero_points = all_data_this_condition[:,[0,1]].mean(axis = 1)			
			all_data_this_condition = np.array([a - z for (a, z) in zip (all_data_this_condition, zero_points)])
			all_data_conditions.append(all_data_this_condition.mean(axis = 0))
			rnge = np.linspace(0,all_data_conditions[-1].shape[0]/sample_rate, all_data_conditions[-1].shape[0])
			sems = (all_data_this_condition.std(axis = 0)/np.sqrt(all_data_this_condition.shape[0]))
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
		pl.savefig(self.stageFolder(stage = 'processed/eye/figs/') + 'pupil_evolution_per_condition.pdf')
		# shell()
	
	
	def roi_data_from_hdf(self, h5file, run, roi_wildcard, data_type, postFix = ['mcf']):
		"""
		drags data from an already opened hdf file into a numpy array, concatenating the data_type data across voxels in the different rois that correspond to the roi_wildcard
		"""
		this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]
		try:
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			roi_names = []
			for roi_name in h5file.iterNodes(where = '/' + this_run_group_name, classname = 'Group'):
				if len(roi_name._v_name.split('.')) > 1:
					hemi, area = roi_name._v_name.split('.')
					if roi_wildcard == area:
						roi_names.append(roi_name._v_name)
			if len(roi_names) == 0:
				self.logger.info('No rois corresponding to ' + roi_wildcard + ' in group ' + this_run_group_name)
				return None
		except NoSuchNodeError:
			# import actual data
			self.logger.info('No group ' + this_run_group_name + ' in this file')
			return None
		
		all_roi_data = []
		for roi_name in roi_names:
			thisRoi = h5file.getNode(where = '/' + this_run_group_name, name = roi_name, classname='Group')
			all_roi_data.append( eval('thisRoi.' + data_type + '.read()') )
		all_roi_data_np = np.hstack(all_roi_data).T
		return all_roi_data_np

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
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data'))
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
			
			deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
			for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
				pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
				time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
			s.set_title('deconvolution' + roi + ' ' + mask_type)
			deco_per_run = []
			for i, rd in enumerate(roi_data_per_run):
				event_data_this_run = event_data_per_run[i] - i * run_duration
				deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
				deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
			deco_per_run = np.array(deco_per_run)
			mean_deco = deco_per_run.mean(axis = 0)
			std_deco = 1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))
			for i in range(0, mean_deco.shape[0]):
				# pl.plot(np.linspace(interval[0],interval[1],mean_deco.shape[1]), mean_deco[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
				s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), time_signals[i] + std_deco[i], time_signals[i] - std_deco[i], color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])
		
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
				pl.plot(np.linspace(0,interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), ts_diff, ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
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
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals)]
	
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
			self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			reward_h5file.createArray(thisRunGroup, r[0], r[-1], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def run_glm_on_hdf5(self, data_type = 'hpf_data', analysis_type = 'per_trial', post_fix_for_text_file = ['all_trials']):
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		super(VisualRewardSession, self).run_glm_on_hdf5(run_list = [self.runList[i] for i in self.conditionDict['reward']], hdf5_file = reward_h5file, data_type = 'hpf_data', analysis_type = 'per_trial', post_fix_for_text_file = ['all_trials'])
		reward_h5file.close()
		

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
	
