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
from pylab import *
from nifti import *

class VisualRewardSession(Session):
	"""
	Analyses for visual reward sessions
	"""
	def create_feat_event_files_one_run(self, run, minimum_blink_duration = 0.01):
		"""
		creates feat analysis event files for reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		if run.condition == 'reward':
			# get EL Data
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
			
			# trials are separated on 'sound and 'contrast' parameters
			sound_trials, visual_trials = np.array(elO.parameter_data['sound'], dtype = 'bool'), np.array(elO.parameter_data['contrast'], dtype = 'bool')
			
			condition_labels = ['visual_sound', 'visual_silence', 'blank_silence', 'blank_sound']
			# conditions are made of boolean combinations
			visual_sound_trials = sound_trials * visual_trials
			visual_silence_trials = visual_trials * (-sound_trials)
			blank_silence_trials = -(visual_trials + sound_trials)
			blank_sound_trials = (-visual_trials) * sound_trials
			
			for (cond, label) in zip([visual_sound_trials, visual_silence_trials, blank_silence_trials, blank_sound_trials], condition_labels):
				try:
					os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]))
				except OSError:
					pass
				np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]), np.array([stimulus_onset_times[cond], np.ones((cond.sum())), np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
	
	def feat_reward_analysis(self, version = ''):
		"""
		Runs feat analysis for all reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			self.create_feat_event_files_one_run(r)
			
			try:
				self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.feat'))
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.feat'))
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.fsf'))
			except OSError:
				pass
				
			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
			thisFeatFile = '/Users/tk/Documents/research/experiments/reward/man/analysis/reward.fsf'
			REDict = {
			'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']), 
			'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']), 	
			'---BLANK_SILENCE_FILE---': self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blank_silence']), 	
			'---BLANK_SOUND_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blank_sound']), 
			'---VISUAL_SILENCE_FILE---':self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['visual_silence']), 	
			'---VISUAL_SOUND_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['visual_sound']), 
			}
			featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.fsf')
			featOp = FEATOperator(inputObject = thisFeatFile)
			if r == [self.runList[i] for i in self.conditionDict['reward']][-1]:
				featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
			else:
				featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
			self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
			# run feat
			featOp.execute()
	
	def project_stats(self, which_file = 'tstat'):
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.feat')
			visual_results_file = os.path.join(this_feat, 'stats', which_file + '1.nii.gz')
			reward_results_file = os.path.join(this_feat, 'stats', which_file + '2.nii.gz')
			
			for (label, f) in zip(['visual', 'reward'], [visual_results_file, reward_results_file]):
				vsO = VolToSurfOperator(inputObject = f)
				ofn = self.runFile(stage = 'processed/mri/', run = r, base = which_file, postFix = [label] )
				ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
				vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
				vsO.execute()
		# mappers also have 1 and 2 for stats files
		for r in [self.runList[i] for i in self.conditionDict['mapper']]:
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.feat')
			visual_results_file = os.path.join(this_feat, 'stats', which_file + '1.nii.gz')
			reward_results_file = os.path.join(this_feat, 'stats', which_file + '2.nii.gz')
			
			for (label, f) in zip(['center', 'surround'], [visual_results_file, reward_results_file]):
				vsO = VolToSurfOperator(inputObject = f)
				ofn = self.runFile(stage = 'processed/mri/', run = r, base = which_file, postFix = [label] )
				ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
				vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
				vsO.execute()
	
	def mask_stats_to_hdf(self, run_type = 'reward'):
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
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('starting table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		else:
			self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
			"""loop over runs, and try to open a group for this run's data"""
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))[1]
			try:
				thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))
			
			"""
			Now, take different stat masks based on the run_type
			"""
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.feat')
			if run_type == 'reward':
				stat_files = {
								'visual_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
								'visual_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
								'visual_cope': os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
								
								'reward_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
								'reward_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
								'reward_cope': os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
								
								'blank_silence': os.path.join(this_feat, 'stats', 'pe3.nii.gz'),
								'blank_sound': os.path.join(this_feat, 'stats', 'pe5.nii.gz'),
								'visual_silence': os.path.join(this_feat, 'stats', 'pe7.nii.gz'),
								'visual_sound': os.path.join(this_feat, 'stats', 'pe9.nii.gz'),
								
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								'input_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'psc', 'hpf']),
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
								
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								'input_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'psc', 'hpf']),
								}
				
			stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))
				
				for (i, sf) in enumerate(stat_files.keys()):
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
					these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
					h5file.createArray(thisRunGroup, sf, these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
				
					# For creating separate groups for all runs
					# try:
					# 	h5file.getNode(where = roinnames, name = sf, classname='Group')
					# 	self.logger.info('group ' + sf + ' already in ' + self.hdf5_filename + ' at ' + this_run_group_name )
					# except NoSuchNodeError:
					# 	# import actual data
					# 	self.logger.info('Adding group ' + sf + ' to ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']))
					# 	thisRunGroup = h5file.createGroup(this_run_group_name, self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']), 'Run ' + str(r.ID) +' imported from ' + this_run_group_name)
		h5file.close()
	
	def hdf5_file(self, run_type):
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('no table file ' + self.hdf5_filename + 'found for stat mask')
			return None
		else:
			self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "r", title = run_type + " file")
		return h5file
	
	
	def roi_data_from_hdf(self, h5file, run, roi_wildcard, data_type):
		"""
		drags data from an already opened hdf file into a numpy array, concatenating the data_type data across voxels in the different rois that correspond to the roi_wildcard
		"""
		this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf']))[1]
		try:
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf']) + ' opened')
			roi_names = []
			for roi_name in h5file.iterNodes(where = '/' + this_run_group_name, classname = 'Group'):
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
		all_roi_data_np = np.hstack(all_roi_data)
		print all_roi_data_np.shape
		return all_roi_data_np

	def correlate_copes_from_run(self, run, rois = ['V1', 'V2', 'V3', 'V4', 'V3A'], copes = ['visual_cope','reward_cope'], plot = True):
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
				mapper_cope = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'center_cope')
				if cope1 != None and cope2 != None:
					if plot:
						pl.plot(mapper_cope, cope2, marker = 'o', ms = 2, mec = 'w', c = 'r', mew = 0.5, alpha = 0.75) # , alpha = 0.25
						pl.plot(mapper_cope, cope1, marker = 'o', ms = 2, mec = 'w', c = 'g', mew = 0.5, alpha = 0.75) # , alpha = 0.25
						s.set_xlabel('mapper visual cope', fontsize=9)
						if rois.index(roi) == 0:
							s.set_ylabel('/'.join(copes), fontsize=9)
					srs = [stats.spearmanr(mapper_cope[0], cope1[0]), stats.spearmanr(mapper_cope[0], cope2[0])]
					corrs[rois.index(roi), 0] = srs[0][0]
					corrs[rois.index(roi), 1] = srs[1][0]
		if plot:
			pl.draw()
			pdf_file = self.runFile(stage = 'processed/mri', run = run, postFix = ['scatter'], extension = '.pdf')
			pdf_file_name = os.path.join(os.path.split(pdf_file)[0], 'figs', os.path.split(pdf_file)[1])
			pl.savefig(pdf_file_name)
		reward_h5file.close()
		mapper_h5file.close()
		return corrs
	
	def correlate_reward_copes(self, rois = ['V1', 'V2d', 'V2v', 'V3d', 'V3v', 'V4', 'V3A'], copes = ['visual_cope','reward_cope'], scatter_plots = False):
		"""
		correlate reward run cope values with one another from all reward runs separately.
		"""
		all_corrs = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			all_corrs.append(self.correlate_copes_from_run(run = r, rois = rois, copes = copes, plot = scatter_plots))
			
		cs = np.array(all_corrs)
		
		fig = pl.figure(figsize = (4, len(rois)*1.5))
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
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'spearman_rho_over_runs.pdf'))
		
		return all_corrs











