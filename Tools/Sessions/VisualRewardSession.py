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
			
			# trials are separated on 'sound' and 'contrast' parameters
			sound_trials, visual_trials = np.array(elO.parameter_data['sound'], dtype = 'bool'), np.array(elO.parameter_data['contrast'], dtype = 'bool')
			
			condition_labels = ['visual_sound', 'visual_silence', 'blank_silence', 'blank_sound']
			# conditions are made of boolean combinations
			visual_sound_trials = sound_trials * visual_trials
			visual_silence_trials = visual_trials * (-sound_trials)
			blank_silence_trials = -(visual_trials + sound_trials)
			blank_sound_trials = (-visual_trials) * sound_trials
			
			# print elO.parameter_data[visual_sound_trials]['sound'], elO.parameter_data[visual_silence_trials]['sound'],elO.parameter_data[blank_silence_trials]['sound'], elO.parameter_data[blank_sound_trials]['sound']
			
			for (cond, label) in zip([visual_sound_trials, visual_silence_trials, blank_silence_trials, blank_sound_trials], condition_labels):
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
				thisFeatFile = '/Users/tk/Documents/research/experiments/reward/man/analysis/reward_more_contrasts.fsf'
				REDict = {
				'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
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
	
	def project_stats(self, which_file = 'zstat', postFix = ['mcf']):
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
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('starting table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		else:
			self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
		
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
								'blank_silence': os.path.join(this_feat, 'stats', 'pe2.nii.gz'),
								'blank_sound': os.path.join(this_feat, 'stats', 'pe3.nii.gz'),
								'visual_silence': os.path.join(this_feat, 'stats', 'pe4.nii.gz'),
								'visual_sound': os.path.join(this_feat, 'stats', 'pe5.nii.gz'),
								
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
								'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'psc', 'hpf']),
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
	
	def hdf5_file(self, run_type):
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('no table file ' + self.hdf5_filename + 'found for stat mask')
			return None
		else:
			# self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "r", title = run_type + " file")
		return h5file
	
	
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
			pdf_file_name = os.path.join(self.stageFolder(stage = 'processed/mri/figs/scatter/'), 'copes_' + str(run.ID) +  '_'.join(copes) + '.pdf')
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
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/scatter/'), 'cope_spearman_rho_over_runs.pdf'))
		
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
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/scatter/'), 'cope_spearman_rho_bar_over_runs' + '_'.join(copes) + '.pdf'))
		
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
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/scatter/'), 'cope_spearman_rho_bar_over_runs_high_conf' + '_'.join(copes) + '.pdf'))
		
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
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'input_data'))
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			nr_runs += 1
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
			
		roi_data = np.hstack(demeaned_roi_data)
		event_data = np.hstack(event_data)
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		print roi_data[mapping_mask,:].mean()
		
		timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		fig = pl.figure(figsize = (7, 3))
		s = fig.add_subplot(111)
		
		if analysis_type == 'deconvolution':
			deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = 12.0)
			for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
				pl.plot(np.linspace(0,10,deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], ['r','r','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = conds[i])
			s.set_title('deconvolution' + roi + ' ' + mask_type + ' ' + analysis_type)
		
		else:
			interval = [-1.5,10.5]
			# zero_timesignals = eraO = EventRelatedAverageOperator(inputObject = np.array([timeseries]), eventObject = event_data[0], interval = interval)
			# zero_time_signal = eraO.run(binWidth = 3.0, stepSize = 1.5)
			for i in range(event_data.shape[0]):
				eraO = EventRelatedAverageOperator(inputObject = np.array([timeseries]), eventObject = event_data[i], interval = interval)
				time_signal = eraO.run(binWidth = 3.0, stepSize = 1.5)
				pl.plot(time_signal[:,0], time_signal[:,1] - time_signal[time_signal[:,0] == 0,1], ['r','r','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = conds[i]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
			s.set_title('event-related average' + roi + ' ' + mask_type + ' ' + analysis_type)
		
		s.set_xlabel('time [s]')
		# s.set_ylabel('percent signal change')
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
			
		reward_h5file.close()
		mapper_h5file.close()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/er/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		pl.show()
	
	def deconvolve(self, threshold = 3.0, rois = ['V1', 'V2d', 'V2v', 'V3d', 'V3v', 'V4', 'V3A'], analysis_type = 'deconvolution'):
		for roi in rois:
			self.deconvolve_roi(roi, threshold, mask_type = 'center_surround_Z', analysis_type = analysis_type, mask_direction = 'pos')
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
			self.deconvolve_roi(roi, threshold, mask_type = 'surround_center_Z', analysis_type = analysis_type, mask_direction = 'pos')
#			self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')

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
		
		input_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, 'input_data')
		
		roi_data = np.zeros((len(stats_types), len(self.conditionDict['reward']), int(mapping_mask.sum())))
		for i, stat in enumerate(stats_types):
			for j, r in enumerate([self.runList[rew] for rew in self.conditionDict['reward']]):
				rd = self.roi_data_from_hdf(reward_h5file, r, roi, stat).ravel()
				sd = self.roi_data_from_hdf(reward_h5file, r, roi, 'input_data').mean(axis = 1).ravel()
				roi_data[i,j] = rd[mapping_mask] / sd[mapping_mask]
		
		reward_h5file.close()
		mapper_h5file.close()
		
		return roi_data
		
	def mean_stats(self, rois = ['V1', 'V2', 'V3', 'V3A', 'V4'], threshold = 2.3, mask_type = 'center_Z', stats_types = ['blank_silence', 'blank_sound', 'visual_silence', 'visual_sound'], mask_direction = 'pos' ):
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
		
		colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
		
		fig = pl.figure(figsize = (12, 4))
		pl.subplots_adjust(left = 0.05, right = 0.97)
		s = fig.add_subplot(111)
		width = 1.0 / (diff_res.shape[1] + 2)
		pl.plot([-1, len(rois) + 1.0], [0,0], 'k', linewidth = 0.5)
		rects = []
		for i in range(diff_res.shape[2]):
			rects.append(pl.bar(np.arange(diff_res.shape[0])+(i*+width), height = diff_res[:,0,i], width = width, yerr = diff_res[:,1,i], color=colors[i], alpha = 0.7, edgecolor = (0.5, 0.5, 0.5), linewidth = 0.0, ecolor = (0.5, 0.5, 0.5), capsize = 0))
		pl.ylabel('beta [a.u.]')
		pl.xticks(np.arange(len(rois))+width, rois )
		s.set_xlim(-0.5, diff_res.shape[0]+.5)
		leg = pl.legend( tuple([r[0] for r in rects]), tuple([st.replace('sound', 'reward').replace('blank','fix') for st in stats_types[1:]]), fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize(9)    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(2.5)  # the legend line width
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'betas.pdf'))
		
		return res

	def correlate_data_from_run(self, run, rois = ['V1', 'V2', 'V3', 'V4', 'V3A'], data_pairs = [[['mapper', 'center_pe'], ['reward', 'visual_cope']], [['mapper', 'center_pe'], ['reward', 'reward_cope']]], plot = True):
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
						cope1 = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, data_pairs[i][0][1])
					elif data_pairs[i][0][0] == 'reward':
						cope1 = self.roi_data_from_hdf(reward_h5file, run, roi, data_pairs[i][0][1])
					
					if data_pairs[i][1][0] == 'mapper':
						cope2 = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, data_pairs[i][1][1])
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
			pdf_file_name = os.path.join(self.stageFolder(stage = 'processed/mri/figs/scatter/'), 'data_scatter_' + str(run.ID) + '.pdf')
			pl.savefig(pdf_file_name)
		reward_h5file.close()
		mapper_h5file.close()
		return corrs
	
	def correlate_data(self, rois = ['V1', 'V2d', 'V2v', 'V3d', 'V3v', 'V4', 'V3A'], data_pairs = [[['mapper', 'center_pe'], ['reward', 'visual_cope']], [['mapper', 'center_pe'], ['reward', 'reward_cope']]], scatter_plots = False):
		"""
		correlate reward run cope values with one another from all reward runs separately.
		"""
		all_corrs = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			all_corrs.append(self.correlate_data_from_run(run = r, rois = rois, data_pairs = data_pairs, plot = scatter_plots))
			
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
			if rois.index(roi) == 3:
				s.set_ylabel('spearman $R$', fontsize=9)
			s.axis([-0.25,5.35,-1,1])
			# designate bad runs:
			s.axvspan(3.75, 4.25, facecolor='y', alpha=0.25, edgecolor = 'w')
			s.axvspan(1.75, 2.25, facecolor='y', alpha=0.25, edgecolor = 'w')
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/scatter/'), 'data_spearman_rho_over_runs.pdf'))
		
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
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/scatter/'), 'data_spearman_rho_bar_over_runs.pdf'))
		
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
		
		input_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, 'input_data')
		
		roi_data = np.zeros((len(stats_types), len(self.conditionDict['reward']), int(mapping_mask.sum())))
		for i, stat in enumerate(stats_types):
			for j, r in enumerate([self.runList[rew] for rew in self.conditionDict['reward']]):
				rd = self.roi_data_from_hdf(reward_h5file, r, roi, stat).ravel()
				sd = self.roi_data_from_hdf(reward_h5file, r, roi, 'input_data').mean(axis = 1).ravel()
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
		
	
	
	



