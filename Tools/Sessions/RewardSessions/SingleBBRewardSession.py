#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Session import * 
from ...Operators.ArrayOperator import *
from ...Operators.EyeOperator import *
from ...other_scripts.circularTools import *
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle
from scipy.stats import *
from SingleRewardSession import *


class SingleBBRewardSession(SingleRewardSession):
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
				flO.execute(wait = False)
	
	def align_feats(self, postFix = ['mcf']):
		"""docstring for align_feats"""
		for r in self.conditionDict['reward']:
			self.setupRegistrationForFeat(self.runFile( stage = 'processed/mri', run = self.runList[r], postFix = postFix, extension = '.feat'), wait_for_execute = False)
	
	def project_stats(self, which_file = 'zstat', postFix = ['mcf']):
		this_feat = self.stageFolder(stage = 'processed/mri/reward/feat/+.gfeat/')
		visual_results_file = os.path.join(this_feat, 'cope1.feat/stats', which_file + '1.nii.gz')
		reward_results_file = os.path.join(this_feat, 'cope2.feat/stats', which_file + '1.nii.gz')
		silent_fix_results_file = os.path.join(this_feat, 'cope3.feat/stats', which_file + '1.nii.gz')
		reward_fix_results_file = os.path.join(this_feat, 'cope4.feat/stats', which_file + '1.nii.gz')
		silent_visual_results_file = os.path.join(this_feat, 'cope5.feat/stats', which_file + '1.nii.gz')
		reward_visual_results_file = os.path.join(this_feat, 'cope6.feat/stats', which_file + '1.nii.gz')
		fix_reward_silence_results_file = os.path.join(this_feat, 'cope7.feat/stats', which_file + '1.nii.gz')
		visual_reward_silence_results_file = os.path.join(this_feat, 'cope8.feat/stats', which_file + '1.nii.gz')
		visual_silence_fix_silence_results_file = os.path.join(this_feat, 'cope9.feat/stats', which_file + '1.nii.gz')
		visual_reward_fix_reward_results_file = os.path.join(this_feat, 'cope10.feat/stats', which_file + '1.nii.gz')
			
		for (label, f) in zip(
								['visual', 'reward', 'fix_silence', 'fix_reward', 'visual_silent', 'visual_reward', 'fix_reward-silence', 'visual_reward-silence', 'visual_silence-fix_silence', 'visual_reward-fix_reward'], 
								[visual_results_file, reward_results_file, silent_fix_results_file, reward_fix_results_file, silent_visual_results_file, reward_visual_results_file, fix_reward_silence_results_file, visual_reward_silence_results_file, visual_silence_fix_silence_results_file, visual_reward_fix_reward_results_file]
								):
			vsO = VolToSurfOperator(inputObject = f)
			ofn = self.stageFolder(stage = 'processed/mri/reward/surf' )
			ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
			vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
			vsO.execute()
	
	def import_deconvolution_responses_from_all_sessions(self, session_1, session_2):
		"""
		"""
		# concatenate older sessions reg to newer session
		cfO = ConcatFlirtOperator(os.path.join(session_1.stageFolder('processed/mri/reg/feat'), 'example_func2highres.mat'))
		cfO.configure(secondInputFile = os.path.join(self.stageFolder('processed/mri/reg/feat'), 'highres2example_func.mat'), 
					outputFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_session_1'], extension = '.mat' ))
		cfO.execute()
		cfO = ConcatFlirtOperator(os.path.join(session_2.stageFolder('processed/mri/reg/feat'), 'example_func2highres.mat'))
		cfO.configure(secondInputFile = os.path.join(self.stageFolder('processed/mri/reg/feat'), 'highres2example_func.mat'), 
					outputFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_session_2'], extension = '.mat' ))
		cfO.execute()
		
		# add fsl results to deco folder so that they are also aligned across sessions.
		session_1.fsl_results_to_deco_folder()
		session_2.fsl_results_to_deco_folder()
		
		session_1_files = subprocess.Popen('ls ' + session_1.stageFolder('processed/mri/reward/deco/') + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		session_2_files = subprocess.Popen('ls ' + session_2.stageFolder('processed/mri/reward/deco/') + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		
		session_1_files = [s for s in session_1_files if os.path.split(s)[1] != 'residuals.nii.gz']
		session_2_files = [s for s in session_2_files if os.path.split(s)[1] != 'residuals.nii.gz']
		
		try:
			os.mkdir(self.stageFolder('processed/mri/reward/stats_older_sessions'))
			os.mkdir(self.stageFolder('processed/mri/reward/stats_older_sessions/exp_1'))
			os.mkdir(self.stageFolder('processed/mri/reward/stats_older_sessions/dual'))
		except OSError:
			pass
		
		for stat_file in session_1_files:
			# apply the transform
			flO = FlirtOperator(inputObject = stat_file, referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf']))
			flO.configureApply(self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_session_1'], extension = '.mat' ), 
									outputFileName = os.path.join(self.stageFolder('processed/mri/reward/stats_older_sessions/exp_1'), os.path.split(stat_file)[1]) )
			flO.execute(wait = False)
		for stat_file in session_2_files:
			# apply the transform
			flO = FlirtOperator(inputObject = stat_file, referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf']))
			flO.configureApply(self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_session_2'], extension = '.mat' ), 
									outputFileName = os.path.join(self.stageFolder('processed/mri/reward/stats_older_sessions/dual'), os.path.split(stat_file)[1]) )
			flO.execute(wait = False)
	
	def mask_other_session_stats_to_hdf(self, run_type = 'reward', postFix = ['mcf']):
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
		h5file = openFile(self.hdf5_filename, mode = "r+", title = run_type + " file")
		self.logger.info('adding to table file ' + self.hdf5_filename)
		
		allFileNames = []
		# dual file names
		allFileNames.extend(subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/reward/stats_older_sessions/dual/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1])
		# first file names
		allFileNames.extend(subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/reward/stats_older_sessions/exp_1/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1])
			
		nifti_files_dict = dict(zip(['_'.join(fn.split('/')[-2:]) for fn in allFileNames] , [NiftiImage(fn) for fn in allFileNames]))
			
		this_run_group_name = 'deconv_results'
		# h5file.removeNode(where = '/', name = this_run_group_name, recursive=1)
		try:
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('deconvolution results file already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = h5file.createGroup("/", this_run_group_name, 'deconvolution results from different sessions')
			
		# add var, dual and first stat files to different folders in there...
		for (roi, roi_name) in zip(rois, roinames):
			try:
				thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
				thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'deconvolution results for roi ' + roi_name)
				
			for (i, sf) in enumerate(nifti_files_dict.keys()):
				try:
					h5file.removeNode(where = thisRunGroup, name = sf[:-7].replace('%',''))
				except NoSuchNodeError:
					pass
				# loop over stat_files and rois
				# to mask the stat_files with the rois:
				imO = ImageMaskingOperator( inputObject = nifti_files_dict[sf], maskObject = roi, thresholds = [0.0] )
				these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
				h5file.createArray(thisRunGroup, sf[:-7].replace('%',''), these_roi_data.astype(np.float32), roi_name + ' data from ' + nifti_files_dict[sf].filename)
			
		h5file.close()
	
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
				
			# general info we want in all hdf files
			stat_files.update({
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'psc', 'tf']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
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
	
	def deconvolve_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'mean'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][-1]]))
		tr, nr_trs = round(niiFile.rtime*100)/100.0, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		reward_h5file = self.hdf5_file('reward')
		# mapper_h5file = self.hdf5_file('mapper')
		
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
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		
		fig = pl.figure(figsize = (9, 5))
		s = fig.add_subplot(211)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		if analysis_type == 'deconvolution':
			interval = [0.0,16.0]
			# nuisance version?
			nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
			nuisance_design.configure(np.array([np.hstack(blink_events)]))
			# shell()
			deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
			deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
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
		# mapper_h5file.close()
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals), np.array(deco_per_run)]
	
	def deconvolve(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution', signal_type = 'mean'):
		results = []
		for roi in rois:
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = signal_type))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'surround_center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type))
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_results' + '_' + signal_type
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_' + signal_type)
				reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_' + signal_type + '_per_run')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, r[0] + '_' + signal_type, r[-2], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.createArray(thisRunGroup, r[0] + '_' + signal_type + '_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	