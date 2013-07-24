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
from ...circularTools import *
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle
from scipy.stats import *
from SingleRewardSession import *
from RewardSession import * 


class DualRewardSession(SingleRewardSession):
	
	def __init__(self, ID, date, project, subject, session_label = 'second', parallelize = True, loggingLevel = logging.DEBUG):
		super(DualRewardSession, self).__init__(ID, date, project, subject, session_label = session_label, parallelize = parallelize, loggingLevel = loggingLevel)

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
			
			if os.path.isfile(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco/'), 'residuals.nii.gz')):
				stat_files.update({
									'deco_residuals': os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco/'), 'residuals.nii.gz'),
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
		
	
	
	def deconvolve_roi(self, roi, threshold = 2.5, mask_type = 'left_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'mean', data_type = 'psc_hpf_data'):
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
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, data_type, postFix = ['mcf','tf']))
			if 'residuals' in data_type:
				roi_data[-1] = roi_data[-1] ** 2
			
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
		
		# timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		if signal_type in ['std', 'var']:
			timeseries = (timeseries - timeseries.mean() ) / timeseries.std()
		
		fig = pl.figure(figsize = (7, 3))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		interval = [0.0,12.0]
			
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
		time_signals = np.array(time_signals).squeeze()
		deco_per_run = np.array(deco_per_run).squeeze()
		mean_deco = deco_per_run.mean(axis = 0).squeeze()
		std_deco = (1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))).squeeze()
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
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + data_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals), deco_per_run]
	
	def deconvolve(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution', signal_type = 'mean', data_type = 'hpf_psc_data' ):
		results = []
		for roi in rois:
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'left_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type, data_type = data_type))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'right_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type, data_type = data_type))
			results.append(self.deconvolve_roi(roi, 4.0, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos'))
			# results.append(self.deconvolve_roi(roi, 4.0, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg'))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_results' + '_' + signal_type + '_' + data_type
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_' + signal_type + '_' + data_type)
				reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_' + signal_type + '_per_run_' + data_type)
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, r[0] + '_' + signal_type + '_' + data_type, r[-2], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.createArray(thisRunGroup, r[0] + '_' + signal_type + '_per_run_' + data_type, r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def deconvolve_conditions_roi(self, roi, threshold = 2.5, analysis_type = 'deconvolution', signal_type = 'mean', data_type = 'psc_hpf_data'):
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
		
		conds = ['rewarded', 'rewarded_side', 'rewarded_orientation', 'null', 'blank_rewarded', 'blank_silence']
		colors = ['r','r','g','g','k','k']
		alphas = [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
		
		event_data = []
		roi_data = []
		blink_events = []
		
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			self.rewarded_stimulus_run(r)
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, data_type, postFix = ['mcf','tf']))
			
			if 'residuals' in data_type:
				print 'deconvolving residuals'
				roi_data[-1] = np.abs(roi_data[-1])
			
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			# the times
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times', postFix = ['mcf','tf'])
			experiment_start_time = (trial_times['trial_phase_timestamps'][0,0,0])
			stim_onsets = (trial_times['trial_phase_timestamps'][:,1,0] - experiment_start_time ) / 1000.0
			
			this_run_events = []
			for cond in r.ordered_conditions:
				this_run_events.append(stim_onsets[cond])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			nr_runs += 1
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(roi_data)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data, r is inherited from earlier for loop
		if r.rewarded_side == 0:
			rewarded_mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'left_Z', postFix = ['mcf','tf'])
			unrewarded_mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'right_Z', postFix = ['mcf','tf'])
		else:
			rewarded_mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'right_Z', postFix = ['mcf','tf'])
			unrewarded_mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'left_Z', postFix = ['mcf','tf'])
		
		# thresholding of mapping data stat values
		rewarded_mapping_mask = rewarded_mapping_data[:,0] > threshold
		unrewarded_mapping_mask = unrewarded_mapping_data[:,0] > threshold
		
		# rewarded_timeseries = roi_data[rewarded_mapping_mask,:].mean(axis = 0)
		# unrewarded_timeseries = roi_data[unrewarded_mapping_mask,:].mean(axis = 0)
		rewarded_timeseries = eval('roi_data[rewarded_mapping_mask,:].' + signal_type + '(axis = 0)')
		unrewarded_timeseries = eval('roi_data[unrewarded_mapping_mask,:].' + signal_type + '(axis = 0)')
		if signal_type in ['std', 'var']:
			rewarded_timeseries = (rewarded_timeseries - rewarded_timeseries.mean() ) / rewarded_timeseries.std()
			unrewarded_timeseries = (unrewarded_timeseries - unrewarded_timeseries.mean() ) / unrewarded_timeseries.std()
		
		fig = pl.figure(figsize = (7, 3))
		
		all_time_signals = []
		all_deco_per_run = []
		for j, timeseries in enumerate([rewarded_timeseries, unrewarded_timeseries]):
		
			s = fig.add_subplot(2,1,j+1)
			s.axhline(0, -10, 30, linewidth = 0.25)
			
			interval = [0.0,12.0]
			time_signals = []
			
			nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
			nuisance_design.configure(np.array([np.vstack(blink_events)]))
			deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
			deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
			
			# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
			for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
				pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), deco.deconvolvedTimeCoursesPerEventTypeNuisance[i], colors[i], alpha = alphas[i], label = conds[i])
				time_signals.append(np.squeeze(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i]))
			s.set_title('deconvolution' + roi)
			
			# deco_per_run = []
			# for i, rd in enumerate(roi_data_per_run):
			# 	nuisance_design = Design(rd.shape[1] * 2, tr/2.0 )
			# 	nuisance_design.configure(np.array([blink_events[j]]))
			# 	
			# 	event_data_this_run = event_data_per_run[i] - i * run_duration
			# 	deco = DeconvolutionOperator(inputObject = rd[[rewarded_mapping_mask,unrewarded_mapping_mask][j],:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
			# 	deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
			# 	
			# 	deco_per_run.append(np.squeeze(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i]))
			time_signals = np.array(time_signals).squeeze()
			# deco_per_run = np.array(deco_per_run).squeeze()
			# mean_deco = deco_per_run.mean(axis = 0).squeeze()
			# std_deco = (1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))).squeeze()
			
			# for i in range(0, mean_deco.shape[0]):
			# 	s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), time_signals[i] + std_deco[i], time_signals[i] - std_deco[i], color = colors[i], alpha = 0.3 * alphas[i])
			
			# all_deco_per_run.append(deco_per_run)
			all_time_signals.append(time_signals)
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
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + analysis_type + '.pdf'))
		
		return [roi + '_conditions_' + analysis_type, event_data, np.concatenate([rewarded_timeseries, unrewarded_timeseries]), np.array(all_time_signals)]
	
	def deconvolve_conditions(self, threshold = 3.5, rois = ['V1', 'V2', 'V3', 'V3AB'], analysis_type = 'deconvolution', signal_type = 'mean', data_type = 'psc_hpf_data'):
		results = []
		for roi in rois:
			results.append(self.deconvolve_conditions_roi(roi, threshold, analysis_type = analysis_type, signal_type = signal_type, data_type = data_type))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_conditions_results' + '_' + signal_type + '_' + data_type
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_' + signal_type + '_' + data_type)
				# reward_h5file.removeNode(where = thisRunGroup, name = r[0]+'_per_run')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, r[0] + '_' + signal_type + '_' + data_type, r[-1], 'deconvolution timecourses results for ' + r[0] + '_' + signal_type + '_' + data_type + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			# reward_h5file.createArray(thisRunGroup, r[0]+'_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	
	def whole_brain_deconvolution(self, deco = True, average_intervals = [[3.5,12],[2,7]], to_surf = True, postFix = ['mcf', 'tf', 'psc']):
		"""
		whole_brain_deconvolution takes all nii files from the reward condition and deconvolves the separate event types
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		nii_file_shape = list(niiFile.data.shape)
		
		nr_reward_runs = len(self.conditionDict['reward'])
		
		# conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		# cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		time_signals = []
		interval = [0.0,16.0]
		
		conds = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW', 'blank_silence', 'blank_rewarded']
		colors = ['r','r','g','g','k','k']
		alphas = [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
		
		if deco:
		
			event_data = []
			nii_data = np.zeros([nr_reward_runs] + nii_file_shape)
			nr_runs = 0
			blink_events = []
			for (j, r) in enumerate([self.runList[i] for i in self.conditionDict['reward']]):
				nii_data[j] = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).data
				this_run_events = []
				for cond in conds:
					this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
				this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
				this_blink_events[:,0] += nr_runs * run_duration
				blink_events.append(this_blink_events)
				this_run_events = np.array(this_run_events) + nr_runs * run_duration
				event_data.append(this_run_events)
				nr_runs += 1
				
			nii_data = nii_data.reshape((nr_reward_runs * nii_file_shape[0], -1))
			event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
			# shell()
			# deco = DeconvolutionOperator(inputObject = nii_data, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
			# nuisance_design = Design(nii_data.shape[0] * 2, tr/2.0 )
			# nuisance_design.configure(np.array([np.vstack(blink_events)]))
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
		for (i, c) in enumerate(conds):
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
					vsO.execute()
				
					for hemi in ['lh','rh']:
						ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
						ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
						ssO.execute(wait = False)
		
		# now create the necessary difference images:
		# only possible if deco has already been run...
		# for i in [0,3,6]:
		# 	for (j, which_times) in enumerate(['reward', 'visual']):
		# 		ipfs = [NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_mean_' + self.deconvolution_labels[i] + '_' + which_times + '.nii.gz')), NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_mean_' + self.deconvolution_labels[i+1] + '_' + which_times + '.nii.gz'))]
		# 		diff_d = ipfs[0].data - ipfs[1].data
		# 	
		# 		ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward'), self.deconvolution_labels[i].split('_')[0] + '_reward_diff' + '_' + which_times + '.nii.gz')
		# 		outputFile = NiftiImage(diff_d)
		# 		outputFile.header = ipfs[0].header
		# 		outputFile.save(ofn)
		# 	
		# 	
		# 		if to_surf:
		# 			vsO = VolToSurfOperator(inputObject = ofn)
		# 			sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
		# 			vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
		# 			vsO.execute()
		# 		
		# 			for hemi in ['lh','rh']:
		# 				ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
		# 				ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
		# 				ssO.execute()
	
	
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
		
		# reorder conditions based on reward, location and orientation
		run.rewarded_orientation = which_stimulus_rewarded % 2
		run.rewarded_side = floor(float(which_stimulus_rewarded)/2.0)
		
		run.order = np.arange(4)
		if run.rewarded_orientation == 1:
			run.order = run.order[[1,0,3,2]]
		if run.rewarded_side == 1:
			run.order = run.order[[2,3,0,1]]
		
		# shell()
		run.ordered_conditions = [run.all_stimulus_trials[i] for i in run.order]
		run.ordered_stimulus_conditions = run.ordered_conditions[:]
		run.ordered_conditions.extend([blank_trials_rewarded, blank_trials_silence])
		
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
			this_run_raw_roi_data = self.roi_data_from_hdf(reward_h5file, r, roi, classification_data_type, postFix = ['mcf','tf'])
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters', postFix = ['mcf','tf'])
			
			left_none_right = np.sign(parameter_data['x_position'])
			CW_none_CCW = np.sign(parameter_data['orientation'])
			left_CW_trials = (left_none_right == -1) * (CW_none_CCW == -1)
			left_CCW_trials = (left_none_right == -1) * (CW_none_CCW == 1)
			right_CW_trials = (left_none_right == 1) * (CW_none_CCW == -1)
			right_CCW_trials = (left_none_right == 1) * (CW_none_CCW == 1)
			# re-order trials based on parameters
			roi_data.append(np.array([this_run_raw_roi_data[:,cond] for cond in [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials]]))
			
			# self.rewarded_stimulus_run(r, postFix = postFix)
			nr_runs += 1
		
		roi_data = np.vstack(np.array(roi_data).transpose(0,3,1,2)).transpose(1,0,2)
		# shell()
		# mapping data
		mapping_data_L = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'left_' + data_type_mask, postFix = ['mcf','tf'])
		mapping_data_R = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'right_' + data_type_mask, postFix = ['mcf','tf'])
		# orientation_mapper_data = np.array([self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi_wildcard = roi, data_type = dt, postFix = ['mcf','tf']) for dt in conditions_data_types]).squeeze()
		
		mapper_run_raw_roi_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, classification_data_type, postFix = ['mcf','tf'])
		parameter_data = self.run_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], 'trial_parameters', postFix = ['mcf','tf'])
		left_none_right = np.sign(parameter_data['x_position'])
		CW_none_CCW = np.sign(parameter_data['orientation'])
		left_CW_trials = (left_none_right == -1) * (CW_none_CCW == -1)
		left_CCW_trials = (left_none_right == -1) * (CW_none_CCW == 1)
		right_CW_trials = (left_none_right == 1) * (CW_none_CCW == -1)
		right_CCW_trials = (left_none_right == 1) * (CW_none_CCW == 1)
		orientation_mapper_data = np.array([this_run_raw_roi_data[:,cond].mean(axis = 1) for cond in [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials]])
		
		reward_h5file.close()
		mapper_h5file.close()
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask_L = mapping_data_L[:,0] > mask_threshold
			mapping_mask_R = mapping_data_R[:,0] > mask_threshold
		else:
			mapping_mask_L = mapping_data_L[:,0] < mask_threshold
			mapping_mask_R = mapping_data_R[:,0] < mask_threshold
		
		left_correlations_ttot = np.array([[self.correlate_patterns(t1,t2) for (t1, t2) in zip(roi_data[i,:-1,mapping_mask_L].T, roi_data[i,1:,mapping_mask_L].T)] for i in range(4)])
		right_correlations_ttot = np.array([[self.correlate_patterns(t1,t2) for (t1, t2) in zip(roi_data[i,:-1,mapping_mask_R].T, roi_data[i,1:,mapping_mask_R].T)] for i in range(4)])
		
		left_correlations_mapper = np.array([[self.correlate_patterns(orientation_mapper_data[i,mapping_mask_L].squeeze(),t.squeeze()) for t in roi_data[i,:,mapping_mask_L].T] for i in range(4)])
		right_correlations_mapper = np.array([[self.correlate_patterns(orientation_mapper_data[i,mapping_mask_R].squeeze(),t.squeeze()) for t in roi_data[i,:,mapping_mask_R].T] for i in range(4)])
		
		return [[left_correlations_mapper, right_correlations_mapper],[left_correlations_ttot, right_correlations_ttot]]
		
	def correlate_patterns_over_time(self, rois = ['V1', 'V2', 'V3', 'V3AB'], classification_data_type = 'per_trial_hpf_data_zscore', data_type_mask = 'Z', mask_threshold = 2.3, mask_direction = 'pos', postFix = ['mcf','tf']):
		
		pattern_corr_mapper_results = []
		pattern_corr_ttot_results = []
		for roi in rois:
			res = self.correlate_patterns_over_time_for_roi(roi = roi, classification_data_type = classification_data_type, data_type_mask = data_type_mask, mask_threshold = mask_threshold, mask_direction = mask_direction, postFix = postFix)
			pattern_corr_mapper_results.append(res[0])
			pattern_corr_ttot_results.append(res[1])
		
		# save to hdf
		
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'pattern_correlations_over_time'
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'pattern time analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for i in range(len(rois)):
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = rois[i]+'_mapper')
				reward_h5file.removeNode(where = thisRunGroup, name = rois[i]+'_ttot')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, rois[i]+'_mapper', pattern_corr_mapper_results[i], 'pattern evolution mapper timecourses results conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.createArray(thisRunGroup, rois[i]+'_ttot', pattern_corr_ttot_results[i], 'per-run deconvolution ttot timecourses results conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
		
		
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

	def pupil_responses_one_run(self, run, frequency, sample_rate = 2000, postFix = ['mcf','tf'], analysis_duration = 10):
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
		parameter_data = trial_parameters
		
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
		trial_phase_timestamps = [trial_times['trial_phase_timestamps'][:,1][cond,0] for cond in [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials, blank_trials_silence, blank_trials_rewarded]]			
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
			
	def pupil_responses(self, sample_rate = 1000, save_all = True):
		"""docstring for pupil_responses"""
		cond_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW', 'blank_silence', 'blank_rewarded']
		colors = ['r','r','g','g','k','k']
		alphas = [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
		
		all_pupil_responses = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			all_pupil_responses.append(self.pupil_responses_one_run(run = r, frequency = 4, sample_rate = sample_rate))
		# self.all_pupil_responses_hs = np.array(all_pupil_responses)
		fig = pl.figure(figsize = (9,4))
		s = fig.add_subplot(1,1,1)
		s.set_ylabel('Z-scored Pupil Size')
		s.set_title('Pupil Size after stimulus onset')
		all_data_conditions = []
		all_data = []
		for i in range(len(cond_labels)):
			all_data_this_condition = np.vstack([all_pupil_responses[j][i] for j in range(len(all_pupil_responses))])
			zero_points = all_data_this_condition[:,[0,1]].mean(axis = 1)
			all_data_this_condition = np.array([a - z for (a, z) in zip (all_data_this_condition, zero_points)])
			all_data_conditions.append(all_data_this_condition.mean(axis = 0))
			all_data.append(all_data_this_condition)
			rnge = np.linspace(0,all_data_conditions[-1].shape[0]/sample_rate, all_data_conditions[-1].shape[0])
			sems = 1.96 * (all_data_this_condition.std(axis = 0)/np.sqrt(all_data_this_condition.shape[0]))
			pl.plot(rnge, all_data_conditions[-1], colors[i], alpha = alphas[i], label = cond_labels[i])
			pl.fill_between(rnge, all_data_conditions[-1]+sems, all_data_conditions[-1]-sems, color = colors[i], alpha = 0.3 * alphas[i])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.75)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		# s = fig.add_subplot(2,1,2)
		# for i in range(0, 4, 2):
		# 	diffs = -(all_data_conditions[i] - all_data_conditions[i+1])
		# 	pl.plot(np.linspace(0,diffs.shape[0]/sample_rate, diffs.shape[0]), diffs, ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2])
		# 	s.set_title('reward signal')
		# leg = s.legend(fancybox = True)
		# leg.get_frame().set_alpha(0.75)
		# if leg:
		# 	for t in leg.get_texts():
		# 	    t.set_fontsize('small')    # the legend text fontsize
		# 	for l in leg.get_lines():
		# 	    l.set_linewidth(3.5)  # the legend line width
		# s.set_xlabel('time [s]')
		# s.set_ylabel('$\Delta$ Z-scored Pupil Size')
		# pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'pupil_evolution_per_condition.pdf'))
		# pl.savefig(os.path.join(self.stageFolder(stage = 'processed/eye/figs/'), 'pupil_evolution_per_condition.pdf'))
		# 
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
	
	def variance_from_whole_brain_residuals(self, time_range_BOLD = [3.0, 9.0], var = True, to_surf = True):
		conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		cond_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW', 'blank_silence', 'blank_rewarded']
			
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
				for cond in cond_labels:
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
		
			opf = NiftiImage(np.array(all_vars))
			opf.header = niiFile.header
			opf.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'variance_residuals.nii.gz'))
		
		if to_surf:
			# vol to surf?
			# for (label, f) in zip(['left', 'right'], [left_file, right_file]):
			ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'variance_residuals.nii.gz')
			vsO = VolToSurfOperator(inputObject = ofn)
			sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
			vsO.configure(frames = dict(zip(['_'+c for c in cond_labels], range(4))), hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
			vsO.execute()
			
			for cond in ['_'+c for c in cond_labels]:
				for hemi in ['lh','rh']:
					ssO = SurfToSurfOperator(vsO.outputFileName + cond + '-' + hemi + '.mgh')
					ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
					ssO.execute()
					
	def fsl_results_to_deco_folder(self, run_type = 'reward', postFix = ['mcf', 'tf','orientation']):
		for j,r in enumerate([self.runList[i] for i in self.conditionDict[run_type]]):
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
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
			for sf in stat_files.keys():
				os.system( 'cp ' + stat_files[sf] + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reward/deco'), 'FSL_' + sf + '_' + str(j) + '.nii.gz'))
				
	
	def cross_correlate_pupil_and_BOLD_for_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', sample_rate = 2000, time_range_BOLD = [5.0, 10.0], time_range_pupil = [0.5, 2.0], stepsize = 0.25, area = '', color = 1.0):
		"""docstring for correlate_pupil_and_BOLD"""
		
		# take data 
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		conds = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW', 'blank_silence', 'blank_rewarded']
		cond_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW', 'blank_silence', 'blank_rewarded']
		my_colors = ['r','r','g','g','k','k']
		alphas = [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
		
		reward_h5file = self.hdf5_file('reward')
		mapper_h5file = self.hdf5_file('mapper')
		
		event_data = []
		roi_data = []
		pupil_data = []
		tr_timings = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data', postFix = ['mcf','tf']))
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			tr_timings.append(np.arange(0, run_duration, tr) + nr_runs * run_duration)
			# take pupil data
			try:
				thisRunGroup = reward_h5file.getNode(where = '/', name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','tf']))[1], classname='Group')
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
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type, postFix = ['mcf','tf'])
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
			
			pl.plot(pupil_trial_data, bold_trial_data, 'o'+my_colors[i], alpha = alphas[i] * 0.2, mec = 'w', mew = 1, ms = 6)
			
			# linear regression for regression lines
			slope, intercept, r_value, p_value, slope_std_error = stats.linregress(pupil_trial_data, bold_trial_data)
			predict_y = intercept + slope * pupil_trial_data
			pl.plot(pupil_trial_data, predict_y, '--'+my_colors[i], alpha = alphas[i], mec = 'w', mew = 1, ms = 6, label = cond_labels[i])
			
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
	
	def calculate_event_history(self, times, parameters):
		"""
		calculate for each trial, the intertrial interval preceding that trial, based on:
		the raw last trial
		the last reward signal in the line
		return the fixation reward trial onsets, with their itis depending on iti, fixation reward and general reward itis.
		"""
		
		left_none_right = np.sign(parameters['x_position'])
		CW_none_CCW = np.sign(parameters['orientation'])
			
		condition_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW']
		left_CW_trials = (left_none_right == -1) * (CW_none_CCW == -1)
		left_CCW_trials = (left_none_right == -1) * (CW_none_CCW == 1)
		right_CW_trials = (left_none_right == 1) * (CW_none_CCW == -1)
		right_CCW_trials = (left_none_right == 1) * (CW_none_CCW == 1)
				
		all_stimulus_trials = [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials]
		all_reward_trials = np.array(parameters['sound'], dtype = bool)
				
		which_trials_rewarded = np.array([(trials * all_reward_trials).sum() > 0 for trials in [left_CW_trials, left_CCW_trials, right_CW_trials, right_CCW_trials]])
		which_stimulus_rewarded = np.arange(4)[which_trials_rewarded]
		stim_trials_rewarded = np.squeeze(np.array(all_stimulus_trials)[which_trials_rewarded])
				
		blank_trials_rewarded = all_reward_trials - stim_trials_rewarded
		blank_trials_silence = -np.array(np.abs(left_none_right) + blank_trials_rewarded, dtype = bool)
		
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
			if i in np.arange(stim_onsets.shape[0])[all_reward_trials]:
				last_reward_time = stim_onsets[i]
			if i in np.arange(stim_onsets.shape[0])[blank_trials_rewarded]:
				last_fix_reward_time = stim_onsets[i]
			if i in np.arange(stim_onsets.shape[0])[stim_trials_rewarded]:
				last_visual_reward_time = stim_onsets[i]
			
		relative_delays = (delays.T - stim_onsets).T
		what_trials_are_sensible = delays.min(axis = 1)!=0.0
		
		raw_itis_of_fix_reward_trials = relative_delays[blank_trials_rewarded * what_trials_are_sensible, 3]
		onsets_fix_reward_trials = stim_onsets[blank_trials_rewarded * what_trials_are_sensible]
		all_reward_itis_of_fix_reward_trials = relative_delays[blank_trials_rewarded * what_trials_are_sensible, 0]
		fixation_reward_itis_fix_reward_trials = relative_delays[blank_trials_rewarded * what_trials_are_sensible, 1]
		
		return onsets_fix_reward_trials, raw_itis_of_fix_reward_trials, all_reward_itis_of_fix_reward_trials, fixation_reward_itis_fix_reward_trials
	
	def deconvolve_interval_roi(self, roi, threshold = 3.5, mask_type = 'center_surround_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'mean', nr_bins = 4, iti_type = 'all_reward', binning_grain = 'session', zero_time_offset = 0.0, add_other_conditions = 'full_design'):
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
		
		iti_data = []
		event_data = []
		other_conditions_event_data = []
		roi_data = []
		blink_events = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			self.rewarded_stimulus_run(r)
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data', postFix = ['mcf','tf']))
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			trial_times = self.run_data_from_hdf(reward_h5file, r, 'trial_times', postFix = ['mcf','tf'])
			parameter_data = self.run_data_from_hdf(reward_h5file, r, 'trial_parameters', postFix = ['mcf','tf'])
			
			onsets_fix_reward_trials, raw_itis_of_fix_reward_trials, all_reward_itis_of_fix_reward_trials, fixation_reward_itis_fix_reward_trials = self.calculate_event_history(trial_times, parameter_data)
			
			events_of_interest = onsets_fix_reward_trials + nr_runs * run_duration
			if iti_type == 'all_reward':
				itis = all_reward_itis_of_fix_reward_trials
			elif iti_type == 'fix_reward':
				itis = fixation_reward_itis_fix_reward_trials
			elif iti_type == 'all_trials':
				itis = raw_itis_of_fix_reward_trials
			
			iti_order = np.argsort(itis)
			stepsize = floor(itis.shape[0]/float(nr_bins))
			if binning_grain == 'run':
				event_data.append([events_of_interest[iti_order[x*stepsize:(x+1)*stepsize]] for x in range(nr_bins)])
				iti_data.append([itis[iti_order[x*stepsize:(x+1)*stepsize]] for x in range(nr_bins)])
			else:
				iti_data.append([itis, events_of_interest])
			
			# the times
			experiment_start_time = (trial_times['trial_phase_timestamps'][0,0,0])
			stim_onsets = (trial_times['trial_phase_timestamps'][:,1,0] - experiment_start_time ) / 1000.0
			
			this_run_events = []
			# conditions but not reward fixation, take out # 4 from the list of conditions
			these_non_fix_reward_conditions = [r.ordered_conditions[k] for k in range(6) if k != 4]
			for cond in these_non_fix_reward_conditions:
				this_run_events.append(stim_onsets[cond])	
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
		
		other_conditions_event_data = [np.concatenate([e[i] for e in other_conditions_event_data]) + zero_time_offset for i in range(len(other_conditions_event_data[0]))]
		# shell()
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		roi_data_per_run = demeaned_roi_data
		roi_data = np.hstack(demeaned_roi_data)
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type, postFix = ['mcf','tf'])
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		# shell()
		fig = pl.figure(figsize = (6, 5))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		colors = [(c, 0, 1-c) for c in np.linspace(0.1,0.9,nr_bins)]
		time_signals = []
		interval = [0.0,12.0]
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.vstack(blink_events)]))
		if add_other_conditions == 'full_design':
			# this next line adds other conditions to the design
			event_data.extend(other_conditions_event_data)
		# shell()
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
		# shell()
		for i in range(0, nr_bins):
			# time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze())
			if add_other_conditions == 'full_design':
				time_signals.append((deco.deconvolvedTimeCoursesPerEventTypeNuisance[i] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[nr_bins+4]).squeeze())
			else:
				time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze())
			
			# shell()
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze()), color = colors[i], alpha = 0.7, label = '%2.1f'%iti_data[i].mean())
		
		
		# the following commented code doesn't factor in blinks as nuisances
		# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
		# 	pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
		# 	time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		s.set_title('deconvolution' + roi + ' ' + mask_type + ' ' + add_other_conditions)
		# deco_per_run = []
		# for i, rd in enumerate(roi_data_per_run):
		# 	event_data_this_run = event_data_per_run[i] - i * run_duration
		# 	deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# 	deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
		# 	# deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		# 	# deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix[i*nr_trs*2:(i+1)*nr_trs*2])
		# 	# deco_per_run.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance)
		# deco_per_run = np.array(deco_per_run)
		# mean_deco = deco_per_run.mean(axis = 0)
		# std_deco = 1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))
		# for i in range(0, mean_deco.shape[0]):
		# 	# pl.plot(np.linspace(interval[0],interval[1],mean_deco.shape[1]), mean_deco[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
		# 	s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), (np.array(time_signals[i]) + std_deco[i].T)[0], (np.array(time_signals[i]) - std_deco[i].T)[0], color = ['k','r'][i], alpha = 0.3 * [0.5, 1.0][i])
		
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
	
	def deconvolve_intervals(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution', signal_type = 'mean', add_other_conditions = 'full_design', nr_bins = 4):
		results = []
		for roi in rois:
			results.append([])
			for itit in ['all_reward', 'fix_reward', 'all_trials']:
				results[-1].append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'left_Z', analysis_type = analysis_type, mask_direction = 'pos', nr_bins = nr_bins, signal_type = signal_type, iti_type = itit, binning_grain = 'session', add_other_conditions = add_other_conditions))
				results[-1].append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'right_Z', analysis_type = analysis_type, mask_direction = 'pos', nr_bins = nr_bins, signal_type = signal_type, iti_type = itit, binning_grain = 'session', add_other_conditions = add_other_conditions))
			# results.append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = signal_type))
			# results.append(self.deconvolve_interval_roi(roi, threshold, mask_type = 'surround_center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type))
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
			# self.deconvolve_roi(roi, -threshold, mask_type = 'surround_Z', analysis_type = analysis_type, mask_direction = 'neg')
		
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_interval_results' + '_' + signal_type
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for res in results:
			for r in res:
				try:
					reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_' + signal_type)
					# reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_' + signal_type + '_per_run')
				except NoSuchNodeError:
					pass
				reward_h5file.createArray(thisRunGroup, r[0] + '_' + signal_type, r[-1], 'interval deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
				# reward_h5file.createArray(thisRunGroup, r[0] + '_' + signal_type + '_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
