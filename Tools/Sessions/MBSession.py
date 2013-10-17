#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Sessions import * 
from pylab import *
from IPython import embed as shell
from itertools import combinations


class MBSession(Session):
	"""
	Template Class for fMRI sessions analysis.
	"""
	def __init__(self, ID, date, project, subject, parallelize = True, loggingLevel = logging.DEBUG, name_appendix = '2.5'):
		# self.session_label = session_label
		super(MBSession, self).__init__(ID, date, project, subject, parallelize = parallelize, loggingLevel = loggingLevel, name_appendix = name_appendix)
	
	
	def feat_event_files_run(self, run, minimum_blink_duration = 0.02):
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

	
	def run_trial_structure(self):
		"""run_trial_structure loops across different conditions and executes their behavioral/trial structure analysis"""
		for cond in self.conditionDict.keys():
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				self.feat_event_files_run(r)

	def physio(self):
		"""physio loops across runs to analyze their physio data"""
		for cond in self.conditionDict.keys():
			for run in [self.runList[i] for i in self.conditionDict[cond]]:
				pO = PhysioOperator(self.runFile(stage = 'processed/hr', run = run, extension = '.log' ))
				nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf', 'tf'] ))
				pO.preprocess_to_continuous_signals(TR = nii_file.rtime, nr_TRs = nii_file.timepoints, filter_width = 0.25, sg_width = 121, sg_order = 3)
	

	def run_feat(self, run, fsf_file_name = 'first.fsf', analysis_folder = '~/projects/MB/analysis/', postFix = ['mcf', 'tf'], run_feat = True, waitForExecute = False):
		"""run_feat creates the fsf file for a given run and runs the feat"""
		try:
			self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
			os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
			os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.fsf'))
		except OSError:
			pass

		# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
		# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
		thisFeatFile = os.path.join(os.path.expanduser(analysis_folder), fsf_file_name)
	
		REDict = {
		'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = run, postFix = postFix), 
		'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = postFix)).timepoints),
		'---TR---':					str(NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = postFix)).rtime),
		'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']), 	
		'---LEFT_CW_FILE---': 		self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['left_CW']), 	
		'---LEFT_CCW_FILE---': 		self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['left_CCW']), 
		'---RIGHT_CW_FILE---':		self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['right_CW']), 	
		'---RIGHT_CCW_FILE---': 	self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['right_CCW']), 
		'---PPU_FILE---': 			self.runFile(stage = 'processed/hr', run = run, extension = '.txt', postFix = ['ppu']), 
		'---PPU_RAW_FILE---': 		self.runFile(stage = 'processed/hr', run = run, extension = '.txt', postFix = ['ppu_raw']), 
		'---RESP_FILE---': 			self.runFile(stage = 'processed/hr', run = run, extension = '.txt', postFix = ['resp']), 
		'---RESP_RAW_FILE---': 		self.runFile(stage = 'processed/hr', run = run, extension = '.txt', postFix = ['resp_raw']), 
		'---MC_PAR_FILE---': 		self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf']), 
		}
		featFileName = self.runFile(stage = 'processed/mri', run = run, extension = '.fsf')
		featOp = FEATOperator(inputObject = thisFeatFile)
		# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
		featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = waitForExecute )
		self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
		# run feat
		featOp.execute()
	
	def run_all_feats(self, postFix = ['mcf', 'tf']):
		for cond in self.conditionDict.keys():
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				self.run_feat(r, postFix = postFix)

	def mask_stats_to_hdf(self, postFix = ['mcf','tf']):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
		
		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri'), 'MB.hdf5')
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = openFile(self.hdf5_filename, mode = "w", title = 'MB.hdf5')
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		for cond in self.conditionDict.keys():
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
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
				stat_files = {
								'LEFT_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
								'LEFT_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
									
								'RIGHT_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
								'RIGHT_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
						
								'L_C_T': os.path.join(this_feat, 'stats', 'tstat3.nii.gz'),
								'L_C_Z': os.path.join(this_feat, 'stats', 'zstat3.nii.gz'),
								
								'R_C_T': os.path.join(this_feat, 'stats', 'tstat4.nii.gz'),
								'R_C_Z': os.path.join(this_feat, 'stats', 'zstat4.nii.gz'),
								
								'BLINKS_T': os.path.join(this_feat, 'stats', 'tstat5.nii.gz'),
								'BLINKS_Z': os.path.join(this_feat, 'stats', 'zstat5.nii.gz'),
								'BLINKS_cope': os.path.join(this_feat, 'stats', 'cope5.nii.gz'),
						
								'left_answer_T': os.path.join(this_feat, 'stats', 'tstat10.nii.gz'),
								'left_answer_Z': os.path.join(this_feat, 'stats', 'zstat10.nii.gz'),
								'left_answer_cope': os.path.join(this_feat, 'stats', 'cope10.nii.gz'),
								
								'right_answer_T': os.path.join(this_feat, 'stats', 'tstat11.nii.gz'),
								'right_answer_Z': os.path.join(this_feat, 'stats', 'zstat11.nii.gz'),
								'right_answer_cope': os.path.join(this_feat, 'stats', 'cope11.nii.gz'),
								
								'physio_F': os.path.join(this_feat, 'stats', 'fstat1.nii.gz'),
								'answer_F': os.path.join(this_feat, 'stats', 'fstat2.nii.gz'),
						
								}
			
				# general info we want in all hdf files
				stat_files.update({
									'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
									'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf', 'psc']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									'mbs_psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf', 'mbs', 'psc']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									'mbs_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf', 'mbs']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									# for these final two, we need to pre-setup the retinotopic mapping data
				})
			
				stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
		
				for (roi, roi_name) in zip(rois, roinames):
					if roi.data.sum() > 0:
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
						self.logger.debug('roi %s did not contain valid voxels for masking' % roi)
		h5file.close()
		
	def deconvolve_roi(self, roi, cond, threshold = 2.5, mask_type = 'LEFT_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'mean', data_type = 'psc_hpf_data', sample_duration = 1):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[cond][0]]))
		tr, nr_trs = niiFile.rtime / 1000.0, niiFile.timepoints
		run_duration = tr * nr_trs
		
		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri'), 'MB.hdf5')
		h5file = openFile(self.hdf5_filename, mode = "r", title = 'MB.hdf5')
		
		event_conds = ['left','right']
		colors = ['r','g']
		alphas = [0.5, 0.5]
		
		event_data = []
		roi_data = []
		blink_events = []
		nuisance_data = []
		
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict[cond]]:
			roi_data.append(self.roi_data_from_hdf(h5file, r, roi, data_type, postFix = ['mcf','tf']))
			if 'residuals' in data_type:
				roi_data[-1] = roi_data[-1] ** 2
			# blinks and nuisances
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			
			nuisance_data.append(np.vstack([np.loadtxt(f).T for f in self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu']), self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu_raw']), self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp']), self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp_raw']), self.runFile(stage = 'processed/mri', run = r, extension = '.par', postFix = ['mcf'])]).T)
			
			this_run_events = []
			for event_cond in event_conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [event_cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			nr_runs += 1
		
		# demean per run
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		roi_data_per_run = demeaned_roi_data
		roi_data = np.hstack(demeaned_roi_data)
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(h5file, self.runList[self.conditionDict['MB_1'][0]], roi, mask_type, postFix = ['mcf', 'tf'])
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		
		# subtract empty voxels from roi_data from mask for smaller fov scans
		mapping_mask = mapping_mask - (roi_data.mean(axis = 1) == 0)
		
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		if signal_type in ['std', 'var']:
			timeseries = (timeseries - timeseries.mean() ) / timeseries.std()
		
		nuisance_design = Design(timeseries.shape[0], tr / sample_duration, subSamplingRatio = 100)
		nuisance_design.configure([list(np.vstack(blink_events))])
		
		nuisance_design_matrix = nuisance_design.designMatrix
		nuisance_design_matrix = np.vstack((nuisance_design_matrix, np.vstack(nuisance_data).T)).T
		nuisance_design_matrix = np.repeat(nuisance_design_matrix, sample_duration, axis = 0)
		
		fig = pl.figure(figsize = (7, 3))
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		interval = [0.0,12.0]
			
		# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr, deconvolutionInterval = interval[1])
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr / sample_duration, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design_matrix)
		sse = deco.sse()
		event_interval_duration = deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]
		for i in range(len(event_conds)):
			pl.plot(np.linspace(interval[0],interval[1],event_interval_duration), deco.deconvolvedTimeCoursesPerEventTypeNuisance[i], colors[i], alpha = alphas[i], label = event_conds[i])
			time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
			this_event_sse = np.abs(sse[i*event_interval_duration:(i+1)*event_interval_duration])
			# shell()
			s.fill_between(np.linspace(interval[0],interval[1],event_interval_duration), deco.deconvolvedTimeCoursesPerEventTypeNuisance[i,:,0] + this_event_sse, deco.deconvolvedTimeCoursesPerEventTypeNuisance[i,:,0] - this_event_sse, color = colors[i], alpha = alphas[i] * 0.25)
			
		s.set_title('deconvolution' + roi + ' ' + mask_type)
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		h5file.close()
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + cond + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + data_type + '_' + str(sample_duration) + '.pdf'))
		
		return [roi + '_' + cond + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + str(sample_duration) + '_' + data_type, event_data, timeseries, np.array(time_signals)]
	
	def deconvolve(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4', 'pIPS', 'LO1', 'LO2'], analysis_type = 'deconvolution', signal_type = 'mean', data_type = 'mbs_psc_hpf_data', sample_duration = 1 ):
		results = []
		for roi in rois:
			for cond in self.conditionDict.keys():
				results.append(self.deconvolve_roi(roi, cond, threshold, mask_type = 'LEFT_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type, data_type = data_type, sample_duration = sample_duration))
				results.append(self.deconvolve_roi(roi, cond, threshold, mask_type = 'RIGHT_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = signal_type, data_type = data_type, sample_duration = sample_duration))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri'), 'MB.hdf5')
		h5file = openFile(self.hdf5_filename, mode = "r+", title = 'MB.hdf5')
		this_run_group_name = 'deconvolution_results' + '_' + signal_type + '_' + data_type
		try:
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				h5file.removeNode(where = thisRunGroup, name = r[0] + '_' + signal_type + '_' + data_type)
			except NoSuchNodeError:
				pass
			h5file.createArray(thisRunGroup, r[0] + '_' + signal_type + '_' + data_type, r[-2], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		h5file.close()
	
	def compare_stats(self, roi = 'V1'):
		"""docstring for compare_stats"""
		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri'), 'MB.hdf5')
		h5file = openFile(self.hdf5_filename, mode = "r", title = 'MB.hdf5')
		
		conditions = self.conditionDict.keys()
		
		for mt in ['LEFT_Z', 'RIGHT_Z']:
			this_cond_data = []
			for cond in conditions:
				# only a single file 
				run = self.runList[ self.conditionDict[cond][0] ]
				this_cond_data.append(self.roi_data_from_hdf(h5file, run, roi, mt, postFix = ['mcf', 'tf']))
			this_cond_data = np.array(this_cond_data)
			
			fig = pl.figure(figsize = (12,5))
			which_TRs = np.array(list(combinations(range(len(conditions)),2)))
			for i in range(which_TRs.shape[0]):
				s = fig.add_subplot(2,which_TRs.shape[0],i+1, aspect = 'equal')
				pl.plot(np.linspace(-10,15,100), np.linspace(-10,15,100), 'k--', linewidth = 3.0)
				pl.plot(this_cond_data[which_TRs[i,0]], this_cond_data[which_TRs[i,1]], ['r','g','b','m'][i%4] + 'o', alpha = 0.3, ms = 3)
				s.set_xlabel(conditions[which_TRs[i,0]])
				s.set_ylabel(conditions[which_TRs[i,1]])
			
			for i in range(which_TRs.shape[0]):
				s = fig.add_subplot(2,which_TRs.shape[0],i+1+which_TRs.shape[0])
				# pl.plot(np.linspace(-10,15,100), np.linspace(-10,15,100), 'k--', linewidth = 3.0)
				pl.hist(this_cond_data[which_TRs[i,0]] - this_cond_data[which_TRs[i,1]], color = ['r','g','b','m'][i%4], alpha = 0.3, normed = True, bins = 100, histtype = 'step')
				s.axvline((this_cond_data[which_TRs[i,0]] - this_cond_data[which_TRs[i,1]]).mean(), color = ['r','g','b','m'][i%4])
				s.set_xlabel(conditions[which_TRs[i,0]] + ' - ' + conditions[which_TRs[i,1]])
			
			
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' +  'corr_' + mt + '.pdf'))
		
		h5file.close()
		
	def compare_tsnr(self, roi = 'V1', data_type = 'hpf_data'):
		"""docstring for compare_stats"""
		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri'), 'MB.hdf5')
		h5file = openFile(self.hdf5_filename, mode = "r", title = 'MB.hdf5')
		
		conditions = self.conditionDict.keys()
		# areas = ['V1', 'V2', 'V3', 'V3AB', 'V4']
		this_cond_data = []
		for cond in conditions:
			# only a single file 
			run = self.runList[ self.conditionDict[cond][0] ]
			these_data = self.roi_data_from_hdf(h5file, run, roi, data_type, postFix = ['mcf', 'tf'])
			this_cond_data.append( these_data.mean(axis = 1) / these_data.std(axis = 1) )
		this_cond_data = np.array(this_cond_data)
		h5file.close()
		
		# shell()
		
		fig = pl.figure(figsize = (12,5))
		which_TRs = np.array(list(combinations(range(len(conditions)),2)))
		for i in range(which_TRs.shape[0]):
			s = fig.add_subplot(2,which_TRs.shape[0],i+1, aspect = 'equal')
			pl.plot(np.linspace(-40,100,100), np.linspace(-40,100,100), 'k--', linewidth = 3.0)
			pl.plot(this_cond_data[which_TRs[i,0]], this_cond_data[which_TRs[i,1]], ['r','g','b','m'][i%4] + 'o', alpha = 0.3, ms = 3)
			s.set_xlabel(conditions[which_TRs[i,0]])
			s.set_ylabel(conditions[which_TRs[i,1]])
		
		for i in range(which_TRs.shape[0]):
			s = fig.add_subplot(2,which_TRs.shape[0],i+1+which_TRs.shape[0])
			# pl.plot(np.linspace(-10,15,100), np.linspace(-10,15,100), 'k--', linewidth = 3.0)
			pl.hist(this_cond_data[which_TRs[i,0]] - this_cond_data[which_TRs[i,1]], color = ['r','g','b','m'][i%4], alpha = 0.3, normed = True, bins = 100, histtype = 'step')
			s.axvline((this_cond_data[which_TRs[i,0]] - this_cond_data[which_TRs[i,1]]).mean(), color = ['r','g','b','m'][i%4])
			s.set_xlabel(conditions[which_TRs[i,0]] + ' - ' + conditions[which_TRs[i,1]])
		
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' +  data_type + '_' +  'tsnr.pdf'))
		
		
		
	
	