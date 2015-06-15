#!/usr/bin/env python
# encoding: utf-8
"""
MonkeyRewardSession.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Session import * 
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
import seaborn as sn
from RL_model import RL_model
from TD_model import TD_model
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, report_errors, fit_report
from sklearn import linear_model
from scipy import io

class MonkeyRewardSession(Session):
	"""
	Analyses for visual reward sessions
	"""
	def __init__(self, ID, date, project, subject, session_label = 'first', parallelize = True, loggingLevel = logging.DEBUG):
		super(MonkeyRewardSession, self).__init__(ID, date, project, subject, session_label = session_label, parallelize = parallelize, loggingLevel = loggingLevel)

		self.TR = 2.0
		self.event_duration = 0.5

	def setupFiles(self, rawBase, process_eyelink_file = True, date_format = None):
		super(MonkeyRewardSession, self).setupFiles(rawBase = rawBase, process_eyelink_file = process_eyelink_file, date_format = date_format)
		for r in self.runList:
			if hasattr(r, 'eye_file'):
				ExecCommandLine('cp ' + r.eye_file.replace('|', '\|') + ' ' + self.runFile(stage = 'processed/mri', run = r, extension = '.dat', postFix = ['eye'] ) )
			if hasattr(r, 'order_data_file'):
				ExecCommandLine('cp ' + r.order_data_file.replace('|', '\|') + ' ' + self.runFile(stage = 'processed/mri', run = r, extension = '.mat' ) )
			if hasattr(r, 'motion_reg_file'):
				ExecCommandLine('cp ' + r.motion_reg_file.replace('|', '\|') + ' ' + self.runFile(stage = 'processed/mri', run = r, extension = '.mat', postFix = ['mcf'] ) )

	def definition_from_mat(self, run):
		"""
		definition_from_mat takes the .mat file for a given run (argument) and 
		internalizes the names and onset_times in the run object, which is passed by reference.
		"""
		definition = io.loadmat(self.runFile(run = run, stage = 'processed/mri', extension = '.mat'))
		
		run.cond_names = [str(definition['names'][0][i][0]) for i in range(4)]		
		run.cond_onset_times = [definition['onsets'][0][i][0] * self.TR for i in range(4)]

		print ['%.2f'%r[0] for r in run.cond_onset_times]

	def definition_to_fsl_event_file(self, run):
		"""
		definition_to_fsl_event_file takes a given run (argument) and 
		converts the onset_times to separate fsl event text files, named after the condition names.
		"""
		for name, onset_times in zip(run.cond_names, run.cond_onset_times):
			onset_array = np.array([onset_times, np.ones(onset_times.shape[0]) * self.event_duration, np.ones(onset_times.shape[0])]).T
			np.savetxt(self.runFile(run = run, stage = 'processed/mri', extension = '.txt', postFix = [name], base = 'events/'+self.fileNameBaseString), onset_array, fmt = '%3.2f', delimiter = '\t')

	def moco_mat_to_par(self, run):
		this_mat_file = self.runFile(stage = 'processed/mri', run = run, extension = '.mat', postFix = ['mcf'] )
		mot_components = io.loadmat(this_mat_file)['motion_regressor']

		np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf'] ), mot_components, fmt = '%f', delimiter = '\t')

		# shell()

	def all_timings_mocos(self):
		for run in self.runList:
			self.definition_from_mat(run)
			self.definition_to_fsl_event_file(run)
			self.moco_mat_to_par(run)

	def combine_motion_parameters_one_file(self, run):
		files = [
				self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf'] ), 
				self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf', 'dt'] ),
				self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf', 'ddt'] )
				]
		np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf', 'all'] ), np.hstack([np.loadtxt(f) for f in files]), fmt = '%f', delimiter = '\t') 

	def combine_motion_parameters(self):
		for run in self.runList:
			self.combine_motion_parameters_one_file(run)

	def feat_run(self, run, feat_file = 'standard_moco_all.fsf', postFix = [], waitForExecute = True):
		try:
			self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
			os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
			os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.fsf'))
		except OSError:
			pass

		# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
		# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
		thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward/monkey/fsf/' + feat_file
	
		REDict = {
		'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = run, postFix = postFix), 
		'---MOCO_FILE---': 			self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf', 'all'] ), 
		'---fix_Juice_FILE---': 	self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['fix_Juice'], base = 'events/'+self.fileNameBaseString), 
		'---fix_NoJuice_FILE---': 	self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['fix_NoJuice'], base = 'events/'+self.fileNameBaseString),	
		'---visualJuice_FILE---': 	self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['visualJuice'], base = 'events/'+self.fileNameBaseString), 
		'---visualNoJuice_FILE---':	self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['visualNoJuice'], base = 'events/'+self.fileNameBaseString), 	
		}

		featFileName = self.runFile(stage = 'processed/mri', run = run, extension = '.fsf')
		featOp = FEATOperator(inputObject = thisFeatFile)
		# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
		featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = waitForExecute )
		self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
		# run feat
		featOp.execute()

	def eye_register(self):
		try:
			os.mkdir(self.stageFolder(stage = 'processed/mri/reg/feat'))
		except OSError:
			pass
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'example_func2standard.mat' ), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'standard2example_func.mat' ), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'highres2standard.mat' ), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'standard2highres.mat' ), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'example_func2highres.mat' ), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'highres2example_func.mat' ), np.eye(4))

		os.system('cp %s %s'%(self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] ), os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'example_func.nii.gz' )))
		os.system('cp %s %s'%(self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] ), os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'highres.nii.gz' )))
		os.system('cp %s %s'%(self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] ), os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'standard.nii.gz' )))

	def register_feats(self):
		for run in self.runList:
			self.setupRegistrationForFeat(self.runFile(stage = 'processed/mri', run = run, postFix = [], extension = '.feat'), wait_for_execute = False)

	def feat_all(self, nr_processes = 16):
		for i, run in enumerate(self.runList):
			if (i % nr_processes) == (nr_processes-1):
				self.feat_run(run, waitForExecute = True)
			else:
				self.feat_run(run, waitForExecute = False)

	def mask_stats_to_hdf(self, run_type = 'reward', postFix = []):
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
		
		this_feat = self.runFile(stage = 'processed/mri/reward', postFix = postFix, extension = '.gfeat', base = 'feat')
		# shell()

		main_stat_files = {
				'visual_T': os.path.join(this_feat, 'cope5.feat', 'stats', 'tstat1.nii.gz'),
				'visual_Z': os.path.join(this_feat, 'cope5.feat', 'stats', 'zstat1.nii.gz'),
				'visual_cope': os.path.join(this_feat, 'cope5.feat', 'stats', 'cope1.nii.gz'),
			
				'reward_T': os.path.join(this_feat, 'cope6.feat', 'stats', 'tstat1.nii.gz'),
				'reward_Z': os.path.join(this_feat, 'cope6.feat', 'stats', 'zstat1.nii.gz'),
				'reward_cope': os.path.join(this_feat, 'cope6.feat', 'stats', 'cope1.nii.gz'),
			
				'fix_Juice_T': os.path.join(this_feat, 'cope1.feat', 'stats', 'tstat1.nii.gz'),
				'fix_Juice_Z': os.path.join(this_feat, 'cope1.feat', 'stats', 'zstat1.nii.gz'),
				'fix_Juice_cope': os.path.join(this_feat, 'cope1.feat', 'stats', 'cope1.nii.gz'),

				'fix_NoJuice_T': os.path.join(this_feat, 'cope2.feat', 'stats', 'tstat1.nii.gz'),
				'fix_NoJuice_Z': os.path.join(this_feat, 'cope2.feat', 'stats', 'zstat1.nii.gz'),
				'fix_NoJuice_cope': os.path.join(this_feat, 'cope2.feat', 'stats', 'cope1.nii.gz'),

				'visualJuice_T': os.path.join(this_feat, 'cope3.feat', 'stats', 'tstat1.nii.gz'),
				'visualJuice_Z': os.path.join(this_feat, 'cope3.feat', 'stats', 'zstat1.nii.gz'),
				'visualJuice_cope': os.path.join(this_feat, 'cope3.feat', 'stats', 'cope1.nii.gz'),

				'visualNoJuice_T': os.path.join(this_feat, 'cope4.feat', 'stats', 'tstat1.nii.gz'),
				'visualNoJuice_Z': os.path.join(this_feat, 'cope4.feat', 'stats', 'zstat1.nii.gz'),
				'visualNoJuice_cope': os.path.join(this_feat, 'cope4.feat', 'stats', 'cope1.nii.gz'),
			
				}

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
			stat_files = main_stat_files.copy()
		
			# general info we want in all hdf files
			stat_files.update({
								# 'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								'raw_data': self.runFile(stage = 'processed/mri', run = r, postFix = []), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['tf', 'Z']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'Z_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['tf']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								# 'hpf_data_fsl': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
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

	def deconvolve_roi(self, roi, threshold = 3.5, mask_type = 'visualNoJuice_Z', mask_direction = 'pos', signal_type = 'mean', data_type = 'Z_hpf_data', subsampling_factor = 1, interval = [-5.0,15.0]
):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		nr_trs = niiFile.timepoints
		run_duration = self.TR * nr_trs
		

		conds = ['fix_NoJuice','fix_Juice','visualNoJuice','visualJuice']
		cond_labels = conds
		
		reward_h5file = self.hdf5_file('reward')
		
		moco_data = []
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, data_type, postFix = []))
			
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond], base = 'events/'+self.fileNameBaseString))[:-1,0] + interval[0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			
			moco_data.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.par', postFix = ['mcf', 'all'] )))

			nr_runs += 1
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		moco_data = np.vstack(moco_data)

		# mapping data
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, mask_type, postFix = [])
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		
		reward_h5file.close()

		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		# timeseries = np.repeat(timeseries, subsampling_factor)

				
		fig = pl.figure(figsize = (6, 5))
		s = fig.add_subplot(211)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		# nuisance version?
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = self.TR, deconvolutionSampleDuration = self.TR/float(subsampling_factor), deconvolutionInterval = interval[1] - interval[0], run = False)
		deco.runWithConvolvedNuisanceVectors(np.repeat(moco_data, subsampling_factor, axis = 0))
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
			deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = self.TR, deconvolutionSampleDuration = self.TR/float(subsampling_factor), deconvolutionInterval = interval[1] - interval[0])
			deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
			# deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
			# deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix[i*nr_trs*2:(i+1)*nr_trs*2])
			# deco_per_run.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance)
		deco_per_run = np.array(deco_per_run)
		mean_deco = deco_per_run.mean(axis = 0)
		std_deco = 1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))
		# shell()
		for i in range(0, mean_deco.shape[0]):
			s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), (np.array(time_signals[i]) + std_deco[i])[0], (np.array(time_signals[i]) - std_deco[i])[0], color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])

		
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
		s.axhline(0, -10, 30, linewidth = 0.25, color = 'k', alpha = 0.5)
		
		for i in range(0, len(event_data), 2):
			ts_diff = -(time_signals[i] - time_signals[i+1])
			pl.plot(np.linspace(interval[0],interval[1],mean_deco.shape[1]), np.array(ts_diff), ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
			s.set_title('reward signal ' + roi + ' ' + mask_type)
		
		
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
			
		pl.tight_layout()
		# pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + data_type + '.pdf'))
		# shell()
		return [roi + '_' + mask_type + '_' + mask_direction, event_data, timeseries, np.array(time_signals), np.array(deco_per_run), residuals]

	def deconvolve(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V4', 'TE', 'TEO'], signal_type = 'mean', data_type = 'Z_hpf_data'):
		results = []
		# neg_threshold = -1.0 * thres
		# neg_threshold = -neg_threshold
		# print threshold
		for roi in rois:
			results.append(self.deconvolve_roi(roi, threshold = threshold, mask_type = 'center_Z', mask_direction = 'pos', signal_type = signal_type, data_type = data_type))
			results.append(self.deconvolve_roi(roi, threshold = -threshold, mask_type = 'center_Z', mask_direction = 'neg', signal_type = signal_type, data_type = data_type))
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
