#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime, os, sys
from ..Sessions import * 
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from ..Operators.PhysioOperator import *
from pylab import *
from nifti import *
from joblib import Parallel, delayed
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge

def fitBR(design_matrix, timeseries):
	"""docstring for fitBR"""
	br = BayesianRidge(n_iter = 300, compute_score=True)
	br.fit(design_matrix, timeseries)
	return br.coef_


class PRFModelTrial(object):
	"""docstring for PRFModelTrial"""
	def __init__(self, orientation, n_elements, n_samples, sample_duration, bar_width = 0.1):
		super(PRFModelTrial, self).__init__()
		self.orientation = orientation
		self.n_elements = n_elements
		self.n_samples = n_samples
		self.sample_duration = sample_duration
		self.bar_width = bar_width
		
		self.rotation_matrix = np.matrix([[cos(self.orientation), -sin(self.orientation)],[sin(self.orientation), cos(self.orientation)]])
		
		x, y = np.meshgrid(np.linspace(-1,1,self.n_elements), np.linspace(-1,1,self.n_elements))
		self.xy = np.matrix([x.ravel(), y.ravel()]).T  
		self.rotated_xy = np.array(self.xy * self.rotation_matrix)
		self.ecc_test = (np.array(self.xy) ** 2).sum(axis = 1) <= 1.0
	
	def in_bar(self, time = 0):
		"""in_bar, a method, not Ralph."""
		# a bar of self.bar_width width
		position = 2.0 * ((time * (1.0 + self.bar_width / 2.0)) - (0.5 + self.bar_width / 4.0))
		extent = [-self.bar_width/2.0 + position, self.bar_width/2.0 + position] 
		# rotating the xy matrix itself allows us to test only the x component 
		return ((self.rotated_xy[:,0] >= extent[0]) * (self.rotated_xy[:,0] <= extent[1]) * self.ecc_test).reshape((self.n_elements, self.n_elements))
	
	def pass_through(self):
		"""pass_through models a single pass-through of the bar, 
		with padding as in the padding list for start and end."""
		self.pass_matrix = np.array([self.in_bar(i) for i in np.linspace(0.0, 1.0, self.n_samples, endpoint = True)])
	
class PRFModelRun(object):
	"""docstring for PRFModelRun"""
	def __init__(self, run, n_TRs, TR, n_pixel_elements, sample_duration = 0.6, bar_width = 0.1):
		super(PRFModelRun, self).__init__()
		self.run = run
		self.n_TRs = n_TRs
		self.TR = TR
		self.n_pixel_elements = n_pixel_elements
		self.sample_duration = sample_duration
		self.bar_width = bar_width
		
		self.orientation_list = self.run.orientations
	
	def simulate_run(self, save_images_to_file = None):
		"""docstring for simulate_run"""
		
		self.sample_times = np.arange(0, self.n_TRs * self.TR, self.sample_duration)
		
		self.run_matrix = np.zeros((self.sample_times.shape[0], self.n_pixel_elements, self.n_pixel_elements))
		
		for i in range(len(self.orientation_list)): # trials
			samples_in_trial = (self.sample_times > (self.run.trial_times[i][1])) * (self.sample_times < (self.run.trial_times[i][2]))
			if self.run.trial_times[i][0] != 'fix_no_stim':
				pt = PRFModelTrial(orientation = self.orientation_list[i], n_elements = self.n_pixel_elements, n_samples = samples_in_trial.sum(), sample_duration = self.sample_duration, bar_width = self.bar_width)
				pt.pass_through()
				self.run_matrix[samples_in_trial] = pt.pass_matrix
		
		if save_images_to_file != None:
			for i in range(self.run_matrix.shape[0]):
				if i < 200:
					f = pl.figure()
					s = f.add_subplot(111)
					pl.imshow(self.run_matrix[i])
					pl.savefig(save_images_to_file + '_' + str(i) + '.pdf')
			

class PopulationReceptiveFieldMappingSession(Session):
	"""
	Class for population receptive field mapping sessions analysis.
	"""
	def resample_epis(self):
		"""resample_epi resamples the mc'd epi files back to their functional space."""
		# create identity matrix
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg'), 'eye.mtx'), np.eye(4), fmt = '%1.1f')
		
		cmds = []
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			fO = FlirtOperator(inputObject = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'] ),  referenceFileName = self.runFile(stage = 'processed/mri', run = r ))
			fO.configureApply( transformMatrixFileName = os.path.join(self.stageFolder(stage = 'processed/mri/reg'), 'eye.mtx'), outputFileName = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res'] ) ) 
			cmds.append(fO.runcmd)
		
		# run all of these resampling commands in parallel
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers)
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(fo,),(),('subprocess','tempfile',)) for fo in cmds]
		for fo in ppResults:
			fo()
		
		# now put stuff back in the right places
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hr']) )
			os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res']) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) )
			
	
	def create_dilated_cortical_mask(self, dilation_sd = 3.0):
		"""create_dilated_cortical_mask takes the rh and lh cortex files and joins them to one cortex.nii.gz file.
		it then smoothes this mask with fslmaths, using a gaussian kernel. 
		This is then thresholded at > 0.0, in order to create an enlarged cortex mask in binary format.
		"""
		# take rh and lh files and join them.
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'rh.cortex.nii.gz'))
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex.nii.gz'), **{'-add': os.path.join(self.stageFolder('processed/mri/masks/anat'), 'lh.cortex.nii.gz')})
		fmO.execute()
		
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex.nii.gz'))
		fmO.configureSmooth(smoothing_sd = dilation_sd)
		fmO.execute()
		
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_smooth.nii.gz'))
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz'), **{'-bin': ''})
		fmO.execute()
	
	def stimulus_timings(self):
		"""stimulus_timings uses behavior operators to distil:
		- the times at which stimulus presentation began and ended per task type
		- the times at which the task buttons were pressed. 
		"""
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			bO = PopulationReceptiveFieldBehaviorOperator(self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' ))
			bO.trial_times() # sets up all behavior  
			r.trial_times = bO.trial_times
			r.all_button_times = bO.all_button_times
			r.parameters = bO.parameters
			r.tasks = [t.task for t in bO.trials]
			r.orientations = [t.parameters['orientation'] for t in bO.trials]
			tasks = list(np.unique(np.array([tt[0] for tt in r.trial_times])))
			for task in tasks:
				these_trials = np.array([[tt[1], tt[2] - tt[1], 1.0] for tt in r.trial_times if tt[0] == task])
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]), these_trials, fmt = '%3.2f', delimiter = '\t')
				these_buttons = np.array([[float(bt[1]), 0.5, 1.0] for bt in r.all_button_times if bt[0] == task])
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['button', task]), these_buttons, fmt = '%3.2f', delimiter = '\t')
	
	def physio(self):
		"""physio loops across runs to analyze their physio data"""
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			pO = PhysioOperator(self.runFile(stage = 'processed/hr', run = r, extension = '.log' ))
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] ))
			pO.preprocess_to_continuous_signals(TR = nii_file.rtime, nr_TRs = nii_file.timepoints)
	
	def zscore_timecourse_per_condition(self, dilate_width = 2):
		"""fit_voxel_timecourse loops over runs and for each run:
		looks when trials of a certain type occurred, 
		and dilates these times by dilate_width TRs.
		The data in these TRs are then z-scored on a per-task basis,
		and rejoined after which they are saved.
		"""
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		# loop over runs
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			self.logger.info('per-condition Z-score of run %s' % r)
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] ))
			tr_times = np.arange(0, nii_file.timepoints * nii_file.rtime, nii_file.rtime)
			masked_input_data = nii_file.data[:,cortex_mask]
			if not hasattr(r, 'trial_times'):
				self.stimulus_timings()
			tasks = list(np.unique(np.array([tt[0] for tt in r.trial_times])))
			output_data = np.zeros(list(masked_input_data.shape) + [len(tasks)])
			# loop over tasks
			for i, task in enumerate(tasks):
				self.logger.info('Z-scoring of task %s' % task)
				trial_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]))
				which_trs_this_task = np.array([(tr_times > (t[0] - (dilate_width * nii_file.rtime))) * (tr_times < (t[1] + (dilate_width * nii_file.rtime))) for t in np.array([trial_events[:,0] , trial_events[:,0] + trial_events[:,1]]).T]).sum(axis = 0, dtype = bool)
				output_data[which_trs_this_task,:,i] = (masked_input_data[which_trs_this_task] - masked_input_data[which_trs_this_task].mean(axis = 0)) / masked_input_data[which_trs_this_task].std(axis = 0)
			
			output_data = output_data.mean(axis = -1) * len(tasks)
			file_output_data = np.zeros(nii_file.data.shape, dtype = np.float32)
			file_output_data[:,cortex_mask] = output_data
			opf = NiftiImage(file_output_data)
			opf.header = nii_file.header
			self.logger.info('saving output file %s' % self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ'] ))
			opf.save(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ'] ))
	
	def design_matrix(self, method = 'hrf', gamma_hrfType = 'singleGamma', gamma_hrfParameters = {'a': 6, 'b': 0.9}, fir_ratio = 6, n_pixel_elements = 40, sample_duration = 0.6):
		"""design_matrix creates a design matrix for the runs
			method can be hrf or fir. when gamma, we can specify the parameters of gamma and double-gamma, etc.
		"""
		# self.logger.info('creating design matrix with arguments %s' % str(kwargs))
		# get orientations and stimulus timings
		self.stimulus_timings()
		
		self.stim_matrix_list = []
		self.design_matrix_list = []
		self.sample_time_list = []
		self.tr_time_list = []
		for i, r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ'] ))
			mr = PRFModelRun(r, n_TRs = nii_file.timepoints, TR = nii_file.rtime, n_pixel_elements = n_pixel_elements, sample_duration = 0.6, bar_width = 0.15)
			# mr.simulate_run( save_images_to_file = os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'design_' + str(i) ) )
			mr.simulate_run( )
			self.stim_matrix_list.append(mr.run_matrix)
			self.sample_time_list.append(mr.sample_times + i * nii_file.timepoints * nii_file.rtime)
			self.tr_time_list.append(np.arange(0, nii_file.timepoints * nii_file.rtime, nii_file.rtime) + i * nii_file.timepoints * nii_file.rtime)
			
			if method == 'hrf':
				run_design = Design(mr.run_matrix.shape[0], mr.sample_duration, subSamplingRatio = 1)
				run_design.rawDesignMatrix = mr.run_matrix.reshape((mr.run_matrix.shape[0], mr.run_matrix.shape[1] * mr.run_matrix.shape[2])).T
				run_design.convolveWithHRF(hrfType = gamma_hrfType, hrfParameters = gamma_hrfParameters)
				workingDesignMatrix = run_design.designMatrix
			elif method == 'fir':
				new_size = list(mr.run_matrix.shape)
				new_size[0] *= int(fir_ratio)
				new_array = np.zeros(new_size)
				for i in np.arange(mr.run_matrix.shape[0]) * int(fir_ratio):
					new_array[i:i+int(fir_ratio)] = mr.run_matrix[int(floor(i/int(fir_ratio)))]
				workingDesignMatrix = new_array
			
			self.design_matrix_list.append(workingDesignMatrix)
		self.full_design_matrix = np.vstack(self.design_matrix_list)
		self.full_design_matrix = (self.full_design_matrix - self.full_design_matrix.mean(axis = 0) )# / self.full_design_matrix.std(axis = 0)
		self.tr_time_list = np.concatenate(self.tr_time_list)
		self.sample_time_list = np.concatenate(self.sample_time_list)
	
	def whole_brain_fit(self, n_pixel_elements = 50):
		"""docstring for fname"""
		self.design_matrix(n_pixel_elements = n_pixel_elements)
		
		# cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		# cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'single_voxel.nii.gz')).data, dtype = bool)
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'rh.V1.nii.gz')).data, dtype = bool)
		slices = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T[cortex_mask]
		data_list = []
		for i, r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ'] ))
			data_list.append(nii_file.data[:,cortex_mask])
		
		z_data = np.vstack(data_list)
		del(data_list)
		
		# run through slices, each slice having a certain timing
		for sl in np.arange(cortex_mask.shape[0]):
			voxels_in_this_slice = slices == sl
			these_tr_times = self.tr_time_list + (nii_file.rtime / float(cortex_mask.shape[1]))
			# closest sample in designmatrix
			these_samples = np.array([np.argmin(np.abs(self.sample_time_list - t)) for t in these_tr_times])
			these_voxels = z_data[:,voxels_in_this_slice].T
			self.logger.info('starting fitting of slice %d' % sl)
			# do the following loop in parallel using joblib
			
			coefs = Parallel(n_jobs = -2, verbose = 5)(delayed(fitBR)(self.full_design_matrix[these_samples,:], vox_timeseries) for vox_timeseries in these_voxels)
			
			
			
			# shell()
			# for i, vox_timeseries in enumerate(these_voxels): # loop over timeseries data
			# 	self.logger.info('starting fitting of voxel %d in slice %d' % (i, sl))
			# 	
			# 	clf = BayesianRidge(n_iter = 300, compute_score=True)
			# 	# clf = Ridge(alpha=1.0)
			# 	clf.fit(self.full_design_matrix[these_samples,:], vox_timeseries)
			# 	self.logger.info('done fitting of voxel %d in slice %d' % (i, sl))
			# 	f = pl.figure(figsize = (6,6))
			# 	s = f.add_subplot(111, aspect='equal')
			# 	pl.imshow(clf.coef_.reshape((n_pixel_elements, n_pixel_elements)))
			# 	s.set_title('%d_%d' % (sl, i))
			# 	pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'fit_%d_%d.pdf' % (sl, i)))
