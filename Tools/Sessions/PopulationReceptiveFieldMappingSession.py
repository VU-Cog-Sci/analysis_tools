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
import scipy as sp
from scipy.stats import spearmanr
from scipy import ndimage

from nifti import *
from math import *

from joblib import Parallel, delayed
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge, RidgeCV
from skimage import filter, measure

def fitARDRidge(design_matrix, timeseries, n_iter = 100, compute_score=True):
	"""fitARDRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	fitARDRidge is too time-expensive.
	"""
	br = ARDRegression(n_iter = n_iter, compute_score = compute_score)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return br.coef_, srp

def fitBayesianRidge(design_matrix, timeseries, n_iter = 50, compute_score = False, verbose = True):
	"""fitBayesianRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	if n_iter == 0:
		br = BayesianRidge(compute_score = compute_score, verbose = verbose)
	else:
		br = BayesianRidge(n_iter = n_iter, compute_score = compute_score, verbose = verbose)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return br.coef_, srp

def fitRidge(design_matrix, timeseries, alpha = 1e7):
	"""fitRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	br = Ridge(alpha = alpha)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return br.coef_, srp
	
def fitRidgeCV(design_matrix, timeseries, alphas = None):
	"""fitRidgeCV fits a design matrix to a given timeseries using
	built-in cross-validation.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	if alphas == None:
		alphas = np.logspace(0.001,10,10)
	br = RidgeCV(alphas = alphas)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return br.coef_, srp


def normalize_histogram(input_array, mask_array = None):
	if mask_array == None:
		mask_array = input_array != 0.0
	
	return (input_array - input_array[mask_array].min()) / (input_array[mask_array].max() - input_array[mask_array].min())

# def convert_2d_array_to_points(input_array, mask_array = None, nr_points = 100000):
# 	if mask_array = None:
# 		mask_array = input_array != 0.0
# 	
# 	indices = np.linspace(0, 1, input_array.shape[0], endpoint = True)
# 	n_a = normalize_histogram(input_array, mask_array)

def analyze_PRF_from_spatial_profile(spatial_profile_array, upscale = 5, diagnostics_plot = True, contour_level = 0.8):
	"""analyze_PRF_from_spatial_profile tries to fit a PRF 
	to the spatial profile of spatial beta values from the ridge regression """
	n_pixel_elements = sqrt(spatial_profile_array.shape[0])
	# upsample five-fold and gaussian smooth with upscale factor
	us_spatial_profile = ndimage.interpolation.zoom(spatial_profile_array.reshape((n_pixel_elements, n_pixel_elements)), upscale)
	uss_spatial_profile = ndimage.gaussian_filter(us_spatial_profile, upscale*2)
	
	maximum = ndimage.measurements.maximum_position(uss_spatial_profile)
	canny_edges = filter.canny(us_spatial_profile, sigma=upscale)
	# find level at which to take contour:
	contour_threshold = spatial_profile_array[np.argsort(spatial_profile_array)[round(contour_level * spatial_profile_array.shape[0])]]
	contour_edges = measure.find_contours(uss_spatial_profile, contour_threshold)
	
	# shell()
	max_norm = 2.0 * (np.array(maximum) / (n_pixel_elements*upscale)) - 1.0
	max_comp =  np.complex(max_norm[0], max_norm[1])
	
	# max_x, max_y = 2.0 * (maximum[0] / (n_pixel_elements*upscale)) - 1.0, (maximum[1] / (n_pixel_elements*upscale)) - 1.0
	# ecc = sqrt(max_x**2 + max_y**2)
	# polar = arctan(max_x/(max_y+0.0000001))
	
	if diagnostics_plot:
		f = pl.figure(figsize = (10, 9))
		s = f.add_subplot(221)
		pl.imshow(us_spatial_profile)
		pl.plot([maximum[1]], [maximum[0]], 'ko')
		s.set_title('original')
		s.axis([0,200,0,200])
		s = f.add_subplot(222)
		pl.imshow(uss_spatial_profile)
		pl.plot([maximum[1]], [maximum[0]], 'ko')
		s.set_title('gaussian smoothed')
		s.axis([0,200,0,200])
		s = f.add_subplot(223)
		pl.imshow(canny_edges, cmap=pl.cm.gray)
		pl.plot([maximum[1]], [maximum[0]], 'ko')
		s.set_title('canny edge image')
		s.axis([0,200,0,200])
		s = f.add_subplot(224)
		pl.imshow(uss_spatial_profile)
		pl.plot([maximum[1]], [maximum[0]], 'ko')
		s.set_title('contour image')
		for n, contour in enumerate(contour_edges):
		    pl.plot(contour[:, 1], contour[:, 0], linewidth=2, color = 'gray')
		s.axis([0,200,0,200])
		pl.show()
	
	return max_comp
		
		

def fit_gaussian(coef_array, method = 'ML'):
	"""
	fit_gaussian fits a gaussian distribution to a two-dimensional histogram (coef_array).
	It uses the argument method to decide what method to use. 
	"""
	# normalize histogram
	n_a = normalize_histogram(coef_array)
	
	
	
	

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
	def resample_epis(self, condition = 'PRF'):
		"""resample_epi resamples the mc'd epi files back to their functional space."""
		# create identity matrix
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg'), 'eye.mtx'), np.eye(4), fmt = '%1.1f')
		self.logger.info('resampling epis back to functional space')
		cmds = []
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
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
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hr']) )
			os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res']) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) )
			
	
	def create_dilated_cortical_mask(self, dilation_sd = 0.5, label = 'cortex'):
		"""create_dilated_cortical_mask takes the rh and lh cortex files and joins them to one cortex.nii.gz file.
		it then smoothes this mask with fslmaths, using a gaussian kernel. 
		This is then thresholded at > 0.0, in order to create an enlarged cortex mask in binary format.
		"""
		self.logger.info('creating dilated %s mask with sd %f'%(label, dilation_sd))
		# take rh and lh files and join them.
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'rh.' + label + '.nii.gz'))
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), '' + label + '.nii.gz'), **{'-add': os.path.join(self.stageFolder('processed/mri/masks/anat'), 'lh.' + label + '.nii.gz')})
		fmO.execute()
		
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), '' + label + '.nii.gz'))
		fmO.configureSmooth(smoothing_sd = dilation_sd)
		fmO.execute()
		
		fmO = FSLMathsOperator(fmO.outputFileName)
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), label + '_dilated_mask.nii.gz'), **{'-bin': ''})
		fmO.execute()
	
	def stimulus_timings(self, condition = 'PRF'):
		"""stimulus_timings uses behavior operators to distil:
		- the times at which stimulus presentation began and ended per task type
		- the times at which the task buttons were pressed. 
		"""
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
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
	
	def physio(self, condition = 'PRF'):
		"""physio loops across runs to analyze their physio data"""
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			pO = PhysioOperator(self.runFile(stage = 'processed/hr', run = r, extension = '.log' ))
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] ))
			pO.preprocess_to_continuous_signals(TR = nii_file.rtime, nr_TRs = nii_file.timepoints)
	
	# def physio_retroicor(self, condition = 'PRF', nr_dummies = 6):
	# 	"""physio loops across runs to analyze their physio data"""
	# 	for r in [self.runList[i] for i in self.conditionDict[condition]]:
	# 		self.retroicor_run(r, nr_dummies = nr_dummies)

	def GLM_for_nuisances(self, condition = 'PRF'):
		"""GLM_for_nuisances takes a diverse set of nuisance regressors,
		runs a GLM on them in order to run further PRF analysis on the 
		residuals after GLM. It assumes physio, motion correction and 
		stimulus_timings have been run beforehand, as it uses the output
		text files of these procedures.
		"""
		self.stimulus_timings()
		# physio regressors
		physio_list = []
		mcf_list = []
		trial_times_list = []
		total_trs  = 0
		for j, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] ))
			# moco and physiology regressors are per-TR regressors that need no convolution anymore.
			# physio_list.append(np.array([
			# 	np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp']) ),
			# 	np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu']) ) 
			# 	]))
			# new piece of code for retroicor regressors
			physio_list.append(np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['regressors']) ).T)
				
			mcf_list.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' )))
			# final regressor captures instruction-related variance that may otherwise cause strong responses in periphery
			# trial_times are single events that have to still be convolved with HRF
			trial_times_list.extend([[[(j * nii_file.rtime * nii_file.timepoints) + tt[1] -1.5, 0.5, 1.0]] for tt in r.trial_times]) # changed the occurrence of this event to -4.5 to -1.5...
			# lateron, this will also have pupil size and the occurrence of saccades in there.
			
			
			total_trs += nii_file.timepoints
		
		# to arrays with these regressors
		mcf_list = np.vstack(mcf_list).T
		physio_list = np.hstack(physio_list)
		
		# create a design matrix and convolve 
		run_design = Design(total_trs, nii_file.rtime, subSamplingRatio = 10)
		run_design.configure(trial_times_list)
		# joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, mcf_list, physio_list]).T)
		# joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, physio_list]).T)
		# only using the mc and physio now
		joined_design_matrix = np.mat(np.vstack([mcf_list, physio_list]).T)
		# joined_design_matrix = np.mat(physio_list.T)
		# shell()
		self.logger.info('nuisance and trial_onset design_matrix of dimensions %s'%(str(joined_design_matrix.shape)))
		# take data
		data_list = []
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			# self.logger.info('per-condition Z-score of run %s' % r)
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ'] ))
			data_list.append(nii_file.data[:,cortex_mask])
		data_list = np.vstack(data_list)
		# now we run the GLM
		self.logger.info('nifti data loaded from %s for nuisance/trial onset analysis'%(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ'] )))
		betas = ((joined_design_matrix.T * joined_design_matrix).I * joined_design_matrix.T) * np.mat(data_list.T).T
		residuals = data_list - (np.mat(joined_design_matrix) * np.mat(betas))
		
		self.logger.info('GLM finished; outputting data to %s and %s'%(os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ', 'res']))[-1], os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ', 'betas']))[-1]))
		# and now, back to image files
		for i, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			output_data_res = np.zeros(nii_file.data.shape, dtype = np.float32)
			output_data_res[:,cortex_mask] = residuals[i*nii_file.data.shape[0]:(i+1)*nii_file.data.shape[0],:]
			
			res_nii_file = NiftiImage(output_data_res)
			res_nii_file.header = nii_file.header
			res_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ', 'res']))
			
			output_data_betas = np.zeros([betas.shape[0]]+list(cortex_mask.shape), dtype = np.float32)
			output_data_betas[:,cortex_mask] = betas
			
			betas_nii_file = NiftiImage(output_data_betas)
			betas_nii_file.header = nii_file.header
			betas_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ', 'betas']))
			
		# shell()
	
	def zscore_timecourse_per_condition(self, dilate_width = 2, condition = 'PRF', postFix = ['mcf', 'sgtf']):
		"""fit_voxel_timecourse loops over runs and for each run:
		looks when trials of a certain type occurred, 
		and dilates these times by dilate_width TRs.
		The data in these TRs are then z-scored on a per-task basis,
		and rejoined after which they are saved.
		"""
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		# loop over runs
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			self.logger.info('per-condition Z-score of run %s' % r)
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
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
			self.logger.info('saving output file %s' % self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['prZ'] ))
			opf.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['prZ'] ))
	
	def design_matrix(self, method = 'hrf', gamma_hrfType = 'singleGamma', gamma_hrfParameters = {'a': 6, 'b': 0.9}, fir_ratio = 6, n_pixel_elements = 40, sample_duration = 0.6, plot_diagnostics = True, ssr = 5, condition = 'PRF'):
		"""design_matrix creates a design matrix for the runs
		using the PRFModelRun and PRFTrial classes. The temporal grain
		of the model is specified by sample_duration. In our case, the 
		stimulus was refreshed every 600 ms. 
		method can be hrf or fir. when gamma, we can specify 
		the parameters of gamma and double-gamma, etc.
		FIR fitting is still to be implemented, as the shape of
		the resulting design matrix will differ from the HRF version.
		"""
		# get orientations and stimulus timings
		self.stimulus_timings()
		self.logger.info('design_matrix of %d pixel elements and %1.2f s sample_duration'%(n_pixel_elements, sample_duration))
		
		self.stim_matrix_list = []
		self.design_matrix_list = []
		self.sample_time_list = []
		self.tr_time_list = []
		self.trial_start_list = []
		for i, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'prZ'] ))
			mr = PRFModelRun(r, n_TRs = nii_file.timepoints, TR = nii_file.rtime, n_pixel_elements = n_pixel_elements, sample_duration = sample_duration, bar_width = 0.15)
			mr.simulate_run( )
			self.stim_matrix_list.append(mr.run_matrix)
			self.sample_time_list.append(mr.sample_times + i * nii_file.timepoints * nii_file.rtime)
			self.tr_time_list.append(np.arange(0, nii_file.timepoints * nii_file.rtime, nii_file.rtime) + i * nii_file.timepoints * nii_file.rtime)
			self.trial_start_list.append(np.array(np.array(r.trial_times)[:,1], dtype = float) + i * nii_file.timepoints * nii_file.rtime)
			
			if method == 'hrf':
				run_design = Design(mr.run_matrix.shape[0], mr.sample_duration, subSamplingRatio = ssr)
				rdm = mr.run_matrix.reshape((mr.run_matrix.shape[0], mr.run_matrix.shape[1] * mr.run_matrix.shape[2])).T
				run_design.rawDesignMatrix = np.repeat(mr.run_matrix, ssr, axis=0).reshape((-1,n_pixel_elements*n_pixel_elements)).T
				
				run_design.convolveWithHRF(hrfType = gamma_hrfType, hrfParameters = gamma_hrfParameters)
				workingDesignMatrix = run_design.designMatrix
				# shell()
			elif method == 'fir':
				new_size = list(mr.run_matrix.shape)
				new_size[0] *= int(fir_ratio)
				new_array = np.zeros(new_size)
				for i in np.arange(mr.run_matrix.shape[0]) * int(fir_ratio):
					new_array[i:i+int(fir_ratio)] = mr.run_matrix[int(floor(i/int(fir_ratio)))]
				workingDesignMatrix = new_array
			
			# shell()
			self.design_matrix_list.append(workingDesignMatrix)
		self.full_design_matrix = np.hstack(self.design_matrix_list).T
		self.full_design_matrix = np.array(self.full_design_matrix - self.full_design_matrix.mean(axis = 0) )# / self.full_design_matrix.std(axis = 0)
		self.tr_time_list = np.concatenate(self.tr_time_list)
		self.sample_time_list = np.concatenate(self.sample_time_list)
		self.trial_start_list = np.concatenate(self.trial_start_list)
		self.logger.info('design_matrix of shape %s created, of which %d are valid stimulus locations'%(str(self.full_design_matrix.shape), int((self.full_design_matrix.sum(axis = 0) != 0).sum())))
		
		# shell()
		if plot_diagnostics:
			f = pl.figure(figsize = (10, 3))
			s = f.add_subplot(111)
			pl.plot(self.full_design_matrix.sum(axis = 1))
			pl.plot(self.trial_start_list / sample_duration, np.ones(self.trial_start_list.shape), 'ko')
			s.set_title('original')
			s.axis([0,200,0,200])
			
		
	
	def fit_PRF(self, n_pixel_elements = 30, mask_file_name = 'single_voxel', postFix = ['mcf', 'sgtf', 'prZ', 'res'], n_jobs = 15, task_conditions = ['fix'], condition = 'PRF', sample_duration = 0.6, save_all_data = True): # cortex_dilated_mask
		"""fit_PRF creates a design matrix for the full experiment, 
		with n_pixel_elements determining the amount of singular pixels in the display in each direction.
		fit_PRF uses a parallel joblib implementation of the Bayesian Ridge Regression from sklearn
		http://en.wikipedia.org/wiki/Ridge_regression
		http://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression
		mask_single_file allows the user to set a binary mask to mask the functional data
		for fitting of individual regions.
		"""
		# we need a design matrix.
		self.design_matrix(n_pixel_elements = n_pixel_elements, condition = condition, sample_duration = sample_duration)
		valid_regressors = self.full_design_matrix.sum(axis = 0) != 0
		self.full_design_matrix = self.full_design_matrix[:,valid_regressors]
		
		mask_file = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), mask_file_name + '.nii.gz'))
		cortex_mask = np.array(mask_file.data, dtype = bool)
		
		# numbered slices, for sub-TR timing to follow stimulus timing. 
		slices = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T[cortex_mask]
		slices_in_full = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T
		
		data_list = []
		for i, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			data_list.append(nii_file.data[:,cortex_mask])
			self.TR = nii_file.rtime
			tasks = list(np.unique(np.array([tt[0] for tt in r.trial_times])))
			
		z_data = np.array(np.vstack(data_list), dtype = np.float32)
		# get rid of the raw data list that will just take up memory
		del(data_list)
		
		# do the separation based on condition
		# loop over tasks
		task_tr_times = np.zeros((len(tasks), self.tr_time_list.shape[0]))
		task_sample_times = np.zeros((len(tasks), self.sample_time_list.shape[0]))
		dilate_width = 5.0 # in seconds
		for i, task in enumerate(tasks):
			add_time_for_previous_runs = 0.0
			trial_events = []
			for j, r in enumerate([self.runList[k] for k in self.conditionDict['PRF']]):
				this_nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
				trial_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]))[:,0] + add_time_for_previous_runs)
				trial_duration = np.median(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]))[:,1])
				add_time_for_previous_runs += this_nii_file.rtime * this_nii_file.timepoints
			trial_events = np.concatenate(trial_events)
			task_tr_times[i] = np.array([(self.tr_time_list > (t - dilate_width)) * (self.tr_time_list < (t + dilate_width + trial_duration)) for t in trial_events]).sum(axis = 0, dtype = bool)
			task_sample_times[i] = np.array([(self.sample_time_list > (t - dilate_width)) * (self.sample_time_list < (t + dilate_width + trial_duration)) for t in trial_events]).sum(axis = 0, dtype = bool)
		# what conditions are we asking for?
		if task_conditions == ['all']:
			selected_tr_times = task_tr_times.sum(axis = 0, dtype = bool)
			selected_sample_times = task_sample_times.sum(axis = 0, dtype = bool)
		else: # only one condition is selected, which means we must add fix_no_stim
			all_conditions = task_conditions + ['fix_no_stim']
			selected_tr_times = task_tr_times[[tasks.index(c) for c in all_conditions]].sum(axis = 0, dtype = bool)
			selected_sample_times = task_sample_times[[tasks.index(c) for c in all_conditions]].sum(axis = 0, dtype = bool)
		
		# set up empty arrays for saving the data
		all_coefs = np.zeros([int(valid_regressors.sum())] + list(cortex_mask.shape))
		all_corrs = np.zeros([2] + list(cortex_mask.shape))
		
		self.logger.info('PRF model fits on %d voxels' % int(cortex_mask.sum()))
		# run through slices, each slice having a certain timing
		for sl in np.arange(cortex_mask.shape[0]):
			voxels_in_this_slice = (slices == sl)
			voxels_in_this_slice_in_full = (slices_in_full == sl)
			if voxels_in_this_slice.sum() > 0:
				these_tr_times = self.tr_time_list + sl * (self.TR / float(cortex_mask.shape[1]))
				these_voxels = z_data[:,voxels_in_this_slice].T
				# closest sample in designmatrix
				these_samples = np.array([np.argmin(np.abs(self.sample_time_list - t)) for t in these_tr_times]) 
				this_design_matrix = np.array(self.full_design_matrix[these_samples[selected_tr_times],:], dtype = np.float64, order = 'F')
				if save_all_data:
					save_design_matrix = np.zeros((this_design_matrix.shape[0],n_pixel_elements * n_pixel_elements))
					save_design_matrix[:,valid_regressors] = this_design_matrix
					np.save(os.path.join(self.stageFolder('processed/mri/PRF/'), 'design_matrix_%ix%i_%s'%(n_pixel_elements, n_pixel_elements, task_conditions[0])), save_design_matrix)
				
				# loop across voxels in this slice in parallel using joblib, 
				# fitBayesianRidge returns coefficients of results, and spearman correlation R and p as a 2-tuple
				self.logger.info('starting fitting of slice %d, with %d voxels' % (sl, int((cortex_mask * voxels_in_this_slice_in_full).sum())))
				# res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitBayesianRidge)(self.full_design_matrix[these_samples,:], vox_timeseries) for vox_timeseries in these_voxels)
				res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitRidge)(self.full_design_matrix[these_samples,:], vox_timeseries, alpha = 1e6) for vox_timeseries in these_voxels)
				# res = [fitRidge(self.full_design_matrix[these_samples,:], vox_timeseries, alpha = 1e6, n_jobs = n_jobs) for vox_timeseries in these_voxels]
				self.logger.info('done fitting of slice %d, with %d voxels' % (sl, int((cortex_mask * voxels_in_this_slice_in_full).sum())))
				if mask_file_name == 'single_voxel':
					pl.figure()
					pl.imshow(all_coefs[:, cortex_mask * voxels_in_this_slice_in_full].reshape((n_pixel_elements,n_pixel_elements)))
					pl.show()
				all_coefs[:, cortex_mask * voxels_in_this_slice_in_full] = np.array([r[0] for r in res]).T
				all_corrs[:, cortex_mask * voxels_in_this_slice_in_full] = np.array([r[1] for r in res]).T
				
		
		output_coefs = np.zeros([n_pixel_elements ** 2] + list(cortex_mask.shape))
		output_coefs[valid_regressors] = all_coefs
		
		# shell()
		
		self.logger.info('saving coefficients and correlations of PRF fits')
		coef_nii_file = NiftiImage(output_coefs)
		coef_nii_file.header = mask_file.header
		coef_nii_file.save(os.path.join(self.stageFolder('processed/mri/PRF/'), 'coefs_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '.nii.gz'))
		
		# replace infs in correlations with the maximal value of the rest of the array.
		all_corrs[np.isinf(all_corrs)] = all_corrs[-np.isinf(all_corrs)].max() + 1.0
		corr_nii_file = NiftiImage(all_corrs)
		corr_nii_file.header = mask_file.header
		corr_nii_file.save(os.path.join(self.stageFolder('processed/mri/PRF/'), 'corrs_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '.nii.gz'))
	
		if save_all_data:
			all_data = np.zeros([selected_tr_times.sum()] + list(cortex_mask.shape))
			all_data[:,cortex_mask] = z_data[selected_tr_times]
			
			data_nii_file = NiftiImage(all_data)
			data_nii_file.header = mask_file.header
			data_nii_file.save(os.path.join(self.stageFolder('processed/mri/PRF/'), 'data_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '.nii.gz'))
			
			np.save(os.path.join(self.stageFolder('processed/mri/PRF/'), 'data_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0]), z_data[selected_tr_times])
	
	
	#
	#	For fitting of receptive fields; we fit full covariance matrices of at least one Gaussian distribution.
	#	One gaussian can be fit using least-squares as per scipy example: http://code.google.com/p/agpy/source/browse/trunk/agpy/gaussfitter.py
	#	Or, conversely, we can use a standard maximum-likelihood fit using our choice of minimization technique
	#	If mixture model fitting is necessary, we use the sklearn implementation of the DPGMM.
	#	This will automatically find the number of distributions in the data - instead of us having to specify which. The conservative leaning of this
	#	technique makes sure that if we have only a single peak, it will find only one single PRF in the histogram.
	#	Note; for most of this, data have to be resampled to singular data points instead of the histogram shape they have now. 
	#	
	
	
	def results_to_surface(self, file_name = 'corrs_cortex', output_file_name = 'polar', frames = {'f':1}, condition = 'PRF'):
		"""docstring for results_to_surface"""
		vsO = VolToSurfOperator(inputObject = os.path.join(self.stageFolder('processed/mri/%s/'%condition), file_name + '.nii.gz'))
		ofn = os.path.join(self.stageFolder('processed/mri/%s/surf/'%condition), output_file_name )
		vsO.configure(frames = frames, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
		vsO.execute()
	
	
	def RF_fit(self, mask_file = 'cortex_dilated_mask', stat_threshold = -10.0, n_jobs = 28, run_fits = True, condition = 'PRF'):
		"""select_voxels_for_RF_fit takes the voxels with high stat values
		and tries to fit a PRF model to their spatial selectivity profiles.
		it takes the images from the mask_file result file, and uses stat_threshold
		to select all voxels crossing a p-value (-log10(p)) threshold.
		"""
		
		if run_fits:
			stats_data = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + mask_file + '.nii.gz')).data
			spatial_data = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'coefs_' + mask_file + '.nii.gz')).data
		
			stat_mask = stats_data[1] > stat_threshold
			voxel_spatial_data_to_fit = spatial_data[:,stat_mask]
			self.logger.info('starting fitting of prf shapes')
			res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(analyze_PRF_from_spatial_profile)(vox_spatial_data, diagnostics_plot = False) for vox_spatial_data in voxel_spatial_data_to_fit.T)
		
			max_comp = np.array(res)
		
			polar = np.angle(max_comp)
			ecc = np.abs(max_comp)
			real = np.real(max_comp)
			imag = np.imag(max_comp)
		
			all_res = np.hstack([polar, ecc, real, imag])
			all_res = all_res.reshape([4] + list(stats_data.shape[1:]))
		
			self.logger.info('saving prf parameters to polar and ecc')
		
			all_res_file = NiftiImage(all_res)
			all_res_file.header = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + mask_file + '.nii.gz')).header
			all_res_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'all_res.nii.gz'))
		
		self.logger.info('converting prf values to surfaces')
		# reproject the original stats
		self.results_to_surface(file_name = 'corrs_cortex_dilated_mask', output_file_name = condition, frames = {'_f':1}, condition = condition)
		# and the spatial values
		self.results_to_surface(file_name = 'all_res', output_file_name = condition, frames = {'_polar':0, '_ecc':1, '_real':2, '_imag':3}, condition = condition)
	
	def makeTiffsFromCondition(self, condition, results_file, y_rotation = 90.0, exit_when_ready = 1 ):
		thisFeatFile = os.path.join(os.environ['ANALYSIS_HOME'], 'Tools/other_scripts/redraw_retmaps.tcl' )
		for hemi in ['lh','rh']:
			REDict = {
			'---HEMI---': hemi,
			'---CONDITION---': condition, 
			'---CONDITIONFILENAME---': results_file, 
			'---FIGPATH---': os.path.join(self.stageFolder(stage = 'processed/mri/'), condition, 'surf'),
			'---NAME---': self.subject.standardFSID,
			'---BASE_Y_ROTATION---': str(y_rotation),
			'---EXIT---': str(exit_when_ready),
			}
			rmtOp = RetMapReDrawOperator(inputObject = thisFeatFile)
			redrawFileName = os.path.join(self.stageFolder(stage = 'processed/mri/scripts'), hemi + '_' + condition.replace('/', '_') + '.tcl')
			rmtOp.configure( REDict = REDict, redrawFileName = redrawFileName, waitForExecute = False )
			# run 
			rmtOp.execute()
	
