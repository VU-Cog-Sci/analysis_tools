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
import numpy as np
import scipy as sp
from scipy.stats import spearmanr
from scipy import ndimage

from nifti import *
from math import *

from joblib import Parallel, delayed
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge, RidgeCV, ElasticNet, ElasticNetCV
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

def fitElasticNetCV(design_matrix, timeseries, verbose = True, l1_ratio = [.1, .5, .7, .9, .95, .99, 1]):
	"""fitBayesianRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	ecv = ElasticNetCV(verbose = verbose, l1_ratio = l1_ratio, n_jobs = 28)
	ecv.fit(design_matrix, timeseries)
	
	predicted_signal = ecv.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return ecv.coef_, srp


def fitRidge(design_matrix, timeseries, alpha = 1.0):
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
		alphas = np.logspace(0.001,10,100)
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

def analyze_PRF_from_spatial_profile(spatial_profile_array, spatial_profile_array_fix, upscale = 5.0, diagnostics_plot = True, contour_level = 0.9, voxel_no = 1, band = 75, cond = cond):
	"""analyze_PRF_from_spatial_profile tries to fit a PRF 
	to the spatial profile of spatial beta values from the ridge regression """
	# shell()
	n_pixel_elements = int(sqrt(spatial_profile_array.shape[0]))
	# upsample five-fold and gaussian smooth with upscale factor
	us_spatial_profile = ndimage.interpolation.zoom(spatial_profile_array.reshape((n_pixel_elements, n_pixel_elements)), upscale)
	us_spatial_profile_fix = ndimage.interpolation.zoom(spatial_profile_array_fix.reshape((n_pixel_elements, n_pixel_elements)), upscale)
	us_spatial_profile_banded = np.zeros((int(n_pixel_elements*upscale + band + 1),int(n_pixel_elements*upscale+ band + 1)))
	us_spatial_profile_banded[(band+1.0)/2:(band+1.0)/2+n_pixel_elements*upscale,(band+1)/2:(band+1)/2+n_pixel_elements*upscale] = us_spatial_profile
	uss_spatial_profile_banded = ndimage.gaussian_filter(us_spatial_profile_banded, upscale*2)
	uss_spatial_profile = ndimage.gaussian_filter(us_spatial_profile, upscale*2)
	uss_spatial_profile_fix = ndimage.gaussian_filter(us_spatial_profile, upscale*2)
	
	maximum = ndimage.measurements.maximum_position(uss_spatial_profile)
	maximum_fix = ndimage.measurements.maximum_position(uss_spatial_profile)
	maximum_for_select = ndimage.measurements.maximum_position(uss_spatial_profile_banded)
	
	# normalize data
	shell()
	# ussn_spatial_profile = spatial_profile_array/spatial_profile_array[]
	# ussn_spatial_profile_fix = spatial_profile_array_fix/spatial_profile_array
	# find level at which to take contour:
	
	contour_threshold = spatial_profile_array_fix[np.argsort(spatial_profile_array)[round(contour_level * spatial_profile_array.shape[0])]]
	contour_edges = measure.find_contours(uss_spatial_profile_banded, contour_threshold)
	
	max_norm = 2.0 * (np.array(maximum) / (n_pixel_elements*upscale)) - 1.0
	max_comp =  np.complex(max_norm[0], max_norm[1])
	
	# convert contours to image and fill for fix
	contour_array = np.zeros((int(n_pixel_elements*upscale + band + 1),int(n_pixel_elements*upscale+ band + 1)))
	for contour in contour_edges:
		for i in range(len(contour)):
			contour_array[int(contour[i,0]),int(contour[i,1])] = 1

	contour_array_filled = ndimage.binary_fill_holes(contour_array)
	# contour_array_filled = fill_contours(contour_array = contour_array, dimension = int(n_pixel_elements*upscale-1), voxel_no = voxel_no, uss_spatial_profile = uss_spatial_profile)
	contour_labels = ndimage.label(contour_array_filled)[0]
	label_index = contour_labels[(maximum_for_select)]
	# implement measure.regionprops.filled_area
	PRF = zeros((int(n_pixel_elements*upscale + band + 1),int(n_pixel_elements*upscale+ band + 1)))
	PRF[contour_labels==label_index] = 1
	surf = (sum(PRF)/(n_pixel_elements**2*upscale**2))*100
	
	if surf>100:
		f = pl.figure(figsize = (12,4))
		s = f.add_subplot(131)
		pl.imshow(us_spatial_profile)
		pl.plot([maximum[1]], [maximum[0]], 'ko')
		s.set_title('original')
		s.axis([0,n_pixel_elements*upscale,0,n_pixel_elements*upscale])
		s = f.add_subplot(132)
		pl.imshow(uss_spatial_profile_banded)
		pl.plot([maximum_for_select[1]], [maximum_for_select[0]], 'ko')
		s.set_title('gaussian smoothed with contour')
		for n, contour in enumerate(contour_edges):
			pl.plot(contour[:, 1], contour[:, 0], linewidth=2, color = 'gray')
		s.axis([0,n_pixel_elements*upscale+band+1,0,n_pixel_elements*upscale+band+1])
		s = f.add_subplot(133)
		pl.imshow(PRF)
		s.axis([0,n_pixel_elements*upscale+band+1,0,n_pixel_elements*upscale+band+1])
		pl.plot([maximum_for_select[1]], [maximum_for_select[0]], 'ko')
		s.set_title('PRF with maximum')
		# pl.show()
		# plt.savefig(os.path.join('/home/shared/PRF/data/AS/AS_090414/processed/mri/figs/failed_contours_2/' + cond + '/' + 'failed_contour' + '_' + str(voxel_no) + '.pdf'))
		
	return max_comp, surf

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
	
	def GLM_for_nuisances(self, condition = 'PRF', physiology_type = 'RETROICOR', postFix = ['mcf', 'sgtf', 'prZ']):
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
		button_times_list = []
		total_trs  = 0
		for j, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			# moco and physiology regressors are per-TR regressors that need no convolution anymore.
			if physiology_type == 'RETROICOR':
				physio_list.append(np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['regressors']) ).T)
			else:
				physio_list.append(np.array([
					np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp']) ),
					np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu']) ),
					np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp', 'raw']) ),
					np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu', 'raw']) ) 
					]))
				
			mcf_list.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' )))
			# final regressor captures instruction-related variance that may otherwise cause strong responses in periphery
			# trial_times are single events that have to still be convolved with HRF
			# trial_times_list.extend([[[(j * nii_file.rtime * nii_file.timepoints) + tt[1] -1.5, 0.5, 1.0]] for tt in r.trial_times]) # changed the occurrence of this event to -4.5 to -1.5...
			button_times_list.extend([[[(j * nii_file.rtime * nii_file.timepoints) + float(tt[1]), 0.5, 1.0]] for tt in r.all_button_times]) # changed the occurrence of this event to -4.5 to -1.5...
			# lateron, this will also have pupil size and the occurrence of saccades in there.
			
			
			total_trs += nii_file.timepoints
		
		# to arrays with these regressors
		mcf_list = np.vstack(mcf_list).T
		physio_list = np.hstack(physio_list)
		
		# check for weird nans and throw out those columns
		physio_list = physio_list[-np.array(np.isnan(physio_list).sum(axis = 1), dtype = bool),:]
		# shell()
		# create a design matrix and convolve 
		run_design = Design(total_trs, nii_file.rtime, subSamplingRatio = 10)
		# run_design.configure(trial_times_list)
		run_design.configure([np.array(button_times_list).squeeze()])
		joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, mcf_list, physio_list]).T)
		# joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, mcf_list]).T)
		# joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, physio_list]).T)
		# only using the mc and physio now
		# joined_design_matrix = np.mat(np.vstack([mcf_list, physio_list]).T)
		# joined_design_matrix = np.mat(physio_list.T)
		# shell()
		# shell()
		self.logger.info('nuisance and trial_onset design_matrix of dimensions %s'%(str(joined_design_matrix.shape)))
		# take data
		data_list = []
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			# self.logger.info('per-condition Z-score of run %s' % r)
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			data_list.append(nii_file.data[:,cortex_mask])
		data_list = np.vstack(data_list)
		# now we run the GLM
		self.logger.info('nifti data loaded from %s for nuisance/trial onset analysis'%(self.runFile(stage = 'processed/mri', run = r, postFix = postFix )))
		betas = ((joined_design_matrix.T * joined_design_matrix).I * joined_design_matrix.T) * np.mat(data_list.T).T
		residuals = data_list - (np.mat(joined_design_matrix) * np.mat(betas))
		
		self.logger.info('GLM finished; outputting data to %s and %s'%(os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res']))[-1], os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['betas']))[-1]))
		# and now, back to image files
		for i, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			output_data_res = np.zeros(nii_file.data.shape, dtype = np.float32)
			output_data_res[:,cortex_mask] = residuals[i*nii_file.data.shape[0]:(i+1)*nii_file.data.shape[0],:]
			
			res_nii_file = NiftiImage(output_data_res)
			res_nii_file.header = nii_file.header
			res_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res']))
			
			output_data_betas = np.zeros([betas.shape[0]]+list(cortex_mask.shape), dtype = np.float32)
			output_data_betas[:,cortex_mask] = betas
			
			betas_nii_file = NiftiImage(output_data_betas)
			betas_nii_file.header = nii_file.header
			betas_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['betas']))
			
		# shell()	

	def GLM_for_nuisances_per_run(self):
		"""GLM_for_nuisances takes a diverse set of nuisance regressors,
		runs a GLM on them in order to run further PRF analysis on the 
		residuals after GLM. It assumes physio, motion correction and 
		stimulus_timings have been run beforehand, as it uses the output
		text files of these procedures.
		"""
		self.stimulus_timings()
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		for j, r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] ))
			self.logger.info('nifti data loaded from %s for nuisance/trial onset analysis'%(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] )))
			# moco and physiology regressors are per-TR regressors that need no convolution anymore.
			physio = np.array([
				np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp']) ),
				np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu']) ) 
				])
				
			mcf = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' ))
			# final regressor captures instruction-related variance that may otherwise cause strong responses in periphery
			# trial_times are single events that have to still be convolved with HRF
			instruct_times = [[[tt[1] - 3.0, 3.0, 1.0]] for tt in r.trial_times]
			trial_onset_times = [[[tt[1], 0.5, 1.0]] for tt in r.trial_times]
			# lateron, this will also have pupil size and the occurrence of saccades in there.
			
			run_design = Design(nii_file.timepoints, nii_file.rtime, subSamplingRatio = 10)
			run_design.configure(np.vstack([instruct_times, trial_onset_times]), hrfType = 'doubleGamma', hrfParameters = {'a1' : 6, 'a2' : 12, 'b1' : 0.9, 'b2' : 0.9, 'c' : 0.35})
			joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, mcf.T, physio]).T)
			
			f = pl.figure(figsize = (10, 10))
			s = f.add_subplot(111)
			pl.imshow(joined_design_matrix)
			s.set_title('nuisance design matrix')
			pl.savefig(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'], base = 'nuisance_design', extension = '.pdf' ))
			
			self.logger.info('nuisance and trial_onset design_matrix of dimensions %s for run %s'%(str(joined_design_matrix.shape), r))
			betas = ((joined_design_matrix.T * joined_design_matrix).I * joined_design_matrix.T) * np.mat(nii_file.data[:,cortex_mask].T).T
			residuals = nii_file.data[:,cortex_mask] - (np.mat(joined_design_matrix) * np.mat(betas))
			
			self.logger.info('nuisance GLM finished; outputting residuals to %s'%os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'res']))[-1])
			output_data_res = np.zeros(nii_file.data.shape, dtype = np.float32)
			output_data_res[:,cortex_mask] = residuals
			
			res_nii_file = NiftiImage(output_data_res)
			res_nii_file.header = nii_file.header
			res_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'res']))
			
			self.logger.info('nuisance GLM finished; outputting betas to %s'%os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'betas']))[-1])
			output_data_betas = np.zeros([betas.shape[0]]+list(cortex_mask.shape), dtype = np.float32)
			output_data_betas[:,cortex_mask] = betas
			
			betas_nii_file = NiftiImage(output_data_betas)
			betas_nii_file.header = nii_file.header
			betas_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf', 'betas']))
		
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

	
	def design_matrix(self, method = 'hrf', gamma_hrfType = 'singleGamma', gamma_hrfParameters = {'a': 6, 'b': 0.9}, fir_ratio = 6, n_pixel_elements = 40, sample_duration = 0.6, plot_diagnostics = False, ssr = 5, condition = 'PRF', save_design_matrix = False):
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

		if save_design_matrix:
			with open(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'design_matrix_%1.1f_%ix%i_%s.pickle'%(sample_duration, n_pixel_elements, n_pixel_elements, method)), 'w') as f:
				pickle.dump({'tr_time_list' : self.tr_time_list, 'full_design_matrix' : self.full_design_matrix, 'sample_time_list' : self.sample_time_list, 'trial_start_list' : self.trial_start_list} , f)
		
	def stats_to_mask(self, mask_file_name, postFix = ['mcf', 'sgtf', 'prZ', 'res'], condition = 'PRF', task_condition = ['all'], threshold = 5.0):
		"""stats_to_mask takes the stats from an initial fitting and converts it to a anatomical mask, and places it in the masks/anat folder"""
		input_file = os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_condition[0] + '.nii.gz')
		p_values = NiftiImage(input_file).data[1] > threshold
		self.logger.info('statistic mask created for threshold %2.2f, resulting in %i voxels' % (threshold, int(p_values.sum())))
		output_image = NiftiImage(np.array(p_values, dtype = np.int16))
		output_image.header = NiftiImage(input_file).header
		output_image.save(os.path.join(self.stageFolder('processed/mri/masks/anat'), mask_file_name + '_' + task_condition[0] + '.nii.gz'))

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
				these_samples = np.array([np.argmin(np.abs(self.sample_time_list - t)) for t in these_tr_times[selected_tr_times]]) 
				this_design_matrix = np.array(self.full_design_matrix[these_samples,:], dtype = np.float64, order = 'F')
				if save_all_data:
					save_design_matrix = np.zeros((this_design_matrix.shape[0],n_pixel_elements * n_pixel_elements))
					save_design_matrix[:,valid_regressors] = this_design_matrix
					np.save(os.path.join(self.stageFolder('processed/mri/PRF/'), 'design_matrix_%ix%i_%s'%(n_pixel_elements, n_pixel_elements, task_conditions[0])), save_design_matrix)
				
				# shell()
				# loop across voxels in this slice in parallel using joblib, 
				# fitBayesianRidge returns coefficients of results, and spearman correlation R and p as a 2-tuple
				self.logger.info('starting fitting of slice %d, with %d voxels and %d timepoints' % (sl, int((cortex_mask * voxels_in_this_slice_in_full).sum()), int(these_samples.shape[0])))
				res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitBayesianRidge)(self.full_design_matrix[these_samples,:], vox_timeseries) for vox_timeseries in these_voxels[:,selected_tr_times])
				# res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitRidge)(self.full_design_matrix[these_samples,:], vox_timeseries, alpha = 1e6) for vox_timeseries in these_voxels)
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
		coef_nii_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'coefs_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '-' + condition + '.nii.gz'))
		
		# replace infs in correlations with the maximal value of the rest of the array.
		all_corrs[np.isinf(all_corrs)] = all_corrs[-np.isinf(all_corrs)].max() + 1.0
		corr_nii_file = NiftiImage(all_corrs)
		corr_nii_file.header = mask_file.header
		corr_nii_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '-' + condition + '.nii.gz'))
	
		if save_all_data:
			all_data = np.zeros([selected_tr_times.sum()] + list(cortex_mask.shape))
			all_data[:,cortex_mask] = z_data[selected_tr_times]
			
			data_nii_file = NiftiImage(all_data)
			data_nii_file.header = mask_file.header
			data_nii_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'data_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '-' + condition + '.nii.gz'))
			
			np.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'data_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '-' + condition), z_data[selected_tr_times])
	
	
	
	def results_to_surface(self, file_name = 'corrs_cortex', output_file_name = 'polar', frames = {'_f':1}, smooth = 0.0, condition = 'PRF'):
		"""docstring for results_to_surface"""
		vsO = VolToSurfOperator(inputObject = os.path.join(self.stageFolder('processed/mri/%s/'%condition), file_name + '.nii.gz'))
		ofn = os.path.join(self.stageFolder('processed/mri/%s/surf/'%condition), output_file_name )
		vsO.configure(frames = frames, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = smooth, surfType = 'paint'  )
		vsO.execute()
	
	def mask_results_to_surface(self, stat_file = '', value_file= '', threshold = 5.0, stat_frame = 1, fill_value = -3.15):
		"""Mask the results, polar.nii.gz, for instance, with the threshold and convert to surface format for easy viewing"""
		if not os.path.isfile(os.path.join(self.stageFolder('processed/mri/%s/'%condition), stat_file + '.nii.gz')) or not os.path.isfile(os.path.join(self.stageFolder('processed/mri/%s/'%condition), value_file + '.nii.gz')):
			self.logger.error('images for mask_results_to_surface %s %s are not files' % (stat_file, value_file))
		
		val_data = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), value_file + '.nii.gz')).data
		stat_mask = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), stat_file + '.nii.gz')).data[stat_frame] < threshold
		val_data[:,stat_mask] = fill_value
		
		op_nii_file = NiftiImage(val_data)
		op_nii_file.header = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), value_file + '.nii.gz')).header
		op_nii_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), value_file + '_%2.2f.nii.gz'%threshold) )
		
		self.results_to_surface(file_name = value_file + '_%2.2f'%threshold, output_file_name = condition, frames = {'_polar':0, '_ecc':1, '_real':2, '_imag':3, 'surf': 4})
		
	
	def RF_fit(self, mask_file = 'cortex_dilated_mask', postFix = ['mcf','sgtf','prZ','res'], task_condition = 'all', anat_mask = 'cortex_dilated_mask', stat_threshold = -10.0, n_jobs = 28, run_fits = True, condition = 'PRF'):
		"""select_voxels_for_RF_fit takes the voxels with high stat values
		and tries to fit a PRF model to their spatial selectivity profiles.
		it takes the images from the mask_file result file, and uses stat_threshold
		to select all voxels crossing a p-value (-log10(p)) threshold.
		"""
		
		anat_mask = os.path.join(self.stageFolder('processed/mri/'), 'masks', 'anat', anat_mask + '.nii.gz')
		filename = mask_file + '_' + '_'.join(postFix + [task_condition]) + '-%s'%condition
		if run_fits:
			stats_data_cond = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + filename + '.nii.gz')).data
			spatial_data_cond = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'coefs_' + filename + '.nii.gz')).data
			spatial_data_fix = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'coefs_cortex_dilated_mask_all_mcf_sgtf_prZ_res_fix-PRF.nii.gz')).data
			stats_data_fix = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_cortex_dilated_mask_all_mcf_sgtf_prZ_res_fix-PRF.nii.gz')).data

			anat_mask_data = NiftiImage(anat_mask).data > 0
			
			stat_mask_cond = stats_data_cond[1] > stat_threshold
			stat_mask_fix = stats_data_fix[1] > stat_threshold
			
			voxel_spatial_data_to_fit_cond = spatial_data_cond[:,stat_mask_cond * anat_mask_data]
			voxel_spatial_data_to_fit_fix = spatial_data_fix[:,stat_mask_fix * anat_mask_data]
			self.logger.info('starting fitting of prf shapes')
			# res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(analyze_PRF_from_spatial_profile)(vox_spatial_data_cond, vox_spatial_data_fix, diagnostics_plot = False, band = 75, voxel_no = i, cond=cond) for i, vox_spatial_data in enumerate(voxel_spatial_data_to_fit.T))
			for i, vox_spatial_data in enumerate(voxel_spatial_data_to_fit.T):
				analyze_PRF_from_spatial_profile(vox_spatial_data_cond, vox_spatial_data_fix, diagnostics_plot = False, band = 75, voxel_no = i, cond=cond) 

	
			max_comp = np.array(res)[:,0]
			surf = np.real(res)[:,1]
		
			polar = np.angle(max_comp)
			ecc = np.abs(max_comp)
			real = np.real(max_comp)
			imag = np.imag(max_comp)
			
			# shell()
			
			prf_res = np.vstack([polar, ecc, real, imag, surf])
			
			empty_res = np.zeros([5] + [np.array(stats_data.shape[1:]).prod()])
			empty_res[:,(stat_mask * anat_mask_data).ravel()] = prf_res
			
			all_res = empty_res.reshape([5] + list(stats_data.shape[1:]))
		
			self.logger.info('saving prf parameters to polar and ecc')
		
			all_res_file = NiftiImage(all_res)
			all_res_file.header = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + filename + '.nii.gz')).header
			all_res_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'results_' + filename + '.nii.gz'))
		
		self.logger.info('converting prf values to surfaces')
		for sm in [0,2,5]: # different smoothing values.
			# reproject the original stats
			self.results_to_surface(file_name = 'corrs_' + filename, output_file_name = 'corrs_' + filename + '_' + str(sm), frames = {'_f':1}, smooth = sm, condition = condition)
			# and the spatial values
			self.results_to_surface(file_name = 'results_' + filename, output_file_name = 'results_' + filename + '_' + str(sm), frames = {'_polar':0, '_ecc':1, '_real':2, '_imag':3, '_surf':4}, smooth = sm, condition = condition)
			
			# but now, we want to do a surf to vol for the smoothed real and imaginary numbers.
			self.surface_to_polar(filename = os.path.join(self.stageFolder('processed/mri/%s/surf/'%condition), 'results_' + filename + '_' + str(sm) ))

			
	def surface_to_polar(self, filename, condition = 'PRF'):
		"""surface_to_polar takes a (smoothed) surface file for both real and imaginary parts and re-converts it to polar and eccentricity angle."""
		self.logger.info('converting %s from (smoothed) surface to nii back to surface')
		for hemi in ['lh','rh']:
			for component in ['real', 'imag']:
				svO = SurfToVolOperator(inputObject = filename + '_' + component + '-' + hemi + '.mgh' )
				svO.configure(templateFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[condition][0]], postFix = ['mcf']), hemispheres = [hemi], register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), fsSubject = self.subject.standardFSID, outputFileName = filename + '_' + component + '.nii.gz', threshold = 0.5, surfType = 'paint')
				print svO.runcmd
				svO.execute()
				# shell()
				
			# now, there's a pair of imag and real nii files for this hemisphere. Let's open them and make polar and eccen phases before re-transforming to surface. 
			complex_values = NiftiImage(filename + '_real-' + hemi + '.nii.gz').data + 1j * NiftiImage(filename + '_imag-' + hemi + '.nii.gz').data
		
			comp = NiftiImage(np.array([np.angle(complex_values), np.abs(complex_values)]))
			comp.header = NiftiImage(filename + '_real-' + hemi + '.nii.gz').header
			comp.save(filename + '_polecc-' + hemi + '.nii.gz')
		
		# add the two polecc files together
		addO = FSLMathsOperator(filename + '_polecc-' + 'lh' + '.nii.gz')
		addO.configureAdd(add_file = filename + '_polecc-' + 'rh' + '.nii.gz', outputFileName = filename + '_polecc.nii.gz')
		addO.execute()
		
		# self.results_to_surface(file_name = filename + '_polecc.nii.gz', output_file_name = filename, frames = , smooth = 0)
		vsO = VolToSurfOperator(inputObject = os.path.join(self.stageFolder('processed/mri/PRF/'), filename + '_polecc.nii.gz'))
		# ofn = os.path.join(self.stageFolder('processed/mri/PRF/surf/'), output_file_name )
		vsO.configure(frames = {'_polar':0, '_ecc':1}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = filename + '_sm', threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
		vsO.execute()

		
	
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
	
	def mask_stats_to_hdf(self, condition = 'PRF', mask_file = 'cortex_dilated_mask_all', postFix = ['mcf','sgtf','prZ','res']):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
		
		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri/%s'%condition), condition + '.hdf5')
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = open_file(self.hdf5_filename, mode = "w", title = condition + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = open_file(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		this_run_group_name = 'prf'
		try:
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = h5file.createGroup("/", this_run_group_name, '')
			
		stat_files = {}
		for c in ['fix','all','color','speed','sf','orient']:
			"""loop over runs, and try to open a group for this run's data"""
		
			"""
			Now, take different stat masks based on the run_type
			"""
			
			for res_type in ['results', 'coefs', 'corrs']:
				filename = mask_file + '_' + '_'.join(postFix + [c]) + '-' + condition
				stat_files.update({c+'_'+res_type: os.path.join(self.stageFolder('processed/mri/%s'%condition), res_type + '_' + filename + '.nii.gz')})
			
		
		stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
		
		for (roi, roi_name) in zip(rois, roinames):
			try:
				thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
				thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'ROI ' + roi_name +' imported' )
		
			for (i, sf) in enumerate(stat_files.keys()):
				# loop over stat_files and rois
				# to mask the stat_files with the rois:
				imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
				these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
				h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
		
		h5file.close()

	def prf_data_from_hdf(self, roi = 'v2d', condition = 'PRF', base_task_condition = 'fix', comparison_task_conditions = ['fix', 'color', 'sf', 'speed', 'orient'], corr_threshold = 0.1):
		self.logger.info('starting prf data correlations from region %s'%roi)
		results_frames = {'polar':0, 'ecc':1, 'real':2, 'imag':3, 'surf':4}
		stats_frames = {'corr': 0, '-logp': 1}

		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri/%s'%condition), condition + '.hdf5')
		h5file = open_file(self.hdf5_filename, mode = "r", title = condition + " file")

		# data to be correlated
		base_task_data = self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi, data_type = base_task_condition + '_results')
		all_comparison_task_data = [self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi, data_type = c + '_results') for c in comparison_task_conditions]

		# correlations on which to base the tasks
		base_task_corr = self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi, data_type = base_task_condition + '_corrs')
		all_comparison_task_corr = [self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi, data_type = c + '_corrs') for c in comparison_task_conditions]

		# shell()
		h5file.close()

		# create and apply the mask. 
		mask = base_task_corr[:,0] > corr_threshold
		mask = mask * (base_task_data[:,results_frames['ecc']] < 0.6)
		base_task_data, all_comparison_task_data = base_task_data[mask, :], np.array([ac[mask, :] for ac in all_comparison_task_data])
		base_task_corr, all_comparison_task_corr = base_task_corr[mask, 0], np.array([ac[mask, 0] for ac in all_comparison_task_corr])

		order = np.argsort(base_task_data[:,results_frames['ecc']])
		kern =  stats.norm.pdf( np.linspace(-.25,.25,int(round(base_task_data.shape[0] / 10)) ))
  		sm_ecc = np.convolve( base_task_data[:,results_frames['ecc']][order], kern / kern.sum(), 'valid' )  
		# shell()

		# scatter plots for results frames
		colors = [(c, 1-c, 1-c) for c in np.linspace(0.0,1.0,len(comparison_task_conditions))]
		mcs = ['o', 'v', 's', '>', '<']
		f = pl.figure(figsize = (8,7))
		for j, res_type in enumerate(['ecc']): # , 'surf'
			s = f.add_subplot(1,1,1+j)
			for i, tc in enumerate(comparison_task_conditions):
				pl.plot(base_task_data[:,results_frames['ecc']], all_comparison_task_data[i][:,results_frames[res_type]], c = colors[i], marker = 'o', linewidth = 0, alpha = 0.1, mec = 'w', ms = 1.5)
				sm_signal = np.convolve( all_comparison_task_data[i][:,results_frames[res_type]][order], kern / kern.sum(), 'valid' )
				pl.plot(sm_ecc, sm_signal, c = colors[i], linewidth = 3.5, alpha = 0.75, label = comparison_task_conditions[i] )
			s.set_title(roi + ' ' + res_type)

			if j == 1:
				s.set_ylim([0,15])
			else:
				s.set_ylim([0,0.65])
				leg = s.legend(fancybox = True, loc = 'best')
				leg.get_frame().set_alpha(0.5)
				if leg:
					for t in leg.get_texts():
					    t.set_fontsize('small')    # the legend text fontsize
					for l in leg.get_lines():
					    l.set_linewidth(3.5)  # the legend line width

			s.set_xlim([0,0.6])
			simpleaxis(s)
			spine_shift(s)
			s.set_xlabel('eccentricity of %s condition'%base_task_condition)
			s.set_ylabel(res_type)


		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/'), 'figs', roi + '.pdf'))
		



