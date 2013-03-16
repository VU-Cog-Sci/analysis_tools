#!/usr/bin/env python
# encoding: utf-8
"""
exp.py

Created by Jan Willem de Gee on 2011-02-16.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
import os, sys, datetime
import subprocess, logging

thisFolder = '/Research/Pupil/PupilExperiment1/'
analysisFolder = os.path.join(thisFolder, 'analysis')
sys.path.append( analysisFolder )
sys.path.append( os.environ['ANALYSIS_HOME'] )

from Tools.Sessions import *
import Tools.Project as Project
from Tools.Subjects.Subject import *
from Tools.Run import *
from Tools import functions_jw

from Tools.Operators.EyeOperator import EyelinkOperator

import scipy as sp
import scipy.stats as stats
import scipy.signal as signal
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import bottleneck
import pandas as pd

from subprocess import *
from pylab import *
from numpy import *
from tables import *
from math import *
from pypsignifit import *
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate
from scipy.signal import butter, filtfilt

import statsmodels.api as sm

import rpy2.robjects as robjects
import rpy2.rlike.container as rlc

from IPython import embed as shell

class preprocessing():
	
	"""
	Make_HDF5:
	- Make HDF5 from raw edf files. All runs are put in one hdf5 file.
	
	Preprocessing:
	- Detect ITI's (+1 seconds; every trial started with 2.5-4.5 s delay, so no useful info in first second of trial), and make pupil diameter 0.
	- Detect blinks, and make pupil diameter 0.
	- Detect whether ITI and blinks are within 0.5 s of each other. If so, merge them by making 0.  
	- Remove 'zero-gaps' (blinks + ITI) by linear interpolation. 
	- High pass, low pass, band pass pupil diameter.
	- Detect omissions (per trial) based on eye movements.
	- Detect omissions (per trial) based on blinks.
	- Zscore pupil diameter.
	- Create time-locked arrays.
	- Make regressor files.
	"""
	
	def __init__(self, subject, experiment, version, this_dir, sample_rate, downsample_rate):
		self.subject = subject
		self.experiment = experiment
		self.version = version
		self.this_dir = this_dir
		self.sample_rate = sample_rate
		self.downsample_rate = downsample_rate
	
	
	def make_HDF5(self):
		### MAKE HDF5 ###
		
		# what files are there to analyze?
		os.chdir(self.this_dir)
		edf_files = subprocess.Popen('ls ' + self.subject + '*.edf', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0:-1]
		msg_files = subprocess.Popen('ls ' + self.subject + '*.msg', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0:-1]
		
		edf_files_no_ext = [f.split('.edf')[0] for f in edf_files]
		msg_files_no_ext = [f.split('.msg')[0] for f in msg_files]
		
		if self.experiment == 1:
			order = np.argsort(np.arange(0,len(edf_files))[np.array([int(e.split(self.subject)[1].split('_')[0]) for e in edf_files]) - 1])
		if self.experiment == 2:
			if self.subject == 'td':
				order = np.array([9,0,1,2,3,4,5,6,7,8])
			else:
				order = np.argsort(np.arange(0,len(edf_files)))
		
		eyelink_fos = []
		for f in order:
		# check which files are already split - and get their data. 
		# if they haven't been split, split then and then import their data
			if edf_files_no_ext[f] not in msg_files_no_ext:
				elo = EyelinkOperator( inputObject = edf_files_no_ext[f]+'.edf', split = True )
			else:
				elo = EyelinkOperator( inputObject = edf_files_no_ext[f]+'.edf', split = False )
				# elo.self.hdf5_filename = os.path.join(os.path.split(elo.inputObject)[0])
			elo.loadData(get_gaze_data = True)
			eyelink_fos.append(elo)
		
		for (i, elo) in enumerate(eyelink_fos):
			elo.processIntoTable(os.path.join(self.this_dir, self.subject + '.hdf5'), name = 'run_' + str(i), compute_velocities = False)
			elo.import_parameters()
	
	
	def preproces_run(self, this_run):
		
		self.this_run = this_run
		os.chdir(self.this_dir)
		
		h5f = openFile((self.subject + '.hdf5'), mode = "r+" ) # mode = "r" means 'read only'.
		search_name = 'run_' + str(self.this_run)
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if search_name == r._v_name:
				run = r
				break
		
		## READ DATA
		trial_times = run.trial_times.read()
		gaze_timestamps = run.gaze_data.read()[:,0] 
		raw_pupil_diameter = run.gaze_data.read()[:,3]
		trial_parameters = run.trial_parameters.read()
		blink_data = run.blinks_from_EL.read()
		
		# Compute time of first trial interval, and make raw_pupil_diameter equal to mean raw_pupil_diameter. We do this because we do not want to start with 0's. 
		times_iti_first = (gaze_timestamps > trial_times['trial_start_EL_timestamp'][0]) * (gaze_timestamps < trial_times['trial_phase_timestamps'][0,0,0]+1000)
		times_iti_first[0:1999] = True
		raw_pupil_diameter[times_iti_first] = bottleneck.nanmean(raw_pupil_diameter[(gaze_timestamps > trial_times['trial_phase_timestamps'][0,1,0] - 1000) * (gaze_timestamps < trial_times['trial_phase_timestamps'][0,2,0])])
		
		###################################################################################################################
		## DETECT BLINKS AND MISSING DATA! ################################################################################
		
		# lost data will appear as a 0.0 instead of anything sensible, and is not detected as blinks. 
		# we will detect them, string them together so that they coalesce into reasonable events, and
		# add them to the blink array.	
		zero_edges = np.arange(raw_pupil_diameter.shape[0])[np.diff(( raw_pupil_diameter < 0.1 ))]
		# zero_edges = zero_edges[1:]
		if zero_edges.shape[0] == 0:
			pass
		else:
			zero_edges = zero_edges[:int(2 * floor(zero_edges.shape[0]/2.0))].reshape(-1,2)
			zero_edges = zero_edges[np.argsort(zero_edges[:,1])]
			new_ze = [zero_edges[0]]
			for ze in zero_edges[1:]:
				if (ze[0] - new_ze[-1][-1])/self.sample_rate < 0.1:
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
			
		## Set all blinks and zero edges to 0!
		for bl in range(blink_data.shape[0]):
			raw_pupil_diameter[(gaze_timestamps>blink_data[bl][1])*(gaze_timestamps<blink_data[bl][3])] = 0
		
		## Set start of run back to mean (we don't want to start with zeros.)
		raw_pupil_diameter[times_iti_first] = bottleneck.nanmean(raw_pupil_diameter[(gaze_timestamps > trial_times['trial_phase_timestamps'][0,1,0] - 1000) * (gaze_timestamps < trial_times['trial_phase_timestamps'][0,2,0])])
		
		###################################################################################################################
		## DETECT ZERO EDGES! (we just created) ###########################################################################
		
		zero_edges = np.arange(raw_pupil_diameter.shape[0])[np.diff(( raw_pupil_diameter < 0.1 ))]
		# zero_edges = zero_edges[1:]
		if zero_edges.shape[0] == 0:
			pass
		else:
			zero_edges = zero_edges[:int(2 * floor(zero_edges.shape[0]/2.0))].reshape(-1,2)
			zero_edges = zero_edges[np.argsort(zero_edges[:,1])]
			new_ze = [zero_edges[0]]
			for ze in zero_edges[1:]:
				if (ze[0] - new_ze[-1][-1])/self.sample_rate < 0.1:
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
		
		blink_data = ze_to_blinks
		
		###################################################################################################################
		## LINEAR INTERPOLATION OF BLINKS AND ITI's! ######################################################################
		
		# these are the time points (defined in samples for now...) from which we take the levels on which to base the interpolation
		points_for_interpolation = np.array([[-100],[100]])
		interpolation_time_points = np.zeros((blink_data.shape[0], points_for_interpolation.ravel().shape[0]))
		interpolation_time_points[:,[0]] = np.tile(blink_data['start_timestamp'], 1).reshape((1,-1)).T
		interpolation_time_points[:,[1]] = np.tile(blink_data['end_timestamp'], 1).reshape((1,-1)).T
		# blinks may start or end before or after sampling has begun or stopped
		interpolation_time_points = np.where(interpolation_time_points < gaze_timestamps[-1], interpolation_time_points, gaze_timestamps[-1])
		interpolation_time_points = np.where(interpolation_time_points > gaze_timestamps[0], interpolation_time_points, gaze_timestamps[0])
		# apparently the above doesn't actually work, and resulting in nan interpolation results. We'll just throw these out.
		# this last-ditch attempt rule might work out the nan errors after interpolation
		interpolation_time_points = np.array([itp for itp in interpolation_time_points if ((itp == gaze_timestamps[0]).sum() == 0) and ((itp == gaze_timestamps[-1]).sum() == 0)])
		# itp = itp[(itp != gaze_timestamps[0]) + (itp != gaze_timestamps[-1])]
		# correct for the fucking eyelink not keeping track of fucking time
		# interpolation_time_points = np.array([[np.arange(gaze_timestamps.shape[0])[gaze_timestamps >= interpolation_time_points[i,j]][0] for j in range(points_for_interpolation.ravel().shape[0])] for i in range(interpolation_time_points.shape[0])])
		# convert everything to indices
		interpolation_time_points = np.array([[np.arange(gaze_timestamps.shape[0])[gaze_timestamps >= interpolation_time_points[i,j]][0] for j in range(points_for_interpolation.ravel().shape[0])] for i in range(interpolation_time_points.shape[0])])
		
		# Check for neighbouring blinks! (wihtin 250 ms):
		double1 = np.ones(interpolation_time_points.shape[0], dtype = bool)
		double2 = np.ones(interpolation_time_points.shape[0], dtype = bool)
		
		for bl in range(len(interpolation_time_points)):
			try:
				if (interpolation_time_points[bl+1,0] - interpolation_time_points[bl,1]) <= 250:
					double1[bl+1] = False
				if (interpolation_time_points[bl+1,0] - interpolation_time_points[bl,1]) <= 250:
					double2[bl] = False
				
				# print( int(a[bl+1,0] - a[bl,1]) )
			
			except IndexError:
				pass
		
		
		interpolation_time_points2a = interpolation_time_points[double1,0]
		interpolation_time_points2b = interpolation_time_points[double2,1]
		interpolation_time_points2 = np.zeros((sum(double1),2))
		interpolation_time_points2[:,0] = interpolation_time_points2a - 100
		interpolation_time_points2[:,1] = interpolation_time_points2b + 100
		interpolation_time_points2 = np.array(interpolation_time_points2, dtype = int)
		
		if interpolation_time_points2[-1,1] > gaze_timestamps.shape[0]:
			interpolation_time_points2[-1,1] = gaze_timestamps.shape[0]-1
		
		for itp in interpolation_time_points2:
			# interpolate
			# spline = interpolate.InterpolatedUnivariateSpline(itp,raw_pupil_diameter[itp])
			lin = interpolate.interp1d(itp,raw_pupil_diameter[itp])
		
			# replace with interpolated data
			raw_pupil_diameter[itp[0]:itp[-1]] = lin(np.arange(itp[0],itp[-1]))
		
		###################################################################################################################
		## FILTER THE SIGNAL! #############################################################################################
		
		# High pass:
		hp_frequency = 0.05
		hp_cof_sample = hp_frequency / (raw_pupil_diameter.shape[0] / self.sample_rate / 2)
		bhp, ahp = butter(3, hp_cof_sample, btype='high')
		hp_c_raw_pupil_diameter = filtfilt(bhp, ahp, raw_pupil_diameter)
		# Low pass:
		lp_frequency = 4.0
		lp_cof_sample = lp_frequency / (raw_pupil_diameter.shape[0] / self.sample_rate / 2)
		blp, alp = butter(3, lp_cof_sample)
		lp_c_raw_pupil_diameter = filtfilt(blp, alp, raw_pupil_diameter)
		# Band pass:
		lp_hp_c_raw_pupil_diameter = filtfilt(blp, alp, hp_c_raw_pupil_diameter)
		# Linear detrending low passed data:
		raw_pupil_diameter_detrend_lin = sm.tsa.detrend(lp_c_raw_pupil_diameter, order = 1)
		# Polynomial detrending low passed data:
		raw_pupil_diameter_detrend_poly = sm.tsa.detrend(lp_c_raw_pupil_diameter, order = 2)
		
		###################################################################################################################
		## OMISSIONS BASED ON EYEMOVEMENTS! ###############################################################################
		
		# Trials are excluded if (1) large saccades ->150px- happened in decision interval, or (2) small saccades ->75px- happened in more than 10% of decision interval.
		
		x = run.gaze_data.read()[:,1]
		y = run.gaze_data.read()[:,2]
		middle_x = 0
		middle_y = 0
		cut_off = 75
		
		## Times decision interval:
		timesss = []
		duration = 0
		for j in range(len(trial_parameters['trial_nr'])):
			timesss.append((gaze_timestamps > trial_times['trial_phase_timestamps'][j,1,0]) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,2,0]))
			if sum(timesss[j]) > duration:
				duration = sum(timesss[j])
		
		omission_indices_sac = np.zeros( len(trial_parameters['trial_nr']), dtype = bool)
		
		x_matrix = []
		y_matrix = []
		for i in range(len(trial_parameters['trial_nr'])):
			x_matrix.append(x[timesss[i]]) 
			x_matrix[i] = x_matrix[i] - bottleneck.nanmean(x_matrix[i])
			y_matrix.append(y[timesss[i]]) 
			y_matrix[i] = y_matrix[i] - bottleneck.nanmean(y_matrix[i])
		
			if ((x_matrix[i] < -175).sum() > 0) or ((x_matrix[i] > 175).sum() > 0):
				omission_indices_sac[i] = True
			if ( ( (x_matrix[i] > (middle_x + cut_off)).sum() + (x_matrix[i] < (middle_x - cut_off)).sum() ) / float(sum(timesss[i])) * 100.0 ) > 10:
				omission_indices_sac[i] = True
			if ((y_matrix[i] < -175).sum() > 0) or ((y_matrix[i] > 175).sum() > 0):
				omission_indices_sac[i] = True
			if ( ( (y_matrix[i] > (middle_y + cut_off)).sum() + (y_matrix[i] < (middle_y - cut_off)).sum() ) / float(sum(timesss[i])) * 100.0 ) > 10:
				omission_indices_sac[i] = True
		
		try:
			run.omission_indices_sac.remove()
		except NodeError, LeafError:
			pass
		h5f.createArray(run, 'omission_indices_sac', omission_indices_sac, 'omission indices based on eye-movements')
		
		###################################################################################################################
		## OMISSIONS BASED ON BLINKS! #####################################################################################
		
		# Trials are excluded if blink happened in timewindow: 0.5s before decision interval - 0.5s after decision interval.
		
		omission_indices_blinks = np.zeros( len(trial_parameters['trial_nr']), dtype = bool)
		for i in range(len(trial_parameters['trial_nr'])):
			for j in range(len(blink_data)):
				if blink_data[j][1] > trial_times['trial_phase_timestamps'][i,1,0]-500:
					if blink_data[j][3] < trial_times['trial_phase_timestamps'][i,2,0]+500:
						omission_indices_blinks[i] = True
		
		try:
			run.omission_indices_blinks.remove()
		except NodeError, LeafError:
			pass
		h5f.createArray(run, 'omission_indices_blinks', omission_indices_blinks, 'omission indices based on blinks')
		
		###################################################################################################################
		## SORT TRIALS IN SEPERATE SDT CATEGORIES! ########################################################################
		
		if self.experiment == 1:
			omission_indices = np.array([(trial_parameters['answer'] == 0) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) + (trial_parameters['confidence'] == -1)
		if self.experiment == 2:
			omission_indices = np.array([(trial_parameters['answer'] == 0) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool)
		omission_indices = omission_indices + omission_indices_sac + omission_indices_blinks
		omission_indices[0:2] = True
		print(sum(omission_indices))
		
		target_indices = np.array([(trial_parameters['target_present_in_stimulus'] == 1) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) * [-omission_indices]
		no_target_indices = np.array([(trial_parameters['target_present_in_stimulus'] == 0) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) * [-omission_indices]
		correct_indices = np.array([(trial_parameters['correct'] == 1) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) * [-omission_indices]
		incorrect_indices = np.array([(trial_parameters['correct'] == 0) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) * [-omission_indices]
		if self.version == 1:
			answer_yes_indices = np.array([(trial_parameters['answer'] == -1) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) * [-omission_indices]
			answer_no_indices = np.array([(trial_parameters['answer'] == 1) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) * [-omission_indices]
		if self.version == 2:
			answer_yes_indices = np.array([(trial_parameters['answer'] == 1) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) * [-omission_indices]
			answer_no_indices = np.array([(trial_parameters['answer'] == -1) for trial_numbers in trial_parameters]).sum(axis = 0, dtype = bool) * [-omission_indices]
		hit_indices = target_indices * answer_yes_indices
		fa_indices = no_target_indices * answer_yes_indices
		cr_indices = no_target_indices * answer_no_indices
		miss_indices = target_indices * answer_no_indices
		
		try:
			run.SDT_indices.remove()
		except NodeError, LeafError:
			pass
		
		bla = [hit_indices[0], fa_indices[0], miss_indices[0], cr_indices[0]]
		h5f.createArray(run, 'SDT_indices', bla, 'hit indices, fa indices, miss indices, cr indices')
		
		###################################################################################################################
		## Z-SCORE THE FILTERED SIGNALS! ##################################################################################
		
		# Window of interest:
		include_indices = np.zeros(gaze_timestamps.shape, dtype = bool)
		include_start_end_indices = np.array([[(include['trial_phase_timestamps'][1,0]-500), (include['trial_phase_timestamps'][2,0]+1500)] for include in trial_times[-omission_indices]], dtype = int)
		for bl in include_start_end_indices:
			include_indices[ (np.argmax( ((gaze_timestamps== bl[0])+(gaze_timestamps== bl[0]+1)) )) : (np.argmax( ((gaze_timestamps== bl[1])+(gaze_timestamps== bl[1]+1)) )) ] = True
		# Low pass (z-score based on window of interest):
		pupil_mean_lp = np.array(lp_c_raw_pupil_diameter[include_indices]).mean()
		pupil_std_lp = lp_c_raw_pupil_diameter[include_indices].std()
		pupil_zscore_lp = (lp_c_raw_pupil_diameter - pupil_mean_lp) / pupil_std_lp # Possible because vectorized.
		# High pass (z-score based on window of interest):
		pupil_mean_hp = np.array(hp_c_raw_pupil_diameter[include_indices]).mean()
		pupil_std_hp = hp_c_raw_pupil_diameter[include_indices].std()
		pupil_zscore_hp = (hp_c_raw_pupil_diameter - pupil_mean_hp) / pupil_std_hp # Possible because vectorized.
		# Band pass (z-score based on window of interest):
		pupil_mean_lp_hp = np.array(lp_hp_c_raw_pupil_diameter[include_indices]).mean()
		pupil_std_lp_hp = lp_hp_c_raw_pupil_diameter[include_indices].std()
		pupil_zscore_lp_hp = (lp_hp_c_raw_pupil_diameter - pupil_mean_lp_hp) / pupil_std_lp_hp # Possible because vectorized.
		# Lin detrended (z-score based on window of interest):
		pupil_mean_detrend_lin = np.array(raw_pupil_diameter_detrend_lin[include_indices]).mean()
		pupil_std_detrend_lin = raw_pupil_diameter_detrend_lin[include_indices].std()
		pupil_zscore_detrend_lin = (raw_pupil_diameter_detrend_lin - pupil_mean_detrend_lin) / pupil_std_detrend_lin # Possible because vectorized.
		# poly detrended (z-score based on window of interest):
		pupil_mean_detrend_poly = np.array(raw_pupil_diameter_detrend_poly[include_indices]).mean()
		pupil_std_detrend_poly = raw_pupil_diameter_detrend_poly[include_indices].std()
		pupil_zscore_detrend_poly = (raw_pupil_diameter_detrend_poly - pupil_mean_detrend_poly) / pupil_std_detrend_poly # Possible because vectorized.
		# For GLM 1 (band pass, z-score on COMPLETE signal):
		GLM_pupil_mean_lp_hp = np.array(lp_hp_c_raw_pupil_diameter).mean()
		GLM_pupil_std_lp_hp = lp_hp_c_raw_pupil_diameter.std()
		GLM_pupil_zscore_lp_hp = (lp_hp_c_raw_pupil_diameter - GLM_pupil_mean_lp_hp) / GLM_pupil_std_lp_hp # Possible because vectorized.
		# For GLM 2 (poly detrend, z-score on COMPLETE signal):
		GLM_pupil_mean_detrend_poly = np.array(raw_pupil_diameter_detrend_poly).mean()
		GLM_pupil_std_detrend_poly = raw_pupil_diameter_detrend_poly.std()
		GLM_pupil_zscore_detrend_poly = (raw_pupil_diameter_detrend_poly - GLM_pupil_mean_detrend_poly) / GLM_pupil_std_detrend_poly # Possible because vectorized.
		
		try:
			run.pupil_data_filtered.remove()
		except NodeError, LeafError:
			pass
		
		# Downsample and in hdf5 file:
		gaze_timestamps_down = gaze_timestamps[0::int(self.downsample_rate)]
		pupil_zscore_lp_down = sp.signal.decimate(pupil_zscore_lp,int(self.downsample_rate),1)
		pupil_zscore_hp_down = sp.signal.decimate(pupil_zscore_hp,int(self.downsample_rate),1)
		pupil_zscore_lp_hp_down = sp.signal.decimate(pupil_zscore_lp_hp,int(self.downsample_rate),1)
		pupil_zscore_detrend_lin_down = sp.signal.decimate(pupil_zscore_detrend_lin,int(self.downsample_rate),1)
		pupil_zscore_detrend_poly_down = sp.signal.decimate(pupil_zscore_detrend_poly,int(self.downsample_rate),1)
		GLM_pupil_zscore_lp_hp_down = sp.signal.decimate(GLM_pupil_zscore_lp_hp,int(self.downsample_rate),1)
		GLM_pupil_zscore_detrend_poly_down = sp.signal.decimate(GLM_pupil_zscore_detrend_poly,int(self.downsample_rate),1)
		
		all_filtered_pupil_data = np.vstack((gaze_timestamps_down, pupil_zscore_lp_down, pupil_zscore_hp_down, pupil_zscore_lp_hp_down, pupil_zscore_detrend_lin_down, pupil_zscore_detrend_poly_down, GLM_pupil_zscore_lp_hp_down, GLM_pupil_zscore_detrend_poly_down))
		h5f.createArray(run, 'pupil_data_filtered', all_filtered_pupil_data, '(0) gaze timestamps, z-scored pupil diameter (1) low passed, (2) high passed, (3) band passed, (4) lin detrend, (5) poly detrend, (6) GLM band passed, (7) GLM poly detrend')
		
		# # NOT downsampled stuff to hdf5 file:
		# all_filtered_pupil_data = np.vstack((gaze_timestamps, pupil_zscore_lp, pupil_zscore_hp, pupil_zscore_lp_hp))
		# h5f.createArray(run, 'pupil_data_filtered', all_filtered_pupil_data, 'Z-scored pupil diameter, low passed, high passed, low+high passed, low passed-zscored on full run')
		
		###################################################################################################################
		## COMPUTE D PRIME! ###############################################################################################
		
		d_prime, criterion = functions_jw.SDT_measures_per_subject(subject = self.subject, target_indices_joined = target_indices, no_target_indices_joined = no_target_indices, hit_indices_joined = hit_indices, fa_indices_joined = fa_indices)
		print('d-prime: ' + str(d_prime))
		print('criterion: ' + str(criterion))
		
		###################################################################################################################
		## CALCULATE DECISION TIMES! ######################################################################################
		
		decision_time = np.ones(shape((trial_parameters['trial_nr'])))
		for j in range(len(trial_parameters['trial_nr'])):
			decision_time[j] = ( sum( (gaze_timestamps > trial_times['trial_phase_timestamps'][j,1,0]) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,2,0]) ) )
		print('mean decision time: ' + str(mean(decision_time)))
		
		###################################################################################################################
		## MAKE TIME LOCKED ARRAY'S! ######################################################################################
		
		## MAKE STIMULUS LOCKED ARRAY'S
		timesss = []
		duration = 0
		for j in range(len(trial_parameters['trial_nr'])):
			timesss.append((gaze_timestamps > trial_times['trial_phase_timestamps'][j,1,0] - 1000) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,1,0]+5002))
			if sum(timesss[j]) > duration:
				duration = sum(timesss[j])
		a = np.zeros((len(trial_parameters['trial_nr']), duration))
		a[:,:] = NaN
		aa = np.zeros((len(trial_parameters['trial_nr']), duration))
		aa[:,:] = NaN
		aaa = np.zeros((len(trial_parameters['trial_nr']), duration))
		aaa[:,:] = NaN
		for j in range(len(trial_parameters['trial_nr'])):
			# Band pass stimulus locked:
			a[j,:sum(timesss[j])] = pupil_zscore_lp_hp[timesss[j]]
			# Low pass stimulus locked:
			aa[j,:sum(timesss[j])] = pupil_zscore_lp[timesss[j]]
			# Poly detrend stimulus locked:
			aaa[j,:sum(timesss[j])] = pupil_zscore_detrend_poly[timesss[j]]
		a = a[:,0:5999]
		aa = aa[:,0:5999]
		aaa = aaa[:,0:5999]
		stim = [a,aa,aaa]
		try:
			run.stimulus_locked_array.remove()
		except NodeError, LeafError:
			pass
		h5f.createArray(run, 'stimulus_locked_array', stim, 'stimulus locked zscored (0) band passed (0.05-4Hz), (1) low passed (4Hz), (2) poly detrend')
		
		## MAKE RESPONSE LOCKED ARRAY
		timesss = []
		duration = 0
		for j in range(len(trial_parameters['trial_nr'])):
			timesss.append((gaze_timestamps > trial_times['trial_phase_timestamps'][j,2,0] - 3500) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,2,0]+2502))
			if sum(timesss[j]) > duration:
				duration = sum(timesss[j])
		b = np.ones((len(trial_parameters['trial_nr']), duration))
		b[:,:] = NaN 
		bb = np.ones((len(trial_parameters['trial_nr']), duration))
		bb[:,:] = NaN
		bbb = np.ones((len(trial_parameters['trial_nr']), duration))
		bbb[:,:] = NaN
		for j in range(len(trial_parameters['trial_nr'])):
			# Band pass response locked:
			b[j,:sum(timesss[j])] = pupil_zscore_lp_hp[timesss[j]]
			# Low pass response locked:
			bb[j,:sum(timesss[j])] = pupil_zscore_lp[timesss[j]]
			# Poly detrend response locked:
			bbb[j,:sum(timesss[j])] = pupil_zscore_detrend_poly[timesss[j]]
		b = b[:,0:5999]
		bb = bb[:,0:5999]
		bbb = bbb[:,0:5999]
		resp = [b,bb,bbb]
		try:
			run.response_locked_array.remove()
		except NodeError, LeafError:
			pass
		h5f.createArray(run, 'response_locked_array', resp, 'response locked zscored (0) band passed (0.05-4Hz), (1) low passed (4Hz), (2) poly detrend')
		
		## MAKE FEEDBACK LOCKED ARRAY
		if self.experiment == 1:
			timesss = []
			duration = 0
			for j in range(len(trial_parameters['trial_nr'])):
				timesss.append((gaze_timestamps > trial_times['trial_phase_timestamps'][j,5,0] - 1000) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,5,0]+3002))
				if sum(timesss[j]) > duration:
					duration = sum(timesss[j])
			c = np.ones((len(trial_parameters['trial_nr']), duration))
			c[:,:] = NaN 
			cc = np.ones((len(trial_parameters['trial_nr']), duration))
			cc[:,:] = NaN
			ccc = np.ones((len(trial_parameters['trial_nr']), duration))
			ccc[:,:] = NaN
			for j in range(len(trial_parameters['trial_nr'])):
				# Band pass feedback locked:
				c[j,:sum(timesss[j])] = pupil_zscore_lp_hp[timesss[j]]
				# Low pass feedback locked:
				cc[j,:sum(timesss[j])] = pupil_zscore_lp[timesss[j]]
				# Poly detrend feedback locked:
				ccc[j,:sum(timesss[j])] = pupil_zscore_detrend_poly[timesss[j]]
			c = c[:,0:3999]
			cc = cc[:,0:3999]
			ccc = ccc[:,0:3999]
			feed = [c,cc,ccc]
			try:
				run.feedback_locked_array.remove()
			except NodeError, LeafError:
				pass
			h5f.createArray(run, 'feedback_locked_array', feed, 'feedback locked zscored (0) band passed (0.05-4Hz), (1) low passed (4Hz), (2) poly detrend')
		
		###################################################################################################################
		## MAKE GLM REGRESSORS! ###########################################################################################
		
		blink_data = run.blinks_from_EL.read()
		blink_on_times = np.zeros(blink_data.shape[0])
		blink_off_times = np.zeros(blink_data.shape[0])
		for i in range(blink_data.shape[0]):
			blink_on_times[i] = blink_data[i][1]
			blink_off_times[i] = blink_data[i][3]
		
		# Stick blink on
		blink_on = np.zeros((blink_on_times.shape[0],3))
		blink_on[:,0] = (blink_on_times - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
		blink_on[:,1] = 0.1
		blink_on[:,2] = np.ones(blink_on_times.shape[0])
		# Stick blink off
		blink_off = np.zeros((blink_off_times.shape[0],3))
		blink_off[:,0] = (blink_off_times - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
		blink_off[:,1] = 0.1
		blink_off[:,2] = np.ones(blink_off_times.shape[0])
		# Stick trial onset
		trial_on = np.zeros((trial_parameters.shape[0],3))
		trial_on[:,0] = (trial_times['trial_phase_timestamps'][:,0,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
		trial_on[:,1] = 0.1
		trial_on[:,2] = np.ones(trial_parameters.shape[0])
		# Stick stim onset
		stim_on = np.zeros((trial_parameters.shape[0],3))
		stim_on[:,0] = (trial_times['trial_phase_timestamps'][:,1,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
		stim_on[:,1] = 0.1
		stim_on[:,2] = np.ones(trial_parameters.shape[0])
		# Stick stim offset
		stim_off = np.zeros((trial_parameters.shape[0],3))
		stim_off[:,0] = (trial_times['trial_phase_timestamps'][:,2,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
		stim_off[:,1] = 0.1
		stim_off[:,2] = np.ones(trial_parameters.shape[0])
		# Ramp decision interval:
		decision_ramp = np.zeros((trial_parameters.shape[0],3))
		decision_ramp[:,0] = (trial_times['trial_phase_timestamps'][:,1,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate 
		decision_ramp[:,1] = (trial_times['trial_phase_timestamps'][:,2,0] - trial_times['trial_phase_timestamps'][:,1,0]) / self.downsample_rate
		decision_ramp[:,2] = np.ones(trial_parameters.shape[0])
		
		if self.experiment == 1:
			# Sticks confidence
			conf_on = np.zeros((trial_parameters.shape[0],3))
			conf_on[:,0] = (trial_times['trial_phase_timestamps'][:,4,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
			conf_on[:,1] = 0.1
			conf_on[:,2] = np.ones(trial_parameters.shape[0])
			# Sticks feedback
			feed_on = np.zeros((trial_parameters.shape[0],3))
			feed_on[:,0] = (trial_times['trial_phase_timestamps'][:,5,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
			feed_on[:,1] = 0.1
			feed_on[:,2] = np.ones(trial_parameters.shape[0])
			
			# Whole trials
			whole_trials = np.zeros((trial_parameters.shape[0],3))
			whole_trials[:,0] = (trial_times['trial_phase_timestamps'][:,0,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
			whole_trials[:,1] = (trial_times['trial_phase_timestamps'][:,6,0] - trial_times['trial_phase_timestamps'][:,0,0]) / self.downsample_rate
			whole_trials[:,2] = np.ones(trial_parameters.shape[0])
		if self.experiment == 2:
			whole_trials = np.zeros((trial_parameters.shape[0],3))
			whole_trials[:,0] = (trial_times['trial_phase_timestamps'][:,0,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
			whole_trials[:,1] = (trial_times['trial_phase_timestamps'][:,2,0] - trial_times['trial_phase_timestamps'][:,0,0]) / self.downsample_rate
			whole_trials[:,2] = np.ones(trial_parameters.shape[0])
		
		try:
			run.GLM_regressors.remove()
			run.GLM_blink_regressors.remove()
		except NodeError, LeafError:
			pass
		
		if self.experiment == 1:
			bla = [trial_on, stim_on, stim_off, decision_ramp, conf_on, feed_on, whole_trials]
			h5f.createArray(run, 'GLM_regressors', bla, 'regressors: (0) stick at trial onset (spacebar press), (1) stick at stim onset (tone), (2) stick at stim offset (response), (3) ramp decision interval, (4) stick at confidence (response), (5) stick at feedback, (6) whole trials')
		if self.experiment == 2:
			bla = [trial_on, stim_on, stim_off, decision_ramp, whole_trials]
			h5f.createArray(run, 'GLM_regressors', bla, 'regressors: (0) stick at trial onset (spacebar press), (1) stick at stim onset (tone), (2) stick at stim offset (response), (3) ramp decision interval, (4) whole trials')
		bla2 = [blink_on, blink_off]
		h5f.createArray(run, 'GLM_blink_regressors', bla2, 'regressors: (0) stick at blink onset, (2) stick at blink ofset')
		
		h5f.close()
	
	
	def preproces_for_GLM(self, this_run):
		
		len_IRF = 3000
		pupil_IRF = np.zeros(len_IRF)
		for i in range(len_IRF):
			pupil_IRF[i] = functions_jw.IRF_canonical(t=i)
		IRF_downsampled = sp.signal.decimate(pupil_IRF,int(self.downsample_rate),1)
		IRF = IRF_downsampled
		
		#################################################################################################
		# Set working directory and open hdf5 file
		self.this_run = this_run
		os.chdir(self.this_dir)
		
		h5f = openFile((self.subject + '.hdf5'), mode = "r+" ) # mode = "r" means 'read only'.
		search_name = 'run_' + str(self.this_run)
		for r in h5f.iterNodes(where = '/', classname = 'Group'):
			if search_name == r._v_name:
				run = r
				break
		
		###################################################################################################################
		## DATA AND INDICES! ##############################################################################################
		
		# This downsampled times series per run. Z-scored per run based on ALL measurements (except ITI), to use for GLM:
		GLM_pupil_diameter = run.pupil_data_filtered.read()[6,:]
		# Regressor files:
		regressor_blink_on = np.array(run.GLM_blink_regressors.read()[0])
		regressor_blink_off = np.array(run.GLM_blink_regressors.read()[1])
		regressor_trial_on = np.array(run.GLM_regressors.read()[0])
		regressor_stim_on = np.array(run.GLM_regressors.read()[1])
		regressor_stim_off = np.array(run.GLM_regressors.read()[2])
		regressor_ramp = np.array(run.GLM_regressors.read()[3])
		if self.experiment == 1:
			regressor_conf = np.array(run.GLM_regressors.read()[4])
			regressor_feed = np.array(run.GLM_regressors.read()[5])
			regressor_whole_trials = np.array(run.GLM_regressors.read()[6])
		if self.experiment == 2:
			regressor_whole_trials = np.array(run[i].GLM_regressors.read()[4])
		hit = np.array(run.SDT_indices.read()[0])
		fa = np.array(run.SDT_indices.read()[1])
		miss = np.array(run.SDT_indices.read()[2])
		cr = np.array(run.SDT_indices.read()[3])
		
		################################################
		##### Convert blink regressors #################
		
		# Stick at end of blink
		blink_regressor = regressor_blink_off
		
		# # Stick at middle of blink:
		# blink_regressor = []
		# for i in range(len(run)):
		# 	blink_regressor.append( np.zeros((regressor_blink_on[i].shape)) )
		# 	blink_regressor[i][:,0] = (regressor_blink_on[i][:,0] + regressor_blink_off[i][:,0]) / 2.0
		# 	blink_regressor[i][:,1] = 0.1
		# 	blink_regressor[i][:,2] = 1.0
		
		###################################################################################################################
		## MAKE REGRESSORS! ###############################################################################################
		
		# data_stick = np.zeros(2000)
		# data_ramp = np.zeros(2000)
		# data_stick[500] = 1
		# data_stick[1500] = 1
		# data_ramp[250:500] = linspace(0,(2.0/250),250)
		# data_ramp[1250:1500] = linspace(0,(2.0/250),250)
		# stick_convolved = (sp.convolve(data_stick,IRF, 'full'))[:-(IRF.shape[0]-1)]
		# ramp_convolved = (sp.convolve(data_ramp,IRF, 'full'))[:-(IRF.shape[0]-1)]
		# figure()
		# # plot(stick_convolved)
		# plot(ramp_convolved)
		# plot(ramp_convolved+(0.1*stick_convolved))
		# plot(0.5*stick_convolved)
		# legend(('ramp', 'ramp+stick', 'stick'))
		
		len_run = len(GLM_pupil_diameter)
		
		# Blink:
		a, aa = functions_jw.createRegressors(inputObject = blink_regressor, len_run = len_run, pupil_IRF = IRF, type_convolve = 'stick')
		aa = (aa - mean(aa))
		blink_ones = a
		blink_convolved = aa
		
		# Stim on:
		a, aa = functions_jw.createRegressors(inputObject = regressor_stim_on, len_run = len_run, pupil_IRF = IRF, type_convolve = 'stick')
		aa = (aa - mean(aa))
		stim_ones = a
		stim_convolved = aa
		
		# Response:
		a, aa = functions_jw.createRegressors(inputObject = regressor_stim_off[hit], len_run = len_run, pupil_IRF = IRF, type_convolve = 'stick')
		aa = (aa - mean(aa))
		resp_ones_hit = a
		resp_convolved_hit = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_stim_off[fa], len_run = len_run, pupil_IRF = IRF, type_convolve = 'stick')
		aa = (aa - mean(aa))
		resp_ones_fa = a
		resp_convolved_fa = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_stim_off[miss], len_run = len_run, pupil_IRF = IRF, type_convolve = 'stick')
		aa = (aa - mean(aa))
		resp_ones_miss = a
		resp_convolved_miss = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_stim_off[cr], len_run = len_run, pupil_IRF = IRF, type_convolve = 'stick')
		aa = (aa - mean(aa))
		resp_ones_cr = a
		resp_convolved_cr = aa
		
		if self.experiment == 1:
			# Confidence:
			a, aa = functions_jw.createRegressors(inputObject = regressor_conf, len_run = len_run, pupil_IRF = IRF, type_convolve = 'stick')
			aa = (aa - mean(aa))
			conf_ones = a
			conf_convolved = aa
			# Feedback:
			a, aa = functions_jw.createRegressors(inputObject = regressor_feed, len_run = len_run, pupil_IRF = IRF, type_convolve = 'stick')
			aa = (aa - mean(aa))
			feed_ones = a
			feed_convolved = aa
		
		# Ramp up:
		a, aa = functions_jw.createRegressors(inputObject = regressor_ramp[hit], len_run = len_run, pupil_IRF = IRF, type_convolve = 'ramp_up')
		aa = (aa - mean(aa))
		up_ones_hit = a
		up_convolved_hit = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_ramp[fa], len_run = len_run, pupil_IRF = IRF, type_convolve = 'ramp_up')
		aa = (aa - mean(aa))
		up_ones_fa = a
		up_convolved_fa = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_ramp[miss], len_run = len_run, pupil_IRF = IRF, type_convolve = 'ramp_up')
		aa = (aa - mean(aa))
		up_ones_miss = a
		up_convolved_miss = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_ramp[cr], len_run = len_run, pupil_IRF = IRF, type_convolve = 'ramp_up')
		aa = (aa - mean(aa))
		up_ones_cr = a
		up_convolved_cr = aa
		
		# Ramp down:
		a, aa = functions_jw.createRegressors(inputObject = regressor_ramp[hit], len_run = len_run, pupil_IRF = IRF, type_convolve = 'ramp_down')
		aa = (aa - mean(aa))
		down_ones_hit = a
		down_convolved_hit = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_ramp[fa], len_run = len_run, pupil_IRF = IRF, type_convolve = 'ramp_down')
		aa = (aa - mean(aa))
		down_ones_fa = a
		down_convolved_fa = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_ramp[miss], len_run = len_run, pupil_IRF = IRF, type_convolve = 'ramp_down')
		aa = (aa - mean(aa))
		down_ones_miss = a
		down_convolved_miss = aa
		
		a, aa = functions_jw.createRegressors(inputObject = regressor_ramp[cr], len_run = len_run, pupil_IRF = IRF, type_convolve = 'ramp_down')
		aa = (aa - mean(aa))
		down_ones_cr = a
		down_convolved_cr = aa
		
		try:
			run.GLM_regressors_ones.remove()
			run.GLM_regressors_convolved.remove()
		except NodeError, LeafError:
			pass
		
		if self.experiment == 1:
			GLM_regressors_ones_for_HDF5 = np.vstack((blink_ones, stim_ones, resp_ones_hit, resp_ones_fa, resp_ones_miss, resp_ones_cr, conf_ones, feed_ones, down_ones_hit, down_ones_fa, down_ones_miss, down_ones_cr, up_ones_hit, up_ones_fa, up_ones_miss, up_ones_cr))
			GLM_regressors_convolved_for_HDF5 = np.vstack((blink_convolved, stim_convolved, resp_convolved_hit, resp_convolved_fa, resp_convolved_miss, resp_convolved_cr, conf_convolved, feed_convolved, down_convolved_hit, down_convolved_fa, down_convolved_miss, down_convolved_cr, up_convolved_hit, up_convolved_fa, up_convolved_miss, up_convolved_cr))
		if self.experiment == 2:
			GLM_regressors_ones_for_HDF5 = np.vstack((blink_ones, stim_ones, resp_ones_hit, resp_ones_fa, resp_ones_miss, resp_ones_cr, down_ones_hit, down_ones_fa, down_ones_miss, down_ones_cr, up_ones_hit, up_ones_fa, up_ones_miss, up_ones_cr))
			GLM_regressors_convolved_for_HDF5 = np.vstack((blink_convolved, stim_convolved, resp_convolved_hit, resp_convolved_fa, resp_convolved_miss, resp_convolved_cr, down_convolved_hit, down_convolved_fa, down_convolved_miss, down_convolved_cr, up_convolved_hit, up_convolved_fa, up_convolved_miss, up_convolved_cr))
		h5f.createArray(run, 'GLM_regressors_ones', GLM_regressors_ones_for_HDF5, 'GLM regressors ones')
		h5f.createArray(run, 'GLM_regressors_convolved', GLM_regressors_convolved_for_HDF5, 'GLM regressors convolved')
		
		h5f.close()
	
	
class make_dataframe():
	def __init__(self, subject, experiment, version, number_runs, this_dir, sample_rate, downsample_rate):
		self.subject = subject
		self.experiment = experiment
		self.version = version
		self.number_runs = number_runs
		self.this_dir = this_dir
		self.sample_rate = sample_rate
		self.downsample_rate = downsample_rate
	
	def data_frame(self):
		
		# Set working directory and open hdf5 file
		os.chdir(self.this_dir)
		h5f = openFile(( self.subject + '.hdf5'), mode = "r+" ) # mode = "r" means 'read only'
		
		run = []
		for this_run in range(self.number_runs):
			search_name = 'run_' + str(this_run)
			for r in h5f.iterNodes(where = '/', classname = 'Group'):
				if search_name == r._v_name:
					run.append(r)
					break
		
		###################################################################################################################
		## DATA AND INDICES! ##############################################################################################
		
		trial_parameters = []
		gaze_data = []
		times = []
		trial_times = []
		omission_indices_sac = []
		omission_indices_blinks = []
		omission_indices_decision_time = []
		omission_indices = []
		pupil_diameter_lp = []
		pupil_diameter_hp = []
		pupil_diameter_hp_lp = []
		confidence_ratings = []
		decision_time = []
		test_contrast = []
		number_trials_in_run = []
		target_indices = []
		no_target_indices = []
		correct_indices = []
		incorrect_indices = []
		answer_yes_indices = []
		answer_no_indices = []
		hit_indices = []
		fa_indices = []
		cr_indices = []
		miss_indices = []
		stimulus_locked_arrays_lp = []
		response_locked_arrays_lp = []
		feedback_locked_arrays_lp = []
		stimulus_locked_arrays_band = []
		response_locked_arrays_band = []
		feedback_locked_arrays_band = []
		stimulus_locked_arrays_detrend_poly = []
		response_locked_arrays_detrend_poly = []
		feedback_locked_arrays_detrend_poly = []
		regressor_trial_on = []
		regressor_stim_on = []
		regressor_stim_off = []
		regressor_ramp = []
		regressor_conf = []
		regressor_feed = []
		regressor_whole_trials = []
		GLM_pupil_diameter = []
		for i in range(len(run)):
			trial_parameters.append(run[i].trial_parameters.read())
			gaze_data.append(run[i].gaze_data.read())
			times.append(gaze_data[i][:,0]) 
			trial_times.append(run[i].trial_times.read())
			test_contrast.append(max(trial_parameters[i]['test_contrast']))
			
			if self.experiment == 1:
				confidence_ratings.append(trial_parameters[i]['confidence'])
			
			############################################
			################ OMISSIONS #################
			# omissions (no answer in <3 seconds, or confidence rating -1: user skipped trial purposefully):
			if self.experiment == 1:
				omission_indices.append( np.array([(trial_parameters[i]['answer'] == 0) for trial_numbers in trial_parameters[i]]).sum(axis = 0, dtype = bool) + (confidence_ratings[i] == -1))
			if self.experiment == 2:
				omission_indices.append( np.array([(trial_parameters[i]['answer'] == 0) for trial_numbers in trial_parameters[i]]).sum(axis = 0, dtype = bool))
			# omissions due to eyemovements:
			omission_indices_sac.append(run[i].omission_indices_sac.read())
			# omissions due to blinks::
			omission_indices_blinks.append(run[i].omission_indices_blinks.read())
			# omission due to decision time (<250ms):
			decision_time.append( np.zeros(trial_parameters[i]['trial_nr'].shape[0]) )
			omission_indices_decision_time.append( np.zeros(trial_parameters[i]['trial_nr'].shape[0], dtype=bool) )
			for j in range(len(trial_parameters[i]['trial_nr'])):
				decision_time[i][j] = ( sum( (times[i] > trial_times[i]['trial_phase_timestamps'][j,1,0]) * (times[i] < trial_times[i]['trial_phase_timestamps'][j,2,0]) ) )
				if decision_time[i][j] < 250:
					omission_indices_decision_time[i][j] = True
			# Merge all omission indices:
			omission_indices[i] = omission_indices[i] + omission_indices_sac[i] + omission_indices_blinks[i] + omission_indices_decision_time[i]
			# Make first 2 trials of each run omissions:
			omission_indices[i][0:2] = True
			################ OMISSIONS #################
			############################################
			
			decision_time[i] = decision_time[i][-omission_indices[i]]
			if self.experiment == 1:
				confidence_ratings[i] = confidence_ratings[i][-omission_indices[i]]
			number_trials_in_run.append(trial_times[i].shape[0])
			target_indices.append(np.array([(trial_parameters[i]['target_present_in_stimulus'] == 1) for trial_numbers in trial_parameters[i]]).sum(axis = 0, dtype = bool)[-omission_indices[i]])
			no_target_indices.append(-target_indices[i])
			correct_indices.append(np.array([(trial_parameters[i]['correct'] == 1) for trial_numbers in trial_parameters[i]]).sum(axis = 0, dtype = bool)[-omission_indices[i]])
			incorrect_indices.append(-correct_indices[i])
			if self.version == 1:
				answer_yes_indices.append(np.array([(trial_parameters[i]['answer'] == -1) for trial_numbers in trial_parameters[i]]).sum(axis = 0, dtype = bool)[-omission_indices[i]])
				answer_no_indices.append(-answer_yes_indices[i])
			if self.version == 2:
				answer_yes_indices.append(np.array([(trial_parameters[i]['answer'] == 1) for trial_numbers in trial_parameters[i]]).sum(axis = 0, dtype = bool)[-omission_indices[i]])
				answer_no_indices.append(-answer_yes_indices[i])
			hit_indices.append( target_indices[i] * answer_yes_indices[i] )
			fa_indices.append( no_target_indices[i] * answer_yes_indices[i] )
			cr_indices.append( no_target_indices[i] * answer_no_indices[i] )
			miss_indices.append( target_indices[i] * answer_no_indices[i] )
			
			# # # We won't load all the eye data; instead we will get all neccesary eye data from the earlier constructed stimulus-, response- and feedback-locked arrays (below).
			# # This downsampled times series per run. Z-scored per run based on ALL measurements (except ITI), to use for GLM:
			# GLM_pupil_diameter.append(run[i].pupil_data_filtered.read()[4,:])
			
			# In hdf5:: 0 = band ; 1 = lp; 2 = poly detrend
			stimulus_locked_arrays_band.append( np.array( run[i].stimulus_locked_array.read()[0])[-omission_indices[i],:])
			stimulus_locked_arrays_lp.append( np.array( run[i].stimulus_locked_array.read()[1])[-omission_indices[i],:])
			stimulus_locked_arrays_detrend_poly.append( np.array( run[i].stimulus_locked_array.read()[2])[-omission_indices[i],:])
			response_locked_arrays_band.append( np.array( run[i].response_locked_array.read()[0])[-omission_indices[i],:])
			response_locked_arrays_lp.append( np.array( run[i].response_locked_array.read()[1])[-omission_indices[i],:])
			response_locked_arrays_detrend_poly.append( np.array( run[i].response_locked_array.read()[2])[-omission_indices[i],:])
			if self.experiment == 1:
				feedback_locked_arrays_band.append( np.array( run[i].feedback_locked_array.read()[0])[-omission_indices[i],:])
				feedback_locked_arrays_lp.append( np.array( run[i].feedback_locked_array.read()[1])[-omission_indices[i],:])
				feedback_locked_arrays_detrend_poly.append( np.array( run[i].feedback_locked_array.read()[2])[-omission_indices[i],:])
				
			# # Regressor files:
			# regressor_trial_on.append( np.array(run[i].GLM_regressors.read()[0]) )
			# regressor_stim_on.append( np.array(run[i].GLM_regressors.read()[1]) )
			# regressor_stim_off.append( np.array(run[i].GLM_regressors.read()[2]) )
			# regressor_ramp.append( np.array(run[i].GLM_regressors.read()[3]) )
			# if self.experiment == 1:
			# 	regressor_conf.append( np.array(run[i].GLM_regressors.read()[4]) )
			# 	regressor_feed.append( np.array(run[i].GLM_regressors.read()[5]) )
			# 	regressor_whole_trials.append (np.array(run[i].GLM_regressors.read()[6]) )
			# if self.experiment == 2:
			# 	regressor_whole_trials.append (np.array(run[i].GLM_regressors.read()[4]) )
			
		###################################################################################################################
		## JOIN EVERYTHING OVER RUNS! #####################################################################################
		
		omission_indices_joined = np.concatenate(omission_indices, axis=1)
		
		target_indices_joined = np.concatenate(target_indices, axis=1)
		no_target_indices_joined = np.concatenate(no_target_indices, axis=1)
		answer_yes_indices_joined = np.concatenate(answer_yes_indices, axis=1)
		answer_no_indices_joined = np.concatenate(answer_no_indices, axis=1)
		correct_indices_joined = np.concatenate(correct_indices, axis=1)
		incorrect_indices_joined = np.concatenate(incorrect_indices, axis=1)
		
		hit_indices_joined = np.concatenate(hit_indices, axis=1)
		fa_indices_joined = np.concatenate(fa_indices, axis=1)
		cr_indices_joined = np.concatenate(cr_indices, axis=1)
		miss_indices_joined = np.concatenate(miss_indices, axis=1)
		
		if self.experiment == 1:
			confidence_ratings_joined = np.concatenate(confidence_ratings, axis=1)
			confidence_0 = (confidence_ratings_joined == 0)
			confidence_1 = (confidence_ratings_joined == 1)
			confidence_2 = (confidence_ratings_joined == 2)
			confidence_3 = (confidence_ratings_joined == 3)
		
		decision_time_joined = np.concatenate(decision_time, axis=1)
		test_contrast_joined = np.array(test_contrast)
		number_trials_in_run_joined = np.array(number_trials_in_run)
		
		stimulus_locked_array_lp_joined = np.vstack(stimulus_locked_arrays_lp)
		stimulus_locked_array_band_joined = np.vstack(stimulus_locked_arrays_band)
		stimulus_locked_arrays_detrend_poly_joined = np.vstack(stimulus_locked_arrays_detrend_poly)
		response_locked_array_lp_joined = np.vstack(response_locked_arrays_lp)
		response_locked_array_band_joined = np.vstack(response_locked_arrays_band)
		response_locked_arrays_detrend_poly_joined = np.vstack(response_locked_arrays_detrend_poly)
		if self.experiment == 1:
			feedback_locked_array_lp_joined = np.vstack(feedback_locked_arrays_lp)
			feedback_locked_array_band_joined = np.vstack(feedback_locked_arrays_band)
			feedback_locked_arrays_detrend_poly_joined = np.vstack(feedback_locked_arrays_detrend_poly)
		
		# trial_onset_locked_array_joined = np.vstack(trial_onset_locked_arrays)
		
		# Join timeseries of each run:
		# GLM_pupil_diameter_joined = np.concatenate(GLM_pupil_diameter, axis=1)
		
		# PRINT SOME STUFF:
		print('total # trials: ' + str(sum(correct_indices_joined)+sum(incorrect_indices_joined)) )
		print('hits: ' + str(sum(hit_indices_joined)))
		print('fa: ' + str(sum(fa_indices_joined)))
		print('cr: ' + str(sum(cr_indices_joined)))
		print('miss: ' + str(sum(miss_indices_joined)))
		print('omission: ' + str(sum(omission_indices_joined)))
		
		print('mean decision time yes answers: ' + str(mean(decision_time_joined[answer_yes_indices_joined])) )
		print('mean decision time no answers: ' + str(mean(decision_time_joined[answer_no_indices_joined])) )
		
		###################################################################################################################
		## COMPUTE BPD! ###################################################################################################
		
		# BPD is computed as the mean pupil diameter in 500 ms preceding stimulus onset / feedback.
		
		# Seperate arrays (with BPD for individual trials) for every run are joined in the list variables 'bpd' (for bpd before stimulus onset) and 'bpd_feed' (for bpd before feedback).
		# Later, all the elements in these lists are concatenated in the variables 'bpd_joined' and 'bpd_feed_joined'.   
		
		use_this_stim_array = stimulus_locked_arrays_lp
		if self.experiment == 1:
			use_this_feed_array = feedback_locked_arrays_lp
		
		bpd = []
		bpd_feed = []
		for i in range(len(run)):
			# BPD:
			bpd_dum = np.zeros(use_this_stim_array[i].shape[0])
			for j in range(use_this_stim_array[i].shape[0]):
				bpd_dum[j] = ( mean(use_this_stim_array[i][j,500:999]) )
			bpd.append(bpd_dum)
			if self.experiment == 1:
				# BPD_Feed:
				bpd_feed_dum = np.zeros(use_this_feed_array[i].shape[0])
				for k in range(use_this_feed_array[i].shape[0]):
					bpd_feed_dum[k] = ( mean(use_this_feed_array[i][k,500:999]) )
				bpd_feed.append(bpd_feed_dum)
			
		# Bin trials by BPD
		
		bpd_low_indices = []
		bpd_high_indices = []
		
		for i in range(len(run)):
			bpd_low_indices.append( (bpd[i] < np.median(bpd[i])) )
			bpd_high_indices.append( (bpd[i] > np.median(bpd[i])) )
			
		# Join everything: 
		
		bpd_joined = np.concatenate(bpd)
		if self.experiment == 1:
			bpd_feed_joined = np.concatenate(bpd_feed)
		
		bpd_low_indices_joined = np.concatenate(bpd_low_indices)
		bpd_high_indices_joined = np.concatenate(bpd_high_indices)
			
		###################################################################################################################
		## COMPUTE PPD! ###################################################################################################
		
		use_this_resp_array = response_locked_arrays_lp
		use_this_resp_array_joined = response_locked_array_lp_joined
		if self.experiment == 1:
			use_this_feed_array = feedback_locked_arrays_lp
			use_this_feed_array_joined = feedback_locked_array_lp_joined
		
		# PPD is computed by use of three different methods. For every method, the timewindow is 1 sec before response till 1.5 sec after response (captures the peak).
		
		# Seperate arrays (with PPD for individual trials) for every run are joined in the list variables 'ppd', 'ppd_mean' and 'ppd_lin'.
		# Later, all the elements in these lists are concatenated in the variables 'ppd_joined', 'ppd_mean_joined' and 'ppd_lin_joined' (plus the same for ppd after feedback).   
		
		ppd = []
		ppd_mean = []
		ppd_lin = []
		# ppd_lin_S = []
		# pooled_mean = []
		for i in range(len(run)):
			
			# PPD - Method 1 (max minus baseline): 
			ppd_dum =  np.zeros(use_this_resp_array[i].shape[0])
			for j in range(use_this_resp_array[i].shape[0]):
				ppd_dum[j] = bottleneck.nanmax(use_this_resp_array[i][j,2500:5000]) - bpd[i][j]
				# ppd_dum[j] = bottleneck.nanmax(response_locked_arrays[i][j,3500:5000]) - response_locked_arrays[i][j,3425]
			ppd.append(ppd_dum)
			
			# PPD - Method 2 (integration: area under curve):
			ppd_dum =  np.zeros(use_this_resp_array[i].shape[0])
			for j in range(use_this_resp_array[i].shape[0]):
				ppd_dum[j] = bottleneck.nanmean(use_this_resp_array[i][j,2500:5000]) - bpd[i][j]
			ppd_mean.append(ppd_dum)
			
			# PPD - Method 3 (linear projection):
			template = (bottleneck.nanmean(use_this_resp_array_joined[:,2500:5000], axis=0)) - mean(bpd_joined)
			# template = ( (bottleneck.nanmean(response_locked_array_joined[hit_indices_joined,2500:5000], axis=0) - mean(bpd_joined[hit_indices_joined])) + (bottleneck.nanmean(response_locked_array_joined[fa_indices_joined,2500:5000], axis=0) - mean(bpd_joined[fa_indices_joined])) + (bottleneck.nanmean(response_locked_array_joined[miss_indices_joined,2500:5000], axis=0) - mean(bpd_joined[miss_indices_joined])) + (bottleneck.nanmean(response_locked_array_joined[cr_indices_joined,2500:5000], axis=0) - mean(bpd_joined[cr_indices_joined])) ) / 4
			ppd_dum =  np.zeros(use_this_resp_array[i].shape[0])
			for j in range(use_this_resp_array[i].shape[0]):
				ppd_dum[j] = ( np.dot( template, (use_this_resp_array[i][j,2500:5000])-bpd[i][j] ) / np.dot(template, template)  )
			ppd_lin.append(ppd_dum)
			
			# # PPD - Method 3b (linear projection per session):.
			# ppd_dum =  np.zeros(response_locked_arrays[i].shape[0])
			# pooled_mean_ppd = ( (bottleneck.nanmean(response_locked_arrays[i][hit_indices[i],2500:5000], axis=0)) - mean(bpd[i][hit_indices[i]]) + (bottleneck.nanmean(response_locked_arrays[i][fa_indices[i],2500:5000], axis=0)) - mean(bpd[i][fa_indices[i]]) + (bottleneck.nanmean(response_locked_arrays[i][miss_indices[i],2500:5000], axis=0)) - mean(bpd[i][miss_indices[i]]) + (bottleneck.nanmean(response_locked_arrays[i][cr_indices[i],2500:5000], axis=0)) - mean(bpd[i][cr_indices[i]]) ) / 4
			# pooled_mean.append(pooled_mean_ppd)
			# #pooled_mean_ppd = (bottleneck.nanmean(response_locked_array_joined[-omission_indices_joined,2500:4997], axis=0)) - mean(bpd_joined[-omission_indices_joined])
			# for j in range(response_locked_arrays[i].shape[0]):
			# 	ppd_dum[j] = ( np.dot( pooled_mean_ppd, (response_locked_arrays[i][j,2500:5000])-bpd[i][j] ) / np.dot(pooled_mean_ppd, pooled_mean_ppd) )
			# ppd_lin_S.append(ppd_dum)
			
		# PPD's after feedback:
		if self.experiment == 1:
			ppd_feed = []
			ppd_feed_mean = []
			ppd_feed_lin = []
			# ppd_feed_lin_S = []
			for i in range(len(run)):
				
				# PPD - Method 1 (max minus baseline): 
				ppd_dum =  np.zeros(use_this_feed_array[i].shape[0])
				for j in range(use_this_feed_array[i].shape[0]):
					ppd_dum[j] = bottleneck.nanmax(use_this_feed_array[i][j,1000:2500]) - bpd_feed[i][j]
				ppd_feed.append(ppd_dum)
				
				# PPD - Method 2 (mean minus baseline):
				ppd_dum =  np.zeros(use_this_feed_array[i].shape[0])
				for j in range(use_this_feed_array[i].shape[0]):
					ppd_dum[j] = bottleneck.nanmean(use_this_feed_array[i][j,1000:2500]) - bpd_feed[i][j]
				ppd_feed_mean.append(ppd_dum)
				
				# PPD - Method 3 (linear projection):
				template_feed = (bottleneck.nanmean(use_this_feed_array_joined[:,1000:2500], axis=0)) - mean(bpd_feed_joined)
				# template_feed = ( (bottleneck.nanmean(feedback_locked_array_joined[hit_indices_joined,1000:2500], axis=0) - mean(bpd_feed_joined[hit_indices_joined])) + (bottleneck.nanmean(feedback_locked_array_joined[fa_indices_joined,1000:2500], axis=0) - mean(bpd_feed_joined[fa_indices_joined])) + (bottleneck.nanmean(feedback_locked_array_joined[miss_indices_joined,1000:2500], axis=0) - mean(bpd_feed_joined[miss_indices_joined])) + (bottleneck.nanmean(feedback_locked_array_joined[cr_indices_joined,1000:2500], axis=0) - mean(bpd_feed_joined[cr_indices_joined])) ) / 4
				ppd_dum =  np.zeros(use_this_feed_array[i].shape[0])
				for j in range(use_this_feed_array[i].shape[0]):
					ppd_dum[j] = ( np.dot( template_feed, (use_this_feed_array[i][j,1000:2500])-bpd_feed[i][j] ) / np.dot(template_feed, template_feed) )
				ppd_feed_lin.append(ppd_dum)
				
				# # PPD - Method 3 (linear projection per session):
				# # BEWARE!! NO UNBIASED POOLED MEAN BELOW....
				# ppd_dum =  np.zeros(feedback_locked_arrays[i].shape[0])
				# pooled_mean_ppd = (bottleneck.nanmean(response_locked_arrays[i][:,1000:2500], axis=0)) - mean(bpd[i])
				# # pooled_mean_ppd_feed = (bottleneck.nanmean(feedback_locked_array_joined[-omission_indices_joined,1000:2500], axis=0)) - mean(bpd_feed_joined[-omission_indices_joined])
				# for j in range(feedback_locked_arrays[i].shape[0]):
				# 	ppd_dum[j] = ( np.dot( pooled_mean_ppd_feed, (feedback_locked_arrays[i][j,1000:2500])-bpd_feed[i][j] ) / np.dot(pooled_mean_ppd_feed, pooled_mean_ppd_feed) )
				# ppd_feed_lin_S.append(ppd_dum)
		
		# Join everything:
		ppd_joined = np.concatenate(ppd)
		ppd_mean_joined = np.concatenate(ppd_mean)
		ppd_lin_joined = np.concatenate(ppd_lin)
		# ppd_lin_S_joined = np.concatenate(ppd_lin_S)
		if self.experiment == 1:
			ppd_feed_joined = np.concatenate(ppd_feed)
			ppd_feed_mean_joined = np.concatenate(ppd_feed_mean)
			ppd_feed_lin_joined = np.concatenate(ppd_feed_lin)
		
		###################################################################################################################
		## CORRECT PPD FOR RT! ############################################################################################
		
		mean_RT_for_correct = mean(decision_time_joined) 
		# mean_RT_for_correct = (mean(decision_time_joined[hit_indices_joined]) + mean(decision_time_joined[fa_indices_joined]) + mean(decision_time_joined[miss_indices_joined]) + mean(decision_time_joined[cr_indices_joined])) / 4
		ppd_lin_RT_corrected = np.zeros(len(ppd_lin_joined))
		for i in range(len(ppd_lin_joined)):
			ppd_lin_RT_corrected[i] = (ppd_lin_joined[i] * mean_RT_for_correct) / decision_time_joined[i]
		
		###################################################################################################################
		## GET D_PRIME AND CRITERION! #####################################################################################
		
		d_prime_overall, criterion_overall = functions_jw.SDT_measures_per_subject(subject = self.subject, target_indices_joined = target_indices_joined, no_target_indices_joined = no_target_indices_joined, hit_indices_joined = hit_indices_joined, fa_indices_joined = fa_indices_joined)
		d_prime_per_run, criterion_per_run = functions_jw.SDT_measures_per_subject_per_run(subject = self.subject, target_indices = target_indices, no_target_indices = no_target_indices, hit_indices = hit_indices, fa_indices = fa_indices)
		
		###################################################################################################################
		## MAKE DATAFRAME! ################################################################################################
		
		if self.experiment == 1:
			d = {
				'test_contrast_per_run' : pd.Series(test_contrast_joined),
				'd_prime_overall' : pd.Series(d_prime_overall),
				'criterion_overall' : pd.Series(criterion_overall),
				'd_prime_per_run' : pd.Series(d_prime_per_run),
				'criterion_per_run' : pd.Series(criterion_per_run),
				'number_trials_per_run' : pd.Series(number_trials_in_run_joined),
				'stimulus' : pd.Series(np.array(target_indices_joined, dtype = int)),
				'answer' : pd.Series(np.array(answer_yes_indices_joined, dtype = int)),
				'confidence': pd.Series(confidence_ratings_joined),
				'response_time': pd.Series(decision_time_joined),
				'bpd' : pd.Series(bpd_joined),
				'bpd_feed' : pd.Series(bpd_feed_joined),
				'ppd' : pd.Series(ppd_joined),
				'ppd_mean' : pd.Series(ppd_mean_joined),
				'ppd_lin' : pd.Series(ppd_lin_joined),
				'ppd_lin_RT' : pd.Series(ppd_lin_RT_corrected),
				'ppd_feed' : pd.Series(ppd_feed_joined),
				'ppd_feed_mean' : pd.Series(ppd_feed_mean_joined),
				'ppd_feed_lin' : pd.Series(ppd_feed_lin_joined)
				}
			
		if self.experiment == 2:
			d = {
				'test_contrast_per_run' : pd.Series(test_contrast_joined),
				'd_prime_overall' : pd.Series(d_prime_overall),
				'criterion_overall' : pd.Series(criterion_overall),
				'd_prime_per_run' : pd.Series(d_prime_per_run),
				'criterion_per_run' : pd.Series(criterion_per_run),
				'number_trials_per_run' : pd.Series(number_trials_in_run_joined),
				'stimulus' : pd.Series(np.array(target_indices_joined, dtype = int)),
				'answer' : pd.Series(np.array(answer_yes_indices_joined, dtype = int)),
				# 'confidence': pd.Series(confidence_ratings_joined),
				'response_time': pd.Series(decision_time_joined),
				'bpd' : pd.Series(bpd_joined),
				# 'bpd_feed' : pd.Series(bpd_feed_joined),
				'ppd' : pd.Series(ppd_joined),
				'ppd_mean' : pd.Series(ppd_mean_joined),
				'ppd_lin' : pd.Series(ppd_lin_joined),
				'ppd_lin_RT' : pd.Series(ppd_lin_RT_corrected)
				# 'ppd_feed' : pd.Series(ppd_feed_joined),
				# 'ppd_feed_mean' : pd.Series(ppd_feed_mean_joined),
				# 'ppd_feed_lin' : pd.Series(ppd_feed_lin_joined),
				}
			
		df = pd.DataFrame(d)
		
		###################################################################################################################
		## SAVE STUFF! ################################################################################################
		
		# stim_locked_grand_mean = ( bottleneck.nanmean(stimulus_locked_array_joined[hit_indices_joined,:], axis=0) + bottleneck.nanmean(stimulus_locked_array_joined[fa_indices_joined,:], axis=0) + bottleneck.nanmean(stimulus_locked_array_joined[miss_indices_joined,:], axis=0) + bottleneck.nanmean(stimulus_locked_array_joined[cr_indices_joined,:], axis=0) ) / 4
		# stim_locked_grand_sem = ( (bottleneck.nanstd(stimulus_locked_array_joined[hit_indices_joined,:], axis=0) / sp.sqrt(stimulus_locked_array_joined[hit_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(stimulus_locked_array_joined[fa_indices_joined,:], axis=0) / sp.sqrt(stimulus_locked_array_joined[fa_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(stimulus_locked_array_joined[miss_indices_joined,:], axis=0) / sp.sqrt(stimulus_locked_array_joined[miss_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(stimulus_locked_array_joined[cr_indices_joined,:], axis=0) / sp.sqrt(stimulus_locked_array_joined[cr_indices_joined,:].shape[0]) ) ) / 4
		# resp_locked_grand_mean = ( bottleneck.nanmean(response_locked_array_joined[hit_indices_joined,:], axis=0) + bottleneck.nanmean(response_locked_array_joined[fa_indices_joined,:], axis=0) + bottleneck.nanmean(response_locked_array_joined[miss_indices_joined,:], axis=0) + bottleneck.nanmean(response_locked_array_joined[cr_indices_joined,:], axis=0) ) / 4
		# resp_locked_grand_sem = ( (bottleneck.nanstd(response_locked_array_joined[hit_indices_joined,:], axis=0) / sp.sqrt(response_locked_array_joined[hit_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(response_locked_array_joined[fa_indices_joined,:], axis=0) / sp.sqrt(response_locked_array_joined[fa_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(response_locked_array_joined[miss_indices_joined,:], axis=0) / sp.sqrt(response_locked_array_joined[miss_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(response_locked_array_joined[cr_indices_joined,:], axis=0) / sp.sqrt(response_locked_array_joined[cr_indices_joined,:].shape[0]) ) ) / 4
		
		stim_locked_grand_mean = bottleneck.nanmean(stimulus_locked_array_band_joined[:,:], axis=0)
		stim_locked_grand_sem = bottleneck.nanstd(stimulus_locked_array_band_joined[:,:], axis=0) / sp.sqrt(stimulus_locked_array_band_joined[:,:].shape[0])
		resp_locked_grand_mean = bottleneck.nanmean(response_locked_array_band_joined[:,:], axis=0)
		resp_locked_grand_sem = bottleneck.nanstd(response_locked_array_band_joined[:,:], axis=0) / sp.sqrt(response_locked_array_band_joined[:,:].shape[0])
		if self.experiment == 1:
			feed_locked_grand_mean = bottleneck.nanmean(feedback_locked_array_band_joined[:,:], axis=0)
			feed_locked_grand_sem = bottleneck.nanstd(feedback_locked_array_band_joined[:,:], axis=0) / sp.sqrt(feedback_locked_array_band_joined[:,:].shape[0])
			
			pupil_IRF = feed_locked_grand_mean[500:3990]
			pupil_IRF_downsampled = sp.signal.decimate(pupil_IRF,int(self.downsample_rate),1)
		
		# Save to folder called 'ALL':
		os.chdir('/Research/PUPIL/data_ALL/')
		
		df.save(str(self.subject) + '_data')
		
		np.save(str(self.subject) + '_stimulus_locked_array_lp_joined', stimulus_locked_array_lp_joined)
		np.save(str(self.subject) + '_response_locked_array_lp_joined', response_locked_array_lp_joined)
		if self.experiment == 1:
			np.save(str(self.subject) + '_feedback_locked_array_lp_joined', feedback_locked_array_lp_joined)
			np.save(str(self.subject) + '_feed_locked_grand_mean', feed_locked_grand_mean)
			np.save(str(self.subject) + '_feed_locked_grand_sem', feed_locked_grand_sem)
			np.save(self.subject + '_pupil_IRF', pupil_IRF)
			np.save(self.subject + '_pupil_IRF_downsampled', pupil_IRF_downsampled)
		
		np.save(str(self.subject) + '_stim_locked_grand_mean', stim_locked_grand_mean)
		np.save(str(self.subject) + '_stim_locked_grand_sem', stim_locked_grand_sem)
		np.save(str(self.subject) + '_resp_locked_grand_mean', resp_locked_grand_mean)
		np.save(str(self.subject) + '_resp_locked_grand_sem', resp_locked_grand_sem)
		
		np.save(str(self.subject) + '_stimulus_locked_array_band_joined', stimulus_locked_array_band_joined)
		np.save(str(self.subject) + '_response_locked_array_band_joined', response_locked_array_band_joined)
		if self.experiment == 1:
			np.save(str(self.subject) + '_feedback_locked_array_band_joined', feedback_locked_array_band_joined)
			
		np.save(str(self.subject) + '_stimulus_locked_arrays_detrend_poly_joined', stimulus_locked_arrays_detrend_poly_joined)
		np.save(str(self.subject) + '_response_locked_arrays_detrend_poly_joined', response_locked_arrays_detrend_poly_joined)
		if self.experiment == 1:
			np.save(str(self.subject) + '_feedback_locked_arrays_detrend_poly_joined', feedback_locked_arrays_detrend_poly_joined)
		
		
		h5f.close()
	
	



class within_subjects_stats():
	
	def __init__(self, subject, experiment, version, number_runs, this_dir, sample_rate, downsample_rate):
		self.subject = subject
		self.experiment = experiment
		self.version = version
		self.number_runs = number_runs
		self.this_dir = this_dir
		self.sample_rate = sample_rate
		self.downsample_rate = downsample_rate
		
		os.chdir(self.this_dir)
		
		df = pd.load(self.subject + '_data')
		self.stimulus_locked_array = np.load(self.subject + '_stimulus_locked_array_lp_joined.npy')
		self.response_locked_array = np.load(self.subject + '_response_locked_array_lp_joined.npy')
		if self.experiment == 1:
			self.feedback_locked_array = np.load(self.subject + '_feedback_locked_array_lp_joined.npy')
		
		# Over session:
		self.d_prime_overall = np.array(df['d_prime_overall'])[0]
		self.criterion_overall = np.array(df['criterion_overall'])[0]
		# Over runs:
		self.test_contrast_per_run = np.array(df['test_contrast_per_run'])
		self.test_contrast_per_run = self.test_contrast_per_run[-np.isnan(self.test_contrast_per_run)]
		self.number_trials_per_run = np.array(df['number_trials_per_run'])
		self.number_trials_per_run = self.number_trials_per_run[-np.isnan(self.number_trials_per_run)]
		self.d_prime_per_run = np.array(df['d_prime_per_run'])
		self.d_prime__perrun = self.d_prime_per_run[-np.isnan(self.d_prime_per_run)]
		self.criterion_per_run = np.array(df['criterion_per_run'])
		self.criterion_per_run = self.criterion_per_run[-np.isnan(self.criterion_per_run)]
		self.number_of_runs = len(self.test_contrast_per_run)
		# Over trials:
		self.response_time = np.array(df['response_time'])
		self.stimulus_present = np.array(df['stimulus'], dtype = bool)
		self.answer_yes = np.array(df['answer'], dtype = bool)
		self.answer_no = -self.answer_yes
		self.correct = (self.stimulus_present*self.answer_yes) + (-self.stimulus_present*self.answer_no)
		self.incorrect = -self.correct
		self.hit = self.answer_yes*self.correct
		self.fa = self.answer_yes*self.incorrect
		self.miss = self.answer_no*self.incorrect
		self.cr = self.answer_no*self.correct
		if self.experiment == 1:
			self.confidence = df['confidence']
			self.confidence_0 = np.array(self.confidence == 0)
			self.confidence_1 = np.array(self.confidence == 1)
			self.confidence_2 = np.array(self.confidence == 2)
			self.confidence_3 = np.array(self.confidence == 3)
			self.bpd_feed = np.array(df['bpd_feed'])
			self.ppd_feed = np.array(df['ppd_feed'])
			self.ppd_feed_mean = np.array(df['ppd_feed_mean'])
			self.ppd_feed_lin = np.array(df['ppd_feed_lin'])
		self.bpd = np.array(df['bpd'])
		self.ppd = np.array(df['ppd'])
		self.ppd_mean = np.array(df['ppd_mean'])
		self.ppd_lin = np.array(df['ppd_lin'])
		self.ppd_lin_RT = np.array(df['ppd_lin_RT'])
		
		os.chdir(self.this_dir + '/figures/')
	
	
	def PPR_amplitude_within_stats(self, use_ppd_lin = True):
		
		# ppd_measure = ppd_lin
		# 
		# # Permutation Tests:
		# indices_to_test1 = [hit, miss]
		# indices_to_test2 = [fa, cr]
		# observed_mean_difference = []
		# perm_results = []
		# significance = []
		# for i in range(len(indices_to_test1)):
		# 	group1 = ppd_measure[indices_to_test1[i]]
		# 	group2 = ppd_measure[indices_to_test2[i]]
		# 	output = functions_jw.permutationTest(group1 = group1, group2 = group2, nrand = 5000)
		# 	observed_mean_difference.append(output[0]) 
		# 	perm_results.append(output[1])
		# 	significance.append(output[2])
		# 
		
		if use_ppd_lin == True:
			ppd_measure = self.ppd_lin
		else:
			ppd_measure = self.ppd_lin_RT
		
		# Permutation Tests:
		indices_to_test1 = [self.hit, self.fa, self.answer_yes, self.correct]
		indices_to_test2 = [self.miss, self.cr, self.answer_no, self.incorrect]
		observed_mean_difference = []
		perm_results = []
		significance = []
		for i in range(len(indices_to_test1)):
			group1 = ppd_measure[indices_to_test1[i]]
			group2 = ppd_measure[indices_to_test2[i]]
			output = functions_jw.permutationTest(group1 = group1, group2 = group2, nrand = 5000)
			observed_mean_difference.append(output[0]) 
			perm_results.append(output[1])
			significance.append(output[2])
		
		## PPD's at response SDT:
		fig = functions_jw.sdt_barplot(subject = self.subject, hit = ppd_measure[self.hit], fa = ppd_measure[self.fa], miss = ppd_measure[self.miss], cr = ppd_measure[self.cr], p1 = significance[0], p2=significance[1])
		ylabel('PPR amplitude (linearly projected)', size = 10)
		if use_ppd_lin == True:
			title('Mean PPR amplitude', size = 12)
			pp = PdfPages('Exp' + str(self.experiment) + '_PPR_bars_1_' + self.subject + '.pdf')
		else:
			title('Mean PPR amplitude RT corrected', size = 12)
			pp = PdfPages('Exp' + str(self.experiment) + '_PPR_bars_3_' + self.subject + '.pdf')
		fig.savefig(pp, format='pdf')
		pp.close()
		
		## PPD's at response yes vs no and corr. vs incorr.:
		fig = functions_jw.sdt_barplot(subject = self.subject, hit = ppd_measure[self.hit], fa = ppd_measure[self.fa], miss = ppd_measure[self.miss], cr = ppd_measure[self.cr], p1 = significance[2], p2=significance[3], type_plot = 2)
		ylabel('PPR amplitude (linearly projected)', size = 10)
		if use_ppd_lin == True:
			title('Mean PPR amplitude', size = 12)
			pp = PdfPages('Exp' + str(self.experiment) + '_PPR_bars_2_' + self.subject + '.pdf')
		else:
			title('Mean PPR amplitude RT corrected', size = 12)
			pp = PdfPages('Exp' + str(self.experiment) + '_PPR_bars_4_' + self.subject + '.pdf')
		fig.savefig(pp, format='pdf')
		pp.close()
		
		# # ROC Analysis:
		# indices_to_test1 = [answer_yes, correct]
		# indices_to_test2 = [answer_no, incorrect]
		# out_i = []
		# out_p = []
		# for i in range(len(indices_to_test1)):
		# 	group1 = ppd_measure[indices_to_test1[i]]
		# 	group2 = ppd_measure[indices_to_test2[i]]
		# 	output = functions_jw.roc_analysis(group1 = group1, group2 = group2, nrand=1000)
		# 	out_i.append(output[0])
		# 	out_p.append(output[1])
		# 
		# 
		# # ROC analysis over time:
		# out_i_time = []
		# out_p_time = []
		# for i in range(response_locked_array.shape[1]):
		# 	out_i_time.append(functions_jw.roc_analysis(group1 = response_locked_array[answer_yes,i], group2 = response_locked_array[answer_no,i], nrand=1))
		# out_i_time = np.array(out_i_time)
	
	
	def BPD_within_stats(self):
		
		# Permutation Tests:
		indices_to_test1 = [self.hit, self.fa, self.answer_yes, self.correct]
		indices_to_test2 = [self.miss, self.cr, self.answer_no, self.incorrect]
		observed_mean_difference = []
		perm_results = []
		significance_bpd = []
		for i in range(len(indices_to_test1)):
			group1 = self.bpd[indices_to_test1[i]]
			group2 = self.bpd[indices_to_test2[i]]
			output = functions_jw.permutationTest(group1 = group1, group2 = group2, nrand = 5000)
			observed_mean_difference.append(output[0])
			perm_results.append(output[1])
			significance_bpd.append(output[2])
		
		## BPD's
		fig = functions_jw.sdt_barplot(subject = self.subject, hit = self.bpd[self.hit], fa = self.bpd[self.fa], miss = self.bpd[self.miss], cr = self.bpd[self.cr], p1 = significance_bpd[0], p2=significance_bpd[1])
		ylabel('Baseline pupil diameter (Z)', size = 10)
		title('Mean baseline pupil diameter', size = 12)
		text(0.45, plt.axis()[2] - ((plt.axis()[2] - plt.axis()[3])/8), 'p = ' + str(significance_bpd[0]))
		text(1.45, plt.axis()[2] - ((plt.axis()[2] - plt.axis()[3])/8), 'p = ' + str(significance_bpd[1]))
		pp = PdfPages('Exp' + str(self.experiment) + '_BPD_bars_' + self.subject + '.pdf')
		fig.savefig(pp, format='pdf')
		pp.close()
		
	
	
	def response_figures(self):
		
		#********************************************************************************
		#************************ CORRECT RESPONSE MATRICES FOR BPD's *******************
		#********************************************************************************
		#*    'pupil change' is computed by taking the time series of each seperate     *
		#*         trial, and subtract the associated bpd from every data point         *
		#********************************************************************************
		
		for i in range(self.stimulus_locked_array.shape[0]):
			self.stimulus_locked_array[i,:] = self.stimulus_locked_array[i,:] - self.bpd[i]
		
		for i in range(self.response_locked_array.shape[0]):
			self.response_locked_array[i,:] = self.response_locked_array[i,:] - self.bpd[i]
		
		if self.experiment == 1:
			for i in range(self.feedback_locked_array.shape[0]):
				self.feedback_locked_array[i,:] = self.feedback_locked_array[i,:] - self.bpd_feed[i]
		
		# for i in range(response_locked_array.shape[0]):
		# 	response_locked_array[i,:] = response_locked_array[i,:] - response_locked_array[i,3250]
		# 
		# sum(response_locked_array[:,3250])
		
		#********************************************************************************
		#***************************** PUPIL RESPONSE FIGURE  ***************************
		#********************************************************************************
		#*       Stimulus-, response and feedback-locked pupillary response plots:      *
		#********************************************************************************
		
		stim_data = self.stimulus_locked_array[:,500:-1000]
		resp_data = self.response_locked_array[:,250:-500]
		if self.experiment == 1:
			feed_data = self.feedback_locked_array[:,500:2499]
		condition = [self.hit, self.fa, self.miss, self.cr]
		
		stim_means = []
		stim_sems = []
		resp_means = []
		resp_sems = []
		feed_means = []
		feed_sems = []
		for i in range(len(condition)):
			stim_means.append( bottleneck.nanmean(stim_data[condition[i]], axis=0))
			stim_sems.append( (bottleneck.nanstd(stim_data[condition[i]], axis=0)) / sp.sqrt(condition[i].sum()) )
			resp_means.append( bottleneck.nanmean(resp_data[condition[i]], axis=0))
			resp_sems.append( (bottleneck.nanstd(resp_data[condition[i]], axis=0)) / sp.sqrt(condition[i].sum()) )
			if self.experiment == 1:
				feed_means.append( bottleneck.nanmean(feed_data[condition[i]], axis=0))
				feed_sems.append( (bottleneck.nanstd(feed_data[condition[i]], axis=0)) / sp.sqrt(condition[i].sum()) )
		
		# Make the plt.plot
		if self.experiment == 1:
			figure_mean_pupil_locked_to_stimulus_response_feedback_SDT = plt.figure(figsize=(4, 9))
		if self.experiment == 2:
			figure_mean_pupil_locked_to_stimulus_response_feedback_SDT = plt.figure(figsize=(4, 6))
		hspace = 0.45
		left = 0.2
		
		# Stimulus 
		if self.experiment == 1:
			a = plt.subplot(311)
		if self.experiment == 2:
			a = plt.subplot(211)
		xa = np.arange(-499,4000)
		for i in range(len(condition)):
			plt.plot(xa, stim_means[i], linewidth=2, color = ['r','r','b','b'][i], alpha = [1,0.5,0.5,1][i])
			plt.fill_between( xa, (stim_means[i]+stim_sems[i]), (stim_means[i]-stim_sems[i]), color = ['r','r','b','b'][i], alpha=0.1 )
		plt.axvline(sp.mean(self.response_time[self.answer_yes]), -1, 1, color = 'r', linestyle = '--', alpha = 0.5)
		plt.axvline(sp.mean(self.response_time[self.answer_no]), -1, 1, color = 'b', linestyle = '--', alpha = 0.5)
		plt.axvline(0, -1, 1, linewidth=1)
		# plt.text(sp.mean(response_time[answer_yes])+30,plt.axis()[3]-0.05,"'yes!'", size=18)
		# plt.text(sp.mean(response_time[answer_no])+30,plt.axis()[3]-0.05,"'no!'", size=18)
		plt.xlim( (-500, 4000) )
		leg = plt.legend(["HIT; " + str(self.hit.sum()) + " trials", "FA; " + str(self.fa.sum()) + " trials", "MISS; " + str(self.miss.sum()) + " trials", "CR; " + str(self.cr.sum()) + " trials"], loc = 2, fancybox = True)
		leg.get_frame().set_alpha(0.9)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize(7)    # the legend text fontsize
			for l in leg.get_lines():
				l.set_linewidth(2)  # the legend line width
		plt.tick_params(axis='both', which='major', labelsize=8)
		# plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit.sum()) + " trials", "FA; " + str(fa.sum()) + " trials", "MISS; " + str(miss.sum()) + " trials", "CR; " + str(cr.sum()) + " trials"], loc = 2)
		simpleaxis(a)
		spine_shift(a) 
		subplots_adjust(hspace = hspace, left = left)
		plt.xticks([0,1000,2000,3000,4000], [0,1,2,3,4])
		plt.title('Stimulus locked PPR', size=12)
		plt.ylabel("PPR (Z)", size=10)
		plt.xlabel("Time from stimulus (s)", size=10)
		plt.hist(self.response_time[self.answer_yes], bins=20, rwidth=0.8, weights = np.ones(condition[0].sum()+condition[1].sum()) / 2500 * ((axis()[3] - axis()[2])*3), bottom = axis()[2], color = 'r', alpha = 0.5)
		plt.hist(self.response_time[self.answer_no], bins=20, rwidth=0.8, weights = np.ones(condition[2].sum()+condition[3].sum()) / 2500 * ((axis()[3] - axis()[2])*3), bottom = axis()[2], color = 'b', alpha = 0.5)
		gca().spines["bottom"].set_linewidth(.5)
		gca().spines["left"].set_linewidth(.5)
		
		# Response
		if self.experiment == 1:
			b = plt.subplot(312)
		if self.experiment == 2:
			b = plt.subplot(212)
		xb = np.arange(-3249,2000)
		for i in range(len(condition)):
			plt.plot(xb, resp_means[i], linewidth=2, color = ['r','r','b','b'][i], alpha = [1,0.5,0.5,1][i])
			plt.fill_between( xb, (resp_means[i]+resp_sems[i]), (resp_means[i]-resp_sems[i]), color = ['r','r','b','b'][i], alpha=0.1 )
		plt.axvline(0-sp.mean(self.response_time[self.answer_yes]), -1, 1, color = 'r', linestyle = '--', alpha = 0.5)
		plt.axvline(0-sp.mean(self.response_time[self.answer_no]), -1, 1, color = 'b', linestyle = '--', alpha = 0.5)
		plt.axvline(0, -1, 1, linewidth=1)
		# plt.text(0-sp.mean(response_time[answer_yes])+10,0.15,"'yes!'")
		# plt.text(0-sp.mean(response_time[answer_no])+10,0.35,"'no!'")
		plt.xlim( (-3225, 1500) )
		plt.tick_params(axis='both', which='major', labelsize=8)
		# plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit.sum()) + " trials", "FA; " + str(fa.sum()) + " trials", "CR; " + str(cr.sum()) + " trials", "MISS; " + str(miss.sum()) + " trials"], loc = 2)
		bottom = 0.1
		simpleaxis(b)
		spine_shift(b)
		subplots_adjust(hspace = hspace, left = left)
		plt.xticks([-3000,-2000,-1000,0,1000], [-3,-2,-1,0,1])
		plt.title('Response locked PPR', size=12)
		plt.ylabel("PPR (Z)", size=10)
		plt.xlabel("Time from report (s)", size=10)
		plt.hist(0-self.response_time[self.answer_yes], bins=20, rwidth=0.8, weights = np.ones(condition[0].sum()+condition[1].sum()) / 2500 * ((axis()[3] - axis()[2])*3), bottom = plt.axis()[3]-0.2, color = 'r', alpha = 0.5)
		plt.hist(0-self.response_time[self.answer_no], bins=20, rwidth=0.8, weights = np.ones(condition[2].sum()+condition[3].sum()) / 2500 * ((axis()[3] - axis()[2])*3), bottom = plt.axis()[3]-0.3, color = 'b', alpha = 0.5)
		gca().spines["bottom"].set_linewidth(.5)
		gca().spines["left"].set_linewidth(.5)
		
		if self.experiment == 1:
			# Feedback
			c = plt.subplot(313)
			xc = np.arange(-499,1500)
			for i in range(len(condition)):
				plt.plot(xc, feed_means[i], linewidth=2, color = ['r','r','b','b'][i], alpha = [1,0.5,0.5,1][i])
				plt.fill_between( xc, (feed_means[i]+feed_sems[i]), (feed_means[i]-feed_sems[i]), color = ['r','r','b','b'][i], alpha=0.1 )
			plt.tick_params(axis='both', which='major', labelsize=8)
			plt.axvline(0, -1, 1, linewidth=1)
			# plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit[0].sum()) + " trials", "FA; " + str(fa[0].sum()) + " trials", "MISS; " + str(miss[0].sum()) + " trials", "CR; " + str(cr[0].sum()) + " trials"], loc = 2)
			bottom = 0.1
			simpleaxis(c)
			spine_shift(c)
			subplots_adjust(hspace = hspace, left = left)
			plt.xticks([-500,-0,500,1000,1500], [-.5,0,.5,1,1.5])
			plt.title('Feedback locked PPR', size=12)
			plt.ylabel("PPR (Z)", size=10)
			plt.xlabel("Time from feedback (s)", size=10)
			gca().spines["bottom"].set_linewidth(.5)
			gca().spines["left"].set_linewidth(.5)
		
		pp = PdfPages('Exp' + str(self.experiment) + '_PPR_response_figure_' + self.subject + '.pdf')
		figure_mean_pupil_locked_to_stimulus_response_feedback_SDT.savefig(pp, format='pdf')
		pp.close()
		
	
	
	def PPR_feedback_amplitude_within_stats(self):
		
		if self.experiment == 1:
			ppd_measure = self.ppd_feed_lin
			
			# Permutation Tests:
			if self.subject == 'rn':
				indices_to_test1 = [self.hit*self.confidence_0, self.hit*self.confidence_1, self.hit*self.confidence_2, self.cr*(self.confidence_0 + self.confidence_1)]
				indices_to_test2 = [self.hit*self.confidence_1, self.hit*self.confidence_2, self.hit*self.confidence_3, self.cr*(self.confidence_2 + self.confidence_3)]
				# indices_to_test1 = [hit*confidence_0, hit*confidence_1, hit*confidence_2, cr*confidence_0, cr*confidence_1, cr*confidence_2]
				# indices_to_test2 = [hit*confidence_1, hit*confidence_2, hit*confidence_3, cr*confidence_1, cr*confidence_2, cr*confidence_3]
			
			if self.subject == 'jwg':
				indices_to_test1 = [self.hit*(self.confidence_0 + self.confidence_1), self.cr*(self.confidence_0 + self.confidence_1)]
				indices_to_test2 = [self.hit*(self.confidence_2 + self.confidence_3), self.cr*(self.confidence_2 + self.confidence_3)]	
			
			observed_mean_difference_feed = []
			perm_results_feed = []
			significance_feed = []
			for i in range(len(indices_to_test1)):
				group1 = ppd_measure[indices_to_test1[i]]
				group2 = ppd_measure[indices_to_test2[i]]
				output = functions_jw.permutationTest(group1 = group1, group2 = group2, nrand = 5000)
				observed_mean_difference_feed.append(output[0]) 
				perm_results_feed.append(output[1])
				significance_feed.append(output[2])
			
			## PPD's after feedback:
			## plot_PPDs_feed3 for JWG, plot_PPDs_feed2 for RN
			pp = PdfPages('Exp' + str(self.experiment) + '_PPR_feedback_bars_1_' + self.subject + '.pdf')
			if self.subject == 'jwg':
				functions_jw.plot_PPDs_feed3(subject = self.subject, ppd_feed = ppd_measure, hit = self.hit, cr = self.cr, confidence_0 = self.confidence_0, confidence_1 = self.confidence_1, confidence_2 = self.confidence_2, confidence_3 = self.confidence_3, p1 = significance_feed[0], p2=significance_feed[1]).savefig(pp, format='pdf')
			if self.subject == 'rn':	
				functions_jw.plot_PPDs_feed2(subject = self.subject, ppd_feed = ppd_measure, hit = self.hit, cr = self.cr, confidence_0 = self.confidence_0, confidence_1 = self.confidence_1, confidence_2 = self.confidence_2, confidence_3 = self.confidence_3, p1 = significance_feed[0], p2=significance_feed[1], p3=significance_feed[2], p4=significance_feed[3]).savefig(pp, format='pdf')
			pp.close()
			
			# PPD's after feedback:
			# Here all PPD's per SDT condition per confidence bin are showed. (possible significant differences are not indicated)
			pp = PdfPages('Exp' + str(self.experiment) + '_PPR_feedback_bars_2_' + self.subject + '.pdf')
			functions_jw.plot_PPDs_feed(subject = self.subject, ppd_feed = ppd_measure, hit = self.hit, fa = self.fa, miss = self.miss, cr = self.cr, confidence_0 = self.confidence_0, confidence_1 = self.confidence_1, confidence_2 = self.confidence_2, confidence_3 = self.confidence_3).savefig(pp, format='pdf')
			pp.close()
			
			# # ROC Analysis:
			# indices_to_test1 = [incorrect]
			# indices_to_test2 = [correct]
			# out_i_feed = []
			# out_p_feed = []
			# for i in range(len(indices_to_test1)):
			# 	group1 = ppd_feed[indices_to_test1[i]]
			# 	group2 = ppd_feed[indices_to_test2[i]]
			# 	output = functions_jw.roc_analysis(group1 = group1, group2 = group2, nrand=1000)
			# 	out_i_feed.append(output[0])
			# 	out_p_feed.append(output[1])
	
	
	def pupil_GLM(self):
		
		# Set working directory and open hdf5 file
		this_dir = '/Research/PUPIL/PupilExperiment1/data/' + self.subject + '/'
		os.chdir(this_dir)
		h5f = openFile(( self.subject + '.hdf5'), mode = "r" ) # mode = "r" means 'read only'
		
		run = []
		for this_run in range(self.number_runs):
			search_name = 'run_' + str(this_run)
			for r in h5f.iterNodes(where = '/', classname = 'Group'):
				if search_name == r._v_name:
					run.append(r)
					break
		
		#################################################################################
		
		include = []
		for i in range(len(run)):
			
			# # measured:
			# pupil_diameter_lp_down.append(run[i].pupil_data_filtered.read()[1,:])
			# GLM_pupil_zscore_lp_hp_down.append(run[i].pupil_data_filtered.read()[4,:])
			# # predicted:
			# predicted_pupil.append( (blink_convolved[i]*mean(betas_blink)) + (stim_convolved[i]*mean(betas_stim)) + (resp_convolved[i]*mean(betas_resp)) + (conf_convolved[i]*mean(betas_conf)) + (feed_convolved[i]*mean(betas_feed)) + (down_convolved[i]*mean(betas_down)) + (up_convolved[i]*mean(betas_up)) )
			
			# times:
			gaze_timestamps = run[i].pupil_data_filtered.read()[0,:]
			trial_times = run[i].trial_times.read()
			trial_parameters = run[i].trial_parameters.read()
			
			timesss = []
			duration = 0
			include_indices = np.zeros(gaze_timestamps.shape[0], dtype = bool)
			for j in range(len(trial_parameters['trial_nr'])):
				timesss.append((gaze_timestamps > trial_times['trial_phase_timestamps'][j,1,0] - 500) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,1,0]))
				if sum(timesss[j]) > duration:
					duration = sum(timesss[j])
				include_indices = include_indices + timesss[j]
			
			include.append(include_indices)
		
		include_joined = np.concatenate(include, axis = 0)
		
		###################################################################################################################
		## DATA AND INDICES! ##############################################################################################
		
		blink_convolved = []
		blink_convolved2 = []
		stim_convolved = []
		stim_convolved2 = []
		resp_convolved = []
		resp_convolved2 = []
		resp_convolved_hit = []
		resp_convolved2_hit = []
		resp_convolved_fa = []
		resp_convolved2_fa = []
		resp_convolved_miss = []
		resp_convolved2_miss = []
		resp_convolved_cr = []
		resp_convolved2_cr = []
		conf_convolved = []
		conf_convolved2 = []
		feed_convolved = []
		feed_convolved2 = []
		down_convolved = []
		down_convolved2 = []
		down_convolved_hit = []
		down_convolved2_hit = []
		down_convolved_fa = []
		down_convolved2_fa = []
		down_convolved_miss = []
		down_convolved2_miss = []
		down_convolved_cr = []
		down_convolved2_cr = []
		up_convolved = []
		up_convolved2 = []
		up_convolved_hit = []
		up_convolved2_hit = []
		up_convolved_fa = []
		up_convolved2_fa = []
		up_convolved_miss = []
		up_convolved2_miss = []
		up_convolved_cr = []
		up_convolved2_cr = []
		GLM_pupil_diameter = []
		GLM_pupil_diameter2 = []
		
		for i in range(len(run)):
			
			# This downsampled times series per run. Z-scored per run based on ALL measurements (except ITI), to use for GLM:
			GLM_pupil_diameter.append(run[i].pupil_data_filtered.read()[6,:])
			GLM_pupil_diameter2.append(GLM_pupil_diameter[i]- mean(GLM_pupil_diameter[i][include[i]]))
			# blinks:
			blink_convolved.append( np.array(run[i].GLM_regressors_convolved.read()[0]) )
			blink_convolved2.append( blink_convolved[i]- mean(blink_convolved[i][include[i]]) )
			# stimulus:
			stim_convolved.append( np.array(run[i].GLM_regressors_convolved.read()[1]) )
			stim_convolved2.append( stim_convolved[i]- mean(stim_convolved[i][include[i]]) )
			# response:
			resp_convolved_hit.append( np.array(run[i].GLM_regressors_convolved.read()[2]) )
			resp_convolved2_hit.append( resp_convolved_hit[i]- mean(resp_convolved_hit[i][include[i]]) )
			resp_convolved_fa.append( np.array(run[i].GLM_regressors_convolved.read()[3]) )
			resp_convolved2_fa.append( resp_convolved_fa[i]- mean(resp_convolved_fa[i][include[i]]) )
			resp_convolved_miss.append( np.array(run[i].GLM_regressors_convolved.read()[4]) )
			resp_convolved2_miss.append( resp_convolved_miss[i]- mean(resp_convolved_miss[i][include[i]]) )
			resp_convolved_cr.append( np.array(run[i].GLM_regressors_convolved.read()[5]) )
			resp_convolved2_cr.append( resp_convolved_cr[i]- mean(resp_convolved_cr[i][include[i]]) )
			resp_convolved.append( (resp_convolved_hit[i]+resp_convolved_fa[i]+resp_convolved_miss[i]+resp_convolved_cr[i]) )
			resp_convolved2.append( resp_convolved[i] - mean(resp_convolved[i][include[i]]) )
			# confidence:
			conf_convolved.append( np.array(run[i].GLM_regressors_convolved.read()[6]) )
			conf_convolved2.append( conf_convolved[i]- mean(conf_convolved[i][include[i]]) )
			# feedback:
			feed_convolved.append( np.array(run[i].GLM_regressors_convolved.read()[7]) )
			feed_convolved2.append( feed_convolved[i]- mean(feed_convolved[i][include[i]]) )
			# down:
			down_convolved_hit.append( np.array(run[i].GLM_regressors_convolved.read()[8]) )
			down_convolved2_hit.append(down_convolved_hit[i]- mean(down_convolved_hit[i][include[i]]) )
			down_convolved_fa.append( np.array(run[i].GLM_regressors_convolved.read()[9]) )
			down_convolved2_fa.append( down_convolved_fa[i]- mean(down_convolved_fa[i][include[i]]) )
			down_convolved_miss.append( np.array(run[i].GLM_regressors_convolved.read()[10]) )
			down_convolved2_miss.append( down_convolved_miss[i]- mean(down_convolved_miss[i][include[i]]) )
			down_convolved_cr.append( np.array(run[i].GLM_regressors_convolved.read()[11]) )
			down_convolved2_cr.append( down_convolved_cr[i]- mean(down_convolved_cr[i][include[i]]) )
			down_convolved.append( (down_convolved_hit[i]+down_convolved_fa[i]+down_convolved_miss[i]+down_convolved_cr[i]) )
			down_convolved2.append( down_convolved[i] - mean(down_convolved[i][include[i]]) )
			# up:
			up_convolved_hit.append( np.array(run[i].GLM_regressors_convolved.read()[12]) )
			up_convolved2_hit.append( up_convolved_hit[i]- mean(up_convolved_hit[i][include[i]]) )
			up_convolved_fa.append( np.array(run[i].GLM_regressors_convolved.read()[13]) )
			up_convolved2_fa.append( up_convolved_fa[i]- mean(up_convolved_fa[i][include[i]]) )
			up_convolved_miss.append( np.array(run[i].GLM_regressors_convolved.read()[14]) )
			up_convolved2_miss.append( up_convolved_miss[i]- mean(up_convolved_miss[i][include[i]]) )
			up_convolved_cr.append( np.array(run[i].GLM_regressors_convolved.read()[15]) )
			up_convolved2_cr.append( up_convolved_cr[i]- mean(up_convolved_cr[i][include[i]]) )
			up_convolved.append( (up_convolved_hit[i]+up_convolved_fa[i]+up_convolved_miss[i]+up_convolved_cr[i]) )
			up_convolved2.append( up_convolved[i] - mean(up_convolved[i][include[i]]) )
		
		#####################
		## Across runs:  ####
		#####################
		# betas = ((designMatrix.T * designMatrix).I * designMatrix.T) * np.mat(pupil).T
		# residuals = np.mat(pupil).T - (designMatrix * np.mat(betas))
		
		betas_blink = []
		betas_stim = []
		betas_resp = []
		betas_resp_hit = []
		betas_resp_fa = []
		betas_resp_miss = []
		betas_resp_cr = []
		betas_conf = []
		betas_feed = []
		betas_down = []
		betas_down_hit = []
		betas_down_fa = []
		betas_down_miss = []
		betas_down_cr = []
		betas_up = []
		betas_up_hit = []
		betas_up_fa = []
		betas_up_miss = []
		betas_up_cr = []
		t_values = []
		p_values = []
		residuals = []
		for i in range(len(run)):
		
			pupil = GLM_pupil_diameter2[i]
			pupil = np.array(pupil, dtype= np.float32)
			pupil = np.matrix(pupil)
			
			designMatrix = np.array(np.vstack((blink_convolved2[i], stim_convolved2[i], resp_convolved2_hit[i], resp_convolved2_fa[i], resp_convolved2_miss[i], resp_convolved2_cr[i], conf_convolved2[i], feed_convolved2[i], down_convolved2_hit[i], down_convolved2_fa[i], down_convolved2_miss[i], down_convolved2_cr[i], up_convolved2_hit[i], up_convolved2_fa[i], up_convolved2_miss[i], up_convolved2_cr[i])), dtype = np.float32)
			designMatrix = np.mat(designMatrix)
			designMatrix = np.matrix(designMatrix).T
			
			# GLM:
			GLM = sm.GLM(pupil,designMatrix)
			GLM_results = GLM.fit()
			GLM_results.summary()
			
			# if split_up_by_SDT == False:
			# 	betas_blink.append( GLM_results.params[0] )
			# 	betas_stim.append( GLM_results.params[1] )
			# 	betas_resp.append( GLM_results.params[2] )
			# 	betas_conf.append( GLM_results.params[3] )
			# 	betas_feed.append( GLM_results.params[4] )
			# 	betas_down.append( GLM_results.params[5] )
			# 	betas_up.append( GLM_results.params[6] )
			
			betas_blink.append( GLM_results.params[0] )
			betas_stim.append( GLM_results.params[1] )
			betas_resp_hit.append( GLM_results.params[2] )
			betas_resp_fa.append( GLM_results.params[3] )
			betas_resp_miss.append( GLM_results.params[4] )
			betas_resp_cr.append( GLM_results.params[5] )
			betas_conf.append( GLM_results.params[6] )
			betas_feed.append( GLM_results.params[7] )
			betas_down_hit.append( GLM_results.params[8] )
			betas_down_fa.append( GLM_results.params[9] )
			betas_down_miss.append( GLM_results.params[10] )
			betas_down_cr.append( GLM_results.params[11] )
			betas_up_hit.append( GLM_results.params[12] )
			betas_up_fa.append( GLM_results.params[13] )
			betas_up_miss.append( GLM_results.params[14] )
			betas_up_cr.append( GLM_results.params[15] )
			
			t_values.append( GLM_results.tvalues )
			p_values.append( GLM_results.pvalues )
			residuals.append( GLM_results.resid_response )
		
		# Recombine separate betas for SDT condition into one: 
		betas_resp_matrix = np.vstack( (np.array(betas_resp_hit), np.array(betas_resp_fa), np.array(betas_resp_miss), np.array(betas_resp_cr)) ) 
		betas_resp = mean(betas_resp_matrix, axis=0)
		betas_down_matrix = np.vstack( (np.array(betas_down_hit), np.array(betas_down_fa), np.array(betas_down_miss), np.array(betas_down_cr)) ) 
		betas_down = mean(betas_down_matrix, axis=0)
		betas_up_matrix = np.vstack( (np.array(betas_up_hit), np.array(betas_up_fa), np.array(betas_up_miss), np.array(betas_up_cr)) ) 
		betas_up = mean(betas_up_matrix, axis=0)
		
		# T-test:
		t_test1 = ttest_1samp(np.array(betas_resp)-np.array(betas_stim), 0)
		t_test2 = ttest_1samp(np.array(betas_up)-np.array(betas_down), 0)
		
		
		# if split_up_by_SDT == True:
		# 	t_test1 = ttest_1samp(np.array(betas_resp_hit)-np.array(betas_resp_miss), 0)
		# 	t_test2 = ttest_1samp(np.array(betas_resp_fa)-np.array(betas_resp_cr), 0)
		# 	t_test3 = ttest_1samp(np.array(betas_up_hit)-np.array(betas_up_miss), 0)
		# 	t_test4 = ttest_1samp(np.array(betas_up_fa)-np.array(betas_up_cr), 0)
		
		# t_test1 = ttest_1samp(np.array(betas_blink),0)
		# t_test2 = ttest_1samp(np.array(betas_stim),0)
		# t_test3 = ttest_1samp(np.array(betas_resp),0)
		# t_test4 = ttest_1samp(np.array(betas_conf),0)
		# t_test5 = ttest_1samp(np.array(betas_feed),0)
		# t_test6 = ttest_1samp(np.array(betas_down),0)
		# t_test7 = ttest_1samp(np.array(betas_up),0)
		
		#####################################################################################
		###### PLOT #########################################################################
		
		os.chdir(self.this_dir + '/figures/')
		
		# BARPLOT BETAS:
		pp = PdfPages('Exp' + str(self.experiment) + '_GLM_betas_1_' + self.subject + '.pdf')
		fig = functions_jw.GLM_betas_barplot(self.subject, betas_stim, betas_resp, betas_down, betas_up, betas_feed, t_test1[1], t_test2[1])
		ylabel('beta value', size = '10')
		title('GLM', size = '12')
		fig.savefig(pp, format='pdf')
		pp.close()
		
		# PREDICTED VERSUS MEASURED:
		predicted_pupil = []
		stim_locked_lp_hp_down = []
		stim_locked_predicted_pupil = []
		resp_locked_lp_hp_down = []
		resp_locked_predicted_pupil = []
		
		for i in range(len(run)):
			
			# measured
			GLM_pupil_diameter2[i]
			# predicted:
			predicted_pupil.append( (blink_convolved2[i]*mean(betas_blink)) + (stim_convolved2[i]*mean(betas_stim)) + (resp_convolved2[i]*mean(betas_resp)) + (conf_convolved2[i]*mean(betas_conf)) + (feed_convolved2[i]*mean(betas_feed)) + (down_convolved2[i]*mean(betas_down)) + (up_convolved2[i]*mean(betas_up)) )
			
			# times:
			gaze_timestamps = run[i].pupil_data_filtered.read()[0,:]
			trial_times = run[i].trial_times.read()
			trial_parameters = run[i].trial_parameters.read()
			
			## MAKE STIMULUS LOCKED ARRAY'S
			timesss = []
			duration = 0
			for j in range(len(trial_parameters['trial_nr'])):
				timesss.append((gaze_timestamps > trial_times['trial_phase_timestamps'][j,1,0] - 1000) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,1,0]+5002))
				if sum(timesss[j]) > duration:
					duration = sum(timesss[j])
			
			# stim_locked_lp_hp_down:
			a = np.zeros((len(trial_parameters['trial_nr']), duration))
			a[:,:] = NaN
			for j in range(len(trial_parameters['trial_nr'])):
				a[j,:sum(timesss[j])] = GLM_pupil_diameter2[i][timesss[j]]
			stim_locked_lp_hp_down.append( a[:,0:200] )	
			
			# stim_locked_predicted_pupil:
			a = np.zeros((len(trial_parameters['trial_nr']), duration))
			a[:,:] = NaN
			for j in range(len(trial_parameters['trial_nr'])):
				a[j,:sum(timesss[j])] = predicted_pupil[i][timesss[j]]
			stim_locked_predicted_pupil.append( a[:,0:200] )
			
			## MAKE RESPONSE LOCKED ARRAY
			timesss = []
			duration = 0
			for j in range(len(trial_parameters['trial_nr'])):
				timesss.append((gaze_timestamps > trial_times['trial_phase_timestamps'][j,2,0] - 3500) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,2,0]+2502))
				if sum(timesss[j]) > duration:
					duration = sum(timesss[j])
			
			# resp_locked_lp_hp_down:
			b = np.ones((len(trial_parameters['trial_nr']), duration))
			b[:,:] = NaN 
			for j in range(len(trial_parameters['trial_nr'])):
				b[j,:sum(timesss[j])] = GLM_pupil_diameter2[i][timesss[j]]
			resp_locked_lp_hp_down.append( b[:,0:200] )
			
			# resp_locked_predicted_pupil
			b = np.ones((len(trial_parameters['trial_nr']), duration))
			b[:,:] = NaN 
			for j in range(len(trial_parameters['trial_nr'])):
				b[j,:sum(timesss[j])] = predicted_pupil[i][timesss[j]]
			resp_locked_predicted_pupil.append( b[:,0:200] )
		
		stim_locked_lp_hp_down_joined = np.vstack(stim_locked_lp_hp_down)
		stim_locked_predicted_pupi_joined = np.vstack(stim_locked_predicted_pupil)
		resp_locked_lp_hp_down_joined = np.vstack(resp_locked_lp_hp_down)
		resp_locked_predicted_pupi_joined = np.vstack(resp_locked_predicted_pupil)
		
		
		# ########################
		# #### R-squared #########
		# stim_measured_concatenate = np.concatenate(stim_locked_lp_hp_down_joined)
		# stim_predicted_concatenate = np.concatenate(stim_locked_predicted_pupi_joined)
		# resp_measured_concatenate = np.concatenate(resp_locked_lp_hp_down_joined)
		# resp_predicted_concatenate = np.concatenate(resp_locked_predicted_pupi_joined)
		# r_squared_stim = round(stats.linregress(stim_measured_concatenate, stim_predicted_concatenate)[2]**2,3)
		# r_squared_resp = round(stats.linregress(resp_measured_concatenate, resp_predicted_concatenate)[2]**2,3)
		
		# Plot:
		mean2 = bottleneck.nanmean(stim_locked_lp_hp_down_joined, axis=0)
		mean3 = bottleneck.nanmean(stim_locked_predicted_pupi_joined, axis=0)
		mean5 = bottleneck.nanmean(resp_locked_lp_hp_down_joined, axis=0)
		mean6 = bottleneck.nanmean(resp_locked_predicted_pupi_joined, axis=0)
		sem2 = bottleneck.nanstd(stim_locked_lp_hp_down_joined, axis=0) / sp.sqrt(stim_locked_lp_hp_down_joined.shape[0])
		sem3 = bottleneck.nanstd(stim_locked_predicted_pupi_joined, axis=0) / sp.sqrt(stim_locked_predicted_pupi_joined.shape[0])
		sem5 = bottleneck.nanstd(resp_locked_lp_hp_down_joined, axis=0) / sp.sqrt(resp_locked_lp_hp_down_joined.shape[0])
		sem6 = bottleneck.nanstd(resp_locked_predicted_pupi_joined, axis=0) / sp.sqrt(resp_locked_predicted_pupi_joined.shape[0])
		
		r_squared_stim = round(stats.linregress(mean2, mean3)[2]**2,3)
		r_squared_resp = round(stats.linregress(mean5, mean6)[2]**2,3)
		
		figure_predicted_measured = plt.figure(figsize=(4, 6))
		hspace = 0.40
		left = 0.2
		# Stimulus 
		a = plt.subplot(211)
		xa = np.arange(-40,160)
		plt.plot(xa, mean2, linewidth=2, color = 'k', alpha = 0.25)
		plt.fill_between( xa, mean2+sem2, mean2-sem2, color = 'k', alpha=0.1 )
		plt.plot(xa, mean3, linewidth=2, color = 'r', alpha = 0.80)
		plt.fill_between( xa, mean3+sem3, mean3-sem3, color = 'b', alpha=0.1 )
		plt.axvline(0, -1, 1, linewidth=1)
		plt.xlim( (-40, 160) )
		leg = plt.legend(["Measured pupil", "Predicted pupil"], loc = 2, fancybox = True)
		leg.get_frame().set_alpha(0.9)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize(7)    # the legend text fontsize
			for l in leg.get_lines():
				l.set_linewidth(2)  # the legend line width
		plt.tick_params(axis='both', which='major', labelsize=8)
		# plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit.sum()) + " trials", "FA; " + str(fa.sum()) + " trials", "MISS; " + str(miss.sum()) + " trials", "CR; " + str(cr.sum()) + " trials"], loc = 2)
		simpleaxis(a)
		spine_shift(a)
		plt.xticks([-40,0,40,80,120,160], [-1,0,1,2,3,4]) 
		subplots_adjust(hspace = hspace, left = left, bottom = 0.1)
		plt.title('Stimulus locked PPR', size=12)
		plt.ylabel("PPR (Z)", size=10)
		plt.xlabel("Time from stimulus (s)", size=10)
		plt.text(-20, 0.3, 'R-squared = ' + str(r_squared_stim))
		gca().spines["bottom"].set_linewidth(.5)
		gca().spines["left"].set_linewidth(.5)
		bottom = 0.1
		
		# Response
		b = plt.subplot(212)
		xb = np.arange(-140,60)
		plt.plot(xb, mean5, linewidth=2, color = 'k', alpha = 0.25)
		plt.fill_between( xb, mean5+sem5, mean5-sem5, color = 'k', alpha=0.1 )
		plt.plot(xb, mean6, linewidth=2, color = 'r', alpha = 0.80)
		plt.fill_between( xb, mean6+sem6, mean6-sem6, color = 'b', alpha=0.1 )
		plt.axvline(0, -1, 1, linewidth=1)
		plt.xlim( (-140, 60) )
		plt.tick_params(axis='both', which='major', labelsize=8)
		simpleaxis(b)
		spine_shift(b)
		plt.xticks([-120,-80,-40,0,40], [-3,-2,-1,0,1]) 
		subplots_adjust(hspace = hspace, left = left, bottom = 0.1)
		plt.title('Response locked PPR', size=12)
		plt.ylabel("PPR (Z)", size=10)
		plt.xlabel("Time from report (s)", size=10)
		plt.text(-120, 0.3, 'R-squared = ' + str(r_squared_resp))
		gca().spines["bottom"].set_linewidth(.5)
		gca().spines["left"].set_linewidth(.5)
		
		pp = PdfPages('Exp' + str(self.experiment) + '_GLM_predicted_measured_response_figure_' + self.subject + '.pdf')
		figure_predicted_measured.savefig(pp, format='pdf')
		pp.close()
		
		# #########################
		# # Combined over runs: ###
		# #########################
		# 
		# # Join regressors of each run:
		# # regressor_trial_on_joined = np.concatenate(regressor_trial_on, axis=1)
		# # regressor_ramp_joined = np.concatenate(regressor_ramp, axis=1)
		# # regressor_whole_trials_joined = np.concatenate(regressor_whole_trials, axis=1)
		# 
		# blink_convolved_joined = np.concatenate(blink_convolved2, axis=1)
		# stim_convolved_joined = np.concatenate(stim_convolved2, axis=1)
		# resp_convolved_hit_joined = np.concatenate(resp_convolved2_hit, axis=1)
		# resp_convolved_fa_joined = np.concatenate(resp_convolved2_fa, axis=1)
		# resp_convolved_miss_joined = np.concatenate(resp_convolved2_miss, axis=1)
		# resp_convolved_cr_joined = np.concatenate(resp_convolved2_cr, axis=1)
		# conf_convolved_joined = np.concatenate(conf_convolved2, axis=1)
		# feed_convolved_joined = np.concatenate(feed_convolved2, axis=1)
		# down_convolved_hit_joined = np.concatenate(down_convolved2_hit, axis=1)
		# down_convolved_fa_joined = np.concatenate(down_convolved2_fa, axis=1)
		# down_convolved_miss_joined = np.concatenate(down_convolved2_miss, axis=1)
		# down_convolved_cr_joined = np.concatenate(down_convolved2_cr, axis=1)
		# up_convolved_hit_joined = np.concatenate(up_convolved2_hit, axis=1)
		# up_convolved_fa_joined = np.concatenate(up_convolved2_fa, axis=1)
		# up_convolved_miss_joined = np.concatenate(up_convolved2_miss, axis=1)
		# up_convolved_cr_joined = np.concatenate(up_convolved2_cr, axis=1)
		# pupil_joined = np.concatenate(GLM_pupil_diameter2, axis=1)
		# 
		# pupil_joined = np.array(pupil_joined, dtype= np.float32)
		# pupil_joined = np.matrix(pupil_joined)
		# 
		# designMatrix = np.array(np.vstack((blink_convolved_joined, stim_convolved_joined, resp_convolved_hit_joined, resp_convolved_fa_joined, resp_convolved_miss_joined, resp_convolved_cr_joined, conf_convolved_joined, feed_convolved_joined, down_convolved_hit_joined, down_convolved_fa_joined, down_convolved_miss_joined, down_convolved_cr_joined, up_convolved_hit_joined, up_convolved_fa_joined, up_convolved_miss_joined, up_convolved_cr_joined)), dtype= np.float32)
		# designMatrix = np.mat(designMatrix)
		# designMatrix = np.matrix(designMatrix).T
		# 
		# # betas = ((designMatrix.T * designMatrix).I * designMatrix.T) * np.mat(pupil_joined).T
		# # residuals = np.mat(pupil_joined).T - (designMatrix * np.mat(betas))
		# 
		# GLM = sm.GLM(pupil_joined,designMatrix)
		# GLM_results = GLM.fit()
		# GLM_results.summary()
		# 
		# betas = GLM_results.params
		# t_values = GLM_results.tvalues
		# p_values = GLM_results.pvalues
		# residuals = GLM_results.resid_response
		# 
		# contrast_matrix1 = np.array([1,0,0,0,0,0,0])
		# contrast_matrix2 = np.array([0,1,0,0,0,0,0])
		# contrast_matrix3 = np.array([0,0,1,0,0,0,0])
		# contrast_matrix4 = np.array([0,0,0,1,0,0,0])
		# contrast_matrix5 = np.array([0,0,0,0,1,0,0])
		# contrast_matrix6 = np.array([0,0,0,0,0,1,0])
		# contrast_matrix7 = np.array([0,0,0,0,0,0,1])
		# contrast_matrix8 = np.array([0,-1,1,0,0,0,0])
		# contrast_matrix9 = np.array([0,0,0,0,0,1,-1])
		# 
		# contrast1 = GLM_results.f_test(contrast_matrix1)
		# contrast2 = GLM_results.f_test(contrast_matrix2)
		# contrast3 = GLM_results.f_test(contrast_matrix3)
		# contrast4 = GLM_results.f_test(contrast_matrix4)
		# contrast5 = GLM_results.f_test(contrast_matrix5)
		# contrast6 = GLM_results.f_test(contrast_matrix6)
		# contrast7 = GLM_results.f_test(contrast_matrix7)
		# contrast8 = GLM_results.f_test(contrast_matrix8)
		# contrast9 = GLM_results.f_test(contrast_matrix9)
	


class across_subject_stats():
	
	def __init__(self, experiment, this_dir, sample_rate, downsample_rate):
		
		self.experiment = experiment
		self.this_dir = this_dir
		self.sample_rate = sample_rate
		self.downsample_rate = downsample_rate
		
		if self.experiment == 1:
			self.subject = ('jwg', 'rn', 'dh', 'dl')
		if self.experiment == 2:
			subject = ('jw', 'ml', 'te', 'ln', 'td', 'ch', 'lm', 'al', 'dli', 'kr', 'vp', 'js', 'ek', 'dho', 'tk')
		
		# cut-off 35 trials:
		# self.subject = ('jw', 'ml', 'te', 'ln', 'td', 'lm', 'al', 'dli', 'vp', 'ek', 'dho', 'tk')
		# # cut-off 40 trials:
		# self.subject = ('jw', 'ml', 'te', 'ln', 'td', 'al', 'vp', 'ek', 'tk')
		
		os.chdir(self.this_dir)
		
		# LIST VARIABLES (one entry per subject):
		df = []
		self.stimulus_present = []
		self.answer_yes = []
		self.answer_no = []
		self.correct = []
		self.incorrect = []
		self.hit = []
		self.fa = []
		self.miss = []
		self.cr = []
		self.bpd = []
		self.ppd = []
		self.ppd_mean = []
		self.ppd_lin = []
		self.ppd_lin_RT = []
		self.ppd_feed_lin = []
		self.response_time = []
		self.confidence = []
		self.confidence_0 = []
		self.confidence_1 = []
		self.confidence_2 = []
		self.confidence_3 = []
		self.d_prime_per_run = []
		self.d_prime_overall = []
		self.criterion_per_run = []
		self.criterion_overall = []
		for i in range(len(self.subject)):
			df.append(pd.load(self.subject[i] + '_data'))
			self.stimulus_present.append( np.array(df[i]['stimulus'], dtype = bool))
			self.answer_yes.append( np.array(df[i]['answer'], dtype = bool))
			self.answer_no.append( -self.answer_yes[i])
			self.correct.append( (self.stimulus_present[i]*self.answer_yes[i]) + (-self.stimulus_present[i]*self.answer_no[i])) 
			self.incorrect.append( -self.correct[i])
			self.hit.append( self.answer_yes[i]*self.correct[i])
			self.fa.append( self.answer_yes[i]*self.incorrect[i])
			self.miss.append( self.answer_no[i]*self.incorrect[i])
			self.cr.append( self.answer_no[i]*self.correct[i])
			if self.experiment == 1:
				self.confidence.append( df[i]['confidence'])
				self.confidence_0.append( np.array(self.confidence[i] == 0))
				self.confidence_1.append( np.array(self.confidence[i] == 1))
				self.confidence_2.append( np.array(self.confidence[i] == 2))
				self.confidence_3.append( np.array(self.confidence[i] == 3))
			self.response_time.append( np.array(df[i]['response_time']))
			self.bpd.append( np.array(df[i]['bpd']))
			self.ppd.append( np.array(df[i]['ppd']))
			self.ppd_mean.append( np.array(df[i]['ppd_mean']))
			self.ppd_lin.append( np.array(df[i]['ppd_lin']))
			self.ppd_lin_RT.append( np.array(df[i]['ppd_lin_RT']))
			if self.experiment == 1:
				self.ppd_feed_lin.append( np.array(df[i]['ppd_feed_lin']))
			self.d_prime_per_run.append( np.array(df[i]['d_prime_per_run']))
			self.d_prime_per_run[i] = self.d_prime_per_run[i][-np.isnan(self.d_prime_per_run[i])]
			self.criterion_per_run.append( np.array(df[i]['criterion_per_run']))
			self.criterion_per_run[i] = self.criterion_per_run[i][-np.isnan(self.criterion_per_run[i])]
			self.criterion_overall.append(np.array(df[i]['criterion_overall'])[0])
			self.d_prime_overall.append(np.array(df[i]['d_prime_overall'])[0])
		
		# CONCATENATED ACCROSS SUBJECTS:
		self.YES = np.array(np.concatenate(self.answer_yes))
		self.NO = np.array(np.concatenate(self.answer_no))
		self.CORRECT = np.array(np.concatenate(self.correct))
		self.INCORRECT = -self.CORRECT
		self.PRESENT = np.array(np.concatenate(self.stimulus_present))
		self.ABSENT = -self.PRESENT
		self.PPD = np.concatenate(self.ppd)
		self.PPD_MEAN = np.concatenate(self.ppd_mean)
		self.PPD_LIN = np.concatenate(self.ppd_lin)
		self.PPD_LIN_RT = np.concatenate(self.ppd_lin_RT)
		if self.experiment == 1:
			self.PPD_FEED_LIN = np.concatenate(self.ppd_feed_lin)
			
		self.SUBJECT = []
		for i in range(len(self.subject)):
			self.SUBJECT.append(np.repeat(i,len(self.ppd_lin[i])))
		self.SUBJECT = np.concatenate(self.SUBJECT, axis = 1)
		
		##########################################
		## INDICES BASED ON CRITERION ############
		subject_criterion = np.array(self.criterion_overall)
		median_subject_criterion = median(subject_criterion)
		self.liberal_indices = subject_criterion < median_subject_criterion
		self.conservative_indices = subject_criterion >= median_subject_criterion
	
	
	def PPR_amplitude_across_stats(self):
		
		os.chdir(self.this_dir + '/figures/')
		
		use = self.ppd_lin
		# if str(this_ppr_measure)
		# 	for_pp = 'ppd_lin_'
		
		hit_means = zeros(len(self.subject))
		fa_means = zeros(len(self.subject))
		miss_means = zeros(len(self.subject))
		cr_means = zeros(len(self.subject))
		yes_means = zeros(len(self.subject))
		no_means = zeros(len(self.subject))
		correct_means = zeros(len(self.subject))
		incorrect_means = zeros(len(self.subject))
		
		# Normalize:
		# for i in range(len(self.subject)):
		# 	hit_means[i] = mean(use[i][hit[i]]) / ((mean(use[i][hit[i]]) + mean(use[i][fa[i]]) + mean(use[i][miss[i]]) + mean(use[i][cr[i]])) / 4)
		# 	fa_means[i] = mean(use[i][fa[i]]) / ((mean(use[i][hit[i]]) + mean(use[i][fa[i]]) + mean(use[i][miss[i]]) + mean(use[i][cr[i]])) / 4)
		# 	miss_means[i] = mean(use[i][miss[i]]) / ((mean(use[i][hit[i]]) + mean(use[i][fa[i]]) + mean(use[i][miss[i]]) + mean(use[i][cr[i]])) / 4)
		# 	cr_means[i] = mean(use[i][cr[i]]) / ((mean(use[i][hit[i]]) + mean(use[i][fa[i]]) + mean(use[i][miss[i]]) + mean(use[i][cr[i]])) / 4)
		# 	yes_means[i] = mean(use[i][answer_yes[i]]) / ((mean(use[i][answer_yes[i]]) + mean(use[i][answer_no[i]])) / 2 )
		# 	no_means[i] = mean(use[i][answer_no[i]]) / ((mean(use[i][answer_yes[i]]) + mean(use[i][answer_no[i]])) / 2 )
		# 	correct_means[i] = mean(use[i][correct[i]]) / ((mean(use[i][correct[i]]) + mean(use[i][incorrect[i]])) / 2 )
		# 	incorrect_means[i] = mean(use[i][incorrect[i]]) / ((mean(use[i][correct[i]]) + mean(use[i][incorrect[i]])) / 2 )
		for i in range(len(self.subject)):
			hit_means[i] = mean(use[i][self.hit[i]])
			fa_means[i] = mean(use[i][self.fa[i]])
			miss_means[i] = mean(use[i][self.miss[i]])
			cr_means[i] = mean(use[i][self.cr[i]])
			yes_means[i] = mean(use[i][self.answer_yes[i]])
			no_means[i] = mean(use[i][self.answer_no[i]])
			correct_means[i] = mean(use[i][self.correct[i]])
			incorrect_means[i] = mean(use[i][self.incorrect[i]])
			
		# hit_means = hit_means[conservative]
		# fa_means = fa_means[conservative]
		# miss_means = miss_means[conservative]
		# cr_means = cr_means[conservative]
		# yes_means = yes_means[conservative]
		# no_means = no_means[conservative]
		# correct_means = correct_means[conservative]
		# incorrect_means = incorrect_means[conservative]
		
		# t-test:
		p1 = ttest_rel(hit_means, miss_means)[1]
		p2 = ttest_rel(fa_means, cr_means)[1]
		p3 = ttest_rel(yes_means, no_means)[1]
		p4 = ttest_rel(correct_means, incorrect_means)[1]
		# wilcoxon:
		p5 = wilcoxon(hit_means, miss_means)[1]
		p6 = wilcoxon(fa_means, cr_means)[1]
		p7 = wilcoxon(yes_means, no_means)[1]
		p8 = wilcoxon(correct_means, incorrect_means)[1]
		
		fig1 = functions_jw.sdt_barplot(subject = 'Across', hit = hit_means, fa = fa_means, miss = miss_means, cr = cr_means, p1 = p1, p2=p2, values = True)
		ylabel('PPR amplitude (linearly projected)', size = '10')
		title(str(len(self.subject)) + ' subjects - mean mean PPR amplitude', size = '12')
		pp = PdfPages(('Exp' + str(self.experiment) + '_PPR_bars_AccrosSubjects1_T-tests.pdf'))
		fig1.savefig(pp, format='pdf')
		pp.close()
		
		fig2 = functions_jw.sdt_barplot(subject = 'Across', hit = hit_means, fa = fa_means, miss = miss_means, cr = cr_means, p1 = p3, p2=p4, values = True, type_plot = 2)
		ylabel('PPR amplitude (linearly projected)', size = '10')
		title(str(len(self.subject)) + ' subjects - mean mean PPR amplitude', size = '12')
		pp = PdfPages('Exp' + str(self.experiment) + '_PPR_bars_AccrosSubjects2_T-tests.pdf')
		fig2.savefig(pp, format='pdf')
		pp.close()
		
		fig3 = functions_jw.sdt_barplot(subject = 'Across', hit = hit_means, fa = fa_means, miss = miss_means, cr = cr_means, p1 = p5, p2=p6, values = True)
		ylabel('PPR amplitude (linearly projected)', size = '10')
		title(str(len(self.subject)) + ' subjects - mean mean PPR amplitude', size = '12')
		pp = PdfPages(('Exp' + str(self.experiment) + '_PPR_bars_AccrosSubjects1_Wilcoxon.pdf'))
		fig3.savefig(pp, format='pdf')
		pp.close()
		
		fig4 = functions_jw.sdt_barplot(subject = 'Across', hit = hit_means, fa = fa_means, miss = miss_means, cr = cr_means, p1 = p7, p2=p8, values = True, type_plot = 2)
		ylabel('PPR amplitude (linearly projected)', size = '10')
		title(str(len(self.subject)) + ' subjects - mean PPR amplitude', size = '12')
		pp = PdfPages('Exp' + str(self.experiment) + '_PPR_bars_AccrosSubjects2_Wilcoxon.pdf')
		fig4.savefig(pp, format='pdf')
		pp.close()
	
	
	def BPD_across_stats(self):
		
		os.chdir(self.this_dir + '/figures/')
		
		use = self.bpd
		
		hit_means = zeros(len(self.subject))
		fa_means = zeros(len(self.subject))
		miss_means = zeros(len(self.subject))
		cr_means = zeros(len(self.subject))
		yes_means = zeros(len(self.subject))
		no_means = zeros(len(self.subject))
		correct_means = zeros(len(self.subject))
		incorrect_means = zeros(len(self.subject))
		for i in range(len(self.subject)):
			hit_means[i] = mean(use[i][self.hit[i]])
			fa_means[i] = mean(use[i][self.fa[i]])
			miss_means[i] = mean(use[i][self.miss[i]])
			cr_means[i] = mean(use[i][self.cr[i]])
			yes_means[i] = mean(use[i][self.answer_yes[i]])
			no_means[i] = mean(use[i][self.answer_no[i]])
			correct_means[i] = mean(use[i][self.correct[i]])
			incorrect_means[i] = mean(use[i][self.incorrect[i]])
		
		# t-test:
		p1 = ttest_rel(hit_means, miss_means)[1]
		p2 = ttest_rel(fa_means, cr_means)[1]
		p3 = ttest_rel(yes_means, no_means)[1]
		p4 = ttest_rel(correct_means, incorrect_means)[1]
		
		# wilcoxon:
		p5 = wilcoxon(hit_means, miss_means)[1]
		p6 = wilcoxon(fa_means, cr_means)[1]
		p7 = wilcoxon(yes_means, no_means)[1]
		p8 = wilcoxon(correct_means, incorrect_means)[1]
		
		fig = functions_jw.sdt_barplot(subject = 'All', hit = hit_means, fa = fa_means, miss = miss_means, cr = cr_means, p1 = p1, p2=p2, values = True)
		ylabel('PPR amplitude (linearly projected)', size = '10')
		title(str(len(self.subject)) + ' subjects - mean BPD', size = '12')
		# ylim((-0.3,0.3))
		pp = PdfPages(('Exp' + str(self.experiment) + '_BPD_bars_AccrosSubjects1_T-tests.pdf'))
		fig.savefig(pp, format='pdf')
		pp.close()
		
		fig = functions_jw.sdt_barplot(subject = 'All', hit = hit_means, fa = fa_means, miss = miss_means, cr = cr_means, p1 = p3, p2=p4, values = True, type_plot = 2)
		ylabel('PPR amplitude (linearly projected)', size = '10')
		title(str(len(self.subject)) + ' subjects - mean BPD', size = '12')
		# ylim((-0.3,0.3))
		pp = PdfPages('Exp' + str(self.experiment) + '_BPD_bars_AccrosSubjects2_T-tests.pdf')
		fig.savefig(pp, format='pdf')
		pp.close()
		
		fig = functions_jw.sdt_barplot(subject = 'All', hit = hit_means, fa = fa_means, miss = miss_means, cr = cr_means, p1 = p5, p2=p6, values = True)
		ylabel('PPR amplitude (linearly projected)', size = '10')
		title(str(len(self.subject)) + ' subjects - mean BPD', size = '12')
		# ylim((-0.3,0.3))
		pp = PdfPages(('Exp' + str(self.experiment) + '_BPD_bars_AccrosSubjects1_Wilcoxon.pdf'))
		fig.savefig(pp, format='pdf')
		pp.close()
		
		fig = functions_jw.sdt_barplot(subject = 'All', hit = hit_means, fa = fa_means, miss = miss_means, cr = cr_means, p1 = p7, p2=p8, values = True, type_plot = 2)
		ylabel('PPR amplitude (linearly projected)', size = '10')
		title(str(len(self.subject)) + ' subjects - mean BPD', size = '12')
		# ylim((-0.3,0.3))
		pp = PdfPages('Exp' + str(self.experiment) + '_BPD_bars_AccrosSubjects2_Wilcoxon.pdf')
		fig.savefig(pp, format='pdf')
		pp.close()
	
	
	def ANOVA(self):
		
		# ANOVA ACCROSS SUBJECTS:
		d = rlc.OrdDict([ ('present', robjects.IntVector(list(np.asarray(self.PRESENT.ravel(), dtype=int)))), ('yes', robjects.IntVector(list(np.asarray(self.YES.ravel(), dtype=int)))), ('correct', robjects.IntVector(list(np.asarray(self.CORRECT.ravel(), dtype=int)))), ('subject', robjects.IntVector(list(self.SUBJECT.ravel()))), ('PPD_LIN', robjects.FloatVector(list(self.PPD.ravel()))), ('PPD_LIN_RT', robjects.FloatVector(list(self.PPD_LIN_RT.ravel()))) ])
		robjects.r.assign('dataf', robjects.DataFrame(d))
		robjects.r('attach(dataf)')
		
		# OPTION 1:
		res1 = robjects.r('res1 = summary(aov(PPD_LIN ~ factor(present)*factor(yes) + Error(factor(subject)), dataf))')
		robjects.r('print(res1)')
		res2 = robjects.r('res2 = summary(aov(PPD_LIN ~ factor(correct)*factor(yes) + Error(factor(subject)), dataf))')
		robjects.r('print(res2)')
		
		# OPTION 2: ((CORRECT I THINK!))
		res3 = robjects.r('res3 = summary(aov(PPD_LIN ~ factor(yes) + Error(factor(subject)/factor(yes)), dataf))')
		robjects.r('print(res3)')
		res4 = robjects.r('res4 = summary(aov(PPD_LIN ~ factor(present) + Error(factor(subject)/factor(present)), dataf))')
		robjects.r('print(res4)')
		res5 = robjects.r('res5 = summary(aov(PPD_LIN ~ factor(correct) + Error(factor(subject)/factor(correct)), dataf))')
		robjects.r('print(res5)')
		
		# OPTION 3:
		res6 = robjects.r('res6 = summary(aov(PPD_LIN ~ factor(present)*factor(yes) + Error(factor(subject)/(factor(present)*factor(yes))), data = dataf))')
		robjects.r('print(res6)')
		res7 = robjects.r('res7 = summary(aov(PPD_LIN ~ factor(correct)*factor(yes) + Error(factor(subject)/(factor(correct)*factor(yes))), data = dataf))')
		robjects.r('print(res7)')	
	
	
	def correlation_PPRa_BPD(self):
		
		"Last plot all the way down is the pooled across subjects one."
		
		os.chdir(self.this_dir + '/figures/')
		
		ppd_measures = self.ppd_lin
		bpd_measures = self.bpd
		reg_slope = []
		reg_intercept = []
		reg_p = []
		reg_r = []
		
		fig = plt.figure(figsize=(8,16))	
		
		# Subject 1:
		ppd = ppd_measures[0]
		bpd = bpd_measures[0]
		slope, intercept, r_value, p_value, std_err = stats.linregress(bpd,ppd)
		(m,b) = sp.polyfit(bpd, ppd, 1)
		phasic_pupil_diameter_p = sp.polyval([m,b], bpd)
		ax2 = plt.subplot2grid( (4,2), (2,0), colspan=1, rowspan=1 )
		ax2.plot(bpd,phasic_pupil_diameter_p, color = 'k', linewidth = 1.5)
		ax2.scatter(bpd, ppd, color='#808080', alpha = 0.75)
		ax2.set_title('subject 1', size = 12)
		plt.tick_params(axis='both', which='major', labelsize=10)
		if round(p_value,5) < 0.005:
			ax2.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
		else:	
			ax2.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
		simpleaxis(ax2)
		spine_shift(ax2)
		# ax2.set_xlabel('BPD (Z)', size = 'x-large')
		ax2.set_ylabel('PPR amplitude (linearly projected)', size = 10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		reg_slope.append(slope)
		reg_intercept.append(intercept)
		reg_r.append(r_value)
		reg_p.append(p_value)
		# Subject 2:
		ppd = ppd_measures[1]
		bpd = bpd_measures[1]
		slope, intercept, r_value, p_value, std_err = stats.linregress(bpd,ppd)
		(m,b) = sp.polyfit(bpd, ppd, 1)
		phasic_pupil_diameter_p = sp.polyval([m,b], bpd)
		ax3 = plt.subplot2grid((4,2), (2,1), colspan=1, rowspan=1 )
		ax3.plot(bpd,phasic_pupil_diameter_p, color = 'k', linewidth = 1.5)
		ax3.scatter(bpd, ppd, color='#808080', alpha = 0.75)
		# ax3.set_ylabel('PPD - linearly projected', size = 'x-large')
		ax3.set_title('subject 2', size = 12)
		plt.tick_params(axis='both', which='major', labelsize=10)
		if round(p_value,5) < 0.005:
			ax3.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
		else:	
			ax3.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
		simpleaxis(ax3)
		spine_shift(ax3)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		reg_slope.append(slope)
		reg_intercept.append(intercept)
		reg_r.append(r_value)
		reg_p.append(p_value)
		# Subject 3:
		ppd = ppd_measures[2]
		bpd = bpd_measures[2]
		slope, intercept, r_value, p_value, std_err = stats.linregress(bpd,ppd)
		(m,b) = sp.polyfit(bpd, ppd, 1)
		phasic_pupil_diameter_p = sp.polyval([m,b], bpd)
		ax4 = plt.subplot2grid((4,2), (3,0), colspan=1, rowspan=1)
		ax4.plot(bpd,phasic_pupil_diameter_p, color = 'k', linewidth = 1.5)
		ax4.scatter(bpd, ppd, color='#808080', alpha = 0.75)
		ax4.set_title('subject 3', size = 12)
		plt.tick_params(axis='both', which='major', labelsize=10)
		if round(p_value,5) < 0.005:
			ax4.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
		else:	
			ax4.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
		simpleaxis(ax4)
		spine_shift(ax4)
		ax4.set_xlabel('BPD', size = 10)
		ax4.set_ylabel('PPR amplitude (linearly projected)', size = 10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		reg_slope.append(slope)
		reg_intercept.append(intercept)
		reg_r.append(r_value)
		reg_p.append(p_value)
		# Subject 4:
		ppd = ppd_measures[3]
		bpd = bpd_measures[3]
		slope, intercept, r_value, p_value, std_err = stats.linregress(bpd,ppd)
		(m,b) = sp.polyfit(bpd, ppd, 1)
		phasic_pupil_diameter_p = sp.polyval([m,b], bpd)
		ax5 = plt.subplot2grid((4,2), (3,1), colspan=1, rowspan=1)
		ax5.plot(bpd,phasic_pupil_diameter_p, color = 'k', linewidth = 1.5)
		ax5.scatter(bpd, ppd, color='#808080', alpha = 0.75)
		ax5.set_title('subject 4', size = 12)
		plt.tick_params(axis='both', which='major', labelsize=10)
		if round(p_value,5) < 0.005:
			ax5.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
		else:	
			ax5.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
		simpleaxis(ax5)
		spine_shift(ax5)
		ax5.set_xlabel('BPD', size = 10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		reg_slope.append(slope)
		reg_intercept.append(intercept)
		reg_r.append(r_value)
		reg_p.append(p_value)
		# Across subjects:
		reg_slope = mean(reg_slope)
		reg_intercept = mean(reg_intercept)
		reg_plot = reg_intercept + (reg_slope * linspace(-5,6))
		p_value = mean(reg_p)
		r_value = mean(reg_r)
		ax1 = plt.subplot2grid((4,2), (0,0), colspan=2, rowspan=2)
		ax1.scatter(np.concatenate(bpd_measures), np.concatenate(ppd_measures), color='#808080', alpha = 0.75)
		ax1.plot(linspace(-5,6), reg_plot, color = 'k', linewidth = 1.5)
		plt.tick_params(axis='both', which='major', labelsize=10)
		ax1.set_title("Correlation PPR amplitude and BPD, across subjects", size = 12)
		if round(p_value,5) < 0.005:
			ax1.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/16), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/16),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
		else:	
			ax1.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/16), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/16),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
		simpleaxis(ax1)
		spine_shift(ax1)
		ax1.set_xlabel('BPD', size = 10)
		ax1.set_ylabel('PPR amplitude (linearly projected)', size = 10)
		plt.gca().spines["bottom"].set_linewidth(.5)
		plt.gca().spines["left"].set_linewidth(.5)
		
		pp = PdfPages('Exp' + str(self.experiment) + '_Correlation_PPRa_BPD_AcrossSubjects.pdf')
		fig.savefig(pp, format='pdf')
		pp.close()
	
	
	def collapsed_response_figure(self):
		
		os.chdir(self.this_dir)
		
		## COLLAPSED ACCROSS ALL TRIALS PLOT:
		a1 = np.load("jwg_resp_locked_grand_mean.npy")[250:-500]
		a2 = np.load("jwg_resp_locked_grand_sem.npy")[250:-500]
		b1 = np.load("rn_resp_locked_grand_mean.npy")[250:-500]
		b2 = np.load("rn_resp_locked_grand_sem.npy")[250:-500]
		c1 = np.load("dh_resp_locked_grand_mean.npy")[250:-500]
		c2 = np.load("dh_resp_locked_grand_sem.npy")[250:-500]
		d1 = np.load("dl_resp_locked_grand_mean.npy")[250:-500]
		d2 = np.load("dl_resp_locked_grand_sem.npy")[250:-500]
		
		os.chdir(self.this_dir + '/figures/')
		
		# Response
		fig = figure(figsize=(4,4))
		a = plt.subplot(111)
		xb = np.arange(-3249,2000)
		p1, = plt.plot(xb, a1, color = 'r', linewidth=2)
		p2, = plt.plot(xb, b1, color = 'b', linewidth=2)
		p3, = plt.plot(xb, c1, color = 'g', linewidth=2)
		p4, = plt.plot(xb, d1, color = 'y', linewidth=2)
		plt.fill_between( xb, (a1+a2), (a1-a2), alpha=0.1, color = 'r' )
		plt.fill_between( xb, (b1+b2), (b1-b2), alpha=0.1, color = 'b' )
		plt.fill_between( xb, (c1+c2), (c1-c2), alpha=0.1, color = 'g' )
		plt.fill_between( xb, (d1+d2), (d1-d2), alpha=0.1, color = 'y' )
		plt.axvline(-3000, -1, 1, color = 'k', alpha = 0.25, linestyle = '--')
		plt.axvline(-1500, -1, 1, color = 'k', alpha = 0.25, linestyle = '--')
		plt.axvline(0, -1, 1, linewidth=1)
		simpleaxis(a)
		spine_shift(a)
		plt.xticks([-3000,-2000,-1000,0,1000], [-3,-2,-1,0,1])
		plt.xlim( (-3225, 1500) )
		plt.title('Response locked PPR pooled across trials', size=12)
		plt.ylabel("PPR (Z)", size=10)
		plt.xlabel("Time from report (s)", size=10)
		plt.legend([p1, p2, p3, p4], ["JWG", "RN", "DH", "DL"], loc = 2)
		gca().spines["bottom"].set_linewidth(.5)
		gca().spines["left"].set_linewidth(.5)
		subplots_adjust(bottom = 0.15, top = 0.925, left = 0.2)
		
		pp = PdfPages('Exp' + str(self.experiment) + '_PPR_response_figure_AcrossSubjects.pdf')
		fig.savefig(pp, format='pdf')
		pp.close()
		
	
















