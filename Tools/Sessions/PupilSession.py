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

from IPython import embed as shell

class pupil_session():
	def __init__(self, subject, experiment, version, this_dir):
		self.subject = subject
		self.experiment = experiment
		self.version = version
		self.this_dir = this_dir
	
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
		
		###################################################################################################################
		## LOOK FOR ITT'S AND CHECK FOR 'NEIGHBOURING' BLINKS! ############################################################
		
		# shell()
		
		# Compute time of inter trial intervals, and make raw_pupil_diameter equal to zero: 
		times_iti = np.zeros(gaze_timestamps.shape[0], dtype = bool)
		for m in range(trial_times.shape[0]):
			times_iti = times_iti + (gaze_timestamps > trial_times['trial_start_EL_timestamp'][m]-200) * (gaze_timestamps < trial_times['trial_phase_timestamps'][m,0,0]+1000)
		raw_pupil_diameter[times_iti] = 0
		
		# Compute time of first trial interval, and make raw_pupil_diameter equal to mean raw_pupil_diameter. We do this because we do not want to start with 0's. 
		times_iti_first = (gaze_timestamps > trial_times['trial_start_EL_timestamp'][0]) * (gaze_timestamps < trial_times['trial_phase_timestamps'][0,0,0]+1000)
		times_iti_first[0:999] = True
		raw_pupil_diameter[times_iti_first] = bottleneck.nanmean(raw_pupil_diameter[(gaze_timestamps > trial_times['trial_phase_timestamps'][0,1,0] - 1000) * (gaze_timestamps < trial_times['trial_phase_timestamps'][0,2,0])])
		
		# Sliding window to test for blinks right next to iti's / other blinks!
		for n in range(raw_pupil_diameter.shape[0]):
			if (raw_pupil_diameter[n] == 0):
				try:
					if (raw_pupil_diameter[n+1] != 0): 
						if sum((raw_pupil_diameter[n+1:n+500] == 0)) > 1:
							raw_pupil_diameter[n+1:n+500] = 0
				except IndexError:
						pass
		
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
		
		###################################################################################################################
		## LINEAR INTERPOLATION OF BLINKS AND ITI's! ######################################################################
		
		# these are the time points (defined in samples for now...) from which we take the levels on which to base the interpolation
		points_for_interpolation = np.array([[-150],[150]])
		interpolation_time_points = np.zeros((blink_data.shape[0], points_for_interpolation.ravel().shape[0]))
		interpolation_time_points[:,[0]] = np.tile(blink_data['start_timestamp'], 1).reshape((1,-1)).T + points_for_interpolation[0]
		interpolation_time_points[:,[1]] = np.tile(blink_data['end_timestamp'], 1).reshape((1,-1)).T + points_for_interpolation[1]
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
		
		for itp in interpolation_time_points:
			# interpolate
			# spline = interpolate.InterpolatedUnivariateSpline(itp,raw_pupil_diameter[itp])
			lin = interpolate.interp1d(itp,raw_pupil_diameter[itp])
		
			# replace with interpolated data
			raw_pupil_diameter[itp[0]:itp[-1]] = lin(np.arange(itp[0],itp[-1]))
		
		###################################################################################################################
		## FILTER THE SIGNAL! #############################################################################################
		
		# High pass:
		hp_frequency = 0.1
		hp_cof_sample = hp_frequency / (raw_pupil_diameter.shape[0] / self.sample_rate / 2)
		bhp, ahp = butter(3, hp_cof_sample, btype='high')
		
		# Low pass:
		lp_frequency = 4.0
		lp_cof_sample = lp_frequency / (raw_pupil_diameter.shape[0] / self.sample_rate / 2)
		blp, alp = butter(3, lp_cof_sample)
		
		# Band pass:
		lp_c_raw_pupil_diameter = filtfilt(blp, alp, raw_pupil_diameter)
		hp_c_raw_pupil_diameter = filtfilt(bhp, ahp, raw_pupil_diameter)
		lp_hp_c_raw_pupil_diameter = filtfilt(blp, alp, hp_c_raw_pupil_diameter)
		
		GLM_pupil_mean_lp_hp = np.array(lp_hp_c_raw_pupil_diameter[-times_iti]).mean()
		GLM_pupil_std_lp_hp = lp_hp_c_raw_pupil_diameter[-times_iti].std()
		GLM_pupil_zscore_lp_hp = (lp_hp_c_raw_pupil_diameter - GLM_pupil_mean_lp_hp) / GLM_pupil_std_lp_hp # Possible because vectorized.
		
		# f = figure(figsize = (12,8))
		# f.add_subplot(2,1,1)
		# plot(times_iti[::100]*2, color = 'r', alpha=0.25)
		# plot(forGLM_pupil_zscore_lp_hp[::100])
		# f.add_subplot(2,1,2)
		# plot(forGLM_pupil_zscore_lp[::100])
		# show()
		
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
		
		###################################################################################################################
		## Z-SCORE THE FILTERED SIGNALS! ##################################################################################
		
		include_indices = np.zeros(gaze_timestamps.shape, dtype = bool)
		include_start_end_indices = np.array([[(include['trial_phase_timestamps'][1,0]-500), (include['trial_phase_timestamps'][2,0]+1500)] for include in trial_times[-omission_indices]], dtype = int)
		for bl in include_start_end_indices:
			include_indices[ (np.argmax( ((gaze_timestamps== bl[0])+(gaze_timestamps== bl[0]+1)) )) : (np.argmax( ((gaze_timestamps== bl[1])+(gaze_timestamps== bl[1]+1)) )) ] = True
		
		pupil_mean_lp = np.array(lp_c_raw_pupil_diameter[include_indices]).mean()
		pupil_std_lp = lp_c_raw_pupil_diameter[include_indices].std()
		pupil_zscore_lp = (lp_c_raw_pupil_diameter - pupil_mean_lp) / pupil_std_lp # Possible because vectorized.
		
		pupil_mean_hp = np.array(hp_c_raw_pupil_diameter[include_indices]).mean()
		pupil_std_hp = hp_c_raw_pupil_diameter[include_indices].std()
		pupil_zscore_hp = (hp_c_raw_pupil_diameter - pupil_mean_hp) / pupil_std_hp # Possible because vectorized.
		
		pupil_mean_lp_hp = np.array(lp_hp_c_raw_pupil_diameter[include_indices]).mean()
		pupil_std_lp_hp = lp_hp_c_raw_pupil_diameter[include_indices].std()
		pupil_zscore_lp_hp = (lp_hp_c_raw_pupil_diameter - pupil_mean_lp_hp) / pupil_std_lp_hp # Possible because vectorized.
		
		try:
			run.pupil_data_filtered.remove()
		except NodeError, LeafError:
			pass
		# Downsampled stuff to hdf5 file:
		all_filtered_pupil_data = np.vstack((gaze_timestamps[0::int(self.downsample_rate)], sp.signal.decimate(pupil_zscore_lp,int(self.downsample_rate),1), sp.signal.decimate(pupil_zscore_hp,int(self.downsample_rate),1), sp.signal.decimate(pupil_zscore_lp_hp,int(self.downsample_rate),1), sp.signal.decimate(GLM_pupil_zscore_lp_hp,int(self.downsample_rate),1)))
		h5f.createArray(run, 'pupil_data_filtered', all_filtered_pupil_data, 'Z-scored pupil diameter (1) low passed, (2) high passed, (3) low+high passed, (4) low+high passed-zscored on full run')
		
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
		# Fill array with the pupil diameters:
		for j in range(len(trial_parameters['trial_nr'])):
			a[j,:sum(timesss[j])] = pupil_zscore_lp_hp[timesss[j]]
		a = a[:,0:5999]
		
		aa = np.zeros((len(trial_parameters['trial_nr']), duration))
		aa[:,:] = NaN
		# Fill array with the pupil diameters:
		for j in range(len(trial_parameters['trial_nr'])):
			aa[j,:sum(timesss[j])] = pupil_zscore_lp[timesss[j]]
		aa = aa[:,0:5999]
		
		stim = [a,aa]
		
		try:
			run.stimulus_locked_array.remove()
		except NodeError, LeafError:
			pass
		h5f.createArray(run, 'stimulus_locked_array', stim, 'stimulus locked z-scored pupil diameter low (4Hz) + high passed (0.1Hz), and only low passed (4Hz)')
		
		
		## MAKE RESPONSE LOCKED ARRAY
		timesss = []
		duration = 0
		for j in range(len(trial_parameters['trial_nr'])):
			timesss.append((gaze_timestamps > trial_times['trial_phase_timestamps'][j,2,0] - 3500) * (gaze_timestamps < trial_times['trial_phase_timestamps'][j,2,0]+2502))
			if sum(timesss[j]) > duration:
				duration = sum(timesss[j])
		
		b = np.ones((len(trial_parameters['trial_nr']), duration))
		b[:,:] = NaN 
		# Fill array with the pupil diameters:
		for j in range(len(trial_parameters['trial_nr'])):
			b[j,:sum(timesss[j])] = pupil_zscore_lp_hp[timesss[j]]
		b = b[:,0:5999]
		
		bb = np.ones((len(trial_parameters['trial_nr']), duration))
		bb[:,:] = NaN 
		# Fill array with the pupil diameters:
		for j in range(len(trial_parameters['trial_nr'])):
			bb[j,:sum(timesss[j])] = pupil_zscore_lp[timesss[j]]
			# bb[j,:sum(timesss[j])] = raw_pupil_diameter[timesss[j]]
		bb = bb[:,0:5999]
		
		resp = [b,bb]
		
		try:
			run.response_locked_array.remove()
		except NodeError, LeafError:
			pass
		h5f.createArray(run, 'response_locked_array', resp, 'response locked z-scored pupil diameter low (4Hz) + high passed (0.1Hz), and only low passed (4Hz)')
		
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
			# Fill array with the pupil diameters:
			for j in range(len(trial_parameters['trial_nr'])):
				c[j,:sum(timesss[j])] = pupil_zscore_lp_hp[timesss[j]]
			c = c[:,0:3999]
		
			cc = np.ones((len(trial_parameters['trial_nr']), duration))
			cc[:,:] = NaN 
			# Fill array with the pupil diameters:
			for j in range(len(trial_parameters['trial_nr'])):
				cc[j,:sum(timesss[j])] = pupil_zscore_lp[timesss[j]]
			cc = cc[:,0:3999]
		
			feed = [c,cc]
		
			try:
				run.feedback_locked_array.remove()
			except NodeError, LeafError:
				pass
			h5f.createArray(run, 'feedback_locked_array', feed, 'feedback locked z-scored pupil diameter low (4Hz) + high passed (0.1Hz), and only low passed (4Hz)')
		
		###################################################################################################################
		## MAKE GLM REGRESSORS! ###########################################################################################
		
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
		
		whole_trials = np.zeros((trial_parameters.shape[0],3))
		whole_trials[:,0] = (trial_times['trial_phase_timestamps'][:,0,0] - trial_times['trial_start_EL_timestamp'][0]) / self.downsample_rate
		whole_trials[:,1] = (trial_times['trial_phase_timestamps'][:,6,0] - trial_times['trial_phase_timestamps'][:,0,0]) / self.downsample_rate
		whole_trials[:,2] = np.ones(trial_parameters.shape[0])
		
		if self.experiment == 1:
			bla = [trial_on, stim_on, stim_off, decision_ramp, conf_on, feed_on, whole_trials]
		if self.experiment == 2:
			bla = [trial_on, stim_on, stim_off, decision_ramp, whole_trials]
		
		try:
			run.GLM_regressors.remove()
		except NodeError, LeafError:
			pass
		h5f.createArray(run, 'GLM_regressors', bla, 'regressors: (1) stick at trial onset (spacebar press), (2) stick at stim onset (tone), (3) stick at stim offset (response), (4) ramp decision interval, (5) stick at confidence (response), (6) stick at feedback, (7) whole trials')
		
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
		h5f = openFile(( self.subject + '.hdf5'), mode = "r" ) # mode = "r" means 'read only'
		
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
		stimulus_locked_arrays = []
		response_locked_arrays = []
		feedback_locked_arrays = []
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
			
			# # We won't load all the eye data; instead we will get all neccesary eye data from the earlier constructed stimulus-, response- and feedback-locked arrays (below).
			# This downsampled times series per run. Z-scored per run based on ALL measurements (except ITI), to use for GLM:
			GLM_pupil_diameter.append(run[i].pupil_data_filtered.read()[4,:])
			
			# In hdf5:: 0 = hp+lp ; 1 = lp
			stimulus_locked_arrays.append( np.array( run[i].stimulus_locked_array.read()[1])[-omission_indices[i],:])
			response_locked_arrays.append( np.array( run[i].response_locked_array.read()[1])[-omission_indices[i],:])
			if self.experiment == 1:
				feedback_locked_arrays.append( np.array( run[i].feedback_locked_array.read()[1])[-omission_indices[i],:])
				
			# Regressor files:
			regressor_trial_on.append( np.array(run[i].GLM_regressors.read()[0]) )
			regressor_stim_on.append( np.array(run[i].GLM_regressors.read()[1]) )
			regressor_stim_off.append( np.array(run[i].GLM_regressors.read()[2]) )
			regressor_ramp.append( np.array(run[i].GLM_regressors.read()[3]) )
			if self.experiment == 1:
				regressor_conf.append( np.array(run[i].GLM_regressors.read()[4]) )
				regressor_feed.append( np.array(run[i].GLM_regressors.read()[5]) )
			regressor_whole_trials.append (np.array(run[i].GLM_regressors.read()[5]) )
			
			
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
		
		stimulus_locked_array_joined = np.vstack(stimulus_locked_arrays)
		response_locked_array_joined = np.vstack(response_locked_arrays)
		if self.experiment == 1:
			feedback_locked_array_joined = np.vstack(feedback_locked_arrays)
		# trial_onset_locked_array_joined = np.vstack(trial_onset_locked_arrays)
		
		# Join timeseries of each run:
		GLM_pupil_diameter_joined = np.concatenate(GLM_pupil_diameter, axis=1)
		
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
		
		bpd = []
		bpd_feed = []
		for i in range(len(run)):
			# BPD:
			bpd_dum = np.zeros(stimulus_locked_arrays[i].shape[0])
			for j in range(stimulus_locked_arrays[i].shape[0]):
				bpd_dum[j] = ( bottleneck.nanmean(stimulus_locked_arrays[i][j,500:999]) )
			bpd.append(bpd_dum)
			if self.experiment == 1:
				# BPD_Feed:
				bpd_feed_dum = np.zeros(feedback_locked_arrays[i].shape[0])
				for k in range(feedback_locked_arrays[i].shape[0]):
					bpd_feed_dum[k] = ( bottleneck.nanmean(feedback_locked_arrays[i][k,500:999]) )
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
			ppd_dum =  np.zeros(response_locked_arrays[i].shape[0])
			for j in range(response_locked_arrays[i].shape[0]):
				ppd_dum[j] = bottleneck.nanmax(response_locked_arrays[i][j,2500:5000]) - bpd[i][j]
				# ppd_dum[j] = bottleneck.nanmax(response_locked_arrays[i][j,3500:5000]) - response_locked_arrays[i][j,3425]
			ppd.append(ppd_dum)
			
			# PPD - Method 2 (integration: area under curve):
			ppd_dum =  np.zeros(response_locked_arrays[i].shape[0])
			for j in range(response_locked_arrays[i].shape[0]):
				ppd_dum[j] = bottleneck.nanmean(response_locked_arrays[i][j,2500:5000]) - bpd[i][j]
			ppd_mean.append(ppd_dum)
			
			# PPD - Method 3 (linear projection):
			template = (bottleneck.nanmean(response_locked_array_joined[:,2500:5000], axis=0)) - mean(bpd_joined)
			# template = ( (bottleneck.nanmean(response_locked_array_joined[hit_indices_joined,2500:5000], axis=0) - mean(bpd_joined[hit_indices_joined])) + (bottleneck.nanmean(response_locked_array_joined[fa_indices_joined,2500:5000], axis=0) - mean(bpd_joined[fa_indices_joined])) + (bottleneck.nanmean(response_locked_array_joined[miss_indices_joined,2500:5000], axis=0) - mean(bpd_joined[miss_indices_joined])) + (bottleneck.nanmean(response_locked_array_joined[cr_indices_joined,2500:5000], axis=0) - mean(bpd_joined[cr_indices_joined])) ) / 4
			ppd_dum =  np.zeros(response_locked_arrays[i].shape[0])
			for j in range(response_locked_arrays[i].shape[0]):
				ppd_dum[j] = ( np.dot( template, (response_locked_arrays[i][j,2500:5000])-bpd[i][j] ) / np.dot(template, template)  )
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
				ppd_dum =  np.zeros(feedback_locked_arrays[i].shape[0])
				for j in range(feedback_locked_arrays[i].shape[0]):
					ppd_dum[j] = bottleneck.nanmax(feedback_locked_arrays[i][j,1000:2500]) - bpd_feed[i][j]
				ppd_feed.append(ppd_dum)
				
				# PPD - Method 2 (mean minus baseline):
				ppd_dum =  np.zeros(feedback_locked_arrays[i].shape[0])
				for j in range(feedback_locked_arrays[i].shape[0]):
					ppd_dum[j] = bottleneck.nanmean(feedback_locked_arrays[i][j,1000:2500]) - bpd_feed[i][j]
				ppd_feed_mean.append(ppd_dum)
				
				# PPD - Method 3 (linear projection):
				template_feed = (bottleneck.nanmean(feedback_locked_array_joined[:,1000:2500], axis=0)) - mean(bpd_feed_joined)
				# template_feed = ( (bottleneck.nanmean(feedback_locked_array_joined[hit_indices_joined,1000:2500], axis=0) - mean(bpd_feed_joined[hit_indices_joined])) + (bottleneck.nanmean(feedback_locked_array_joined[fa_indices_joined,1000:2500], axis=0) - mean(bpd_feed_joined[fa_indices_joined])) + (bottleneck.nanmean(feedback_locked_array_joined[miss_indices_joined,1000:2500], axis=0) - mean(bpd_feed_joined[miss_indices_joined])) + (bottleneck.nanmean(feedback_locked_array_joined[cr_indices_joined,1000:2500], axis=0) - mean(bpd_feed_joined[cr_indices_joined])) ) / 4
				ppd_dum =  np.zeros(feedback_locked_arrays[i].shape[0])
				for j in range(feedback_locked_arrays[i].shape[0]):
					ppd_dum[j] = ( np.dot( template_feed, (feedback_locked_arrays[i][j,1000:2500])-bpd_feed[i][j] ) / np.dot(template_feed, template_feed) )
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
		
		stim_locked_grand_mean = ( bottleneck.nanmean(stimulus_locked_array_joined[hit_indices_joined,:], axis=0) + bottleneck.nanmean(stimulus_locked_array_joined[fa_indices_joined,:], axis=0) + bottleneck.nanmean(stimulus_locked_array_joined[miss_indices_joined,:], axis=0) + bottleneck.nanmean(stimulus_locked_array_joined[cr_indices_joined,:], axis=0) ) / 4
		stim_locked_grand_sem = ( (bottleneck.nanstd(stimulus_locked_array_joined[hit_indices_joined,:], axis=0) / sp.sqrt(stimulus_locked_array_joined[hit_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(stimulus_locked_array_joined[fa_indices_joined,:], axis=0) / sp.sqrt(stimulus_locked_array_joined[fa_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(stimulus_locked_array_joined[miss_indices_joined,:], axis=0) / sp.sqrt(stimulus_locked_array_joined[miss_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(stimulus_locked_array_joined[cr_indices_joined,:], axis=0) / sp.sqrt(stimulus_locked_array_joined[cr_indices_joined,:].shape[0]) ) ) / 4
		
		resp_locked_grand_mean = ( bottleneck.nanmean(response_locked_array_joined[hit_indices_joined,:], axis=0) + bottleneck.nanmean(response_locked_array_joined[fa_indices_joined,:], axis=0) + bottleneck.nanmean(response_locked_array_joined[miss_indices_joined,:], axis=0) + bottleneck.nanmean(response_locked_array_joined[cr_indices_joined,:], axis=0) ) / 4
		resp_locked_grand_sem = ( (bottleneck.nanstd(response_locked_array_joined[hit_indices_joined,:], axis=0) / sp.sqrt(response_locked_array_joined[hit_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(response_locked_array_joined[fa_indices_joined,:], axis=0) / sp.sqrt(response_locked_array_joined[fa_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(response_locked_array_joined[miss_indices_joined,:], axis=0) / sp.sqrt(response_locked_array_joined[miss_indices_joined,:].shape[0]) ) + (bottleneck.nanstd(response_locked_array_joined[cr_indices_joined,:], axis=0) / sp.sqrt(response_locked_array_joined[cr_indices_joined,:].shape[0]) ) ) / 4
		
		if self.experiment == 1:
			pupil_IRF = ( bottleneck.nanmean(feedback_locked_array_joined[hit_indices_joined,1000:3000], axis=0) + bottleneck.nanmean(feedback_locked_array_joined[fa_indices_joined,1000:3000], axis=0) + bottleneck.nanmean(feedback_locked_array_joined[miss_indices_joined,1000:3000], axis=0) + bottleneck.nanmean(feedback_locked_array_joined[cr_indices_joined,1000:3000], axis=0) ) / 4
			if pupil_IRF[0] < 0:
				pupil_IRF = pupil_IRF + abs(pupil_IRF[0])
			if pupil_IRF[0] > 0:
				pupil_IRF = pupil_IRF - abs(pupil_IRF[0])
			pupil_IRF_downsampled = sp.signal.decimate(pupil_IRF, int(self.downsample_rate),1) 
		
		# Save to folder called 'ALL':
		if self.experiment == 1:
			os.chdir('/Research/PUPIL/PupilExperiment1/data/ALL/')
		if self.experiment == 2:
			os.chdir('/Research/PUPIL/PupilExperiment2/data/ALL/')
		
		df.save(str(self.subject) + '_data')
		
		np.save(str(self.subject) + '_stimulus_locked_array', stimulus_locked_array_joined)
		np.save(str(self.subject) + '_response_locked_array', response_locked_array_joined)
		
		if self.experiment == 1:
			np.save(str(self.subject) + '_feedback_locked_array', feedback_locked_array_joined)
			np.save(str(self.subject) + '_pupil_IRF', pupil_IRF)
			np.save(str(self.subject) + '_pupil_IRF_downsampled', pupil_IRF_downsampled)
		
		np.save(str(self.subject) + '_stim_locked_grand_mean', stim_locked_grand_mean)
		np.save(str(self.subject) + '_stim_locked_grand_sem', stim_locked_grand_sem)
		
		np.save(str(self.subject) + '_resp_locked_grand_mean', resp_locked_grand_mean)
		np.save(str(self.subject) + '_resp_locked_grand_sem', resp_locked_grand_sem)
		
		
		
		# shell()
		# 
		# 
		# ###################################################################################################################
		# ## MAKE REGRESSORS! ###############################################################################################
		# 
		# regr_trial_on_stick = []
		# regr_stim_on_stick = []
		# regr_stim_off_stick = []
		# regr_ramp_down = []
		# regr_ramp_up = []
		# regr_conf_stick = []
		# regr_feed_stick = []
		# exclude_from_regressors_indices = []
		# exclude_from_regressors_indices2 = []
		# for i in range(len(run)):
		# 	
		# 	regr_trial_on_stick.append( np.zeros(pupil_diameter_lp[i].shape[0]) )
		# 	for j in range(regressor_trial_on[i].shape[0]):
		# 		regr_trial_on_stick[i][round(regressor_trial_on[i][j,0]/self.downsample_rate)] = 1.0
		# 		regr_trial_on_stick[i] = regr_trial_on_stick[i]-mean(regr_trial_on_stick[i])
		# 		
		# 	regr_stim_on_stick.append( np.zeros(pupil_diameter_lp[i].shape[0]) )
		# 	for j in range(regressor_stim_on[i].shape[0]):
		# 		regr_stim_on_stick[i][round(regressor_stim_on[i][j,0]/self.downsample_rate)] = 1.0
		# 		regr_stim_on_stick[i] = regr_stim_on_stick[i]-mean(regr_stim_on_stick[i])
		# 		
		# 	regr_stim_off_stick.append( np.zeros(pupil_diameter_lp[i].shape[0]) )
		# 	for j in range(regressor_stim_off[i].shape[0]):
		# 		regr_stim_off_stick[i][round(regressor_stim_off[i][j,0]/self.downsample_rate)] = 1.0
		# 		regr_stim_off_stick[i] = regr_stim_off_stick[i]-mean(regr_stim_off_stick[i])
		# 		
		# 	regr_ramp_down.append( np.zeros(pupil_diameter_lp[i].shape[0]) )
		# 	for j in range(regressor_ramp[i].shape[0]):
		# 		regr_ramp_down[i][round(regressor_stim_on[i][j,0]/self.downsample_rate):round(regressor_stim_off[i][j,0]/self.downsample_rate)] = linspace(0,1,(round(regressor_stim_off[i][j,0]/self.downsample_rate)-round(regressor_stim_on[i][j,0]/self.downsample_rate)) )
		# 		regr_ramp_down[i] = regr_ramp_down[i]-mean(regr_ramp_down[i])
		# 		
		# 	regr_ramp_up.append( np.zeros(pupil_diameter_lp[i].shape[0]) )
		# 	for j in range(regressor_ramp[i].shape[0]):
		# 		regr_ramp_up[i][round(regressor_stim_on[i][j,0]/self.downsample_rate):round(regressor_stim_off[i][j,0]/self.downsample_rate)] = linspace(0,-1,(round(regressor_stim_off[i][j,0]/self.downsample_rate)-round(regressor_stim_on[i][j,0]/self.downsample_rate)) )
		# 		regr_ramp_up[i] = regr_ramp_up[i]-mean(regr_ramp_up[i])
		# 		
		# 	if self.experiment == 1:
		# 		regr_conf_stick.append( np.zeros(pupil_diameter_lp[i].shape[0]) )
		# 		for j in range(regressor_conf[i].shape[0]):
		# 			regr_conf_stick[i][round(regressor_conf[i][j,0]/self.downsample_rate)] = 1.0
		# 			regr_conf_stick[i] = regr_conf_stick[i]-mean(regr_conf_stick[i])
		# 			
		# 		regr_feed_stick.append( np.zeros(pupil_diameter_lp[i].shape[0]) )
		# 		for j in range(regressor_feed[i].shape[0]):
		# 			regr_feed_stick[i][round(regressor_feed[i][j,0]/self.downsample_rate)] = 1.0
		# 			regr_feed_stick[i] = regr_feed_stick[i]-mean(regr_feed_stick[i])
		# 			
		# 	# # MAKE EXCLUDE INDICES:
		# 	# # Exlcude first two trials of each run:
		# 	# exclude_from_regressors_indices2.append( np.zeros(pupil_diameter_lp[i].shape[0], dtype = bool) )
		# 	# for j in range(2):
		# 	# 	exclude_from_regressors_indices2[i][round(regressor_trial_on[i][j,0]/downsample_rate):round(regressor_feed[i][j,0]/downsample_rate)+(2000/25)] = True
		# 	# 
		# 	# # Exclude ITI's
		# 	# exclude_from_regressors_indices.append( np.zeros(pupil_diameter_lp[i].shape[0], dtype = bool) )
		# 	# for j in range(regressor_feed[i].shape[0]):
		# 	# 	try:
		# 	# 		exclude_from_regressors_indices[i][round(regressor_feed[i][j,0]/downsample_rate)+(1500/25):round(regressor_trial_on[i][j+1,0]/downsample_rate)+(2000/25)] = True
		# 	# 	except:
		# 	# 		pass
		# 
		# 
		# 
		# 
		# 
		# 	# Join regressors of each run:
		# 	regr_trial_on_stick_joined = np.concatenate(regr_trial_on_stick, axis=1)
		# 	regr_stim_on_stick_joined = np.concatenate(regr_stim_on_stick, axis=1)
		# 	regr_stim_off_stick_joined = np.concatenate(regr_stim_off_stick, axis=1)
		# 	regr_ramp_up_joined = np.concatenate(regr_ramp_up, axis=1)
		# 	regr_ramp_down_joined = np.concatenate(regr_ramp_down, axis=1)
		# 	if self.experiment == 1:
		# 		regr_conf_stick_joined = np.concatenate(regr_conf_stick, axis=1)
		# 		regr_feed_stick_joined = np.concatenate(regr_feed_stick, axis=1)		
		# 	# exclude_from_regressors_indices_joined = np.concatenate(exclude_from_regressors_indices, axis=1)
		# 	# exclude_from_regressors_indices2_joined = np.concatenate(exclude_from_regressors_indices2, axis=1)
		
		
		
		
		
		
		
		
		
		
		
		# ClOSE HDF5 FILE:
		
		h5f.close()
