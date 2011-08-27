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
import matplotlib.cm as cm
from pylab import *
from nifti import *

class LatencyRemappingSession(Session):
	def saccade_latency_analysis_all_runs(self):
		self.mapper_saccade_data = []
		for r in [self.runList[i] for i in self.conditionDict['Mapper']]:
			self.mapper_saccade_data.append(self.saccade_latency_analysis_one_run(r))
		self.remapping_saccade_data = []
		for r in [self.runList[i] for i in self.conditionDict['Remapping']]:
			self.remapping_saccade_data.append(self.saccade_latency_analysis_one_run(r))
	
	def parameter_analysis_all_runs(self):
		self.mapper_parameter_data = []
		for r in [self.runList[i] for i in self.conditionDict['Mapper']]:
			self.mapper_parameter_data.append(self.parameter_analysis_one_run(r))
		self.remapping_parameter_data = []
		for r in [self.runList[i] for i in self.conditionDict['Remapping']]:
			self.remapping_parameter_data.append(self.parameter_analysis_one_run(r))
	
	def saccade_latency_analysis_one_run(self, run):
		if run.condition == 'Remapping':
			# get EL Data
			elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
			elO.import_parameters(run_name = 'bla')
			el_saccades = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'saccades')[0] # just a single list instead of more than one....
			first_saccades_not_too_fast = [(el_saccades[i][:]['start_timestamp'] - el_gaze_data[i][0,0]) > 200.0 for i in range(len(el_gaze_data))] # if not this, early microsaccades count as full early saccades but they're not. 200 ms is arbitrary but looks okay from the data.
			el_gaze_data = elO.get_EL_samples_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'gaze_x')[0] # just a single list instead of more than one....
			phase_durations_of_interest = np.array([np.diff(pt) for pt in elO.timings['trial_phase_timestamps'][:,1:,0]])
			# start plotting info
			f = pl.figure(figsize = (8,2))
			sf = f.add_subplot(1,1,1)
			sf.set_xlabel('time, [ms]', fontsize=9)
			sf.set_ylabel('position on screen')
			sf.set_xlim([0,2000])
			for i in range(len(el_gaze_data)):
				el_gaze_data[i] = np.array(el_gaze_data[i])
				pl.plot(el_gaze_data[i][::10,0] - el_gaze_data[i][0,0], el_gaze_data[i][::10,1], alpha = 0.3, linewidth = 1.5, c = 'k')
			sf = sf.twinx()
			pl.hist(phase_durations_of_interest[:,0], edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'r', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'trial_phase_0' )
			pl.hist([el_saccades[i][first_saccades_not_too_fast[i]]['start_timestamp'][0] - el_gaze_data[i][0,0] for i in range(len(el_gaze_data))], edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'g', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'saccade' )
			pl.hist([el_saccades[i][first_saccades_not_too_fast[i]]['start_timestamp'][0] - el_gaze_data[i][0,0] - phase_durations_of_interest[i,0] for i in range(len(el_gaze_data))], edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'b', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'lag' )
#			sf.set_xlabel('time, [ms]', fontsize=9)
			sf.set_ylabel('histo')
			sf.axis([0,2000,0,40])
			sf.legend()
			pl.draw()
			# save figure and save saccade latencies per trial
			pl.savefig(self.runFile(stage = 'processed/eye', run = run, postFix = ['sacc_lat'], extension = '.pdf'))
			run.saccade_latencies = np.array([el_saccades[i][first_saccades_not_too_fast[i]]['start_timestamp'][0] - el_gaze_data[i][0,0] - phase_durations_of_interest[i,0] for i in range(len(el_gaze_data))])
		
		elif run.condition == 'Mapper':
			pass
			# print el_saccades, elO.timings.dtype, elO.timings
	
	def parameter_analysis_one_run(self, run):
		elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
		elO.import_parameters(run_name = 'bla')
		print run.condition
		if run.condition == 'Mapper':
			####
			####	run retinotopic area mapping. do the main pattern-based GLM internally. 
			####
			trial_types = np.sign((elO.parameter_data[:]['contrast_L'] - elO.parameter_data[:]['contrast_R']) * elO.parameter_data[:]['stim_eccentricity'])
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf','hpf'], extension = '.nii.gz'))
			tr, nrsamples = niiFile.rtime, niiFile.timepoints
			stim_durations = np.ones((trial_types.shape[0])) * 2
			stim_onsets = np.arange(0,nrsamples*tr,tr*2)
			design = np.vstack((stim_onsets, stim_durations, trial_types))
			stim_locations = np.unique(trial_types)
			segmented_design = [] # [[[0,tr * nrsamples]]]	# the mean value for regression
			for i in range(3):
				this_location_design = design[:,design[2] == stim_locations[i]]
				this_location_design[2] = 1
				# print out design file for fsl analysis
				np.savetxt( self.runFile(stage = 'processed/mri', run = run, postFix = ['design',str(i)], extension = '.txt'), this_location_design.T, fmt = '%3.1f', delimiter = '\t')
				segmented_design.append(this_location_design.T)
			segmented_design.append([[0,tr * nrsamples, 1.0]])
			# don't do this stuff now. let's just take fsl's output
			if False:
				irO = ImageRegressOperator( niiFile, segmented_design )
				res = irO.execute()
				tstats = [ b for b in res['betas'] ]#  / np.sqrt(res['sse'])
				tstats.append(res['sse'])
				tstats = np.array(tstats, dtype = np.float32) 
				pl.figure()
				pl.plot(irO.design.designMatrix)
				pl.show()
				tFile = NiftiImage(tstats)
				tFile.filename = self.runFile(stage = 'processed/mri', run = run, postFix = ['T'], extension = '.nii.gz')
				tFile.header = niiFile.header
				tFile.save()
		
		if run.condition == 'Remapping':
			pass
			
			
		