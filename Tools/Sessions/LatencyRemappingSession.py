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
	
	def saccade_latency_analysis_one_run(self, run, plot = True):
		if run.condition == 'Remapping':
			# get EL Data
			elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
			elO.import_parameters(run_name = 'bla')
			el_saccades = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'saccades')[0] # just a single list instead of more than one....
			el_gaze_data = elO.get_EL_samples_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'gaze_x')[0] # just a single list instead of more than one....
			first_saccades_not_too_fast = [(el_saccades[i][:]['start_timestamp'] - el_gaze_data[i][0,0]) > 200.0 for i in range(len(el_gaze_data))] # if not this, early microsaccades count as full early saccades but they're not. 200 ms is arbitrary but looks okay from the data.
			phase_durations_of_interest = np.array([np.diff(pt) for pt in elO.timings['trial_phase_timestamps'][:,1:,0]])
			# incorporate saccade knowledge into run.
			run.saccade_stimulus_latencies = np.array([el_saccades[i][first_saccades_not_too_fast[i]]['start_timestamp'][0] - el_gaze_data[i][0,0] - phase_durations_of_interest[i,0] for i in range(len(el_gaze_data))])
			run.timings = elO.timings
			run.parameter_data = elO.parameter_data
			run.events = elO.events
			if plot:
			# start plotting info
				f = pl.figure(figsize = (8,2))
				sf = f.add_subplot(1,1,1)
				sf.set_xlabel('time, [ms]', fontsize=9)
				sf.set_ylabel('position on screen')
				sf.set_xlim([0,1000])
				for i in range(len(el_gaze_data)):
					el_gaze_data[i] = np.array(el_gaze_data[i])
					pl.plot(el_gaze_data[i][::10,0] - el_gaze_data[i][0,0], el_gaze_data[i][::10,1], alpha = 0.3, linewidth = 1.5, c = 'k')
				sf = sf.twinx()
				pl.hist(phase_durations_of_interest[:,0], edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'r', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'trial_phase_0' )
				pl.hist([el_saccades[i][first_saccades_not_too_fast[i]]['start_timestamp'][0] - el_gaze_data[i][0,0] for i in range(len(el_gaze_data))], edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'g', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'saccade' )
				pl.hist([el_saccades[i][first_saccades_not_too_fast[i]]['start_timestamp'][0] - el_gaze_data[i][0,0] - phase_durations_of_interest[i,0] for i in range(len(el_gaze_data))], edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'b', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'lag' )
	#			sf.set_xlabel('time, [ms]', fontsize=9)
				sf.set_ylabel('histo')
				sf.axis([0,1000,0,40])
				sf.legend()
				pl.draw()
				# save figure and save saccade latencies per trial
				pl.savefig(self.runFile(stage = 'processed/eye', run = run, postFix = ['sacc_lat'], extension = '.pdf'))
			
		
		elif run.condition == 'Mapper':
			pass
	
	def amplitude_analysis_all_runs(self, run_length = 480):
		self.mapper_amplitude_data = []
		for r in [self.runList[i] for i in self.conditionDict['Mapper']]:
			self.mapper_amplitude_data.append(self.amplitude_analysis_one_run(r))
		self.remapping_amplitude_data = []
		for (k, r) in zip(range(len(self.conditionDict['Remapping'])), [self.runList[i] for i in self.conditionDict['Remapping']]):
			self.remapping_amplitude_data.append(self.amplitude_analysis_one_run(r))
			self.remapping_amplitude_data[-1][0] += k * run_length
			self.remapping_amplitude_data[-1][1] += k * run_length
		all_early_saccade_times = np.hstack([i[0] for i in self.remapping_amplitude_data])
		all_late_saccade_times = np.hstack([i[1] for i in self.remapping_amplitude_data])
		# should change the median split to a 4-way split or something
		eventData = [all_early_saccade_times, all_late_saccade_times]
		f = pl.figure()
		plotnr = 1
		areas = ['V1','V2','V3','V3AB','V4']
		for i in range(len(areas)):
			print areas[i]
			s = f.add_subplot(len(areas),1,plotnr)
			roiData = self.gatherRIOData([areas[i]], whichRuns = self.conditionDict['Remapping'], whichMask = '_center' )
			roiDataM = roiData.mean(axis = 1)
			for e in range(len(eventData)):
				eraOp = EventRelatedAverageOperator(inputObject = np.array([roiDataM]), TR = 2.0, eventObject = eventData[e], interval = [-5.0,15.0])
				zero_index = np.arange(eraOp.intervalRange.shape[0])[np.abs(eraOp.intervalRange).min() == np.abs(eraOp.intervalRange)]
				d = eraOp.run(binWidth = 4.0, stepSize = 0.5)
				pl.plot(d[:,0], d[:,1]-d[zero_index,1], c = ['r','g'][e], alpha = 0.75)
				pl.fill_between(d[:,0], (d[:,1]-d[zero_index,1]) - (d[:,2]/np.sqrt(d[:,3])), (d[:,1]-d[zero_index,1]) + (d[:,2]/np.sqrt(d[:,3])), color = ['r','g'][e], alpha = 0.1)
			s.set_title(areas[i])
			s.axis([-5,15,-0.1,0.1])
			plotnr += 1
		pl.show()
	
	def amplitude_analysis_one_run(self, run):
		if run.condition == 'Remapping':
			if not hasattr(run, 'saccade_stimulus_latencies'):
				# this will prepare for our present analysis
				self.saccade_latency_analysis_one_run(run, plot = False)
			# look at times. 
			run.first_TR_timestamp = run.events[run.events[:]['unicode'] == 't'][0]['EL_timestamp']
			run.stimulus_on_times =  run.timings['trial_phase_timestamps'][:,0,0]-run.first_TR_timestamp
			run.saccade_instruction_times =  run.timings['trial_phase_timestamps'][:,1,0]-run.first_TR_timestamp
			run.stimulus_off_times =  run.timings['trial_phase_timestamps'][:,2,0]-run.first_TR_timestamp
			
			above_median, below_median = run.saccade_stimulus_latencies > np.median(run.saccade_stimulus_latencies), run.saccade_stimulus_latencies <= np.median(run.saccade_stimulus_latencies)
			return [(run.stimulus_off_times[above_median] + run.saccade_stimulus_latencies[above_median])/1000.0, (run.stimulus_off_times[below_median] + run.saccade_stimulus_latencies[below_median])/1000.0]
			
		elif run.condition == 'Mapper':
			pass
			
	
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
			
			
		