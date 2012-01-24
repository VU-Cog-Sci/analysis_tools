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
			run.first_saccades_not_too_fast = [(el_saccades[i][:]['start_timestamp'] - el_gaze_data[i][0,0]) > 200.0 for i in range(len(el_gaze_data))] # if not this, early microsaccades count as full early saccades but they're not. 200 ms is arbitrary but looks okay from the data.
			run.first_saccades_not_too_small = [(el_saccades[i][:]['start_x'] - el_saccades[i][:]['end_x']) > 300.0 for i in range(len(el_gaze_data))] # if not this, early microsaccades count as full early saccades but they're not. 300 pix is arbitrary but looks okay from the data.
			run.first_saccades_not_too_small_not_too_fast = [run.first_saccades_not_too_fast[i] * run.first_saccades_not_too_small[i] for i in range(len(el_gaze_data))] 
			phase_durations_of_interest = np.array([np.diff(pt) for pt in elO.timings['trial_phase_timestamps'][:,1:,0]])
			# incorporate saccade knowledge into run.
			run.saccade_times = np.array([el_saccades[i][run.first_saccades_not_too_fast[i]]['start_timestamp'][0] - el_gaze_data[i][0,0] for i in range(len(el_gaze_data))])
			run.saccade_stimulus_latencies = np.array([el_saccades[i][run.first_saccades_not_too_fast[i]]['start_timestamp'][0] - el_gaze_data[i][0,0] - phase_durations_of_interest[i,0] for i in range(len(el_gaze_data))])
			run.timings = elO.timings
			run.parameter_data = elO.parameter_data
			run.events = elO.events
			run.saccades = el_saccades
			run.gaze_data = el_gaze_data
			
			run.correct_saccades = []
			for i in range(len(el_gaze_data)):
				if len(run.saccades[i]) == 1:
					x_diff = run.saccades[i][run.first_saccades_not_too_small_not_too_fast[i]]['start_x'] - run.saccades[i][run.first_saccades_not_too_small_not_too_fast[i]]['end_x']
				else:
					x_diff = run.saccades[i][run.first_saccades_not_too_small_not_too_fast[i][0]]['start_x'] - run.saccades[i][run.first_saccades_not_too_small_not_too_fast[i][0]]['end_x']
				if ((np.sign(np.mod(run.ID,2))*2)-1) == run.parameter_data[i]['saccade_direction']:	# if the 'red' instruction meant a rightward saccade in this run
					if np.sign( x_diff ) > 0:# correct saccade?
						run.correct_saccades.append(True)
					else:
						run.correct_saccades.append(False)
				else:
					if np.sign( x_diff ) < 0:# correct saccade?
						run.correct_saccades.append(True)
					else:
						run.correct_saccades.append(False)
			run.correct_saccades = np.array(run.correct_saccades, dtype = bool)
			
			if plot:
			# start plotting info
				f = pl.figure(figsize = (8,2))
				sf = f.add_subplot(1,1,1)
				sf.set_xlabel('time, [ms]', fontsize=9)
				sf.set_ylabel('position on screen')
				sf.set_xlim([0,1000])
				for i in range(len(el_gaze_data)):
					el_gaze_data[i] = np.array(el_gaze_data[i])
					color = ['r','k'][int(run.correct_saccades[i])]
					pl.plot(el_gaze_data[i][::10,0] - el_gaze_data[i][0,0], el_gaze_data[i][::10,1], alpha = 0.3, linewidth = 1.5, c = color)
				sf = sf.twinx()
				pl.hist(phase_durations_of_interest[:,0], edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'r', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'trial_phase_0' )
				pl.hist(run.saccade_times, edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'g', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'saccade' )
				pl.hist(run.saccade_stimulus_latencies, edgecolor = (0.0,0.0,0.0), alpha = 0.25, color = 'b', normed = False, rwidth = 0.5, histtype = 'stepfilled', label = 'lag' )
				sf.set_xlabel('time, [ms]', fontsize=9)
				sf.set_ylabel('histo')
				sf.axis([0,1000,0,40])
				leg = sf.legend(fancybox = True, loc = 2)
				leg.get_frame().set_alpha(0.5)
				pl.draw()
				# save figure and save saccade latencies per trial
				pl.savefig(self.runFile(stage = 'processed/eye', run = run, postFix = ['sacc_lat'], extension = '.pdf'))
			
		
		elif run.condition == 'Mapper':
			elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
			elO.import_parameters(run_name = 'bla')
			el_saccades = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'saccades')[0] # just a single list instead of more than one....
			el_gaze_data = elO.get_EL_samples_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'gaze_x')[0] # just a single list instead of more than one....
			phase_durations_of_interest = np.array([np.diff(pt) for pt in elO.timings['trial_phase_timestamps'][:,1:,0]])
			# incorporate knowledge into run.
			run.timings = elO.timings
			run.parameter_data = elO.parameter_data
			run.events = elO.events
			run.saccades = el_saccades
			run.gaze_data = el_gaze_data
	
	def amplitude_analysis_all_runs(self, run_length = 480, analysis_type = 'dec', mask = '_center', nr_bins = 4):
		self.mapper_amplitude_data = []
		for r in [self.runList[i] for i in self.conditionDict['Mapper']]:
			self.mapper_amplitude_data.append(self.amplitude_analysis_one_run(r))
		self.remapping_amplitude_data = []
		for (k, r) in zip(range(len(self.conditionDict['Remapping'])), [self.runList[i] for i in self.conditionDict['Remapping']]):
			self.remapping_amplitude_data.append(self.amplitude_analysis_one_run(r, nr_bins = nr_bins))
			for i in range(len(self.remapping_amplitude_data[-1])):
				self.remapping_amplitude_data[-1][i] += k * run_length
		all_saccade_times_by_latency = [np.hstack([i[k] for i in self.remapping_amplitude_data]) for k in range(len(self.remapping_amplitude_data[-1]))]
		mean_saccade_latency_per_bin = np.array([k.mean() for k in all_saccade_times_by_latency])
		# should change the median split to a 4-way split or something
		eventData = all_saccade_times_by_latency
		f = pl.figure(figsize = (7,12))
		plotnr = 1
		areas = ['V1','V2','V3','V3AB','V4','inferiorparietal','superiorparietal']
		colors = [(i/2.0,0.0,1-fmod(i,1.0)) for i in np.linspace(0,2,len(all_saccade_times_by_latency))]
		for i in range(len(areas)):
			print areas[i]
			s = f.add_subplot(len(areas),1,plotnr)
			roiData = self.gatherRIOData([areas[i]], whichRuns = self.conditionDict['Remapping'], whichMask = mask )
			print roiData.shape
			roiDataM = roiData.mean(axis = 1)
			if analysis_type == 'era':
				all_results_this_area = []
				for e in range(len(eventData)):
					eraOp = EventRelatedAverageOperator(inputObject = np.array([roiDataM]), TR = 2.0, eventObject = eventData[e], interval = [-0.0,10.0])
					zero_index = np.arange(eraOp.intervalRange.shape[0])[np.abs(eraOp.intervalRange).min() == np.abs(eraOp.intervalRange)]
					d = eraOp.run(binWidth = 2.0, stepSize = 0.5)
					all_results_this_area.append(d[:,1])
					times = d[:,0] 
				pl.imshow(np.array(all_results_this_area), extent = [times[0], times[-1], 0, len(eventData)])
				s = s.twinx()
				for e in range(len(eventData)):
					pl.plot(times, np.array(all_results_this_area)[e], c = colors[e])
			elif analysis_type == 'dec':
				decOp = DeconvolutionOperator(inputObject = np.array(roiDataM), eventObject = eventData, TR = 2.0, deconvolutionSampleDuration = 1.0, deconvolutionInterval = 10.0)
				# import pdb; pdb.set_trace()
				for e in range(len(eventData)):
					pl.plot(decOp.deconvolvedTimeCoursesPerEventType[e], c = colors[e])
			s.set_title(areas[i])
			plotnr += 1
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'era_visual_areas'+ mask +'.pdf'))
	
	def amplitude_analysis_one_run(self, run, nr_bins = 2):
		if run.condition == 'Remapping':
			if not hasattr(run, 'saccade_stimulus_latencies'):
				# this will prepare for our present analysis
				self.saccade_latency_analysis_one_run(run, plot = False)
			# look at times. 
			run.first_TR_timestamp = run.events[run.events[:]['unicode'] == 't'][0]['EL_timestamp']
			run.stimulus_on_times =  run.timings['trial_phase_timestamps'][:,0,0]-run.first_TR_timestamp
			run.saccade_instruction_times =  run.timings['trial_phase_timestamps'][:,1,0]-run.first_TR_timestamp
			run.stimulus_off_times =  run.timings['trial_phase_timestamps'][:,2,0]-run.first_TR_timestamp
			
			latency_indices = np.argsort(run.saccade_stimulus_latencies)
			# latency_indices = np.argsort(run.saccade_times)
			bin_width = np.floor((latency_indices.shape[0])/nr_bins)
			bin_limits = np.arange(0, latency_indices.shape[0], bin_width)[:nr_bins]
			
			#return_times = [(run.stimulus_on_times[latency_indices[bin:bin+bin_width]])/1000.0 for bin in bin_limits]
			#return_times = [(run.stimulus_on_times[latency_indices[bin:bin+bin_width]] + run.saccade_stimulus_latencies[latency_indices[bin:bin+bin_width]])/1000.0 for bin in bin_limits]
			return_times = [(run.stimulus_off_times[latency_indices[bin:bin+bin_width]] + run.saccade_stimulus_latencies[latency_indices[bin:bin+bin_width]])/1000.0 for bin in bin_limits]
			#return_times = [(run.saccade_instruction_times[latency_indices[bin:bin+bin_width]] + run.saccade_stimulus_latencies[latency_indices[bin:bin+bin_width]])/1000.0 for bin in bin_limits]
			
			for i in range(len(return_times)):
				design = np.vstack((return_times[i], np.ones(return_times[i].shape[0]) * 0.2, np.ones(return_times[i].shape[0]) * 1))
				np.savetxt( self.runFile(stage = 'processed/mri', run = run, postFix = ['design_sacc',str(i)], extension = '.txt'), design.T, fmt = '%3.2f', delimiter = '\t')
			design = np.vstack((run.stimulus_on_times/1000.0, (run.stimulus_off_times-run.stimulus_on_times)/1000.0, np.ones(run.stimulus_on_times.shape[0]) * 1))
			np.savetxt( self.runFile(stage = 'processed/mri', run = run, postFix = ['design_stim'], extension = '.txt'), design.T, fmt = '%3.2f', delimiter = '\t')
			
			return return_times
			
		elif run.condition == 'Mapper':
			pass
	
	def amplitude_analysis_one_run_all_trials_sep(self, run, mask = '_center', roi = 'V1'):
		if run.condition == 'Remapping':
			if not hasattr(run, 'saccade_stimulus_latencies'):
				# this will prepare for our present analysis
				self.saccade_latency_analysis_one_run(run, plot = False)
			# look at times. 
			run.first_TR_timestamp = run.events[run.events[:]['unicode'] == 't'][0]['EL_timestamp']
			run.stimulus_on_times =  run.timings['trial_phase_timestamps'][:,0,0]-run.first_TR_timestamp
			run.saccade_instruction_times =  run.timings['trial_phase_timestamps'][:,1,0]-run.first_TR_timestamp
			run.stimulus_off_times =  run.timings['trial_phase_timestamps'][:,2,0]-run.first_TR_timestamp
			
			latency_indices = np.argsort(run.saccade_stimulus_latencies)
			# latency_indices = np.argsort(run.saccade_times)
			return_times = np.array([(run.stimulus_off_times[latency_indices[i]] + run.saccade_stimulus_latencies[latency_indices[i]])/1000.0 for i in range(len(latency_indices))])
			correct_return_times = return_times[run.correct_saccades]
			# construct trial-based design
			correct_trial_parameters = run.parameter_data[run.correct_saccades]
			np.stimulus_orientation_indices = np.array([correct_trial_parameters['orientation'] == orient for orient in np.unique(correct_trial_parameters['orientation'])])
			np.what_stimulus_is_remapped_indices = np.array([correct_trial_parameters['saccade_direction'] == direction for direction in np.unique(correct_trial_parameters['saccade_direction'])])
			
			remapped_orientations_indices = np.stimulus_orientation_indices - np.what_stimulus_is_remapped_indices
			remapped_orientations_times = [correct_return_times[ro] for ro in remapped_orientations_indices]
			per_saccade_design = []
			
#			for i in range(len(remapped_orientations_times)):
#				for j in range(len(remapped_orientations_times[i])):
#					per_saccade_design.append([[remapped_orientations_times[i][j], 0.05, 1.0]])
			for i in range(correct_return_times.shape[0]):
				per_saccade_design.append([[correct_return_times[i], 0.05, 1.0]])
			per_saccade_design = np.array(per_saccade_design)
			irO = ImageRegressOperator( NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf','hpf'], extension = '.nii.gz')), per_saccade_design )
			voxel_data = self.gatherRIOData(roi, [run.indexInSession], whichMask = mask) 
			betas, sse, rank, sing = sp.linalg.lstsq( irO.design.designMatrix, voxel_data, overwrite_a = True, overwrite_b = True )
			return [correct_return_times, betas]
			
		elif run.condition == 'Mapper':
			if not hasattr(run, 'parameter_data'):
				# this will prepare for our present analysis
				self.saccade_latency_analysis_one_run(run, plot = False)
			
			stimulus_locations = np.sign((run.parameter_data[:]['contrast_L'] - run.parameter_data[:]['contrast_R']) * run.parameter_data[:]['stim_eccentricity'])
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf','hpf'], extension = '.nii.gz'))
			tr, nrsamples = niiFile.rtime, niiFile.timepoints
			stim_onsets = np.arange(0,nrsamples*tr,tr*2)
			
			stimulus_locations_indices = np.array([stimulus_locations  == tt for tt in np.unique(stimulus_locations)])
			stimulus_orientation_indices = np.array([run.parameter_data[:]['orientation'] == orient for orient in np.unique(run.parameter_data[:]['orientation'])])
			
			training_regressors = []
			trial_types = []
#			i = 0
			for stim_loc in stimulus_locations_indices:
				for stim_orient in stimulus_orientation_indices:
					training_regressors.append([stim_onsets[(stim_loc * stim_orient)], 2.0 * np.ones(stim_onsets[(stim_loc * stim_orient)].shape[0]), np.ones(stim_onsets[(stim_loc * stim_orient)].shape[0])])
			training_regressors = np.array(training_regressors)
			
			irO = ImageRegressOperator( niiFile, [t.T for t in training_regressors] )
			
			voxel_data = self.gatherRIOData(roi, [run.ID], whichMask = mask) 
			betas, sse, rank, sing = sp.linalg.lstsq( irO.design.designMatrix, voxel_data, overwrite_a = False, overwrite_b = False )
			center_predictor = betas[2]-betas[3]
			
			if False:
				f = pl.figure(figsize = (4,10))
				f.add_subplot(511)
				pl.plot(irO.design.designMatrix)
				f.add_subplot(512)
				for i in range(len(irO.design.rawDesignMatrix)):
					pl.plot(irO.design.rawDesignMatrix[i][::irO.design.subSamplingRatio])
				f.add_subplot(513)
				pl.plot(irO.design.hrfKernel[::irO.design.subSamplingRatio])
				f.add_subplot(514)
				plot(np.dot(center_predictor, voxel_data.T))
				f.add_subplot(515)
				plot(irO.design.designMatrix[:,[2,3]])
			
				pl.draw()
			return betas
			
	
	def amplitude_analysis_all_runs_all_trials_sep(self, run_length = 480, mask = '_center'):
		self.mapper_amplitude_data = []
		for r in [self.runList[i] for i in self.conditionDict['Mapper']]:
			self.mapper_amplitude_data.append(self.amplitude_analysis_one_run_all_trials_sep(r))
		self.remapping_amplitude_data = []
		f = pl.figure()
		areas = ['V1','V2','V3','V3AB','V4','inferiorparietal','superiorparietal']
		for roi in areas:
			s = f.add_subplot(len(areas),1,areas.index(roi)+1)
			self.remapping_amplitude_data.append([])
			for (k, r) in zip(range(len(self.conditionDict['Remapping'])), [self.runList[i] for i in self.conditionDict['Remapping']]):
				self.remapping_amplitude_data[-1].append(self.amplitude_analysis_one_run_all_trials_sep(r, mask = mask, roi = roi))
			sac_latencies = np.concatenate([rm[0] for rm in self.remapping_amplitude_data[-1]])
			sac_betas = np.concatenate([rm[1] for rm in self.remapping_amplitude_data[-1]])
			time_order = np.argsort(sac_latencies)
#			pl.plot( sac_latencies[time_order], np.median(sac_betas[time_order], axis = 1) )
			imshow(np.array([np.sort(hmm) for hmm in sac_betas[time_order]]))
			pl.draw()
		pl.show()
	
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
	
			
		