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
	def saccade_latency_analysis_all_runs(self, plot = True, latency_type = 'saccade', lowerbound_saccade_latency = 16.7):
		for r in [self.runList[i] for i in self.conditionDict['remap']]:
			self.saccade_latency_analysis_one_run(r, plot = plot)
		f = pl.figure(figsize = (8,4))
		if latency_type == 'saccade':
			all_latencies = np.concatenate([r.saccade_latencies for r in [self.runList[i] for i in self.conditionDict['remap']]])
		elif latency_type == 'stim_offset_saccade':
			all_latencies = np.concatenate([r.saccadeonset_stimoffset_latencies for r in [self.runList[i] for i in self.conditionDict['remap']]])
		self.all_latencies = all_latencies[all_latencies > lowerbound_saccade_latency]
		pl.hist(all_latencies, fc = 'b', bins = 30, range = [-200, 1800], histtype = 'stepfilled', linewidth = 2.0, alpha = 0.3, normed = True, ec = 'w')
		all_saccade_latencies = np.concatenate([r.saccade_latencies for r in [self.runList[i] for i in self.conditionDict['remap']]])
		all_saccade_latencies = all_latencies[all_saccade_latencies > lowerbound_saccade_latency]
		pl.hist(all_latencies, fc = 'r', bins = 30, range = [-200, 1800], histtype = 'stepfilled', linewidth = 2.0, alpha = 0.3, normed = True, ec = 'w')
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'saccade_latency_histograms.pdf'))
	
	def mapper_feat_analysis(self):
		self.mapper_parameter_data = []
		for r in [self.runList[i] for i in self.conditionDict['mapper']]:
			self.mapper_parameter_data.append(self.mapper_feat_analysis_one_run(r))
	
	def saccade_latency_analysis_one_run(self, run, plot = False):
		if run.condition == 'remap':
			# get EL Data
			elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
			elO.import_parameters(run_name = 'bla')
			
			# take saccades only for the interval after the saccade instruction was shown.
			run.timings = elO.timings
			run.parameter_data = elO.parameter_data
			run.saccades = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [3,5], data_type = 'saccades')[0] # just a single list instead of more than one....
			run.blinks = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,5], data_type = 'blinks')[0] # just a single list instead of more than one....
			run.gaze_data = elO.get_EL_samples_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,5], data_type = 'gaze_x')[0] # just a single list instead of more than one....
			run.gaze_data_during_stim = elO.get_EL_samples_per_trial(run_name = 'bla', trial_ranges = [[0,-1]], trial_phase_range = [2,4], data_type = 'gaze_x')[0] # just a single list instead of more than one....
			run.events = elO.events 
			
			run.saccadeonset_stimoffset_latencies = []
			run.saccadeoffset_stimoffset_latencies = []
			run.correct_saccades = []
			run.saccade_latencies = []
			run.which_saccades = [[None, None] for x in range(len(run.gaze_data))]
			for i in range(len(run.gaze_data)):
				# check whether at any time during the presentation, eye position was on one of the two stimuli
				nonblink_data_indices = run.gaze_data_during_stim[i][:,-1] != 0.0001
				if (np.abs((run.gaze_data_during_stim[i][nonblink_data_indices,-1] - 960)) > 400).sum() > 0:
					self.logger.debug('Run # %d, trial # %d the subject likely looked at one of the two stimuli.' % (run.ID, i))
					run.saccadeonset_stimoffset_latencies.append( -10000 )
					run.saccadeoffset_stimoffset_latencies.append( -10000 )
					run.saccade_latencies.append( -10000 )
					run.correct_saccades.append(False)
					continue
				if len(run.saccades[i]) == 0:
					self.logger.debug('Run # %d, trial # %d no saccade was made' % (run.ID, i))
					run.saccadeonset_stimoffset_latencies.append( -10000 )
					run.saccadeoffset_stimoffset_latencies.append( -10000 )
					run.saccade_latencies.append( -10000 )
					run.correct_saccades.append(False)
					continue
				else:
					self.saccade_dtype = run.saccades[i][0].dtype
					# check saccade amplitude and duration:
					x_diffs = np.abs(run.saccades[i][:]['start_x'] - run.saccades[i][:]['end_x']) > 300
					durs = run.saccades[i][:]['duration'] > 20.0
					
					if (x_diffs * durs).sum() == 0:
						self.logger.debug('Run # %d, trial # %d only microsaccades were made' % (run.ID, i))
						run.saccadeonset_stimoffset_latencies.append( -10000 )
						run.saccadeoffset_stimoffset_latencies.append( -10000 )
						run.saccade_latencies.append( -10000 )
						run.correct_saccades.append(False)
						continue
					# first saccade with sufficient amplitude and duration:
					which_saccade = np.arange(x_diffs.shape[0])[(x_diffs * durs)][0]
					
					run.saccadeonset_stimoffset_latencies.append(run.saccades[i][which_saccade]['start_timestamp'] - run.timings['trial_phase_timestamps'][i,3,0])
					run.saccadeoffset_stimoffset_latencies.append(run.saccades[i][which_saccade]['end_timestamp'] - run.timings['trial_phase_timestamps'][i,3,0])
					run.saccade_latencies.append(run.saccades[i][which_saccade]['start_timestamp'] - run.timings['trial_phase_timestamps'][i,2,0])
					
					x_diff = x_diffs[which_saccade]
					run.which_saccades[i] = [which_saccade, run.saccades[i][which_saccade]]
				
				# check the direction of the saccade
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
			run.saccadeonset_stimoffset_latencies = np.array(run.saccadeonset_stimoffset_latencies)
			run.saccadeoffset_stimoffset_latencies = np.array(run.saccadeoffset_stimoffset_latencies)
			
			if plot:
				f = pl.figure(figsize = (12, len(run.gaze_data)/3.0))
				f.subplots_adjust(wspace = 0.2, hspace = 0.3, left = 0.05, right = 0.95, bottom = 0.05, top = 0.99)
				for i in range(len(run.gaze_data)):
					s = f.add_subplot(ceil(len(run.gaze_data)/3.0), 3, i+1)
					# trial phases:
					s.axvspan(run.timings['trial_phase_timestamps'][i,1,0], run.timings['trial_phase_timestamps'][i,2,0], facecolor='y', alpha=0.25, edgecolor = 'w')
					s.axvspan(run.timings['trial_phase_timestamps'][i,2,0], run.timings['trial_phase_timestamps'][i,3,0], facecolor='y', alpha=0.5, edgecolor = 'w')
					[pl.axvline(x = ts, c = 'k', linewidth = 1.5, alpha = 0.5) for ts in elO.timings['trial_phase_timestamps'][i,:,0]]
					# saccade that counts is:
					if run.which_saccades[i][0] != None:
						pl.axvline(x = run.saccades[i][run.which_saccades[i][0]]['start_timestamp'], c = 'b', linewidth = 3.5, alpha = 0.5)
					# saccade goals:
					pl.axhline(y = 960 + 672 * (run.parameter_data[i]['saccade_direction'] * ((np.sign(np.mod(run.ID,2))*2)-1)), c = 'r', linestyle = '--', linewidth = 2.5, alpha = 0.5)
					pl.axhline(y = 960 - 672 * (run.parameter_data[i]['saccade_direction'] * ((np.sign(np.mod(run.ID,2))*2)-1)), c = 'g', linestyle = '--', linewidth = 2.5, alpha = 0.5)
					# gaze data:
					gd = run.gaze_data[i][::10]
					pl.plot(gd[gd[:,-1]!=0.0001,0], gd[gd[:,-1]!=0.0001,-1], 'k', alpha = 0.7, linewidth = 3.5)
					s.set_ylim([0,1920])
				pl.savefig(self.runFile(stage = 'processed/eye', run = run, postFix = ['position_per_trial'], extension = '.pdf'))
			
		elif run.condition == 'mapper':
			pass
	
	def times_all_runs(self, nr_bins = 2, latency_type = 'saccade'):
		# if not hasattr(self, 'all_latencies'):
		self.saccade_latency_analysis_all_runs(plot = False, latency_type = latency_type)
		self.cutoff_latencies = [self.all_latencies[np.argsort(self.all_latencies)[i * floor(self.all_latencies.shape[0]/nr_bins)]] for i in range(nr_bins)]
		self.cutoff_latencies.append(np.max(self.all_latencies))
		self.cutoff_latencies = np.array([np.array(self.cutoff_latencies)[:-1], np.array(self.cutoff_latencies)[1:]]).T
		
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['remap'][0]], postFix = ['mcf','hpf','psc'], extension = '.nii.gz'))
		tr, nrsamples = niiFile.rtime, niiFile.timepoints
		run_length = tr * nrsamples
		
		binned_sacc_times = []
		for l in self.cutoff_latencies:
			sacc_times = []
			stim_times = []
			for (i, r) in enumerate([self.runList[i] for i in self.conditionDict['remap']]):
				ss_times = self.times_for_run(r, l, latency_type = latency_type)
				ss_times[1] = ss_times[1][ss_times[1]>0]
				sacc_times.append(i*run_length + ss_times[0])
				stim_times.append(i*run_length + ss_times[1])
			binned_sacc_times.append(np.concatenate(sacc_times))
		self.stim_times = np.concatenate(stim_times)	
		self.binned_sacc_times = binned_sacc_times
		
	def deconvolve_roi(self, roi, threshold = 3.5, mask_type = 'center_Z_contrast_joined', mask_direction = 'pos', event_type = 'binned_saccade_onsets', color = 'b', latency_type = 'saccade', permute = False, nr_permutations = 250, sample_interval = 0.375 ):
		if not hasattr(self, 'binned_sacc_times'):
			self.times_all_runs(latency_type = latency_type)
		
		remap_h5file = self.hdf5_file('remap')
		mapper_h5file = self.hdf5_file('mapper')
		
		roi_data = []
		for r in [self.runList[i] for i in self.conditionDict['remap']]:
			roi_data.append(self.roi_data_from_hdf(remap_h5file, r, roi, 'mcf_psc_hpf_data'))
			
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
			
		roi_data_per_run = demeaned_roi_data
		roi_data = np.hstack(demeaned_roi_data)
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
			
		timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		fig = pl.figure(figsize = (9, 3))
		fig.subplots_adjust(wspace = 0.2, hspace = 0.3, left = 0.1, right = 0.95, bottom = 0.1, top = 0.9)
		s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		interval = [0.0,9.0]
		
		if event_type == 'binned_saccade_onsets':
			eventData = self.binned_sacc_times
			labels = [str(i.mean()) for i in self.cutoff_latencies]
		elif event_type == 'stim_onsets':
			eventData = [self.stim_times]
			labels = ['stimulus']
		elif event_type == 'all_saccade_onsets':
			eventData = [np.concatenate(self.binned_sacc_times)]
			labels = ['saccade']
		
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = eventData, TR = 1.5, deconvolutionSampleDuration = sample_interval, deconvolutionInterval = interval[1])
		for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], color, alpha = np.linspace(0.25,1,deco.deconvolvedTimeCoursesPerEventType.shape[0])[i], label = labels[i])
			time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		s.set_title('deconvolution ' + roi + ' ' + mask_type.split('_')[0] + ' ' + event_type)
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
			
		remap_h5file.close()
		mapper_h5file.close()	
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'decon_'+ roi + '_' + mask_type.split('_')[0] + ' ' + latency_type +'.pdf'))
		
		if permute == False:
			return np.array(time_signals)
		else:
			# we will permute the events and split up in two parts, looking at the lag between resultant FIR signals
			original_difference = deco.deconvolvedTimeCoursesPerEventType[0] - deco.deconvolvedTimeCoursesPerEventType[1]
			
			permute_results = np.zeros((nr_permutations, deco.deconvolvedTimeCoursesPerEventType[0].shape[0]))
			# we permute the different event-related responses with 2 bins
			all_sacc_times = np.concatenate(self.binned_sacc_times)
			all_sacc_times = all_sacc_times[:2*floor(all_sacc_times.shape[0]/2.0)]
			for i in range(nr_permutations):
				np.random.shuffle(all_sacc_times)
				deco = DeconvolutionOperator(inputObject = timeseries, eventObject = all_sacc_times.reshape((2,all_sacc_times.shape[0]/2)), TR = 1.5, deconvolutionSampleDuration = sample_interval, deconvolutionInterval = interval[1])
				# preprocess for normalized correlation of timeseries
				permute_results[i] = deco.deconvolvedTimeCoursesPerEventType[0] - deco.deconvolvedTimeCoursesPerEventType[1]
			fig = pl.figure(figsize = (7,6))
			s = fig.add_subplot(211)
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), original_difference, color = 'k', alpha = 1.0, linewidth = 2.0)
			for i in range(nr_permutations):
				pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), permute_results[i], color = 'b', alpha = 0.25, linewidth = 0.75)
			s.axhline(0, -10, 30, linewidth = 0.25)
			s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			s.set_ylabel('% signal change difference')
			s.set_xlabel('time [s]')
			s.set_title(roi)
			s = fig.add_subplot(212)
			diff_ratios = np.array([(p>original_difference[i]).sum()/float(nr_permutations) for (i, p) in enumerate(permute_results.T)])
			pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), diff_ratios, color = 'b', alpha = 0.75, linewidth = 1.25)
			s.axhline(0, -10, 30, linewidth = 0.25)
			s.axhline(0.05, -10, 30, color = 'r', linewidth = 0.75, alpha = 0.25)
			s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			s.set_ylabel('permutation probability')
			s.set_xlabel('time [s]')
			
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'decon_perm_'+ roi + '_' + mask_type.split('_')[0] + '_' + event_type +'.pdf'))
			return [np.array(time_signals), original_difference, permute_results]
	
	def single_trial_glm_film(self, latency_type = 'saccade', types = ('mapper','remap') ):
		basic_film_command = 'film_gls -sa -epith 200 -output_pwdata -v -rn %s %s %s'
		if 'mapper' in types:
			film_commands = []
			# stimulus glm
			for (i, r) in enumerate([self.runList[i] for i in self.conditionDict['mapper']]):
			
				elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'))
				elO.import_parameters(run_name = 'bla')
				r.first_TR_timestamp = elO.events[elO.events[:]['unicode'] == 't'][0]['EL_timestamp']
			
				r.stimulus_on_times =  (elO.timings['trial_phase_timestamps'][:,1,0]-r.first_TR_timestamp) / 1000.0
				r.stimulus_durations = (elO.timings['trial_phase_timestamps'][:,2,0]-elO.timings['trial_phase_timestamps'][:,1,0]) / 1000.0
			
			
				stim_regressors = np.vstack((r.stimulus_on_times, r.stimulus_durations, np.ones(r.stimulus_on_times.shape[0]) * 1)).T.reshape((r.stimulus_on_times.shape[0],1,3))
			
				d = Design(nrTimePoints = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hpf'])).timepoints, rtime = 1.5, subSamplingRatio = 100)
				d.configure( regressors = stim_regressors )
				design_file = self.runFile(stage = 'processed/mri', run = r, postFix = ['single_trial_design'], extension = '.mat')
				design_file_header = """/NumWaves\t%d\n/NumPoints\t%d\n/PPheights\t\t%s\n\n/Matrix\n""" % (d.designMatrix.shape[1], d.designMatrix.shape[0], '\t'.join(['1' for i in range(d.designMatrix.shape[0])]))
				f = open( design_file, 'w')
				f.write(design_file_header)
				f.close()
				f = open( design_file, 'a')
				np.savetxt( f, d.designMatrix, delimiter = '\t', fmt = '%3.2f' ) #, header = design_file_header
				f.close()
				# film_gls fit with autocorrelation correction:
				film_commands.append(basic_film_command % ( os.path.splitext(design_file)[0], os.path.splitext(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hpf'], extension = '.mat'))[0], design_file ))
			
			# parallel implementation
			ppservers = ()
			job_server = pp.Server(ncpus = 4, ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = [job_server.submit(ExecCommandLine,(fgls,),(),('subprocess','tempfile',)) for fgls in film_commands]
			for fgls in ppResults:
				fgls()
			
			job_server.print_stats()
		
		# import pdb; pdb.set_trace()
		if 'remap' in types:
			film_commands = []
			# saccade glm
			for (i, r) in enumerate([self.runList[i] for i in self.conditionDict['remap']]):
				times = self.times_for_run(r, np.array([50, 1500]), latency_type = latency_type)
				ss_times = times[0]
				ss_latencies = times[2]
			
				saccade_regressors = np.vstack((ss_times, np.ones(ss_times.shape[0]) * 0.2, np.ones(ss_times.shape[0]) * 1)).T.reshape((ss_times.shape[0],1,3))
			
				d = Design(nrTimePoints = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hpf'])).timepoints, rtime = 1.5, subSamplingRatio = 100)
				d.configure( regressors = saccade_regressors )
				design_file = self.runFile(stage = 'processed/mri', run = r, postFix = ['single_trial_design'], extension = '.mat')
				design_file_header = """/NumWaves\t%d\n/NumPoints\t%d\n/PPheights\t\t%s\n\n/Matrix\n""" % (d.designMatrix.shape[1], d.designMatrix.shape[0], '\t'.join(['1' for i in range(d.designMatrix.shape[0])]))
				f = open( design_file, 'w')
				f.write(design_file_header)
				f.close()
				f = open( design_file, 'a')
				np.savetxt( f, d.designMatrix, delimiter = '\t', fmt = '%3.2f' ) #, header = design_file_header
				f.close()
				# film_gls fit with autocorrelation correction:
				film_commands.append(basic_film_command % ( os.path.splitext(design_file)[0], os.path.splitext(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hpf'], extension = '.mat'))[0], design_file ))
			
			# parallel implementation
			ppservers = ()
			job_server = pp.Server(ncpus = 4, ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = [job_server.submit(ExecCommandLine,(fgls,),(),('subprocess','tempfile',)) for fgls in film_commands]
			for fgls in ppResults:
				fgls()
			
			job_server.print_stats()
	
	
	def single_saccade_glm_roi(self, roi, threshold = 3.5, mask_type = 'center_Z_contrast_joined', mask_direction = 'pos', latency_type = 'saccade'):
		"""docstring for single_saccade_glm_roi"""
		remap_h5file = self.hdf5_file('remap')
		mapper_h5file = self.hdf5_file('mapper')
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type)
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		betas_for_latencies = []
		all_sacc_latencies = []
		
		
		for (i, r) in enumerate([self.runList[i] for i in self.conditionDict['remap']]):
			times = self.times_for_run(r, np.array([-20000, 10000]), latency_type = latency_type)
			ss_times = times[0]
			ss_latencies = times[2]
			roi_data = self.roi_data_from_hdf(remap_h5file, r, roi, 'mcf_hpf_data')
			demeaned_roi_data = (roi_data.T - roi_data.mean(axis = 1)).T
			
			saccade_regressors = np.vstack((ss_times, np.ones(ss_times.shape[0]) * 0.2, np.ones(ss_times.shape[0]) * 1)).T.reshape((ss_times.shape[0],1,3))
			
			d = Design(nrTimePoints = demeaned_roi_data.shape[1], rtime = 1.5, subSamplingRatio = 100)
			d.configure( regressors = saccade_regressors )
			
			# internal leastsquares
			design = d.designMatrix / d.designMatrix.mean(axis = 0)
			betas, sse, rank, sing = sp.linalg.lstsq( design, demeaned_roi_data[mapping_mask,:].T, overwrite_a = True, overwrite_b = True )
			
			betas_for_latencies.append([ss_latencies, (betas).mean(axis = 1)])
		
		
		remap_h5file.close()
		mapper_h5file.close()
		
		sl = np.concatenate([bnl[0] for bnl in betas_for_latencies])
		bs = np.concatenate([bnl[1] for bnl in betas_for_latencies])
		sr = sp.stats.spearmanr(sl, bs)
		
		fig = pl.figure(figsize = (6, 4))
		s = fig.add_subplot(111)
		pl.plot( sl, bs, marker = 'o', ms = 5, mec = 'r', c = 'w', mew = 1.0, alpha = 0.15, linewidth = 0)
		s.set_title('correlation ' + roi + ' ' + mask_type.split('_')[0])
		s.set_xlabel('stim offset - saccade latency')
		s.set_ylabel('beta value')
		pl.annotate( 'spearmans rho: %0.2f, \n p: %0.3f' % sr, (0,0))
		
		# from scikits.timeseries.lib.moving_funcs import *
		# 
		# smooth_width = 120
		# order = np.argsort(sl)
		# # kern =  stats.norm.pdf( np.linspace(-3.25,3.25,smooth_width) )
		# # sm_signal = np.convolve( bs[order], kern / kern.sum(), 'valid' )
		# # sm_time = np.convolve( sl[order], kern / kern.sum(), 'valid' )
		# # pl.plot(sm_time, sm_signal, 'r', alpha = 0.8, linewidth = 2.0)
		# pl.plot(mov_median(sl[order], smooth_width), mov_median(bs[order], smooth_width), 'r', alpha = 0.8, linewidth = 2.0)
		
		bins = np.array([np.linspace(30,550,12)[:-1],np.linspace(30,550,12)[1:]]).T
		bs_binned = [bs[(sl<a[1])*(sl>a[0])] for a in bins]
		bs_binned_mean = np.array([[b.mean(), b.std()/np.sqrt(b.shape[0])] for b in bs_binned])
		# import pdb; pdb.set_trace()
		
		pl.plot( bins.mean(axis = 1), bs_binned_mean[:,0], 'r', linewidth = 1.5 )
		pl.fill_between( bins.mean(axis = 1), bs_binned_mean[:,0]-bs_binned_mean[:,1], bs_binned_mean[:,0]+bs_binned_mean[:,1], color = 'r', alpha = 0.25)
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'corr '+ roi + ' ' + mask_type.split('_')[0] +'.pdf'))
		
	
	
	def single_saccade_glm(self, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4', 'lateraloccipital', 'superiorparietal']):
		"""docstring for single_saccade_glm"""
		for roi in rois:
			self.single_saccade_glm_roi(roi = roi, threshold = 3.5)
	
	
	def deconvolve(self, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4', 'lateraloccipital', 'superiorparietal'], latency_type = 'saccade', nr_bins = 2, permute = False ):
		"""docstring for deconvolve"""
		roi_data = []
		self.times_all_runs(latency_type = latency_type, nr_bins = nr_bins)
		for roi in rois:
			roi_data.append(self.deconvolve_roi(roi = roi, threshold = 3.5, mask_type = 'center_Z_contrast_joined', mask_direction = 'pos', event_type = 'binned_saccade_onsets', color = 'b', latency_type = latency_type, permute = permute ))
			# self.deconvolve_roi(roi = roi, threshold = 2.3, mask_type = 'left_Z_contrast_joined', mask_direction = 'pos', event_type = 'binned_saccade_onsets', color = 'g' )
			# self.deconvolve_roi(roi = roi, threshold = 2.3, mask_type = 'right_Z_contrast_joined', mask_direction = 'pos', event_type = 'binned_saccade_onsets', color = 'r' )
			# self.deconvolve_roi(roi = roi, threshold = 2.3, mask_type = 'left_Z_contrast_joined', mask_direction = 'pos', event_type = 'all_saccade_onsets', color = 'r' )
			# self.deconvolve_roi(roi = roi, threshold = 2.3, mask_type = 'right_Z_contrast_joined', mask_direction = 'pos', event_type = 'all_saccade_onsets', color = 'r' )
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'dec_' + latency_type + '.npy'),np.array(roi_data))
		
		
	
	def times_for_run(self, run, saccade_latency_range = [0,2000], latency_type = 'saccade'):
		if not hasattr(run, 'saccadeonset_stimoffset_latencies'):
			# this will prepare for our present analysis
			self.saccade_latency_analysis_one_run(run, plot = False)
		# look at times. 
		run.first_TR_timestamp = run.events[run.events[:]['unicode'] == 't'][0]['EL_timestamp']
		run.stimulus_on_times =  run.timings['trial_phase_timestamps'][:,1,0]-run.first_TR_timestamp
		run.saccade_instruction_times =  run.timings['trial_phase_timestamps'][:,2,0]-run.first_TR_timestamp
		run.stimulus_off_times =  run.timings['trial_phase_timestamps'][:,3,0]-run.first_TR_timestamp
		
		if latency_type == 'saccade':
		# trials_for_this_slr = np.arange(run.stimulus_off_times.shape[0])[((run.saccadeonset_stimoffset_latencies > saccade_latency_range[0]) * (run.saccadeonset_stimoffset_latencies < saccade_latency_range[1]))]
			trials_for_this_slr = np.arange(run.stimulus_off_times.shape[0])[((run.saccade_latencies > saccade_latency_range[0]) * (run.saccade_latencies < saccade_latency_range[1]))]
		elif latency_type == 'stim_offset_saccade':
			trials_for_this_slr = np.arange(run.stimulus_off_times.shape[0])[((run.saccadeonset_stimoffset_latencies > saccade_latency_range[0]) * (run.saccadeonset_stimoffset_latencies < saccade_latency_range[1]))]
		
		trial_saccades = np.array([run.which_saccades[s][1] for s in trials_for_this_slr], dtype = self.saccade_dtype)
		saccade_onsets = trial_saccades['start_timestamp'] - run.first_TR_timestamp
		
		design = np.vstack((saccade_onsets / 1000.0, np.ones(saccade_onsets.shape[0]) * 0.2, np.ones(saccade_onsets.shape[0]) * 1))
		np.savetxt( self.runFile(stage = 'processed/mri', run = run, postFix = ['design_sacc','%3.0f-%3.0f'%(saccade_latency_range[0],saccade_latency_range[1])], extension = '.txt'), design.T, fmt = '%3.2f', delimiter = '\t')
		design = np.vstack((run.stimulus_on_times / 1000.0, (run.stimulus_off_times-run.stimulus_on_times) / 1000.0, np.ones(run.stimulus_on_times.shape[0]) * 1))
		np.savetxt( self.runFile(stage = 'processed/mri', run = run, postFix = ['design_stim'], extension = '.txt'), design.T, fmt = '%3.2f', delimiter = '\t')
		
		return [saccade_onsets / 1000.0, run.stimulus_on_times / 1000.0, run.saccadeonset_stimoffset_latencies[trials_for_this_slr]]
	
	
	def mapper_feat_analysis_one_run(self, run, run_feat = True, postFix = ['mcf']):
		elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
		elO.import_parameters(run_name = 'bla')
		####
		####	run retinotopic area mapping. do the main pattern-based GLM internally. 
		####
		trial_types = np.sign((elO.parameter_data[:]['contrast_L'] - elO.parameter_data[:]['contrast_R']) * elO.parameter_data[:]['stim_eccentricity'])
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf','hpf','psc'], extension = '.nii.gz'))
		tr, nrsamples = niiFile.rtime, niiFile.timepoints
		trial_onsets = elO.timings['trial_phase_timestamps'][:,0,-1]-elO.timings['trial_phase_timestamps'][0,0,-1]
		stim_onsets = elO.timings['trial_phase_timestamps'][:,1,-1]-elO.timings['trial_phase_timestamps'][0,1,-1]
		# import pdb; pdb.set_trace()
		stim_durations = np.ones((stim_onsets.shape[0])) * 2
		design = np.vstack((stim_onsets, stim_durations, trial_types))
		# elo.timings[:,1]['']
		stim_locations = np.sort(np.unique(trial_types))[::-1]
		segmented_design = [] # [[[0,tr * nrsamples]]]	# the mean value for regression
		for i in range(3):
			this_location_design = design[:,design[2] == stim_locations[i]]
			this_location_design[2] = 1
			# print out design file for fsl analysis
			np.savetxt( self.runFile(stage = 'processed/mri', run = run, postFix = ['design',str(i)], extension = '.txt'), this_location_design.T, fmt = '%3.1f', delimiter = '\t')
			segmented_design.append(this_location_design.T)
		segmented_design.append([[0,tr * nrsamples, 1.0]])
		# don't do this stuff now. let's just take fsl's output
		
		if run_feat:
			try:
				self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.fsf'))
			except OSError:
				pass
		
			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
			thisFeatFile = '/Volumes/HDD/research/projects/remapping/latency/analysis/mapper.fsf'
			REDict = {
			'---NII_FILE---': 				self.runFile(stage = 'processed/mri', run = run, postFix = postFix), 
			'---EVT_FILE_0---': 			self.runFile(stage = 'processed/mri', run = run, postFix = ['design','0'], extension = '.txt'), 	
			'---EVT_FILE_1---': 			self.runFile(stage = 'processed/mri', run = run, postFix = ['design','1'], extension = '.txt'), 	
			'---EVT_FILE_2---': 			self.runFile(stage = 'processed/mri', run = run, postFix = ['design','2'], extension = '.txt'), 	
			}
			featFileName = self.runFile(stage = 'processed/mri', run = run, extension = '.fsf')
			featOp = FEATOperator(inputObject = thisFeatFile)
			featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
			self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
			# run feat
			featOp.execute()
			
			self.setupRegistrationForFeat(self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
		
	
	
	def mask_stats_to_hdf(self, run_type = 'mapper', postFix = ['mcf']):
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
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('starting table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		else:
			self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
			
		for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
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
			# joined_feat_directory = self.stageFolder(stage = 'processed/mri/mapper/joined.gfeat')
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			if run_type == 'mapper':
				stat_files = {
								'center_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
								'center_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
								'center_cope': os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
								
								'left_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
								'left_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
								'left_cope': os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
								
								'right_T': os.path.join(this_feat, 'stats', 'tstat3.nii.gz'),
								'right_Z': os.path.join(this_feat, 'stats', 'zstat3.nii.gz'),
								'right_cope': os.path.join(this_feat, 'stats', 'cope3.nii.gz'),
								
								'center_contrast_T': os.path.join(this_feat, 'stats', 'tstat4.nii.gz'),
								'center_contrast_Z': os.path.join(this_feat, 'stats', 'zstat4.nii.gz'),
								'center_contrast_cope': os.path.join(this_feat, 'stats', 'cope4.nii.gz'),
								
								'left_contrast_T': os.path.join(this_feat, 'stats', 'tstat5.nii.gz'),
								'left_contrast_Z': os.path.join(this_feat, 'stats', 'zstat5.nii.gz'),
								'left_contrast_cope': os.path.join(this_feat, 'stats', 'cope5.nii.gz'),
								
								'right_contrast_T': os.path.join(this_feat, 'stats', 'tstat6.nii.gz'),
								'right_contrast_Z': os.path.join(this_feat, 'stats', 'zstat6.nii.gz'),
								'right_contrast_cope': os.path.join(this_feat, 'stats', 'cope6.nii.gz'),
								
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								
								}
				
			elif run_type == 'remap':
				# no feat has been run for remapping
				stat_files = {}
			
			# general info we want in all hdf files
			stat_files.update({
								'mcf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'mcf_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hpf']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'mcf_psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'hpf', 'psc']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								# for these final two, we need to pre-setup the retinotopic mapping data
								'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
								'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz'),
								
								'center_T_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope1_tstat1.nii.gz'),
								'center_Z_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope1_zstat1.nii.gz'),
								'left_T_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope2_tstat1.nii.gz'),
								'left_Z_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope2_zstat1.nii.gz'),
								'right_T_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope3_tstat1.nii.gz'),
								'right_Z_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope3_zstat1.nii.gz'),
								
								'center_T_contrast_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope4_tstat1.nii.gz'),
								'center_Z_contrast_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope4_zstat1.nii.gz'),
								'left_T_contrast_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope5_tstat1.nii.gz'),
								'left_Z_contrast_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope5_zstat1.nii.gz'),
								'right_T_contrast_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope6_tstat1.nii.gz'),
								'right_Z_contrast_joined': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat_mapper/'), 'cope6_zstat1.nii.gz'),
								
								# add joined stat files results to all hdf files
			})
			
			stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
			for (roi, roi_name) in zip(rois, roinames):
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
		h5file.close()
	

	def hdf5_file(self, run_type):
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('no table file ' + self.hdf5_filename + 'found for stat mask')
			return None
		else:
			# self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "r", title = run_type + " file")
		return h5file
	

	def roi_data_from_hdf(self, h5file, run, roi_wildcard, data_type, postFix = ['mcf']):
		"""
		drags data from an already opened hdf file into a numpy array, concatenating the data_type data across voxels in the different rois that correspond to the roi_wildcard
		"""
		this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]
		try:
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			roi_names = []
			for roi_name in h5file.iterNodes(where = '/' + this_run_group_name, classname = 'Group'):
				if len(roi_name._v_name.split('.')) > 1:
					hemi, area = roi_name._v_name.split('.')
					if roi_wildcard == area:
						roi_names.append(roi_name._v_name)
			if len(roi_names) == 0:
				self.logger.info('No rois corresponding to ' + roi_wildcard + ' in group ' + this_run_group_name)
				return None
		except NoSuchNodeError:
			# import actual data
			self.logger.info('No group ' + this_run_group_name + ' in this file')
			return None
			
		all_roi_data = []
		for roi_name in roi_names:
			thisRoi = h5file.getNode(where = '/' + this_run_group_name, name = roi_name, classname='Group')
			all_roi_data.append( eval('thisRoi.' + data_type + '.read()') )
		all_roi_data_np = np.hstack(all_roi_data).T
		return all_roi_data_np
	
	
	def project_stats(self, which_file = 'zstat', postFix = ['mcf']):
		for r in [self.runList[i] for i in self.conditionDict['mapper']]:
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			
			center_Z = os.path.join(this_feat, 'stats', which_file+'1.nii.gz')
			left_Z = os.path.join(this_feat, 'stats', which_file+'2.nii.gz')
			right_Z = os.path.join(this_feat, 'stats', which_file+'3.nii.gz')
			
			center_Z_C = os.path.join(this_feat, 'stats', which_file+'4.nii.gz')
			left_Z_C = os.path.join(this_feat, 'stats', which_file+'5.nii.gz')
			right_Z_C = os.path.join(this_feat, 'stats', which_file+'6.nii.gz')
			
			for (label, f) in zip(	['center_Z','left_Z','right_Z','center_Z_C','left_Z_C','right_Z_C'],
									[center_Z,left_Z,right_Z,center_Z_C,left_Z_C,right_Z_C]):
				vsO = VolToSurfOperator(inputObject = f)
				ofn = self.runFile(stage = 'processed/mri/', run = r, base = which_file, postFix = [label] )
				ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
				vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
				vsO.execute()
		
		this_feat = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'gfeat_mapper')
		
		center_Z = os.path.join(this_feat, 'cope1_zstat1.nii.gz')
		left_Z = os.path.join(this_feat, 'cope2_zstat1.nii.gz')
		right_Z = os.path.join(this_feat, 'cope3_zstat1.nii.gz')
		
		center_Z_C = os.path.join(this_feat, 'cope4_zstat1.nii.gz')
		left_Z_C = os.path.join(this_feat, 'cope5_zstat1.nii.gz')
		right_Z_C = os.path.join(this_feat, 'cope6_zstat1.nii.gz')
		
		for (label, f) in zip(	['center_Z','left_Z','right_Z','center_Z_C','left_Z_C','right_Z_C'],
								[center_Z,left_Z,right_Z,center_Z_C,left_Z_C,right_Z_C]):
			vsO = VolToSurfOperator(inputObject = f)
			ofn = self.runFile(stage = 'processed/mri/mapper', run = r, base = which_file, postFix = [label] )
			ofn = os.path.join(self.stageFolder(stage = 'processed/mri/mapper/surf'), 'joined_' + label)
			# ofn = os.path.join(os.path.split(ofn)[0], 'surf/', label)
			vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 2.0, surfType = 'paint'  )
			vsO.execute()

	
	
