#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Session import * 
from ...Operators.ArrayOperator import *
from ...Operators.EyeOperator import *
from ...circularTools import *
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle
from scipy.stats import *
from SingleRewardSession import *
from ...plotting_tools import *

class VariableRewardSession(SingleRewardSession):
	def deconvolve_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'reward'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = round(niiFile.rtime*100)/100.0, niiFile.timepoints
		run_duration = tr * nr_trs
		
		reward_h5file = self.hdf5_file('reward')
		
		self.deconvolution_labels = ['75%_yes', '75%_no', '75%_stim', '50%_yes', '50%_no', '50%_stim', '25%_yes', '25%_no', '25%_stim', 'blank_reward']
		decon_label_grouping = [[0,1,2],[3,4,5],[6,7,8],[-1]]
		colors = [['b--','b','b'],['g--','g','g'],['r--','r','r'], ['k--']]
		alphas = [[1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0]]
		lthns = [[2.0, 2.0, 2.0],[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0]]
		
		event_data = []
		roi_data = []
		nr_runs = 0
		blink_events = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data', postFix = ['mcf']))
			this_run_events = []
			for cond in self.deconvolution_labels:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)
			nr_runs += 1
			
		# join event data for stimulus events per probability bin - the stimulus responses cannot be different for rewarded or unrewarded trials
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, mask_type, postFix = ['mcf'])
		# and close the file
		reward_h5file.close()
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		time_signals = []
		interval = [0.0,16.0]
		
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure(np.array([np.vstack(blink_events)]))
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
		for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
			time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
		
		# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
		# 	time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		time_signals = np.array(time_signals).squeeze()
		# shell()
		fig = pl.figure(figsize = (8, 16))
		s = fig.add_subplot(411)
		s.axhline(0, -10, 30, linewidth = 0.25)
		for i in range(3): # plot stimulus responses
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[decon_label_grouping[i][-1]], colors[i][-1], alpha = alphas[i][-1], linewidth = lthns[i][-1], label = self.deconvolution_labels[decon_label_grouping[i][-1]])
		s.set_title('deconvolution stimulus response' + roi + ' ' + mask_type)		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				# if i == self.which_stimulus_rewarded:
				l.set_linewidth(3.5)  # the legend line width
				# else:
					# l.set_linewidth(2.0)  # the legend line width
		for i in range(3): # plot stimulus responses
		
			s = fig.add_subplot(4,1,2+i)
			s.axhline(0, -10, 30, linewidth = 0.25)
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[decon_label_grouping[i][0]], colors[i][0], alpha = alphas[i][0], linewidth = lthns[i][0], label = self.deconvolution_labels[decon_label_grouping[i][0]])
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[decon_label_grouping[i][1]], colors[i][1], alpha = alphas[i][1], linewidth = lthns[i][1], label = self.deconvolution_labels[decon_label_grouping[i][1]])
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[decon_label_grouping[-1][0]], colors[-1][0], alpha = alphas[-1][0], linewidth = lthns[-1][0], label = self.deconvolution_labels[decon_label_grouping[-1][0]])
			s.set_title('deconvolution reward response' + roi + ' ' + mask_type)		
			s.set_xlabel('time [s]')
			s.set_ylabel('% signal change')
			s.set_xlim([interval[0]-1.5, interval[1]+1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for (i, l) in enumerate(leg.get_lines()):
					# if i == self.which_stimulus_rewarded:
					l.set_linewidth(3.5)  # the legend line width
					# else:
						# l.set_linewidth(2.0)  # the legend line width
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals)] #, deco_per_run]
	
	def deconvolve_plus_glm_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'GLM'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = round(niiFile.rtime*100)/100.0, niiFile.timepoints
		run_duration = tr * nr_trs
		
		reward_h5file = self.hdf5_file('reward')
		
		self.deconvolution_labels = ['75%_yes', '75%_no', '75%_stim', '50%_yes', '50%_no', '50%_stim', '25%_yes', '25%_no', '25%_stim', '75%_delay', '50%_delay', '25%_delay', 'blank_reward' ]
		decon_label_grouping = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12]]
		colors = [['b--','b','b'],['g--','g','g'],['r--','r','r'],['k--','k','k'], ['k--']]
		alphas = [[1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0]]
		lthns = [[2.0, 2.0, 2.0],[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0]]
		
		event_data = []
		roi_data = []
		nr_runs = 0
		blink_events = []
		delay_events = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data', postFix = ['mcf']))
			this_run_events = []
			for cond in self.deconvolution_labels:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond])))	
				this_run_events[-1][:,0] = this_run_events[-1][:,0] + nr_runs * run_duration
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			event_data.append(this_run_events)
			nr_runs += 1
			
		# join event data for stimulus events per probability bin - the stimulus responses cannot be different for rewarded or unrewarded trials
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		reward_event_data = [event_data[i][:,0] for (i,s) in enumerate(self.deconvolution_labels) if 'yes' in s or 'no' in s or 'reward' in s]
		stimulus_event_data = [event_data[i] for (i,s) in enumerate(self.deconvolution_labels) if 'stim' in s]
		delay_event_data = [event_data[i] for (i,s) in enumerate(self.deconvolution_labels) if 'delay' in s]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, mask_type, postFix = ['mcf'])
		# and close the file
		reward_h5file.close()
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		time_signals = []
		interval = [0.0,16.0]
		
		# shell()
		
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		nuisance_design.configure([list(np.vstack(blink_events))])
		# nuisance_design.configure([list(np.vstack(blink_events))], hrfType = 'doubleGamma', hrfParameters = {'a1': 6, 'a2': 12, 'b1': 0.9, 'b2': 0.9, 'c': 0.35})
		
		stimulus_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		stimulus_design.configure(stimulus_event_data)	# standard HRF for stimulus events
		# stimulus_design.configure(stimulus_event_data, hrfType = 'doubleGamma', hrfParameters = {'a1': 6, 'a2': 12, 'b1': 0.9, 'b2': 0.9, 'c': 0.35})	# standard HRF for stimulus events
		
		# non-standard reward HRF for delay events
		delay_design = Design(timeseries.shape[0] * 2, tr/2.0 )
		# delay_design.configure(delay_event_data, hrfType = 'doubleGamma', hrfParameters = {'a1' : 22.32792026, 'a2' : 18.05752151, 'b1' : 0.30113662, 'b2' : 0.37294047, 'c' : 1.21845208})#, hrfType = 'double_gamma', hrfParameters = {'a1':-1.43231888, 'sh1':9.09749517, 'sc1':0.85289563, 'a2':0.14215637, 'sh2':103.37806306, 'sc2':0.11897103}) 22.32792026  18.05752151   0.30113662   0.37294047   1.21845208 {a1 = 22.32792026, a2 = 18.05752151, b1 = 0.30113662, b2 = 0.37294047, c = 1.21845208}
		delay_design.configure(delay_event_data, hrfType = 'singleGamma', hrfParameters = {'a':10.46713698,'b':0.65580082})
		# delay_design.configure(delay_event_data)
		nuisance_design_matrix = np.hstack((stimulus_design.designMatrix, delay_design.designMatrix, nuisance_design.designMatrix))
		
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = reward_event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(nuisance_design_matrix)
		for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
			time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
		
		# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
		# 	time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		time_signals = np.array(time_signals).squeeze()
		# shell()
		fig = pl.figure(figsize = (8, 16))
		s = fig.add_subplot(411)
		s.axhline(0, interval[0]-1.5, interval[1]+1.5, linewidth = 0.25)
		for i in range(3): # plot stimulus responses
			pl.bar(i, deco.deconvolvedTimeCoursesPerEventTypeNuisanceAll[-4:-1][i][0,0], width = 0.25, edgecolor = 'k', color = colors[i][-1], label = ['75%_delay', '50%_delay', '25%_delay'][i])
			pl.bar(i+0.25, deco.deconvolvedTimeCoursesPerEventTypeNuisanceAll[-7:-4][i][0,0], width = 0.25, edgecolor = 'w', color = colors[i][-1], alpha = 0.5, label = ['75%_stim', '50%_stim', '25%_stim'][i])
		# 	pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[i*2], colors[i][-1], alpha = alphas[i][-1], linewidth = lthns[i][-1], label = self.deconvolution_labels[decon_label_grouping[i][-1]])
		s.set_title('stimulus response beta' + roi + ' ' + mask_type)		
		# s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		# s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				# if i == self.which_stimulus_rewarded:
				l.set_linewidth(3.5)  # the legend line width
				# else:
					# l.set_linewidth(2.0)  # the legend line width
		for i in range(3): # plot stimulus responses
		
			s = fig.add_subplot(4,1,2+i)
			s.axhline(0, -10, 30, linewidth = 0.25)
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[i*2], colors[i][0], alpha = alphas[i][0], linewidth = lthns[i][0], label = self.deconvolution_labels[decon_label_grouping[i][0]])
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[i*2 + 1], colors[i][1], alpha = alphas[i][1], linewidth = lthns[i][1], label = self.deconvolution_labels[decon_label_grouping[i][1]])
			pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[-1], colors[-1][0], alpha = alphas[-1][0], linewidth = lthns[-1][0], label = self.deconvolution_labels[decon_label_grouping[-1][0]])
			s.set_title('deconvolution reward response' + roi + ' ' + mask_type)		
			s.set_xlabel('time [s]')
			s.set_ylabel('% signal change')
			s.set_xlim([interval[0]-1.5, interval[1]+1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for (i, l) in enumerate(leg.get_lines()):
					# if i == self.which_stimulus_rewarded:
					l.set_linewidth(3.5)  # the legend line width
					# else:
						# l.set_linewidth(2.0)  # the legend line width
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals), deco.deconvolvedTimeCoursesPerEventTypeNuisanceAll[-7:-1]] #, deco_per_run]
	
	def deconvolve(self, threshold = 3.5, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution'):
		results = []
		for roi in rois:
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = 'reward'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = 'reward'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos', signal_type = 'stim'))
			results.append(self.deconvolve_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg', signal_type = 'stim'))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_results'
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = r[0])
				# reward_h5file.removeNode(where = thisRunGroup, name = r[0]+'_per_run')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, r[0], r[-1], 'deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			# reward_h5file.createArray(thisRunGroup, r[0]+'_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	def deconvolve_plus_glm(self, threshold = 3.5, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'deconvolution'):
		results = []
		for roi in rois:
			results.append(self.deconvolve_plus_glm_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'pos'))
			results.append(self.deconvolve_plus_glm_roi(roi, threshold, mask_type = 'center_Z', analysis_type = analysis_type, mask_direction = 'neg'))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_glm_results'
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = r[0])
				reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_glm_betas')
				# reward_h5file.removeNode(where = thisRunGroup, name = r[0]+'_per_run')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, r[0], r[-2], 'deconvolution glm timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.createArray(thisRunGroup, r[0] + '_glm_betas', r[-1], 'glm deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			# reward_h5file.createArray(thisRunGroup, r[0]+'_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	
	
	def whole_brain_deconvolution(self, deco = True, average_intervals = [[3.5,12],[2,7]], to_surf = True, postFix = ['mcf', 'tf', 'psc']):
		"""
		whole_brain_deconvolution takes all nii files from the reward condition and deconvolves the separate event types
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		nii_file_shape = list(niiFile.data.shape)
		
		nr_reward_runs = len(self.conditionDict['reward'])
		
		# conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		# cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		time_signals = []
		interval = [0.0,16.0]
		
		self.deconvolution_labels = ['75%_yes', '75%_no', '75%_stim', '50%_yes', '50%_no', '50%_stim', '25%_yes', '25%_no', '25%_stim', 'blank_reward']
		decon_label_grouping = [[0,1,2],[3,4,5],[6,7,8],[-1]]
		# colors = [['b--','b','b'],['g--','g','g'],['r--','r','r'], ['k--']]
		# alphas = [[1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0]]
		# lthns = [[2.0, 2.0, 2.0],[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0]]
		if deco:
		
			event_data = []
			nii_data = np.zeros([nr_reward_runs] + nii_file_shape)
			nr_runs = 0
			blink_events = []
			for (j, r) in enumerate([self.runList[i] for i in self.conditionDict['reward']]):
				nii_data[j] = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).data
				this_run_events = []
				for cond in self.deconvolution_labels:
					this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
				this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
				this_blink_events[:,0] += nr_runs * run_duration
				blink_events.append(this_blink_events)
				this_run_events = np.array(this_run_events) + nr_runs * run_duration
				event_data.append(this_run_events)
				nr_runs += 1
				
			nii_data = nii_data.reshape((nr_reward_runs * nii_file_shape[0], -1))
			event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		
			deco = DeconvolutionOperator(inputObject = nii_data, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		
		if to_surf:
			try:
				os.system('rm -rf %s' % (os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'surf')))
				os.mkdir(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'surf'))
			except OSError:
				pass
		for (i, c) in enumerate(self.deconvolution_labels):
			if deco:
				outputdata = deco.deconvolvedTimeCoursesPerEventType[i]
				outputFile = NiftiImage(outputdata.reshape([outputdata.shape[0]]+nii_file_shape[1:]))
				outputFile.header = niiFile.header
				outputFile.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_' + c + '.nii.gz'))
			else:
				outputdata = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_' + c + '.nii.gz')).data
				# average over the interval [5,12] and [2,10] for reward and visual respectively. so, we'll just do [2,12]
			for (j, which_times) in enumerate(['reward', 'visual']):
				timepoints_for_averaging = (np.linspace(interval[0], interval[1], outputdata.shape[0]) < average_intervals[j][1]) * (np.linspace(interval[0], interval[1], outputdata.shape[0]) > average_intervals[j][0])
				meaned_data = outputdata[timepoints_for_averaging].mean(axis = 0)
				outputFile = NiftiImage(meaned_data.reshape(nii_file_shape[1:]))
				outputFile.header = niiFile.header
				ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_mean_' + c + '_' + which_times + '.nii.gz')
				outputFile.save(ofn)
			
				if to_surf:
					# vol to surf?
					# for (label, f) in zip(['left', 'right'], [left_file, right_file]):
					vsO = VolToSurfOperator(inputObject = ofn)
					sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
					vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute()
				
					for hemi in ['lh','rh']:
						ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
						ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
						ssO.execute()
		
		# now create the necessary difference images:
		# only possible if deco has already been run...
		for i in [0,3,6]:
			for (j, which_times) in enumerate(['reward', 'visual']):
				ipfs = [NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_mean_' + self.deconvolution_labels[i] + '_' + which_times + '.nii.gz')), NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_mean_' + self.deconvolution_labels[i+1] + '_' + which_times + '.nii.gz'))]
				diff_d = ipfs[0].data - ipfs[1].data
			
				ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward'), self.deconvolution_labels[i].split('_')[0] + '_reward_diff' + '_' + which_times + '.nii.gz')
				outputFile = NiftiImage(diff_d)
				outputFile.header = ipfs[0].header
				outputFile.save(ofn)
			
			
				if to_surf:
					vsO = VolToSurfOperator(inputObject = ofn)
					sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
					vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute()
				
					for hemi in ['lh','rh']:
						ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
						ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
						ssO.execute()
	
	def whole_brain_deconvolution_plus_glm(self, deco = True, average_intervals = [[2,7]], to_surf = True, postFix = ['mcf', 'tf', 'psc']):
		"""
		whole_brain_deconvolution takes all nii files from the reward condition and deconvolves the separate event types
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = round(niiFile.rtime*100)/100.0, niiFile.timepoints	# workaround for AV's strange reported TR
		run_duration = tr * nr_trs
		nii_file_shape = list(niiFile.data.shape)
		
		nr_reward_runs = len(self.conditionDict['reward'])
		
		# conds = ['blank_silence','blank_sound','visual_silence','visual_sound']
		# cond_labels = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward']
		
		time_signals = []
		interval = [0.0,16.0]
		
		self.deconvolution_labels = ['75%_yes', '75%_no', '75%_stim', '50%_yes', '50%_no', '50%_stim', '25%_yes', '25%_no', '25%_stim', '75%_delay', '50%_delay', '25%_delay', 'blank_reward' ]
		decon_label_grouping = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12]]
		# colors = [['b--','b','b'],['g--','g','g'],['r--','r','r'],['k--','k','k'], ['k--']]
		# alphas = [[1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0]]
		# lthns = [[2.0, 2.0, 2.0],[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0]]
		if deco:
		
			event_data = []
			nii_data = np.zeros([nr_reward_runs] + nii_file_shape)
			nr_runs = 0
			blink_events = []
			delay_events = []
			for (j, r) in enumerate([self.runList[i] for i in self.conditionDict['reward']]):
				nii_data[j] = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).data
				this_run_events = []
				for cond in self.deconvolution_labels:
					this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond])))	
					this_run_events[-1][:,0] = this_run_events[-1][:,0] + nr_runs * run_duration
				this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
				this_blink_events[:,0] += nr_runs * run_duration
				blink_events.append(this_blink_events)
				event_data.append(this_run_events)
				nr_runs += 1
				
			nii_data = nii_data.reshape((nr_reward_runs * nii_file_shape[0], -1))
			
			event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
			reward_event_data = [event_data[i][:,0] for (i,s) in enumerate(self.deconvolution_labels) if 'yes' in s or 'no' in s or 'reward' in s]
			stimulus_event_data = [event_data[i] for (i,s) in enumerate(self.deconvolution_labels) if 'stim' in s]
			delay_event_data = [event_data[i] for (i,s) in enumerate(self.deconvolution_labels) if 'delay' in s]
		
			interval = [0.0,16.0]
			# nuisance version?
			nuisance_design = Design(nr_reward_runs * nii_file_shape[0] * 2, tr/2.0 )
			nuisance_design.configure([list(np.vstack(blink_events))])
			# nuisance_design.configure([list(np.vstack(blink_events))], hrfType = 'doubleGamma', hrfParameters = {'a1': 6, 'a2': 12, 'b1': 0.9, 'b2': 0.9, 'c': 0.35})
		
			stimulus_design = Design(nr_reward_runs * nii_file_shape[0] * 2, tr/2.0 )
			stimulus_design.configure(stimulus_event_data)	# standard HRF for stimulus events
			# stimulus_design.configure(stimulus_event_data, hrfType = 'doubleGamma', hrfParameters = {'a1': 6, 'a2': 12, 'b1': 0.9, 'b2': 0.9, 'c': 0.35})	# standard HRF for stimulus events
		
			# non-standard reward HRF for delay events
			delay_design = Design(nr_reward_runs * nii_file_shape[0] * 2, tr/2.0 )
			# delay_design.configure(delay_event_data, hrfType = 'doubleGamma', hrfParameters = {'a1' : 22.32792026, 'a2' : 18.05752151, 'b1' : 0.30113662, 'b2' : 0.37294047, 'c' : 1.21845208})#, hrfType = 'double_gamma', hrfParameters = {'a1':-1.43231888, 'sh1':9.09749517, 'sc1':0.85289563, 'a2':0.14215637, 'sh2':103.37806306, 'sc2':0.11897103}) 22.32792026  18.05752151   0.30113662   0.37294047   1.21845208 {a1 = 22.32792026, a2 = 18.05752151, b1 = 0.30113662, b2 = 0.37294047, c = 1.21845208}
			delay_design.configure(delay_event_data, hrfType = 'singleGamma', hrfParameters = {'a':10.46713698,'b':0.65580082})
			# delay_design.configure(delay_event_data)
			nuisance_design_matrix = np.hstack((stimulus_design.designMatrix, delay_design.designMatrix, nuisance_design.designMatrix))
			
			deco = DeconvolutionOperator(inputObject = nii_data, eventObject = reward_event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
			deco.runWithConvolvedNuisanceVectors(nuisance_design_matrix)
			
		if to_surf:
			try:
				os.system('rm -rf %s' % (os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'surf')))
				os.mkdir(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'surf'))
			except OSError:
				pass
			
		# stimulus, delay and blink nuisance betas
		if deco:
			nuisance_names = {'stim_75':0, 'stim_50':1, 'stim_25':2, 'delay_75':3, 'delay_50':4, 'delay_25':5, 'blinks': 6}
			for i, nn in enumerate(nuisance_names.keys()):
				outputFile = NiftiImage(np.array(deco.deconvolvedTimeCoursesPerEventTypeNuisanceAll[i+deco.designMatrix.shape[1]]).reshape(nii_file_shape[1:]))
				outputFile.header = niiFile.header
				ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_glm_nuisance_beta_' + nn + '.nii.gz')
				outputFile.save(ofn)
				if to_surf:
					# vol to surf?
					# for (label, f) in zip(['left', 'right'], [left_file, right_file]):
					vsO = VolToSurfOperator(inputObject = ofn)
					sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
					vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute()
				
					for hemi in ['lh','rh']:
						ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
						ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
						ssO.execute()
		
		actual_deconvolved_events = [s for s in self.deconvolution_labels if 'yes' in s or 'no' in s or 'reward' in s]
		for (i, c) in enumerate(actual_deconvolved_events):
			if deco:
				outputdata = deco.deconvolvedTimeCoursesPerEventTypeNuisance[i]
				outputFile = NiftiImage(outputdata.reshape([outputdata.shape[0]]+nii_file_shape[1:]))
				outputFile.header = niiFile.header
				outputFile.save(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_glm_' + c + '.nii.gz'))
			else:
				outputdata = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_glm_' + c + '.nii.gz')).data
				# average over the interval [5,12] and [2,10] for reward and visual respectively. so, we'll just do [2,12]
			for (j, which_times) in enumerate(['reward']):
				timepoints_for_averaging = (np.linspace(interval[0], interval[1], outputdata.shape[0]) < average_intervals[j][1]) * (np.linspace(interval[0], interval[1], outputdata.shape[0]) > average_intervals[j][0])
				meaned_data = outputdata[timepoints_for_averaging].mean(axis = 0)
				outputFile = NiftiImage(meaned_data.reshape(nii_file_shape[1:]))
				outputFile.header = niiFile.header
				ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_glm_mean_' + c + '_' + which_times + '.nii.gz')
				outputFile.save(ofn)
			
				if to_surf:
					# vol to surf?
					# for (label, f) in zip(['left', 'right'], [left_file, right_file]):
					vsO = VolToSurfOperator(inputObject = ofn)
					sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
					vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
					vsO.execute()
				
					for hemi in ['lh','rh']:
						ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
						ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
						ssO.execute()
				
			
		
		# now create the necessary difference images:
		# only possible if deco has already been run...
		# not running this now, doing this differencing on the surface with freesurfer tools.
		# for i in [0,3,6]:
		# 	for (j, which_times) in enumerate(['reward', 'visual']):
		# 		ipfs = [NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_mean_' + self.deconvolution_labels[i] + '_' + which_times + '.nii.gz')), NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/reward'), 'reward_deconv_mean_' + self.deconvolution_labels[i+1] + '_' + which_times + '.nii.gz'))]
		# 		diff_d = ipfs[0].data - ipfs[1].data
		# 	
		# 		ofn = os.path.join(self.stageFolder(stage = 'processed/mri/reward'), self.deconvolution_labels[i].split('_')[0] + '_reward_diff' + '_' + which_times + '.nii.gz')
		# 		outputFile = NiftiImage(diff_d)
		# 		outputFile.header = ipfs[0].header
		# 		outputFile.save(ofn)
		# 	
		# 	
		# 		if to_surf:
		# 			vsO = VolToSurfOperator(inputObject = ofn)
		# 			sofn = os.path.join(os.path.split(ofn)[0], 'surf/', os.path.split(ofn)[1])
		# 			vsO.configure(frames = {'':0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = sofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
		# 			vsO.execute()
		# 		
		# 			for hemi in ['lh','rh']:
		# 				ssO = SurfToSurfOperator(vsO.outputFileName + '-' + hemi + '.mgh')
		# 				ssO.configure(fsSourceSubject = self.subject.standardFSID, fsTargetSubject = 'reward_AVG', hemi = hemi, outputFileName = os.path.join(os.path.split(ssO.inputFileName)[0],  'ss_' + os.path.split(ssO.inputFileName)[1]), insmooth = 5.0 )
		# 				ssO.execute()
	
	def create_feat_event_files_one_run(self, run, minimum_blink_duration = 0.01):
		"""
		creates feat analysis event files for reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		# get EL Data, do blink and timings irrespective of mapper or reward runs
		elO = EyelinkOperator(self.runFile(stage = 'processed/eye', run = run, extension = '.hdf5'))
		elO.import_parameters(run_name = 'bla')
		el_blinks = elO.get_EL_events_per_trial(run_name = 'bla', trial_ranges = [[0,255]], trial_phase_range = [0,4], data_type = 'blinks')[0] # just a single list instead of more than one....
		el_blinks = np.concatenate(el_blinks)
			
		# stimulus onsets are the trial phase indexed by 1
		# 'trial' onsets are indexed by 0
		experiment_start_time = (elO.timings['trial_phase_timestamps'][0,0,0] / 1000)
			
		# blinks
		blink_times = (el_blinks['start_timestamp'] / 1000) - experiment_start_time 
		blinks_during_experiment = blink_times > 0.0
		minimum_blink_duration_indices = (el_blinks['duration'] / 1000) > minimum_blink_duration
		blink_durations, blink_times = (el_blinks['duration'][blinks_during_experiment * minimum_blink_duration_indices] / 1000), blink_times[blinks_during_experiment * minimum_blink_duration_indices]
		
		try:
			os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']))
		except OSError:
			pass
		np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']), np.array([blink_times, blink_durations, np.ones((blink_times.shape[0]))]).T, fmt = '%3.2f', delimiter = '\t')
			
		# stimulus onset thingies
		stimulus_onset_times = (elO.timings['trial_phase_timestamps'][:,1,0] / 1000) - experiment_start_time
		reward_onset_times = (elO.timings['trial_phase_timestamps'][:,2,0] / 1000) - experiment_start_time + elO.parameter_data['reward_delay']
		
		reward_delay_periods = np.array([stimulus_onset_times + 1,  (reward_onset_times - stimulus_onset_times)]).T
		
		self.condition_labels = ['75%_yes', '75%_no', '75%_stim', '50%_yes', '50%_no', '50%_stim', '25%_yes', '25%_no', '25%_stim', 'blank_reward', '75%_delay', '50%_delay', '25%_delay']
		rewarded_trials = elO.parameter_data['sound'] == self.do_i_play_sound[1]
		orientation_trials = [elO.parameter_data['stim_orientation'] == ori for ori in self.orientations_in_order]
		
		# these trials were never rewarded
		no_stim_plus_trials = orientation_trials[-1]
		# high probability trials
		high_stim_plus_trials = orientation_trials[0] * rewarded_trials
		high_stim_min_trials = orientation_trials[0] * -rewarded_trials
		# medium probability trials
		med_stim_plus_trials = orientation_trials[1] * rewarded_trials
		med_stim_min_trials = orientation_trials[1] * -rewarded_trials
		# low probability trials
		low_stim_plus_trials = orientation_trials[2] * rewarded_trials
		low_stim_min_trials = orientation_trials[2] * -rewarded_trials
		
		run.condition_trials = [high_stim_plus_trials, high_stim_min_trials, orientation_trials[0], med_stim_plus_trials, med_stim_min_trials, orientation_trials[1], low_stim_plus_trials, low_stim_min_trials, orientation_trials[2], no_stim_plus_trials, orientation_trials[0], orientation_trials[1], orientation_trials[2]]
				
		for (cond, label) in zip(run.condition_trials, self.condition_labels):
			try:
				os.system('rm ' + self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]))
			except OSError:
				pass
			if label.split('_')[-1] == 'stim':
				times = stimulus_onset_times[cond]
				durations = np.ones((cond.sum()))
			elif label.split('_')[-1] == 'delay':
				times = reward_delay_periods[cond][:,0]
				durations = reward_delay_periods[cond][:,1]
			else:
				times = reward_onset_times[cond]
				durations = np.ones((cond.sum()))
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [label]), np.array([times, durations, np.ones((cond.sum()))]).T, fmt = '%3.2f', delimiter = '\t')
			
	
	def feat_reward_analysis(self, version = '', postFix = ['mcf'], run_feat = False):
		"""
		Runs feat analysis for all reward runs. 
		Takes run and minimum blink duration in seconds as arguments
		"""
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			self.create_feat_event_files_one_run(r)
			
			# not running feats just yet
			if run_feat:
				try:
					self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.fsf'))
				except OSError:
					pass
			
				# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
				# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
				if 'sara' in os.uname():
					thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_more_contrasts.fsf'
				else:
					thisFeatFile = '/Volumes/HDD/research/projects/reward/man/analysis/reward_more_contrasts.fsf'
				
				REDict = {
				'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
				'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
				'---BLINK_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']), 	
				'---BLANK_SILENCE_FILE---': self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blank_silence']), 	
				'---BLANK_SOUND_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blank_sound']), 
				'---VISUAL_SILENCE_FILE---':self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['visual_silence']), 	
				'---VISUAL_SOUND_FILE---': 	self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['visual_sound']), 
				}
				featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.fsf')
				featOp = FEATOperator(inputObject = thisFeatFile)
				# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
				if r == [self.runList[i] for i in self.conditionDict['reward']][-1]:
					featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
				else:
					featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
				self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
				# run feat
				featOp.execute()
	
	def import_stats_from_initial_session(self, example_func_to_highres_file, original_stat_folder, nr_stat_files = 4, stat_file_names = ['cope', 'tstat', 'pe', 'zstat']):
		"""
		"""
		# concatenate older session reg to newer session
		cfO = ConcatFlirtOperator(example_func_to_highres_file)
		cfO.configure(secondInputFile = os.path.join(self.stageFolder('processed/mri/reg/feat'), 'highres2example_func.mat'), 
					outputFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_older_stat_masks'], extension = '.mat' ))
		cfO.execute()
		
		for stat_name in stat_file_names:
			for i in np.arange(nr_stat_files)+1:
				# apply the transform
				flO = FlirtOperator(inputObject = os.path.join(original_stat_folder, stat_name+str(i)+'.nii.gz'), referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf']))
				flO.configureApply(self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_older_stat_masks'], extension = '.mat' ), 
										outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat'), stat_name+str(i)+'.nii.gz') )
				flO.execute()
		
	def create_glm_design_matrix_with_reward_convolution_and_nuisances_for_run(self, run, postFix = ['mcf','tf'], remove = False):
		"""
		This function takes a run, opens its nifti file and runs a glm on it, that incorporates standard HRF responses for visual and trial structure events,
		and negative BOLD HRF responses for reward and non-reward events.
		"""
		if remove:
			# prepare the save location
			try:
				ExecCommandLine('rm -rf %s' % (self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm'])))
			except OSError:
				pass
		try:
			os.mkdir(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']))
		except OSError:
			pass
		self.logger.debug('running glm analysis on run %s' + self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']))
		
		# input file opening
		niiFileName = self.runFile(stage = 'processed/mri', run = run, postFix = postFix)
		niiFile = NiftiImage(niiFileName)
		
		# visual and task timing responses:
		# trial_times = [np.vstack([np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [pf])) for pf in ['%i%%_yes'%perc, '%i%%_no'%perc]]) for perc in [75, 50, 25]]
		stim_times = [np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [pf])) for pf in ['75%_stim', '50%_stim', '25%_stim']]
		blink_times = np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks']))
		
		visual_task_names = ['75%_stim', '50%_stim', '25%_stim', 'blinks']
		# visual_task_names = ['75_tt', '50_tt', '25_tt', '75%_stim', '50%_stim', '25%_stim', 'blinks']
		
		norm_design = Design(nrTimePoints = niiFile.timepoints, rtime = round(niiFile.rtime*100)/100.0)
		# for condition_event_data in trial_times:
		# 	# for i in range(len(condition_event_data)):
		# 	norm_design.addRegressor(condition_event_data)
		for condition_event_data in stim_times:
			# for i in range(len(condition_event_data)):
			norm_design.addRegressor(condition_event_data)
		norm_design.addRegressor(blink_times)
		norm_design.convolveWithHRF()	# standard HRF
		# add derivative regressors to this part of the final design matrix
		norm_design.designMatrix = np.hstack((norm_design.designMatrix, np.vstack((np.zeros((3)), np.diff(norm_design.designMatrix[:,:3], axis = 0)))))
		visual_task_names.extend([s+'_diff' for s in visual_task_names[:3]])
		
		# reward times
		reward_times = [np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [pf])) for pf in ['75%_yes', '50%_yes', '25%_yes', 'blank_reward']]
		no_reward_times = [np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = [pf])) for pf in ['75%_no', '50%_no', '25%_no']]
		
		reward_names = ['75%_yes', '50%_yes', '25%_yes', 'blank_reward', '75%_no', '50%_no', '25%_no']
		
		# design matrix for reward with negative bold convolution kernel
		reward_design = Design(nrTimePoints = niiFile.timepoints, rtime = round(niiFile.rtime*100)/100.0)
		for condition_event_data in reward_times:
			# for i in range(len(condition_event_data)):
			reward_design.addRegressor(condition_event_data)
		for condition_event_data in no_reward_times:
			# for i in range(len(condition_event_data)):
			reward_design.addRegressor(condition_event_data)
		# convolve with double-gamma hrf fitted from the first session's reward response.
		reward_design.convolveWithHRF(hrfType = 'double_gamma', hrfParameters = {'a1':-1.43231888, 'sh1':9.09749517, 'sc1':0.85289563, 'a2':0.14215637, 'sh2':103.37806306, 'sc2':0.11897103})
		
		# motion parameters are their own design matrix
		motion_pars = np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.par', postFix = ['mcf']))
		
		motion_names = [str(i) + '_motion' for i in range(motion_pars.shape[1])]
		
		full_design = np.hstack((norm_design.designMatrix, reward_design.designMatrix, motion_pars))
		full_design_names = np.concatenate([visual_task_names, reward_names, motion_names])
		fd_name_string = ', '.join(full_design_names)
		
		# plot and save the design used.
		f = pl.figure(figsize = (10,8))
		s = f.add_subplot(111)
		im = pl.imshow(full_design)
		im.set_interpolation('nearest')
		im.set_cmap('Greys')
		s.set_title(fd_name_string, fontsize = 5)
		pl.savefig(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'design.pdf'))
		np.savetxt(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'design.txt'), full_design, fmt = '%3.2f', delimiter = '\t')
		run.design_file = os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'design.mat')
		design_file_header = """/NumWaves\t%d\n/NumPoints\t%d\n/PPheights\t\t%s\n\n/Matrix\n""" % (full_design.shape[1], full_design.shape[0], '\t'.join(['1' for i in range(full_design.shape[0])]))
		f = open( run.design_file, 'w')
		f.write(design_file_header)
		f.close()
		f = open( run.design_file, 'a')
		np.savetxt( f, full_design, delimiter = '\t', fmt = '%3.2f' ) #, header = design_file_header
		f.close()
		
		self.full_design_names = full_design_names
	
	def run_glm_from_design_matrix_nipy(self, run, postFix = ['mcf','tf'], design_matrix = None, design_matrix_file = None):
		"""
		Takes a designmatrix file and uses it to run a glm in nipy
		"""
		
		if design_matrix == None:
			if design_matrix_file == None:
				self.logger.error('no valid input for run_glm_from_design_matrix_nipy for run %s' %(repr(run)))
			else:
				design_matrix = np.loadtxt(design_matrix_file)
		
		niiFileName = self.runFile(stage = 'processed/mri', run = run, postFix = postFix)
		niiFile = NiftiImage(niiFileName)
		
		# GLM!
		my_glm = nipy.labs.glm.glm.glm()
		fit_data = (niiFile.data - niiFile.data.mean(axis = 0)).reshape((niiFile.timepoints,-1)).astype(np.float32)	# demean first
		glm = my_glm.fit(fit_data, design_matrix, model="ar1") # , method="kalman"
		
		niiFile.close()
		
		stat_matrix = []
		zscore_matrix = []
		beta_matrix = []
		# contrasts for each of the regressors separately
		for i in range(full_design.shape[-1]):
			this_contrast = np.zeros(full_design.shape[-1])
			this_contrast[i] = 1.0
			stat_matrix.append(my_glm.contrast(this_contrast).stat())
			zscore_matrix.append(my_glm.contrast(this_contrast).zscore())
			beta_matrix.append(my_glm.beta)
		
		# interesting contrast, the difference between reward and no reward per probability.
		diff_yes_no = np.zeros((3, full_design.shape[-1]))
		full_design_names = list(full_design_names)
		full_design_names.extend( ['75%_rew_diff', '50%_rew_diff', '25%_rew_diff'] )
		print full_design_names
		for (i, perc) in enumerate([75, 50, 25]):
			 diff_yes_no[i, full_design_names.index(str(perc)+'%_yes')] = 1
			 diff_yes_no[i, full_design_names.index(str(perc)+'%_no')] = -1
			 stat_matrix.append(my_glm.contrast(diff_yes_no[i]).stat())
			 zscore_matrix.append(my_glm.contrast(diff_yes_no[i]).zscore())
		
		# prepare stat images for saving.
		stat_matrix = np.array(stat_matrix).reshape((np.concatenate(([-1], niiFile.data.shape[1:]))))
		zscore_matrix = np.array(zscore_matrix).reshape((np.concatenate(([-1], niiFile.data.shape[1:]))))
		beta_matrix = np.array(beta_matrix).reshape((np.concatenate(([-1], niiFile.data.shape[1:]))))
		
		# save separate files for all contrasts
		for (i, img) in enumerate(full_design_names):
			# save stat and zscore
			stat_nii = NiftiImage(stat_matrix[i].astype(np.float32))
			stat_nii.header = niiFile.header
			stat_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), img + '_stat.nii.gz'))
			z_nii = NiftiImage(zscore_matrix[i].astype(np.float32))
			z_nii.header = niiFile.header
			z_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), img + '_zscore.nii.gz'))
			if i < beta_matrix.shape[0]:
				beta_nii = NiftiImage(beta_matrix[i].astype(np.float32))
				beta_nii.header = niiFile.header
				beta_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), img + '_betas.nii.gz'))
		
		# all in one file:
		stat_nii = NiftiImage(stat_matrix.astype(np.float32))
		stat_nii.header = niiFile.header
		stat_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'stat.nii.gz'))
		z_nii = NiftiImage(zscore_matrix.astype(np.float32))
		z_nii.header = niiFile.header
		z_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'zscore.nii.gz'))
		beta_nii = NiftiImage(beta_matrix.astype(np.float32))
		beta_nii.header = niiFile.header
		beta_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'betas.nii.gz'))
	
	def create_glm_command_from_design_matrix_fsl(self, run, postFix = ['mcf','tf'], design_matrix_file = None, basic_film_command = 'film_gls -sa -epith 200 -output_pwdata -v -rn %s %s %s'):
		"""run_glm_from_design_matrix_nipy assumes the design_matrix_file is in a folder in which to put the results"""
		return basic_film_command % ( os.path.join(os.path.split(design_matrix_file)[0], 'stats'), self.runFile(stage = 'processed/mri', run = run, postFix = ['mcf','tf'], extension = ''), design_matrix_file )
	
	def glm_with_reward_convolution_and_nuisances(self, postFix = ['mcf','tf'], execute = True):
		"""docstring for fname"""
		self.logger.debug('running glm analysis on all runs')
		film_commands = []
		for r in self.conditionDict['reward']:
			self.create_glm_design_matrix_with_reward_convolution_and_nuisances_for_run(run = self.runList[r], postFix = postFix)
			film_commands.append(self.create_glm_command_from_design_matrix_fsl(run = self.runList[r], postFix = postFix, design_matrix_file = os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[r], extension = '', postFix = ['mcf','glm']), 'design.mat')))
			self.logger.debug('running glm with command ' + film_commands[-1])
		
		if execute:
			# parallel implementation
			ppservers = ()
			job_server = pp.Server(ncpus = 8, ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = [job_server.submit(ExecCommandLine,(fgls,),(),('subprocess','tempfile',)) for fgls in film_commands]
			for fgls in ppResults:
				fgls()
			
			job_server.print_stats()
	
	def contrast_list(self):
		contrasts = [
			['75_diff', {'75%_yes':1, '75%_no': -1}],	# difference between reward and no reward
			['50_diff', {'50%_yes':1, '50%_no': -1}],	# difference between reward and no reward
			['25_diff', {'25%_yes':1, '25%_no': -1}],	# difference between reward and no reward
			['75_stim', {'75%_stim':1 }],				# stimulus contrast
			['50_stim', {'50%_stim':1 }],				# stimulus contrast
			['25_stim', {'25%_stim':1 }],				# stimulus contrast
			['75_yes', {'75%_yes':1 }],					# reward contrast
			['50_yes', {'50%_yes':1 }],					# reward contrast
			['25_yes', {'25%_yes':1 }],					# reward contrast
		]
		return contrasts
	
	def create_contrasts_per_run(self, run, postFix = ['mcf', 'tf']):
		"""calculate_contrasts_per_run takes the output from a film_gls run and uses contrast_mgr to calculate relevant contrasts"""
		self.logger.debug('running contrast analysis on run %s' % (repr(run)))
		# create regressor names:
		self.create_glm_design_matrix_with_reward_convolution_and_nuisances_for_run(run = run, postFix = postFix)
		
		contrast_file_name = os.path.splitext(run.design_file)[0] + '.con'
		
		contrasts = self.contrast_list()
		
		contrast_matrix = np.zeros((len(contrasts), len(self.full_design_names)))
		contrast_names = [c[0] for c in contrasts]
		contrast_file_string = ''
		for i in range(len(contrasts)):
			contrast_matrix[i,[list(self.full_design_names).index(k) for k in contrasts[i][1].keys()]] = contrasts[i][1].values()
			contrast_file_string += '/ContrastName' + str(i+1) + '\t' + '"' + contrasts[i][0] + '"\n'
		contrast_file_string += '/NumWaves\t' + str(int(len(self.full_design_names))) + '\n'
		contrast_file_string += '/NumContrasts\t' + str(int(len(contrasts))) + '\n'
		contrast_file_string += '/PPheights\t\t' + '\t'.join(['1.0000' for i in range(len(contrasts))]) + '\n'
		contrast_file_string += '/RequiredEffect\t\t' + '\t'.join(['2.000' for i in range(len(contrasts))]) + '\n'
		contrast_file_string += '\n\Matrix\n'
		
		run.contrast_matrix = contrast_matrix
		run.contrasts = contrasts
		# f = open( contrast_file_name, 'w')
		# f.write(contrast_file_string)
		# f.close()
		f = open( contrast_file_name, 'w')
		np.savetxt( f, contrast_matrix, delimiter = '\t', fmt = '%3.2f' ) #, header = design_file_header
		f.close()
		# f.open(os.path.splitext(run.design_file)[0] + '.con', 'w')
		# np.savetxt( f, contrast_matrix, delimiter = '\t', fmt = '%3.2f' )
		# f.close()
		
		
		run.contrast_file_name = contrast_file_name
		return 'contrast_mgr -d ' + os.path.join(os.path.split(run.design_file)[0], 'stats') + ' ' + contrast_file_name
	
	def calculate_contrasts_by_contrast_mgr(self, postFix = ['mcf','tf'], execute = True):
		"""docstring for calculate_contrasts"""
		self.logger.debug('running contrasts on all runs')
		con_commands = []
		for r in self.conditionDict['reward']:
			con_commands.append(self.create_contrasts_per_run(run = self.runList[r], postFix = postFix))
			self.logger.debug('running contrast with command ' + con_commands[-1])
			
		if execute:
			# parallel implementation
			ppservers = ()
			job_server = pp.Server(ncpus = 8, ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = [job_server.submit(ExecCommandLine,(fcon,),(),('subprocess','tempfile',)) for fcon in con_commands]
			for fcon in ppResults:
				fcon()
			
			job_server.print_stats()
	
	def calculate_contrasts_by_numpy(self, postFix = ['mcf','tf']):
		self.logger.debug('running contrasts on all runs')
		con_commands = []
		for r in self.conditionDict['reward']:
			con_commands.append(self.create_contrasts_per_run(run = self.runList[r], postFix = postFix))
			self.logger.debug('running contrast with command ' + con_commands[-1])
			ss = NiftiImage(os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[r], extension = '', postFix = ['mcf','glm']), 'stats', 'sigmasquareds.nii.gz'))
			# beta_data = np.array([NiftiImage(os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[r], extension = '', postFix = ['mcf','glm']), 'stats', 'pe' + str(r) + '.nii.gz')).data for r in range(run.contrast_matrix.shape[1])])
			ts = []
			for c in self.runList[r].contrast_matrix:
				which_regressors = (np.arange(c.shape[0])+1)[c!=0]
				nii_files = []
				for regr in which_regressors:
					nii_files.append(os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[r], extension = '', postFix = ['mcf','glm']), 'stats', 'pe' + str(regr) + '.nii.gz'))
				ts.append(np.nan_to_num(np.array([(NiftiImage(nii_files[i]).data * c) for (i, c) in enumerate(c[c!=0])]).sum(axis = 0))) # /ss.data
			contrast_output_nii = NiftiImage(np.array(ts, dtype = np.float32))
			contrast_output_nii.header = ss.header
			contrast_output_nii.save(os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[r], extension = '', postFix = ['mcf','glm']), 'stats', 'contrasts.nii.gz'))
			f = open(os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[r], extension = '', postFix = ['mcf','glm']), 'stats', 'contrasts.txt'), 'w')
			f.write(str(self.runList[r].contrasts))
			f.close()
	
	def mask_stats_to_hdf(self, run_type = 'reward', postFix = ['mcf'], secondary_addition = False):
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
		if os.path.isfile(self.hdf5_filename) and not secondary_addition:
			os.system('rm ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
			self.logger.info('deleting and starting table file ' + self.hdf5_filename)
		elif os.path.isfile(self.hdf5_filename) and secondary_addition:
			h5file = openFile(self.hdf5_filename, mode = "r+", title = run_type + " file")
			self.logger.info('adding to table file ' + self.hdf5_filename)
		elif not os.path.isfile(self.hdf5_filename):
			self.logger.info('creating table file ' + self.hdf5_filename)
			h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		
		if not secondary_addition:
			# create design matrix names by creating a design matrix for the first reward run
			self.create_glm_design_matrix_with_reward_convolution_and_nuisances_for_run(run = self.runList[self.conditionDict[run_type][0]])
		
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
			
				# add parameters, eye data and the like 
				eye_h5file = openFile(self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'), mode = "r")
				eyeGroup = eye_h5file.getNode(where = '/', name = 'bla', classname='Group')
				eyeGroup._f_copyChildren(thisRunGroup) 
				eye_h5file.close()
				"""
				Now, take different stat masks based on the run_type
				"""
				stat_files = {}
			
				# general info we want in all hdf files
				stat_files.update({
									# 'residuals': os.path.join(this_orientation_feat, 'stats', 'res4d.nii.gz'),
									'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf', 'psc']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									'hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf']), 
									# os.path.join(this_orientation_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
									# for these final two, we need to pre-setup the retinotopic mapping data
									'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
									'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz'),
								
									'center_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat1.nii.gz'),
									'center_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat1.nii.gz'),
									'center_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope1.nii.gz'),
									'center_pe': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'pe1.nii.gz'),
								
									'surround_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat2.nii.gz'),
									'surround_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat2.nii.gz'),
									'surround_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope2.nii.gz'),
									'surround_pe': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'pe3.nii.gz'),
								
									'center>surround_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat3.nii.gz'),
									'center>surround_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat3.nii.gz'),
									'center>surround_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope3.nii.gz'),
								
									'surround>center_T': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'tstat4.nii.gz'),
									'surround>center_Z': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'zstat4.nii.gz'),
									'surround>center_cope': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'cope4.nii.gz'),
									
				})
				# now we're going to add the results of film_gls' approximation.
				for (i, name) in enumerate(self.full_design_names):
					stat_files.update({name: os.path.join(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'glm'], extension = ''), 'stats', 'pe' + str(i+1) + '.nii.gz')})
				for name in ['prewhitened_data','res4d','sigmasquareds','contrasts']:
					stat_files.update({name: os.path.join(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'glm'], extension = ''), 'stats', name + '.nii.gz')})
			
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
						# if sf == 'contrasts':
						# 	h5file.createArray('/', roi_name + '_' + this_run_group_name + '_' + 'contrasts', these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
			
		else:	# secondary additions...
			# varFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/reward/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
			# dualFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/reward/stats_older_sessions/dual' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
			# firstFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/reward/stats_older_sessions/exp1' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
			
			# var file names
			allFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/reward/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
			# dual file names
			allFileNames.extend(subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/reward/stats_older_sessions/dual/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1])
			# first file names
			allFileNames.extend(subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/reward/stats_older_sessions/exp_1/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1])
			
			nifti_files_dict = dict(zip(['_'.join(fn.split('/')[-2:]) for fn in allFileNames] , [NiftiImage(fn) for fn in allFileNames]))
			
			this_run_group_name = 'deconv_results'
			# h5file.removeNode(where = '/', name = this_run_group_name, recursive=1)
			try:
				thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('deconvolution results file already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'deconvolution results from different sessions')
			
			# add var, dual and first stat files to different folders in there...
			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'deconvolution results for roi ' + roi_name)
				
				for (i, sf) in enumerate(nifti_files_dict.keys()):
					try:
						h5file.removeNode(where = thisRunGroup, name = sf[:-7].replace('%',''))
					except NoSuchNodeError:
						pass
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					imO = ImageMaskingOperator( inputObject = nifti_files_dict[sf], maskObject = roi, thresholds = [0.0] )
					these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
					h5file.createArray(thisRunGroup, sf[:-7].replace('%',''), these_roi_data.astype(np.float32), roi_name + ' data from ' + nifti_files_dict[sf].filename)
			
		h5file.close()
	
	def compare_glm_stats(self, areas = ['V1', 'V2', 'V3', 'V3AB', 'V4'], threshold = 3.5, mask_direction = 'pos', mask_type = 'center_Z'):
		"""docstring for compare_glm_stats"""
		
		contrasts = self.contrast_list()
		contrast_types = ['reward-no_reward', 'stimulus', 'reward']
		h5file = openFile(os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]), 'reward.hdf5'))
		
		mapper_data = []
		raw_mapper_data = []
		for area in areas:
			raw_mapper_data.append(self.roi_data_from_hdf(h5file, run = self.runList[self.conditionDict['reward'][0]], roi_wildcard = area, data_type = mask_type, postFix = ['mcf']))
			# thresholding of mapping data stat values
			if mask_direction == 'pos':
				mapping_mask = raw_mapper_data[-1][:,0] > threshold
			else:
				mapping_mask = raw_mapper_data[-1][:,0] < threshold
			mapper_data.append(mapping_mask)
		
		contrast_data = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			contrast_data.append([])
			for (i, area) in enumerate(areas):
				contrast_data[-1].append(self.roi_data_from_hdf(h5file, run = r, roi_wildcard = area, data_type = 'contrasts', postFix = ['mcf'])[mapper_data[i],:].mean(axis = 0))
		h5file.close()
		
		# # stats are only the last 9 vectors
		# contrast_data = np.array(contrast_data)
		
		# separate reward and stimulus values on raw data
		self.contrast_data = np.array(contrast_data).reshape((len(self.conditionDict['reward']),len(areas),len(contrast_types),3))
		self.mean_response = self.contrast_data.mean(axis = 0)
		self.std_response = self.contrast_data.std(axis = 0) / sqrt(len(self.conditionDict['reward']))
		# separate reward and stimulus values on raw data
		self.normed_contrast_data = np.array([[self.contrast_data[:,:,j,i] - self.contrast_data[:,:,j,1] for i in range(3)] for j in range(len(contrast_types))]).transpose((2,3,0,1))
		self.normed_mean_response = self.normed_contrast_data.mean(axis = 0)
		self.normed_std_response = self.normed_contrast_data.std(axis = 0) / sqrt(len(self.conditionDict['reward']))
		
		for (j, contrast_type) in enumerate(['reward-no_reward', 'stimulus', 'reward']):
			f = pl.figure(figsize = (9,6))
			s = f.add_subplot(211)
			s.set_title(contrast_type)
			barwidth = 0.25
			for i in range(3):
	 			pl.bar(np.arange(len(areas)) - 3.0*barwidth/2.0 + i * barwidth, self.mean_response[:,j,i], width = barwidth, edgecolor = 'k', color = ['r', 'g', 'b'][i], yerr = self.std_response[:,j,i], capsize = 0, alpha = 0.5, ecolor = 'k', label = contrasts[i+3*j][0])
			s.set_xlim(-0.5, len(areas)-0.5)
			s.set_xlabel('visual area')
			s.set_ylabel('beta weight [A.U.]')
			pl.xticks(np.arange(len(areas)), areas )
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.75)
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for l in leg.get_lines():
				    l.set_linewidth(3.5)  # the legend line width
		
			s = f.add_subplot(212)
			s.set_title(contrast_type + ' normalized')
			barwidth = 0.25
			for i in range(3):
	 			pl.bar(np.arange(len(areas)) - 3.0*barwidth/2.0 + i * barwidth, self.normed_mean_response[:,j,i], width = barwidth, edgecolor = 'k', color = ['r', 'g', 'b'][i], yerr = self.normed_std_response[:,j,i], capsize = 0, alpha = 0.5, ecolor = 'k', label = contrasts[i+3*j][0])
			s.set_xlim(-0.5, len(areas)-0.5)
			s.set_xlabel('visual area')
			s.set_ylabel('beta weight [A.U.], normalized to 50% reward probability')
			pl.xticks(np.arange(len(areas)), areas )
		
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'beta_barchart' + mask_type + '_' + mask_direction + '_' + contrast_type + '.pdf'))
	
	def run_glm_on_hdf5(self, data_type = 'hpf_data', analysis_type = 'from_design', post_fix_for_text_file = ['all_trials'], functionalPostFix = ['mcf'], which_conditions = ['reward','mapper'], contrast_matrix = []):
		# create design
		design = [np.loadtxt(os.path.join(self.runFile(stage = 'processed/mri', run = run, extension = '', postFix = ['mcf','glm']), 'design.txt')) for run in [self.runList[i] for i in self.conditionDict['reward']]]
		contrasts = np.loadtxt(os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]], extension = '', postFix = ['mcf','glm']), 'design.con'))
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		super(VisualRewardSession, self).run_glm_on_hdf5(run_list = [self.runList[i] for i in self.conditionDict['reward']], hdf5_file = reward_h5file, data_type = data_type, analysis_type = analysis_type, post_fix_for_text_file = post_fix_for_text_file, functionalPostFix = functionalPostFix, design = design, contrast_matrix = contrasts)
		reward_h5file.close()
	
	def import_deconvolution_responses_from_all_session(self, session_1, session_2):
		"""
		"""
		# concatenate older sessions reg to newer session
		cfO = ConcatFlirtOperator(os.path.join(session_1.stageFolder('processed/mri/reg/feat'), 'example_func2highres.mat'))
		cfO.configure(secondInputFile = os.path.join(self.stageFolder('processed/mri/reg/feat'), 'highres2example_func.mat'), 
					outputFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_session_1'], extension = '.mat' ))
		cfO.execute()
		cfO = ConcatFlirtOperator(os.path.join(session_2.stageFolder('processed/mri/reg/feat'), 'example_func2highres.mat'))
		cfO.configure(secondInputFile = os.path.join(self.stageFolder('processed/mri/reg/feat'), 'highres2example_func.mat'), 
					outputFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_session_2'], extension = '.mat' ))
		cfO.execute()
		
		session_1_files = subprocess.Popen('ls ' + session_1.stageFolder('processed/mri/reward/') + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		session_2_files = subprocess.Popen('ls ' + session_2.stageFolder('processed/mri/reward/') + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		
		try:
			os.mkdir(self.stageFolder('processed/mri/reward/stats_older_sessions'))
			os.mkdir(self.stageFolder('processed/mri/reward/stats_older_sessions/exp_1'))
			os.mkdir(self.stageFolder('processed/mri/reward/stats_older_sessions/dual'))
		except OSError:
			pass
		
		for stat_file in session_1_files:
			# apply the transform
			flO = FlirtOperator(inputObject = stat_file, referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf']))
			flO.configureApply(self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_session_1'], extension = '.mat' ), 
									outputFileName = os.path.join(self.stageFolder('processed/mri/reward/stats_older_sessions/exp_1'), os.path.split(stat_file)[1]) )
			flO.execute()
		for stat_file in session_2_files:
			# apply the transform
			flO = FlirtOperator(inputObject = stat_file, referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf']))
			flO.configureApply(self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'to_session_2'], extension = '.mat' ), 
									outputFileName = os.path.join(self.stageFolder('processed/mri/reward/stats_older_sessions/dual'), os.path.split(stat_file)[1]) )
			flO.execute()
	
	def pattern_comparisons(self):
		return [
		['dual'],
		['reward_reward_deconv_glm_mean_25_yes_reward','reward_reward_deconv_glm_mean_25_no_reward'],
		['reward_reward_deconv_glm_mean_50_yes_reward','reward_reward_deconv_glm_mean_50_no_reward'],
		['reward_reward_deconv_glm_mean_75_yes_reward','reward_reward_deconv_glm_mean_75_no_reward'],
		['reward_reward_deconv_glm_25_yes'], 
		['reward_reward_deconv_glm_25_no'], 
		['reward_reward_deconv_glm_50_yes'], 
		['reward_reward_deconv_glm_50_no'], 
		['reward_reward_deconv_glm_75_yes'], 
		['reward_reward_deconv_glm_75_no'],
		['reward_reward_deconv_glm_nuisance_beta_stim_25', 'reward_reward_deconv_glm_nuisance_beta_stim_50', 'reward_reward_deconv_glm_nuisance_beta_stim_75'],
		['reward_reward_deconv_glm_nuisance_beta_delay_25', 'reward_reward_deconv_glm_nuisance_beta_delay_50', 'reward_reward_deconv_glm_nuisance_beta_delay_75'],
		['reward_reward_deconv_mean_25_stim_visual', 'reward_reward_deconv_mean_50_stim_visual', 'reward_reward_deconv_mean_75_stim_visual'],
		
		]
	
	def compare_deconvolved_responses_across_sessions_per_roi(self, roi, template = 'exp1' ):
		"""docstring for compare_deconvolved_responses_per_session_per_roi"""
		self.files_for_comparisons = self.pattern_comparisons()
		return [self.compare_deconvolved_responses_across_sessions_per_roi_per_datatype(roi, template = template, comparison_list = fn ) for fn in self.files_for_comparisons]
		
	def compare_deconvolved_responses_across_sessions_per_roi_per_datatype(self, roi, template = 'exp1', comparison_list = [] ):
		h5file = self.hdf5_file('reward')
		this_run_group_name = 'deconv_results'
		# check the rois in the file
		roi_names = []
		for roi_name in h5file.iterNodes(where = '/' + this_run_group_name, classname = 'Group'):
			if len(roi_name._v_name.split('.')) > 1:
				hemi, area = roi_name._v_name.split('.')
				if roi == area:
					roi_names.append(roi_name._v_name)
		if len(roi_names) == 0:
			self.logger.info('No rois corresponding to ' + roi + ' in group ' + this_run_group_name)
			return None
		
		# check for the standard comparison things - standard is the average for the first experiment, but can also be dual
		if template == 'exp1':
			c_file_list = ['exp_1_fix_reward_diff_reward','exp_1_stimulus_reward_diff_reward']
		elif template == 'dual':
			c_file_list = ['dual_reward_deconv_mean_blank_rewarded_reward','dual_reward_deconv_mean_blank_silence_reward']
		ard_np = []
		for i, data_type in enumerate(c_file_list):
			ard = []
			for roi_name in roi_names:
				thisRoi = h5file.getNode(where = '/' + this_run_group_name, name = roi_name, classname='Group')
				ard.append( eval('thisRoi.' + data_type + '.read()') )
			ard_np.append(np.hstack(ard).T.squeeze())
		if template == 'exp1':
			comparison_array = (ard_np[0] + ard_np[1]) / 2.0
		elif template == 'dual':
			comparison_array = ard_np[0] - ard_np[1]
		
		all_roi_data = []
		if 'dual' in comparison_list:
			c_file_list = ['dual_reward_deconv_mean_blank_rewarded_reward','dual_reward_deconv_mean_blank_silence_reward']
			ard_np = []
			for i, data_type in enumerate(c_file_list):
				ard = []
				for roi_name in roi_names:
					thisRoi = h5file.getNode(where = '/' + this_run_group_name, name = roi_name, classname='Group')
					ard.append( eval('thisRoi.' + data_type + '.read()') )
				ard_np.append(np.hstack(ard).T.squeeze())
			all_roi_data.append(ard_np[0] - ard_np[1])
		# get other data:
		all_dt_dict = {}
		for i, data_type in enumerate(comparison_list):
			if data_type != 'dual':
				ard = []
				for roi_name in roi_names:
					thisRoi = h5file.getNode(where = '/' + this_run_group_name, name = roi_name, classname='Group')
					ard.append( eval('thisRoi.' + data_type + '.read()') )
				ard_np = np.hstack(ard).T.squeeze()
				all_roi_data.append(ard_np)
				all_dt_dict.update({data_type: i})
		# and close the file
		h5file.close()
		# shell()
		all_roi_data = np.array(all_roi_data).reshape((-1, comparison_array.shape[0]))
		
		# calculate spearman correlation and projection of patterns in the comparison array.
		return np.array([self.correlate_patterns(comparison_array, da) for da in all_roi_data])
		
	def compare_deconvolved_responses_across_sessions(self, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], force_run = False):
		# fig = pl.figure(figsize = (9, 16))
		# for i, roi in enumerate(rois):
		# 	s1 = fig.add_subplot(len(rois),1,1+i)
		# 	self.compare_deconvolved_responses_across_sessions_per_roi(s1 = s1, roi = roi)
		# pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'spatial_corr.pdf'))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		this_run_group_name = 'spatial_correlation_results'
		
		if force_run:
			results = [self.compare_deconvolved_responses_across_sessions_per_roi(roi) for roi in rois]
		
			reward_h5file = self.hdf5_file('reward', mode = 'r+')
			try:
				thisRunGroup = reward_h5file.removeNode(where = '/', name = this_run_group_name, recursive=1)
				self.logger.info('data file ' + self.hdf5_filename + ' contains ' + this_run_group_name)
			except NoSuchNodeError:
				pass
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'spatial correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
			data = []
			for i,r in enumerate(results):
				self.logger.info('Adding group ' + rois[i] + ' to this file')
				thisRunGroup = reward_h5file.createGroup("/" + this_run_group_name, rois[i], 'spatial correlation analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
				data.append({})
				for j, fc in enumerate(self.files_for_comparisons):
					# shell()
					reward_h5file.createArray(thisRunGroup, 'X'.join(fc), r[j], 'spatial correlation results for ' + 'X'.join(fc) + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
					data[-1].update( { 'X'.join(fc): r[j] } )
			reward_h5file.close()
		else:
			reward_h5file = self.hdf5_file('reward', mode = 'r+')
			data = []
			for r in rois:
				data.append({})
				for data_type in self.pattern_comparisons():
					# shell()
					thisRoi = reward_h5file.getNode(where = '/' + this_run_group_name, name = r, classname='Group')
					data[-1].update( { 'X'.join(data_type): eval('thisRoi.' + 'X'.join(data_type) + '.read()') } )
			reward_h5file.close()
		self.inter_experiment_correlations = data

	def deconvolve_with_correlation_roi(self, roi, threshold = 3.5, mask_type = 'center_Z', mask_direction = 'pos', template = 'exp1', analysis_type = 'correlation', correlation_function = 'spearman', interval = [0.0, 9.0], offsets = {'stim': 0.0, 'delay': -6.0, 'reward': -2.0}):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		tr, nr_trs = round(niiFile.rtime*100)/100.0, niiFile.timepoints
		run_duration = tr * nr_trs
		
		reward_h5file = self.hdf5_file('reward')
		
		self.deconvolution_labels = ['75%_yes', '75%_no', '75%_stim', '50%_yes', '50%_no', '50%_stim', '25%_yes', '25%_no', '25%_stim', '75%_delay', '50%_delay', '25%_delay', 'blank_reward' ]
		decon_label_grouping = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12]]
		colors = [['b--','b','b'],['g--','g','g'],['r--','r','r'],['k--','k','k'], ['k--']]
		alphas = [[1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0, 0.75, 1.0], [1.0]]
		lthns = [[2.0, 2.0, 2.0],[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0]]
		
		event_data = []
		roi_data = []
		nr_runs = 0
		blink_events = []
		delay_events = []
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, 'psc_hpf_data', postFix = ['mcf']))
			this_run_events = []
			for cond in self.deconvolution_labels:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond])))	
				this_run_events[-1][:,0] = this_run_events[-1][:,0] + nr_runs * run_duration
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['blinks']))
			this_blink_events[:,0] += nr_runs * run_duration
			blink_events.append(this_blink_events)
			event_data.append(this_run_events)
			nr_runs += 1
			
		# join event data for stimulus events per probability bin - the stimulus responses cannot be different for rewarded or unrewarded trials
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		reward_event_data = [event_data[i][:,0] for (i,s) in enumerate(self.deconvolution_labels) if 'yes' in s or 'no' in s or 'reward' in s]
		stimulus_event_data = [event_data[i] for (i,s) in enumerate(self.deconvolution_labels) if 'stim' in s]
		delay_event_data = [event_data[i] for (i,s) in enumerate(self.deconvolution_labels) if 'delay' in s]
		
		# check the rois in the file
		this_run_group_name = 'deconv_results'
		roi_names = []
		for roi_name in reward_h5file.iterNodes(where = '/' + this_run_group_name, classname = 'Group'):
			if len(roi_name._v_name.split('.')) > 1:
				hemi, area = roi_name._v_name.split('.')
				if roi == area:
					roi_names.append(roi_name._v_name)
		if len(roi_names) == 0:
			self.logger.info('No rois corresponding to ' + roi + ' in group ' + this_run_group_name)
			return None
		
		# check for the standard comparison things - standard is the average for the first experiment, but can also be dual
		if template == 'exp1':
			c_file_list = ['exp_1_fix_reward_diff_reward','exp_1_stimulus_reward_diff_reward']
		elif template == 'dual':
			c_file_list = ['dual_reward_deconv_mean_blank_rewarded_reward','dual_reward_deconv_mean_blank_silence_reward']
		ard_np = []
		for i, data_type in enumerate(c_file_list):
			ard = []
			for roi_name in roi_names:
				thisRoi = reward_h5file.getNode(where = '/' + this_run_group_name, name = roi_name, classname='Group')
				ard.append( eval('thisRoi.' + data_type + '.read()') )
			ard_np.append(np.hstack(ard).T.squeeze())
		if template == 'exp1':
			comparison_array = (ard_np[0] + ard_np[1]) / 2.0
		elif template == 'dual':
			comparison_array = ard_np[0] - ard_np[1]
		
		# mapping data
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[self.conditionDict['reward'][0]], roi, mask_type, postFix = ['mcf'])
		# and close the file
		reward_h5file.close()
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		# construct time series from signal amplitude in stim and non-stim regions, plus correlation
		if analysis_type == 'correlation':
			# first, just correlation
			if correlation_function == 'spearman':
				all_correlation_timeseries = np.array([self.correlate_patterns(comparison_array[mapping_mask], rd[mapping_mask]) for rd in roi_data.T])[:,0]
			elif correlation_function == 'projection':
				all_correlation_timeseries = np.array([self.correlate_patterns(comparison_array[mapping_mask], rd[mapping_mask]) for rd in roi_data.T])[:,2]
			timeseries = all_correlation_timeseries # roi_data[mapping_mask,:].mean(axis = 0)
		elif analysis_type == 'amplitude':
			timeseries = roi_data[mapping_mask,:].mean(axis = 0)
		
		sample_duration = tr/2.0
		
		# nuisance version?
		nuisance_design = Design(timeseries.shape[0] * 2, sample_duration )
		nuisance_design.configure([list(np.vstack(blink_events))])
		# nuisance_design.configure([list(np.vstack(blink_events))], hrfType = 'doubleGamma', hrfParameters = {'a1': 6, 'a2': 12, 'b1': 0.9, 'b2': 0.9, 'c': 0.35})
		
		stimulus_design = Design(timeseries.shape[0] * 2, sample_duration )
		stimulus_design.configure(stimulus_event_data)	# standard HRF for stimulus events
		# stimulus_design.configure(stimulus_event_data, hrfType = 'doubleGamma', hrfParameters = {'a1': 6, 'a2': 12, 'b1': 0.9, 'b2': 0.9, 'c': 0.35})	# standard HRF for stimulus events
		
		# non-standard reward HRF for delay events
		delay_design = Design(timeseries.shape[0] * 2, sample_duration )
		# delay_design.configure(delay_event_data, hrfType = 'doubleGamma', hrfParameters = {'a1' : 22.32792026, 'a2' : 18.05752151, 'b1' : 0.30113662, 'b2' : 0.37294047, 'c' : 1.21845208})#, hrfType = 'double_gamma', hrfParameters = {'a1':-1.43231888, 'sh1':9.09749517, 'sc1':0.85289563, 'a2':0.14215637, 'sh2':103.37806306, 'sc2':0.11897103}) 22.32792026  18.05752151   0.30113662   0.37294047   1.21845208 {a1 = 22.32792026, a2 = 18.05752151, b1 = 0.30113662, b2 = 0.37294047, c = 1.21845208}
		delay_design.configure(delay_event_data, hrfType = 'singleGamma', hrfParameters = {'a':10.46713698,'b':0.65580082})
		# delay_design.configure(delay_event_data)
		
		if analysis_type == 'correlation':
			nuisance_design_matrix = nuisance_design.designMatrix#np.hstack((stimulus_design.designMatrix, nuisance_design.designMatrix))
		elif analysis_type == 'amplitude':
			nuisance_design_matrix = np.hstack((stimulus_design.designMatrix, nuisance_design.designMatrix, delay_design.designMatrix)) # , delay_design.designMatrix
		
		time_signals = []
		
		stimulus_event_data = [r + offsets['stim'] for r in stimulus_event_data]
		
		# shell()
		stim_and_uncertainty_combined = False
		all_combined = False
		stim_as_full_regressors = True
		if not all_combined:
			if not stim_as_full_regressors:
				if stim_and_uncertainty_combined == False:
					# first run stimulus deconvolution with only blinks design matrix
					deco = DeconvolutionOperator(inputObject = timeseries, eventObject = stimulus_event_data[:], TR = tr, deconvolutionSampleDuration = sample_duration, deconvolutionInterval = interval[1], run = False)
					deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
					for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
						time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
		
					# run reward event data but only before the reward happens, as uncertainty events. This means concatenating the yes and no reward events to percentage classes
					# nuisance regressors are blinks and the like but also stimulus GLM
					uncertainty_event_data = [np.concatenate([reward_event_data[0+i],reward_event_data[1+i]]) + offsets['delay'] for i in [0,2,4]]
					deco = DeconvolutionOperator(inputObject = timeseries, eventObject = uncertainty_event_data, TR = tr, deconvolutionSampleDuration = sample_duration, deconvolutionInterval = interval[1], run = False)
					deco.runWithConvolvedNuisanceVectors(np.hstack((stimulus_design.designMatrix, nuisance_design.designMatrix)))
					for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
						time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
				elif stim_and_uncertainty_combined == True:
					# join events for both stimulus and reward events
					uncertainty_event_data = [np.concatenate([reward_event_data[0+i],reward_event_data[1+i]]) + offsets['delay'] for i in [0,2,4]]
					stimulus_event_data.extend(uncertainty_event_data)
					deco = DeconvolutionOperator(inputObject = timeseries, eventObject = stimulus_event_data, TR = tr, deconvolutionSampleDuration = sample_duration, deconvolutionInterval = interval[1], run = False)
					deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
					for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
						time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
				# now the response to reward, with the reward regressors in there. 
				# Now regressing out as much as possible, including the delay.
				# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = reward_event_data[:] + timeshift_for_deconvolution, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1] - timeshift_for_deconvolution, run = False)
				# deco.runWithConvolvedNuisanceVectors(nuisance_design_matrix)
				deco = DeconvolutionOperator(inputObject = timeseries, eventObject = [r + offsets['reward'] for r in reward_event_data], TR = tr, deconvolutionSampleDuration = sample_duration, deconvolutionInterval = interval[1], run = False)
				deco.runWithConvolvedNuisanceVectors(np.hstack((stimulus_design.designMatrix, nuisance_design.designMatrix, delay_design.designMatrix)))
				for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
					time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
			
			elif stim_as_full_regressors:
				uncertainty_event_data = [np.concatenate([reward_event_data[0+i],reward_event_data[1+i]]) + offsets['delay'] for i in [0,2,4]]
				# stimulus_event_data.extend(uncertainty_event_data)
				reward_event_data_separate = [r + offsets['reward'] for r in reward_event_data]
				uncertainty_event_data.extend(reward_event_data_separate)
				deco = DeconvolutionOperator(inputObject = timeseries, eventObject = uncertainty_event_data, TR = tr, deconvolutionSampleDuration = sample_duration, deconvolutionInterval = interval[1], run = False)
				deco.runWithConvolvedNuisanceVectors(np.hstack((nuisance_design.designMatrix, stimulus_design.designMatrix)))
				for i in range(3): # add stimulus betas to deconvolution results array as if they were time series - flat lines in subsequent plots, that is.
					time_signals.append(np.ones(deco.deconvolvedTimeCoursesPerEventTypeNuisance[0].shape) * deco.deconvolvedNuisanceBetas[i])
				for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
					time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
				
			
		if all_combined:
			# all in one big regression matrix
			uncertainty_event_data = [np.concatenate([reward_event_data[0+i],reward_event_data[1+i]]) + offsets['delay'] for i in [0,2,4]]
			stimulus_event_data.extend(uncertainty_event_data)
			reward_event_data_separate = [r + offsets['reward'] for r in reward_event_data]
			stimulus_event_data.extend(reward_event_data_separate)
			deco = DeconvolutionOperator(inputObject = timeseries, eventObject = stimulus_event_data, TR = tr, deconvolutionSampleDuration = sample_duration, deconvolutionInterval = interval[1], run = False)
			deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)
			for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
				time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i])
			
		
		# shell()
		
		time_signals = np.array(time_signals).squeeze()
		# shell()
		fig = pl.figure(figsize = (8, 16))
		s = fig.add_subplot(411)	# stim figure
		s.axhline(0, -10, 30, linewidth = 0.25)
		s.axvline(-offsets['stim'], -1, 2, linewidth = 0.25)
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[0], colors[0][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '75%_stim')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[1], colors[1][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '50%_stim')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[2], colors[2][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '25%_stim')
		s.set_title('deconvolution stimulus response' + roi + ' ' + mask_type)		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		simpleaxis(s)
		spine_shift(s)
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				# if i == self.which_stimulus_rewarded:
				l.set_linewidth(3.5)  # the legend line width
				# else:
					# l.set_linewidth(2.0)  # the legend line width
		
		s = fig.add_subplot(412)	# stim figure
		s.axhline(0, -10, 30, linewidth = 0.25)
		s.axvline(-offsets['delay'], -1, 2, linewidth = 0.25)
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[3], colors[0][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '75%_delay')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[4], colors[1][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '50%_delay')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[5], colors[2][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '25%_delay')
		s.set_title('deconvolution delay response' + roi + ' ' + mask_type)		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		simpleaxis(s)
		spine_shift(s)
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				# if i == self.which_stimulus_rewarded:
				l.set_linewidth(3.5)  # the legend line width
				# else:
					# l.set_linewidth(2.0)  # the legend line width
		
		s = fig.add_subplot(413)	# stim figure
		s.axhline(0, -10, 30, linewidth = 0.25)
		s.axvline(-offsets['reward'], -1, 2, linewidth = 0.25)
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[6], colors[0][0], alpha = alphas[0][0], linewidth = lthns[0][0], label = '75%_yes')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[8], colors[1][0], alpha = alphas[0][0], linewidth = lthns[0][0], label = '50%_yes')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[10], colors[2][0], alpha = alphas[0][0], linewidth = lthns[0][0], label = '25%_yes')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[7], colors[0][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '75%_no')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[9], colors[1][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '50%_no')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[11], colors[2][1], alpha = alphas[0][0], linewidth = lthns[0][0], label = '25%_no')
		s.set_title('deconvolution full reward response' + roi + ' ' + mask_type)		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		simpleaxis(s)
		spine_shift(s)
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				# if i == self.which_stimulus_rewarded:
				l.set_linewidth(3.5)  # the legend line width
				# else:
					# l.set_linewidth(2.0)  # the legend line width
		
		s = fig.add_subplot(414)	# stim figure
		s.axhline(0, -10, 30, linewidth = 0.25)
		s.axvline(-offsets['reward'], -1, 2, linewidth = 0.25)
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[6]-time_signals[7], colors[0][0], alpha = alphas[0][0], linewidth = lthns[0][0], label = '75%_diff')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[8]-time_signals[9], colors[1][0], alpha = alphas[0][0], linewidth = lthns[0][0], label = '50%_diff')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[10]-time_signals[11], colors[2][0], alpha = alphas[0][0], linewidth = lthns[0][0], label = '25%_diff')
		pl.plot(np.linspace(interval[0], interval[1], time_signals.shape[-1]), time_signals[12], 'k', alpha = alphas[0][0], linewidth = lthns[0][0], label = 'blank_reward')
		s.set_title('deconvolution reward difference response' + roi + ' ' + mask_type)		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		simpleaxis(s)
		spine_shift(s)
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		# self.rewarded_stimulus_run(self.runList[self.conditionDict['reward'][0]])
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for (i, l) in enumerate(leg.get_lines()):
				# if i == self.which_stimulus_rewarded:
				l.set_linewidth(3.5)  # the legend line width
				# else:
					# l.set_linewidth(2.0)  # the legend line width
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '_' + correlation_function + '.pdf'))
		# pl.show()
		
		return [roi + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals), deco.deconvolvedTimeCoursesPerEventTypeNuisanceAll[-7:-1]] #, deco_per_run]
	
	def deconvolve_pattern_plus_glm(self, threshold = 3.5, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], analysis_type = 'correlation', correlation_function = 'projection', interval = [0.0, 9.0], offsets = {'stim': 0.0, 'delay': -6.0, 'reward': -2.0}):
		results = []
		for roi in rois:
			results.append(self.deconvolve_with_correlation_roi(roi, threshold, mask_type = 'center_Z', mask_direction = 'pos', analysis_type = analysis_type, correlation_function = correlation_function, interval = interval, offsets = offsets))
			results.append(self.deconvolve_with_correlation_roi(roi, threshold, mask_type = 'center_Z', mask_direction = 'neg', analysis_type = analysis_type, correlation_function = correlation_function, interval = interval, offsets = offsets))
		# now construct hdf5 table for this whole mess - do the same for glm and pupil size responses
		reward_h5file = self.hdf5_file('reward', mode = 'r+')
		this_run_group_name = 'deconvolution_' + analysis_type + '_glm_results'
		try:
			thisRunGroup = reward_h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			# self.logger.info('data file ' + self.hdf5_filename + ' does not contain ' + this_run_group_name)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = reward_h5file.createGroup("/", this_run_group_name, 'deconvolution analysis conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") )
		
		for r in results:
			try:
				reward_h5file.removeNode(where = thisRunGroup, name = r[0])
				reward_h5file.removeNode(where = thisRunGroup, name = r[0] + '_glm_betas')
				# reward_h5file.removeNode(where = thisRunGroup, name = r[0]+'_per_run')
			except NoSuchNodeError:
				pass
			reward_h5file.createArray(thisRunGroup, r[0], r[-2], 'deconvolution pattern glm timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			reward_h5file.createArray(thisRunGroup, r[0] + '_glm_betas', r[-1], 'pattern glm deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
			# reward_h5file.createArray(thisRunGroup, r[0]+'_per_run', r[-1], 'per-run deconvolution timecourses results for ' + r[0] + 'conducted at ' + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
		reward_h5file.close()
	