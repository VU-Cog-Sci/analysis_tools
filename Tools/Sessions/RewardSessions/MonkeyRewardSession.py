#!/usr/bin/env python
# encoding: utf-8
"""
MonkeyRewardSession.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Session import *
from ...Operators.ArrayOperator import *
from ...Operators.EyeOperator import *
from ...Operators import ArrayOperator, EDFOperator, HDFEyeOperator, EyeSignalOperator
from ...other_scripts.circularTools import *
from ...other_scripts.plotting_tools import *
from ...other_scripts.savitzky_golay import *

from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle
from scipy.stats import *
import pandas as pd
import os
import seaborn as sn
from RL_model import RL_model
from TD_model import TD_model
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, report_errors, fit_report
from sklearn import linear_model
from scipy import io

ET_SAMPLE_RATE = 120.0

class MonkeyRewardSession(Session):

	"""
	Analyses for visual reward sessions
	"""

	def __init__(self, ID, date, project, subject, session_label='first', parallelize=True, loggingLevel=logging.DEBUG):
		super(MonkeyRewardSession, self).__init__(ID, date, project, subject,
												  session_label=session_label, parallelize=parallelize, loggingLevel=loggingLevel)

		self.TR = 2.0
		self.event_duration = 0.5

		self.hdf5_filename = os.path.join(
			self.stageFolder(stage='processed/mri'), 'reward', 'reward.hdf5')

	def setupFiles(self, rawBase, process_eyelink_file=True, date_format=None):
		super(MonkeyRewardSession, self).setupFiles(
			rawBase=rawBase, process_eyelink_file=process_eyelink_file, date_format=date_format)
		for r in self.runList:
			if hasattr(r, 'eye_file'):
				ExecCommandLine('cp ' + r.eye_file.replace('|', '\|') + ' ' + self.runFile(
					stage='processed/mri', run=r, extension='.dat', postFix=['eye']))
			if hasattr(r, 'order_data_file'):
				ExecCommandLine('cp ' + r.order_data_file.replace('|', '\|') +
								' ' + self.runFile(stage='processed/mri', run=r, extension='.mat'))
			if hasattr(r, 'motion_reg_file'):
				ExecCommandLine('cp ' + r.motion_reg_file.replace('|', '\|') + ' ' + self.runFile(
					stage='processed/mri', run=r, extension='.mat', postFix=['mcf']))

	def eye_pre_processing_run(self, run, hp = 0.05, lp = 8.0):
		"""eye_pre_processing_run preprocesses the pupil data.
		It then saves these pupil and xy data in hdf5 format in the monkey's h5_file"""
		eye_data = np.loadtxt(self.runFile(stage='processed/mri', run=run, extension='.dat', postFix=['eye']))
		pupil_signal = eye_data[:,2]
		# detect blinks based on zero acceleration
		x_acc = r_[1,1,np.diff(np.diff(eye_data[:,0]))]
		x_acc_ind = np.abs(x_acc) < 1e-6

		# detect them and put them in a file
		blink_edges = np.arange(x_acc_ind.shape[0])[r_[False, np.diff(x_acc_ind) == 1]]
		blink_edges = blink_edges[:2*int(floor(blink_edges.shape[0]/2.0))].reshape(-1,2)
		blink_times = np.array([blink_edges[:,0] / ET_SAMPLE_RATE , (blink_edges[:,1] - blink_edges[:,0]) / ET_SAMPLE_RATE, np.ones(blink_edges.shape[0])]).T

		np.savetxt(self.runFile(run=run, stage='processed/mri', extension='.txt', postFix=[
					   'blinks'], base='events/' + self.fileNameBaseString), blink_times, fmt='%3.2f', delimiter='\t')

		# now, for pupil signals
		# band-pass filtering of signal, high pass first and then low-pass
		# High pass:
		hp_cof_sample = hp / (pupil_signal.shape[0] / ET_SAMPLE_RATE / 2)
		bhp, ahp = sp.signal.butter(3, hp_cof_sample, btype='high')
		hp_filt_pupil = sp.signal.filtfilt(bhp, ahp, pupil_signal)
		# Low pass:
		lp_cof_sample = lp / (pupil_signal.shape[0] / ET_SAMPLE_RATE / 2)
		blp, alp = sp.signal.butter(3, lp_cof_sample)
		lp_filt_pupil = sp.signal.filtfilt(blp, alp, pupil_signal)
		# Band pass:
		bp_filt_pupil = sp.signal.filtfilt(blp, alp, hp_filt_pupil)

		# we may also add a baseline variable which contains the baseline 
		# by doing 3rd order savitzky-golay filtering, with a width of ~100 s
		# we can use this baseline signal for correlations of phasic and tonic pupil responses, for example
		window_size = ET_SAMPLE_RATE / (hp * 0.25)
		if np.mod(window_size,2)==0.0:
			window_size+=1
		baseline_filt_pupil = savitzky_golay(pupil_signal,window_size, 3)

		# now save all these data 
		with pd.get_store(self.hdf5_filename) as h5_file:
			h5_file.put("/%s/%s/%s"%('eye', self.fileNameBaseString + '_' + str(run.ID), 'xy_gaze'), pd.DataFrame(eye_data[:,[0,1]]))
			h5_file.put("/%s/%s/%s"%('eye', self.fileNameBaseString + '_' + str(run.ID), 'lp_pupil'), pd.Series(lp_filt_pupil))
			h5_file.put("/%s/%s/%s"%('eye', self.fileNameBaseString + '_' + str(run.ID), 'hp_pupil'), pd.Series(hp_filt_pupil))
			h5_file.put("/%s/%s/%s"%('eye', self.fileNameBaseString + '_' + str(run.ID), 'bp_pupil'), pd.Series(bp_filt_pupil))
			h5_file.put("/%s/%s/%s"%('eye', self.fileNameBaseString + '_' + str(run.ID), 'Z_pupil'), pd.Series((bp_filt_pupil-bp_filt_pupil.mean())/bp_filt_pupil.std()))
			h5_file.put("/%s/%s/%s"%('eye', self.fileNameBaseString + '_' + str(run.ID), 'baseline_pupil'), pd.Series(baseline_filt_pupil))

	def eye_pre_processing(self):
		for run in self.runList:
			self.eye_pre_processing_run(run)

	def pupil_deconvolution(self, interval = [-2,10.0], subsampling_factor = 10):
		"""pupil_deconvolution deconvolves pupil responses in a manner similar to the roi-based analysis"""
		# shell()
		# run duration based on the pupil data is one Tr minus the fmri data's duration
		run_duration = 219.0 * 2.0
		conds = ['fix_NoJuice', 'fix_Juice', 'visualNoJuice', 'visualJuice','blinks']

		event_data = []
		pupil_data = []
		nr_runs = 0
		with pd.get_store(self.hdf5_filename) as h5_file:
			for r in [self.runList[i] for i in self.conditionDict['reward']]:
				pupil_data.append(h5_file.get("/%s/%s/%s"%('eye', self.fileNameBaseString + '_' + str(r.ID), 'Z_pupil')))

				this_run_events = []
				for cond in conds:
					# toss out last trial of each type to make sure there are no
					# strange spill-over effects
					this_run_events.append(np.loadtxt(self.runFile(
						stage='processed/mri', run=r, extension='.txt', postFix=[cond], base='events/' + self.fileNameBaseString))[:-1, 0] + interval[0])
				this_run_events = np.array(
					this_run_events) + nr_runs * run_duration
				event_data.append(this_run_events)
				nr_runs += 1

		# per run data kept separately
		event_data_per_run = event_data

		pupil_data_per_run = pupil_data
		pupil_data_per_run = [sp.signal.decimate(pdpr, subsampling_factor) for pdpr in pupil_data_per_run]

		# concatenate data for further analysis
		event_data = [
			np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		event_data = [e[e>0] for e in event_data]

		timeseries = sp.signal.decimate(np.concatenate(pupil_data), subsampling_factor)

		fig = pl.figure(figsize=(6, 5))
		s = fig.add_subplot(211)
		s.axhline(
			0, interval[0] - 0.5, interval[1] + 0.5, linewidth=0.25, color='k')

		time_signals = []
		# nuisance version?
		deco = DeconvolutionOperator(inputObject=timeseries, eventObject=event_data, 
								TR=subsampling_factor/ET_SAMPLE_RATE, deconvolutionSampleDuration=subsampling_factor/ET_SAMPLE_RATE, 
								deconvolutionInterval=interval[1] - interval[0], run=True)
		residuals = deco.residuals()
		timepoints = np.linspace(interval[0], interval[1], deco.deconvolvedTimeCoursesPerEventType.shape[1])
		# shell()
		for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
			time_signals.append(
				deco.deconvolvedTimeCoursesPerEventType[i].squeeze())
			# shell()
			pl.plot(timepoints, np.array(
				deco.deconvolvedTimeCoursesPerEventType[i].squeeze()), ['b', 'b', 'g', 'g', 'k'][i], alpha=[0.5, 1.0, 0.5, 1.0, 0.25][i], label=conds[i])
		
		s.set_title('deconvolution pupil')
		deco_per_run = []
		for i, rd in enumerate(pupil_data_per_run):
			event_data_this_run = event_data_per_run[i] - i * run_duration
			deco = DeconvolutionOperator(inputObject=rd, eventObject=event_data_this_run, TR=subsampling_factor/ET_SAMPLE_RATE, deconvolutionSampleDuration=subsampling_factor/ET_SAMPLE_RATE, deconvolutionInterval=interval[1] - interval[0])
			deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
		deco_per_run = np.array(deco_per_run)
		mean_deco = deco_per_run.mean(axis=0)
		std_deco = 1.96 * deco_per_run.std(axis=0) / sqrt(len(pupil_data_per_run))
		for i in range(0, mean_deco.shape[0]):
			s.fill_between(timepoints, (np.array(time_signals[i]) + std_deco[i])[
						   0], (np.array(time_signals[i]) - std_deco[i])[0], color=['b', 'b', 'g', 'g', 'k'][i], alpha=0.3 * [0.5, 1.0, 0.5, 1.0, 1.0][i])

		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0] - 1.5, interval[1] + 1.5])
		leg = s.legend(fancybox=True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize('small')	# the legend text fontsize
			for l in leg.get_lines():
				l.set_linewidth(3.5)  # the legend line width

		s = fig.add_subplot(212)
		s.axhline(0, -10, 30, linewidth=0.25, color='k', alpha=0.5)

		for i in range(0, 4, 2):
			ts_diff = -(time_signals[i] - time_signals[i + 1])
			pl.plot(timepoints, np.array(ts_diff), ['b', 'b', 'g', 'g', 'k'][i], alpha=[1.0, 0.5, 1.0, 0.5][
					i], label=['fixation', 'visual stimulus'][i / 2])  # - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
			s.set_title('reward signal pupil')

		s.set_xlabel('time [s]')
		s.set_ylabel('$\Delta$ % signal change')
		s.set_xlim([interval[0] - 1.5, interval[1] + 1.5])
		leg = s.legend(fancybox=True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize('small')	# the legend text fontsize
			for l in leg.get_lines():
				l.set_linewidth(3.5)  # the legend line width

		pl.tight_layout()
		# pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage='processed/mri/figs/'),
								'pupil_deconvolution.pdf'))

		with pd.get_store(self.hdf5_filename) as h5_file:
			h5_file.put("/%s/%s"%('pupil_deconvolution', 'residuals'), pd.Series(np.squeeze(np.array(residuals))))
			h5_file.put("/%s/%s"%('pupil_deconvolution', 'time_points'), pd.Series(timepoints))
			h5_file.put("/%s/%s"%('pupil_deconvolution', 'dec_time_course'), pd.DataFrame(np.array(time_signals)))
			h5_file.put("/%s/%s"%('pupil_deconvolution', 'dec_time_course_per_run'), pd.Panel(np.squeeze(deco_per_run)))



	def definition_from_mat(self, run):
		"""
		definition_from_mat takes the .mat file for a given run (argument) and
		internalizes the names and onset_times in the run object, which is passed by reference.
		"""
		definition = io.loadmat(
			self.runFile(run=run, stage='processed/mri', extension='.mat'))

		run.cond_names = [str(definition['names'][0][i][0]) for i in range(4)]
		run.cond_onset_times = [
			definition['onsets'][0][i][0] * self.TR for i in range(4)]

		print ['%.2f' % r[0] for r in run.cond_onset_times]

	def definition_to_fsl_event_file(self, run):
		"""
		definition_to_fsl_event_file takes a given run (argument) and
		converts the onset_times to separate fsl event text files, named after the condition names.
		"""
		for name, onset_times in zip(run.cond_names, run.cond_onset_times):
			onset_array = np.array([onset_times, np.ones(
				onset_times.shape[0]) * self.event_duration, np.ones(onset_times.shape[0])]).T
			np.savetxt(self.runFile(run=run, stage='processed/mri', extension='.txt', postFix=[
					   name], base='events/' + self.fileNameBaseString), onset_array, fmt='%3.2f', delimiter='\t')

	def moco_mat_to_par(self, run):
		this_mat_file = self.runFile(
			stage='processed/mri', run=run, extension='.mat', postFix=['mcf'])
		mot_components = io.loadmat(this_mat_file)['motion_regressor']

		np.savetxt(self.runFile(stage='processed/mri', run=run, extension='.par',
								postFix=['mcf']), mot_components, fmt='%f', delimiter='\t')

		# shell()

	def all_timings_mocos(self):
		for run in self.runList:
			self.definition_from_mat(run)
			self.definition_to_fsl_event_file(run)
			self.moco_mat_to_par(run)

	def feat_run(self, run, feat_file='standard_moco_all.fsf', postFix=[], waitForExecute=True):
		try:
			self.logger.debug(
				'rm -rf ' + self.runFile(stage='processed/mri', run=run, postFix=postFix, extension='.feat'))
			os.system('rm -rf ' + self.runFile(stage='processed/mri',
											   run=run, postFix=postFix, extension='.feat'))
			os.system('rm -rf ' + self.runFile(stage='processed/mri',
											   run=run, postFix=postFix, extension='.fsf'))
		except OSError:
			pass

		# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
		# the order of the REs here, is the order in which they enter the feat.
		# this can be used as further reference for PEs and the like.
		thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward/monkey/fsf/' + \
			feat_file

		REDict = {
			'---NII_FILE---': 			self.runFile(stage='processed/mri', run=run, postFix=postFix),
			'---MOCO_FILE---': 			self.runFile(stage='processed/mri', run=run, extension='.par', postFix=['mcf', 'all']),
			'---fix_Juice_FILE---': 	self.runFile(stage='processed/mri', run=run, extension='.txt', postFix=['fix_Juice'], base='events/' + self.fileNameBaseString),
			'---fix_NoJuice_FILE---': 	self.runFile(stage='processed/mri', run=run, extension='.txt', postFix=['fix_NoJuice'], base='events/' + self.fileNameBaseString),
			'---visualJuice_FILE---': 	self.runFile(stage='processed/mri', run=run, extension='.txt', postFix=['visualJuice'], base='events/' + self.fileNameBaseString),
			'---visualNoJuice_FILE---':	self.runFile(stage='processed/mri', run=run, extension='.txt', postFix=['visualNoJuice'], base='events/' + self.fileNameBaseString),
		}

		featFileName = self.runFile(
			stage='processed/mri', run=run, extension='.fsf')
		featOp = FEATOperator(inputObject=thisFeatFile)
		# no need to wait for execute because we're running the mappers after
		# this sequence - need (more than) 8 processors for this, though.
		featOp.configure(
			REDict=REDict, featFileName=featFileName, waitForExecute=waitForExecute)
		self.logger.debug(
			'Running feat from ' + thisFeatFile + ' as ' + featFileName)
		# run feat
		featOp.execute()

	def eye_register(self):
		try:
			os.mkdir(self.stageFolder(stage='processed/mri/reg/feat'))
		except OSError:
			pass
		np.savetxt(os.path.join(self.stageFolder(
			stage='processed/mri/reg/feat'), 'example_func2standard.mat'), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(
			stage='processed/mri/reg/feat'), 'standard2example_func.mat'), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(
			stage='processed/mri/reg/feat'), 'highres2standard.mat'), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(
			stage='processed/mri/reg/feat'), 'standard2highres.mat'), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(
			stage='processed/mri/reg/feat'), 'example_func2highres.mat'), np.eye(4))
		np.savetxt(os.path.join(self.stageFolder(
			stage='processed/mri/reg/feat'), 'highres2example_func.mat'), np.eye(4))

		os.system('cp %s %s' % (self.runFile(stage='processed/mri/reg', base='forRegistration', postFix=[
				  self.ID]), os.path.join(self.stageFolder(stage='processed/mri/reg/feat'), 'example_func.nii.gz')))
		os.system('cp %s %s' % (self.runFile(stage='processed/mri/reg', base='forRegistration',
											 postFix=[self.ID]), os.path.join(self.stageFolder(stage='processed/mri/reg/feat'), 'highres.nii.gz')))
		os.system('cp %s %s' % (self.runFile(stage='processed/mri/reg', base='forRegistration',
											 postFix=[self.ID]), os.path.join(self.stageFolder(stage='processed/mri/reg/feat'), 'standard.nii.gz')))

	def register_feats(self):
		for run in self.runList:
			self.setupRegistrationForFeat(self.runFile(
				stage='processed/mri', run=run, postFix=[], extension='.feat'), wait_for_execute=False)

	def feat_all(self, nr_processes=16):
		for i, run in enumerate(self.runList):
			if (i % nr_processes) == (nr_processes - 1):
				self.feat_run(run, waitForExecute=True)
			else:
				self.feat_run(run, waitForExecute=False)

	def mask_stats_to_hdf(self, run_type='reward', postFix=[]):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""

		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder(stage='processed/mri/masks/anat/') +
											'*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])

		self.hdf5_filename = os.path.join(self.conditionFolder(
			stage='processed/mri', run=self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = open_file(
			self.hdf5_filename, mode="w", title=run_type + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = open_file(self.hdf5_filename, mode = "a", title = run_type + " file")

		this_feat = self.runFile(
			stage='processed/mri/reward', postFix=postFix, extension='.gfeat', base='feat+')
		# shell()

		main_stat_files = {
			'visual_T': os.path.join(this_feat, 'cope5.feat', 'stats', 'tstat1.nii.gz'),
			'visual_Z': os.path.join(this_feat, 'cope5.feat', 'stats', 'zstat1.nii.gz'),
			'visual_cope': os.path.join(this_feat, 'cope5.feat', 'stats', 'cope1.nii.gz'),

			'reward_T': os.path.join(this_feat, 'cope6.feat', 'stats', 'tstat1.nii.gz'),
			'reward_Z': os.path.join(this_feat, 'cope6.feat', 'stats', 'zstat1.nii.gz'),
			'reward_cope': os.path.join(this_feat, 'cope6.feat', 'stats', 'cope1.nii.gz'),

			'fix_Juice_T': os.path.join(this_feat, 'cope1.feat', 'stats', 'tstat1.nii.gz'),
			'fix_Juice_Z': os.path.join(this_feat, 'cope1.feat', 'stats', 'zstat1.nii.gz'),
			'fix_Juice_cope': os.path.join(this_feat, 'cope1.feat', 'stats', 'cope1.nii.gz'),

			'fix_NoJuice_T': os.path.join(this_feat, 'cope2.feat', 'stats', 'tstat1.nii.gz'),
			'fix_NoJuice_Z': os.path.join(this_feat, 'cope2.feat', 'stats', 'zstat1.nii.gz'),
			'fix_NoJuice_cope': os.path.join(this_feat, 'cope2.feat', 'stats', 'cope1.nii.gz'),

			'visualJuice_T': os.path.join(this_feat, 'cope3.feat', 'stats', 'tstat1.nii.gz'),
			'visualJuice_Z': os.path.join(this_feat, 'cope3.feat', 'stats', 'zstat1.nii.gz'),
			'visualJuice_cope': os.path.join(this_feat, 'cope3.feat', 'stats', 'cope1.nii.gz'),

			'visualNoJuice_T': os.path.join(this_feat, 'cope4.feat', 'stats', 'tstat1.nii.gz'),
			'visualNoJuice_Z': os.path.join(this_feat, 'cope4.feat', 'stats', 'zstat1.nii.gz'),
			'visualNoJuice_cope': os.path.join(this_feat, 'cope4.feat', 'stats', 'cope1.nii.gz'),

		}

		for r in [self.runList[i] for i in self.conditionDict[run_type]]:
			"""loop over runs, and try to open a group for this run's data"""
			this_run_group_name = os.path.split(
				self.runFile(stage='processed/mri', run=r, postFix=postFix))[1]
			try:
				thisRunGroup = h5file.get_node(
					where='/', name=this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage='processed/mri',
															 run=r, postFix=postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info(
					'Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(
					r.ID) + ' imported from ' + self.runFile(stage='processed/mri', run=r, postFix=postFix))

			"""
			Now, take different stat masks based on the run_type
			"""
			stat_files = main_stat_files.copy()

			# general info we want in all hdf files
			stat_files.update({
				# 'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
				# 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
				'raw_data': self.runFile(stage='processed/mri', run=r, postFix=[]),
				# 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
				'hpf_data': self.runFile(stage='processed/mri', run=r, postFix=['tf', 'Z']),
				# 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
				'Z_hpf_data': self.runFile(stage='processed/mri', run=r, postFix=['tf']),
				# 'hpf_data_fsl': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
			})

			stat_nii_files = [NiftiImage(stat_files[sf])
							  for sf in stat_files.keys()]

			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.get_node(
						where="/" + this_run_group_name, name=roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info(
						'Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(
						r.ID) + ' imported from ' + self.runFile(stage='processed/mri', run=r, postFix=postFix))

				for (i, sf) in enumerate(stat_files.keys()):
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					imO = ImageMaskingOperator(
						inputObject=stat_nii_files[i], maskObject=roi, thresholds=[0.0])
					these_roi_data = imO.applySingleMask(
						whichMask=0, maskThreshold=0.0, nrVoxels=False, maskFunction='__gt__', flat=True)
					h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(
						np.float32), roi_name + ' data from ' + stat_files[sf])

		h5file.close()

	def hdf5_file(self, run_type, mode='r'):
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info(
				'no table file ' + self.hdf5_filename + 'found for stat mask')
			return None
		else:
			# self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = open_file(
				self.hdf5_filename, mode=mode, title=run_type + " file")
		return h5file

	def deconvolve_roi(self, roi, threshold=3.5, mask_type='visualNoJuice_Z', mask_direction='pos', signal_type='mean', data_type='Z_hpf_data', subsampling_factor=5, interval=[-1.0, 8.0]
					   ):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file.
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same
		# length.
		niiFile = NiftiImage(self.runFile(
			stage='processed/mri', run=self.runList[self.conditionDict['reward'][0]]))
		nr_trs = niiFile.timepoints
		run_duration = self.TR * nr_trs

		conds = ['fix_NoJuice', 'fix_Juice', 'visualNoJuice', 'visualJuice', 'blinks']
		cond_labels = conds

		reward_h5file = self.hdf5_file('reward')

		moco_data = []
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(
				self.roi_data_from_hdf(reward_h5file, r, roi, data_type, postFix=[]))

			this_run_events = []
			for cond in conds:
				# toss out last trial of each type to make sure there are no
				# strange spill-over effects
				this_run_events.append(np.loadtxt(self.runFile(
					stage='processed/mri', run=r, extension='.txt', postFix=[cond], base='events/' + self.fileNameBaseString))[:-1, 0] + interval[0])
			this_run_events = np.array(
				this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)

			moco_data.append(np.loadtxt(self.runFile(
				stage='processed/mri', run=r, extension='.par', postFix=['mcf', 'all'])))

			nr_runs += 1

		demeaned_roi_data = []
		for rd in roi_data:
			# demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
			# flip sign for MION effects:
			demeaned_roi_data.append(-(rd.T - rd.mean(axis=1)).T)

		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data

		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [
			np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		moco_data = np.vstack(moco_data)

		# mapping data
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[
											  self.conditionDict['reward'][0]], roi, mask_type, postFix=[])
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		elif mask_direction == 'excl':
			mapping_mask = mapping_data[:,0] < threshold
		
		if mask_direction in ('pos','neg'):
			weight_vector = mapping_data[mapping_mask] / np.sum(mapping_data[mapping_mask])
			timeseries = roi_data[mapping_mask,:].T.dot(weight_vector)[:,0]
		elif mask_direction in ('all', 'excl'):
			timeseries = roi_data[mapping_mask,:].mean(axis = 0)

		reward_h5file.close()

		# timeseries = eval(
		# 	'roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		# timeseries = np.repeat(timeseries, subsampling_factor)

		fig = pl.figure(figsize=(6, 5))
		s = fig.add_subplot(211)
		s.axhline(
			0, interval[0] - 0.5, interval[1] + 0.5, linewidth=0.25, color='k')

		time_signals = []
		# nuisance version?
		deco = DeconvolutionOperator(inputObject=timeseries, eventObject=event_data[
									 :], TR=self.TR, deconvolutionSampleDuration=self.TR / float(subsampling_factor), deconvolutionInterval=interval[1] - interval[0], run=False)
		deco.runWithConvolvedNuisanceVectors(
			np.repeat(moco_data, subsampling_factor, axis=0))
		deco.residuals()
		residuals = deco.residuals
		# shell()
		for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
			time_signals.append(
				deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze())
			# shell()
			pl.plot(np.linspace(interval[0], interval[1], deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(
				deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze()), ['b', 'b', 'g', 'g','k'][i], alpha=[0.5, 1.0, 0.5, 1.0, 1.0][i], label=cond_labels[i])
		timepoints = np.linspace(interval[0], interval[1], deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1])
		# the following commented code doesn't factor in blinks as nuisances
		# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
		# 	pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
		# 	time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		s.set_title('deconvolution' + roi + ' ' + mask_type)
		deco_per_run = []
		for i, rd in enumerate(roi_data_per_run):
			event_data_this_run = event_data_per_run[i] - i * run_duration
			deco = DeconvolutionOperator(inputObject=rd[mapping_mask, :].mean(
				axis=0), eventObject=event_data_this_run, TR=self.TR, deconvolutionSampleDuration=self.TR / float(subsampling_factor), deconvolutionInterval=interval[1] - interval[0])
			deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
			# deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
			# deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix[i*nr_trs*2:(i+1)*nr_trs*2])
			# deco_per_run.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance)
		deco_per_run = np.array(deco_per_run)
		mean_deco = deco_per_run.mean(axis=0)
		std_deco = 1.96 * \
			deco_per_run.std(axis=0) / sqrt(len(roi_data_per_run))
		# shell()
		for i in range(0, mean_deco.shape[0]):
			s.fill_between(np.linspace(interval[0], interval[1], mean_deco.shape[1]), (np.array(time_signals[i]) + std_deco[i])[
						   0], (np.array(time_signals[i]) - std_deco[i])[0], color=['b', 'b', 'g', 'g','k'][i], alpha=0.3 * [0.5, 1.0, 0.5, 1.0, 0.5][i])

		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0] - 1.5, interval[1] + 1.5])
		leg = s.legend(fancybox=True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize('small')	# the legend text fontsize
			for l in leg.get_lines():
				l.set_linewidth(3.5)  # the legend line width

		s = fig.add_subplot(212)
		s.axhline(0, -10, 30, linewidth=0.25, color='k', alpha=0.5)

		for i in range(0, 4, 2):
			ts_diff = -(time_signals[i] - time_signals[i + 1])
			pl.plot(np.linspace(interval[0], interval[1], mean_deco.shape[1]), np.array(ts_diff), ['b', 'b', 'g', 'g'][i], alpha=[1.0, 0.5, 1.0, 0.5][
					i], label=['fixation', 'visual stimulus'][i / 2])  # - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
			s.set_title('reward signal ' + roi + ' ' + mask_type)

		s.set_xlabel('time [s]')
		s.set_ylabel('$\Delta$ % signal change')
		s.set_xlim([interval[0] - 1.5, interval[1] + 1.5])
		leg = s.legend(fancybox=True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize('small')	# the legend text fontsize
			for l in leg.get_lines():
				l.set_linewidth(3.5)  # the legend line width

		pl.tight_layout()
		# pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage='processed/mri/figs/'),
								roi + '_' + mask_type + '_' + mask_direction + '_' + data_type + '.pdf'))

		with pd.get_store(self.hdf5_filename) as h5_file:
			h5_file.put("/%s/%s"%('deconvolution' + roi + '_' + mask_type + '_' + mask_direction + '_' + data_type, 'residuals'), pd.Series(np.squeeze(np.array(residuals))))
			h5_file.put("/%s/%s"%('deconvolution' + roi + '_' + mask_type + '_' + mask_direction + '_' + data_type, 'time_points'), pd.Series(timepoints))
			h5_file.put("/%s/%s"%('deconvolution' + roi + '_' + mask_type + '_' + mask_direction + '_' + data_type, 'dec_time_course'), pd.DataFrame(np.array(time_signals)))
			h5_file.put("/%s/%s"%('deconvolution' + roi + '_' + mask_type + '_' + mask_direction + '_' + data_type, 'dec_time_course_per_run'), pd.Panel(np.squeeze(deco_per_run)))

		# shell()
		return [roi + '_' + mask_type + '_' + mask_direction, event_data, timeseries, np.array(time_signals), np.array(deco_per_run), residuals]

	def deconvolve(self, threshold=2.3, rois=['V1', 'V2', 'V3', 'V4', 'TE', 'TEO'], signal_type='mean', data_type='Z_hpf_data'):
		# neg_threshold = -1.0 * thres
		# neg_threshold = -neg_threshold
		# print threshold
		for roi in rois:
			self.deconvolve_roi(roi, threshold=threshold, mask_type='visual_Z',
											   mask_direction='pos', signal_type=signal_type, data_type=data_type)
			self.deconvolve_roi(roi, threshold=-threshold, mask_type='visual_Z',
											   mask_direction='neg', signal_type=signal_type, data_type=data_type)
			self.deconvolve_roi(roi, threshold=threshold, mask_type='visual_Z',
											   mask_direction='all', signal_type=signal_type, data_type=data_type)
			self.deconvolve_roi(roi, threshold=threshold, mask_type='visual_Z',
											   mask_direction='excl', signal_type=signal_type, data_type=data_type)


	def era_roi(self, roi, threshold=3.5, mask_type='visualNoJuice_Z', mask_direction='pos', signal_type='mean', data_type='Z_hpf_data', subsampling_factor=1, interval=[-5.0, 15.0]
				):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file.
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same
		# length.
		niiFile = NiftiImage(self.runFile(
			stage='processed/mri', run=self.runList[self.conditionDict['reward'][0]]))
		nr_trs = niiFile.timepoints
		run_duration = self.TR * nr_trs

		conds = ['fix_NoJuice', 'fix_Juice', 'visualNoJuice', 'visualJuice']
		cond_labels = conds

		reward_h5file = self.hdf5_file('reward')

		moco_data = []
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(
				self.roi_data_from_hdf(reward_h5file, r, roi, data_type, postFix=[]))

			this_run_events = []
			for cond in conds:
				# toss out last trial of each type to make sure there are no
				# strange spill-over effects
				this_run_events.append(np.loadtxt(self.runFile(
					stage='processed/mri', run=r, extension='.txt', postFix=[cond], base='events/' + self.fileNameBaseString))[:-1, 0])
			this_run_events = np.array(
				this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)

			moco_data.append(np.loadtxt(self.runFile(
				stage='processed/mri', run=r, extension='.par', postFix=['mcf', 'all'])))

			nr_runs += 1

		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append((rd.T - rd.mean(axis=1)).T)

		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data

		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [
			np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		moco_data = np.vstack(moco_data)

		# mapping data
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[
											  self.conditionDict['reward'][0]], roi, mask_type, postFix=[])
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:, 0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:, 0].shape, dtype=bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:, 0] < threshold

		reward_h5file.close()

		timeseries = eval(
			'roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		# timeseries = np.repeat(timeseries, subsampling_factor)

		fig = pl.figure(figsize=(6, 5))
		s = fig.add_subplot(211)
		s.axhline(0, -10, 30, linewidth=0.25)

		time_signals = []
		# nuisance version?
		for ed in event_data:
			era = EventRelatedAverageOperator(inputObject=np.array([timeseries]),
											  eventObject=ed, TR=self.TR,
											  interval=interval)
			time_signals.append(era.run())
		# deco.runWithConvolvedNuisanceVectors(np.repeat(moco_data, subsampling_factor, axis = 0))
		# deco.residuals()
		# residuals = deco.residuals
		# shell()
		for i in range(len(time_signals)):
			# shell()
			pl.plot(np.linspace(interval[0], interval[1], len(time_signals[i])), time_signals[i][
					:, 1], ['b', 'b', 'g', 'g'][i], alpha=[0.5, 1.0, 0.5, 1.0][i], label=cond_labels[i])

		# the following commented code doesn't factor in blinks as nuisances
		# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
		# 	pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
		# 	time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
		# s.set_title('deconvolution' + roi + ' ' + mask_type)
		# deco_per_run = []
		# for i, rd in enumerate(roi_data_per_run):
		# 	event_data_this_run = event_data_per_run[i] - i * run_duration
		# 	deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = self.TR, deconvolutionSampleDuration = self.TR/float(subsampling_factor), deconvolutionInterval = interval[1] - interval[0])
		# 	deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
		# 	# deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data_this_run, TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		# 	# deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix[i*nr_trs*2:(i+1)*nr_trs*2])
		# 	# deco_per_run.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance)
		# deco_per_run = np.array(deco_per_run)
		# mean_deco = deco_per_run.mean(axis = 0)
		# std_deco = 1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))
		# # shell()
		# for i in range(0, mean_deco.shape[0]):
		# 	s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), (np.array(time_signals[i]) + std_deco[i])[0], (np.array(time_signals[i]) - std_deco[i])[0], color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])

		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0] - 1.5, interval[1] + 1.5])
		leg = s.legend(fancybox=True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize('small')	# the legend text fontsize
			for l in leg.get_lines():
				l.set_linewidth(3.5)  # the legend line width

		s = fig.add_subplot(212)
		s.axhline(0, -10, 30, linewidth=0.25, color='k', alpha=0.5)

		for i in range(0, len(event_data), 2):
			ts_diff = -(time_signals[i][:, 1] - time_signals[i + 1][:, 1])
			pl.plot(np.linspace(interval[0], interval[1], len(time_signals[i])), np.array(ts_diff), ['b', 'b', 'g', 'g'][i], alpha=[
					1.0, 0.5, 1.0, 0.5][i], label=['fixation', 'visual stimulus'][i / 2])  # - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
			s.set_title('reward signal ' + roi + ' ' + mask_type)

		s.set_xlabel('time [s]')
		s.set_ylabel('$\Delta$ % signal change')
		s.set_xlim([interval[0] - 1.5, interval[1] + 1.5])
		leg = s.legend(fancybox=True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
				t.set_fontsize('small')	# the legend text fontsize
			for l in leg.get_lines():
				l.set_linewidth(3.5)  # the legend line width

		pl.tight_layout()
		# pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage='processed/mri/figs/'),
								roi + '_' + mask_type + '_' + mask_direction + '_' + data_type + '_era.pdf'))
		# shell()
		return [roi + '_' + mask_type + '_' + mask_direction, event_data, timeseries, np.array(time_signals)]

	def deconvolve_and_regress_trials_roi(self, roi, threshold=3.5, mask_type='visual_Z', mask_direction='pos', signal_type='mean', data_type='Z_hpf_data', subsampling_factor=3, interval=[0.0, 8.0]):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file.
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""

		niiFile = NiftiImage(self.runFile(
			stage='processed/mri', run=self.runList[self.conditionDict['reward'][0]]))
		nr_trs = niiFile.timepoints
		run_duration = self.TR * nr_trs

		conds = ['fix_NoJuice', 'fix_Juice', 'visualNoJuice', 'visualJuice']
		cond_labels = conds

		reward_h5file = self.hdf5_file('reward')

		moco_data = []
		event_data = []
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['reward']]:
			roi_data.append(
				self.roi_data_from_hdf(reward_h5file, r, roi, data_type, postFix=[]))

			this_run_events = []
			for cond in conds:
				# toss out last trial of each type to make sure there are no
				# strange spill-over effects
				this_run_events.append(np.loadtxt(self.runFile(
					stage='processed/mri', run=r, extension='.txt', postFix=[cond], base='events/' + self.fileNameBaseString))[:-1, 0] + interval[0])
			this_run_events = np.array(
				this_run_events) + nr_runs * run_duration
			event_data.append(this_run_events)

			moco_data.append(np.loadtxt(self.runFile(
				stage='processed/mri', run=r, extension='.par', postFix=['mcf', 'all'])))

			nr_runs += 1

		demeaned_roi_data = []
		for rd in roi_data:
			# demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
			# flip sign for MION effects:
			demeaned_roi_data.append(-(rd.T - rd.mean(axis=1)).T)

		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data

		roi_data = np.hstack(demeaned_roi_data)
		# event_data = np.hstack(event_data)
		event_data = [
			np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		moco_data = np.vstack(moco_data)

		# mapping data
		mapping_data = self.roi_data_from_hdf(reward_h5file, self.runList[
											  self.conditionDict['reward'][0]], roi, mask_type, postFix=[])
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		elif mask_direction == 'excl':
			mapping_mask = mapping_data[:,0] < threshold
		
		if mask_direction in ('pos','neg'):
			weight_vector = mapping_data[mapping_mask] / np.sum(mapping_data[mapping_mask])
			timeseries = roi_data[mapping_mask,:].T.dot(weight_vector)[:,0]
		elif mask_direction in ('all', 'excl'):
			timeseries = roi_data[mapping_mask,:].mean(axis = 0)


		reward_h5file.close()

		# timeseries = eval(
		# 	'roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')
		# timeseries = np.repeat(timeseries, subsampling_factor)

		fig = pl.figure(figsize=(6, 5))
		s = fig.add_subplot(211)
		s.axhline(
			0, interval[0] - 0.5, interval[1] + 0.5, linewidth=0.25, color='k')

		time_signals = []
		# nuisance version?
		deco = DeconvolutionOperator(inputObject=timeseries, eventObject=event_data[
									 :], TR=self.TR, deconvolutionSampleDuration=self.TR / float(subsampling_factor), deconvolutionInterval=interval[1] - interval[0], run=False)
		deco.runWithConvolvedNuisanceVectors(
			np.repeat(moco_data, subsampling_factor, axis=0))
		deco.residuals()
		residuals = deco.residuals
		# mean stimulus response:
		stim_resp = (((deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visualJuice')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('fix_Juice')]) + (
			deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visualNoJuice')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('fix_NoJuice')])) / 2.0).squeeze()
		# mean reward response:
		rew_resp = (((deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visualJuice')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('visualNoJuice')]) + (
			deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('fix_Juice')] - deco.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('fix_NoJuice')])) / 2.0).squeeze()

		if True:
			f = pl.figure(figsize=(6, 3))
			s = f.add_subplot(1, 1, 1)
			s.set_title(roi + ' ' + 'reward')
			pl.plot(np.linspace(interval[0], interval[1], stim_resp.shape[
					0]), stim_resp, 'k', label='stimulus')
			pl.plot(np.linspace(
				interval[0], interval[1], rew_resp.shape[0]), rew_resp, 'r', label='reward')
			s.set_xlabel('time [s]')
			s.set_ylabel('% signal change')
			# s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			leg = s.legend(fancybox=True)
			leg.get_frame().set_alpha(0.5)
			if leg:
				for t in leg.get_texts():
					t.set_fontsize('small')	# the legend text fontsize
				for l in leg.get_lines():
					l.set_linewidth(3.5)  # the legend line width
			# simpleaxis(s)
			# spine_shift(s)
			# s.set_ylim([-2,2])
			pl.savefig(os.path.join(self.stageFolder(stage='processed/mri/figs/'), roi +
									'_' + mask_type + '_' + mask_direction + '_template_deconvolutions.pdf'))

		rounded_event_array = np.array(
			[np.array(((ev / self.TR) * subsampling_factor), dtype=int) for ev in event_data])
		rounded_event_types = np.array(
			[np.ones(ev.shape) * i for i, ev in enumerate(event_data)])

		nr_trials = np.concatenate(rounded_event_array).shape[0]
		per_trial_design_matrix = np.zeros(
			(nr_trials * 2, residuals.shape[-1]))

		for i in range(nr_trials):
			# stim regressors:
			per_trial_design_matrix[i][
				np.concatenate(rounded_event_array)[i]] = 1.0
			per_trial_design_matrix[i] = np.correlate(
				per_trial_design_matrix[i], stim_resp, 'same')
			per_trial_design_matrix[i] -= per_trial_design_matrix[i].mean()
			# reward regressors:
			per_trial_design_matrix[
				i + nr_trials][np.concatenate(rounded_event_array)[i]] = 1.0
			per_trial_design_matrix[
				i + nr_trials] = np.correlate(per_trial_design_matrix[i + nr_trials], rew_resp, 'same')
			per_trial_design_matrix[
				i + nr_trials] -= per_trial_design_matrix[i + nr_trials].mean()

		# shell()

		# full_per_trial_design_matrix = np.mat(per_trial_design_matrix)
		# full_per_trial_betas = ((full_per_trial_design_matrix.T * full_per_trial_design_matrix).I * full_per_trial_design_matrix.T) * np.mat(deco.residuals).T
		# full_per_trial_betas_no_nuisance = np.array(full_per_trial_betas[:nr_trials*2].reshape(2,-1).T).squeeze()

		clf = linear_model.LinearRegression()
		clf.fit(per_trial_design_matrix.T, deco.residuals.T)
		full_per_trial_betas = clf.coef_
		full_per_trial_betas_no_nuisance = np.array(
			full_per_trial_betas[:nr_trials * 2].reshape(2, -1).T).squeeze()

		trial_info = pd.DataFrame({'stim_betas': full_per_trial_betas_no_nuisance[:, 0], 'reward_betas': full_per_trial_betas_no_nuisance[
								  :, 1], 'event_times': np.concatenate(rounded_event_array), 'event_types': np.concatenate(rounded_event_types)})

		reward_h5file.close()
		# hdf5_filename is now the reward file as that was opened last
		with pd.get_store(self.hdf5_filename) as h5_file:
			h5_file.put("/per_trial_glm_results/%s" % roi + '_' +
						mask_type + '_' + mask_direction + '_' + data_type, trial_info)

	def deconvolve_and_regress_trials(self, threshold=2.3, rois=['V1', 'V2', 'V3', 'V4', 'TE', 'TEO'], signal_type='mean', data_type='Z_hpf_data'):
		"""docstring for deconvolve_and_regress_trials_roi"""
		for roi in rois:
			self.deconvolve_and_regress_trials_roi(
				roi, threshold=threshold, mask_type='visual_Z', mask_direction='pos', signal_type=signal_type, data_type=data_type)
			self.deconvolve_and_regress_trials_roi(
				roi, threshold=-threshold, mask_type='visual_Z', mask_direction='neg', signal_type=signal_type, data_type=data_type)
			self.deconvolve_and_regress_trials_roi(
				roi, threshold=-threshold, mask_type='visual_Z', mask_direction='excl', signal_type=signal_type, data_type=data_type)
			self.deconvolve_and_regress_trials_roi(
				roi, threshold=-threshold, mask_type='visual_Z', mask_direction='all', signal_type=signal_type, data_type=data_type)


	def deconvolve_and_regress_trials_pupil(self, data_type = 'Z_pupil', interval=[0.0, 10.0], subsampling_factor = 20):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file.
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""


		run_duration = 219.0 * 2.0
		conds = ['fix_NoJuice', 'fix_Juice', 'visualNoJuice', 'visualJuice','blinks']
		cond_labels = conds

		event_data = []
		pupil_data = []
		nr_runs = 0
		with pd.get_store(self.hdf5_filename) as h5_file:
			for r in [self.runList[i] for i in self.conditionDict['reward']]:
				pupil_data.append(h5_file.get("/%s/%s/%s"%('eye', self.fileNameBaseString + '_' + str(r.ID), data_type)))

				this_run_events = []
				for cond in conds:
					# toss out last trial of each type to make sure there are no
					# strange spill-over effects
					this_run_events.append(np.loadtxt(self.runFile(
						stage='processed/mri', run=r, extension='.txt', postFix=[cond], base='events/' + self.fileNameBaseString))[:-1, 0] + interval[0])
				this_run_events = np.array(
					this_run_events) + nr_runs * run_duration
				event_data.append(this_run_events)
				nr_runs += 1

		# per run data kept separately
		event_data_per_run = event_data

		pupil_data_per_run = pupil_data
		pupil_data_per_run = [sp.signal.decimate(pdpr, subsampling_factor) for pdpr in pupil_data_per_run]

		# concatenate data for further analysis
		event_data = [
			np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		event_data = [e[e>0] for e in event_data]

		timeseries = sp.signal.decimate(np.concatenate(pupil_data), subsampling_factor)

		deco = DeconvolutionOperator(inputObject=timeseries, eventObject=event_data, 
						TR=subsampling_factor/ET_SAMPLE_RATE, deconvolutionSampleDuration=subsampling_factor/ET_SAMPLE_RATE, 
						deconvolutionInterval=interval[1] - interval[0], run=True)
		residuals = deco.residuals()
		residuals = np.array(residuals).squeeze()

		stim_response = (deco.deconvolvedTimeCoursesPerEventType[2:4].mean(axis = 0) - deco.deconvolvedTimeCoursesPerEventType[:2].mean(axis = 0)).squeeze()
		rew_response =  (deco.deconvolvedTimeCoursesPerEventType[[1,3]].mean(axis = 0) - deco.deconvolvedTimeCoursesPerEventType[[0,2]].mean(axis = 0)).squeeze()

		event_types = [np.ones(evd.shape[0]) * i for i, evd in enumerate(event_data[:4])]
		event_times = np.concatenate(event_data[:4])
		event_order = np.argsort(event_times)
		event_types_ordered = np.concatenate(event_types)[event_order]
		event_times_ordered = event_times[event_order]

		nr_trials = event_times.shape[0]

		per_trial_design_matrix = np.zeros((nr_trials * 2, residuals.shape[0]))

		for i, reg in enumerate(np.arange(nr_trials)):
			event_time_point = int(event_times_ordered[i]*subsampling_factor/ET_SAMPLE_RATE)
			per_trial_design_matrix[i, event_time_point] = 1.0
			per_trial_design_matrix[i] = np.correlate(
				per_trial_design_matrix[i], stim_response, 'same')
			per_trial_design_matrix[i] -= per_trial_design_matrix[i].mean()
			per_trial_design_matrix[i+nr_trials, event_time_point] = 1.0
			per_trial_design_matrix[i+nr_trials] = np.correlate(
				per_trial_design_matrix[i], rew_response, 'same')
			per_trial_design_matrix[i+nr_trials] -= per_trial_design_matrix[i+nr_trials].mean()

		self.logger.info('Starting single-trial GLM with stimulus and reward kernels.')

		clf = linear_model.LinearRegression()
		clf.fit(per_trial_design_matrix.T, residuals.T)
		full_per_trial_betas = clf.coef_
		full_per_trial_betas_no_nuisance = np.array(
			full_per_trial_betas[:nr_trials * 2].reshape(2, -1).T).squeeze()

		trial_info = pd.DataFrame({'stim_betas': full_per_trial_betas_no_nuisance[:, 0], 'reward_betas': full_per_trial_betas_no_nuisance[
								  :, 1], 'event_times': event_times_ordered * subsampling_factor/ET_SAMPLE_RATE, 'event_types': event_types_ordered})

		# reward_h5file.close()
		# hdf5_filename is now the reward file as that was opened last
		with pd.get_store(self.hdf5_filename) as h5_file:
			h5_file.put("/per_trial_glm_results/%s" % data_type, trial_info)

		self.logger.info('Saved single-trial GLM with stimulus and reward kernels to hdf5.')


	def per_trial_glm_history_regression_pupil(self, data_type = 'Z_pupil', which_betas = 'stim', which_trials = 'all'):

		conds = ['fix_NoJuice', 'fix_Juice', 'visualNoJuice', 'visualJuice','blinks']
		cond_labels = conds

		with pd.get_store(self.hdf5_filename) as h5_file:
			trial_info = h5_file.get("/per_trial_glm_results/%s" % data_type)

		event_types_ordered = trial_info['event_types']
		event_times_ordered = trial_info['event_times']

		fix_norewards = np.array(event_types_ordered) == 0
		fix_rewards = np.array(event_types_ordered) == 1
		stim_norewards = np.array(event_types_ordered) == 2
		stim_rewards = np.array(event_types_ordered) == 3
		reward_times = np.array(event_times_ordered[fix_rewards + stim_rewards])
		
		stimulus = stim_rewards + stim_norewards
		rewards = stim_rewards + fix_rewards

		# choose which data
		if which_betas == 'stim':
			betas = np.array(trial_info['stim_betas'])
		elif which_betas == 'reward':
			betas = np.array(trial_info['reward_betas'])
		betas = (betas-np.mean(betas))/np.std(betas)

		if which_trials == 'all':
			trial_selection = np.ones(betas.shape, dtype = bool)
		elif which_trials == 'all_reward':
			trial_selection = fix_rewards + stim_rewards
		elif which_trials == 'all_stims':
			trial_selection = stim_norewards + stim_rewards
		elif which_trials == 'all_fix':
			trial_selection = fix_norewards + fix_rewards
		elif which_trials == 'fix_norewards':
			trial_selection = fix_norewards
		elif which_trials == 'fix_rewards':
			trial_selection = fix_rewards
		elif which_trials == 'stim_norewards':
			trial_selection = stim_norewards
		elif which_trials == 'stim_rewards':
			trial_selection = stim_rewards
		else:
			trial_selection = np.ones(betas.shape, dtype = bool)

		N_BACK = 20
		res = np.zeros((6,N_BACK))

		sn.set(style="ticks")
		f = pl.figure(figsize = (9,5))
		s = f.add_subplot(1,1,1)
		for rew_i, rew in enumerate([fix_norewards, fix_rewards, stim_norewards, stim_rewards, stimulus, rewards ]):
			# dm = np.zeros((N_BACK-1, betas.shape[0]))
			# for tp in np.arange(1,N_BACK):
			# 	dm[tp,tp:] = rew[:-tp]
			dm = np.array([np.roll(rew, i) for i in np.arange(1,N_BACK + 1)])
			for x in range(1,dm.shape[0]):
				dm[x,:x] = 0
			clf = linear_model.LinearRegression()
			clf.fit(dm.T[trial_selection], betas[trial_selection])
			res[rew_i] = clf.coef_
			plot(np.arange(0,N_BACK), res[rew_i], ['b','b','g','g','k','r'][rew_i], alpha = [0.5,1.0,0.5,1.0,1.0,1.0][rew_i])
		# pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		s.set_xlim(xmin=0.5, xmax=N_BACK-0.5)
		pl.legend(cond_labels)
		simpleaxis(s)
		spine_shift(s)
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), which_betas + '_' + which_trials + '_per_trial_history_regression_pupil.pdf'))

		res_pd = pd.DataFrame(res.T, columns = ['fix_no_reward','fix_reward','stimulus_no_reward','stimulus_reward', 'stimulus', 'rewards'])

		with pd.get_store(self.hdf5_filename) as h5_file:
				h5_file.put("/per_trial_history_regression_pupil/%s_%s"%(which_betas, which_trials), res_pd)

	def per_trial_glm_history_regression_pupil_all(self):
		for which_beta in ['stim', 'reward']:
			for which_trials in ['all', 'all_reward', 'all_stims', 'all_fix', 'fix_norewards', 'fix_rewards', 'stim_norewards', 'stim_rewards']:
				self.per_trial_glm_history_regression_pupil(which_betas=which_beta, which_trials=which_trials)


	def per_trial_glm_history_regression_roi(self, roi='V1', mask_type='visual_Z', mask_direction='pos', which_betas='reward', which_trials='all'):
		"""docstring for trial_history_from_per_trial_glm_results"""
		# set up the right reward file
		with pd.get_store(self.hdf5_filename) as h5_file:
			trials = h5_file["/per_trial_glm_results/%s" % roi + '_' +
							 mask_type + '_' + mask_direction + '_' + 'Z_hpf_data']

		conds = ['fix_NoJuice', 'fix_Juice', 'visualNoJuice', 'visualJuice']
		cond_labels = conds

		# shell()
		time_order = np.argsort(trials.event_times)
		time_ordered_trials = trials.iloc[time_order]

		fix_norewards = np.array(time_ordered_trials.event_types) == 0
		fix_rewards = np.array(time_ordered_trials.event_types) == 1
		stim_norewards = np.array(time_ordered_trials.event_types) == 2
		stim_rewards = np.array(time_ordered_trials.event_types) == 3
		reward_times = np.array(
			time_ordered_trials[fix_rewards + stim_rewards].event_times)

		stimulus = stim_rewards + stim_norewards
		rewards = stim_rewards + fix_rewards

		# shell()

		# choose which data
		if which_betas == 'stim':
			betas = np.array(time_ordered_trials.stim_betas)
		elif which_betas == 'reward':
			betas = np.array(time_ordered_trials.reward_betas)
		betas = (betas - np.mean(betas)) / np.std(betas)

		if which_trials == 'all':
			trial_selection = np.ones(betas.shape, dtype=bool)
		elif which_trials == 'all_reward':
			trial_selection = fix_rewards + stim_rewards
		elif which_trials == 'all_stims':
			trial_selection = stim_norewards + stim_rewards
		elif which_trials == 'all_fix':
			trial_selection = fix_norewards + fix_rewards
		elif which_trials == 'fix_norewards':
			trial_selection = fix_norewards
		elif which_trials == 'fix_rewards':
			trial_selection = fix_rewards
		elif which_trials == 'stim_norewards':
			trial_selection = stim_norewards
		elif which_trials == 'stim_rewards':
			trial_selection = stim_rewards
		else:
			trial_selection = np.ones(betas.shape, dtype=bool)

		N_BACK = 20
		res = np.zeros((6, N_BACK))

		sn.set(style="ticks")
		f = pl.figure(figsize=(9, 5))
		s = f.add_subplot(1, 1, 1)
		for rew_i, rew in enumerate([fix_norewards, fix_rewards, stim_norewards, stim_rewards, stimulus, rewards]):
			# dm = np.zeros((N_BACK-1, betas.shape[0]))
			# for tp in np.arange(1,N_BACK):
			# 	dm[tp,tp:] = rew[:-tp]
			dm = np.array([np.roll(rew, i) for i in np.arange(1, N_BACK + 1)])
			for x in range(1, dm.shape[0]):
				dm[x, :x] = 0
			clf = linear_model.LinearRegression()
			clf.fit(dm.T[trial_selection], betas[trial_selection])
			res[rew_i] = clf.coef_
			plot(np.arange(0, N_BACK), res[rew_i], ['b', 'b', 'g', 'g', 'k', 'r'][
				 rew_i], alpha=[0.5, 1.0, 0.5, 1.0, 1.0, 1.0][rew_i])
		# pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color='k')
		s.set_xlim(xmin=0.5, xmax=N_BACK - 0.5)
		pl.legend(cond_labels)
		simpleaxis(s)
		spine_shift(s)
		pl.savefig(os.path.join(self.stageFolder(stage='processed/mri/figs/'), roi + '_' + mask_type + '_' +
								mask_direction + '_' + which_betas + '_' + which_trials + '_per_trial_glm_history_regression_roi.pdf'))

		res_pd = pd.DataFrame(res.T, columns=[
							  'fix_no_reward', 'fix_reward', 'stimulus_no_reward', 'stimulus_reward', 'stimulus', 'rewards'])

		with pd.get_store(self.hdf5_filename) as h5_file:
			h5_file.put("/per_trial_glm_history_regression/%s_%s_%s_%s" %
						(roi, mask_direction, which_betas, which_trials), res_pd)

		return res

	# 'V2', 'V3', 'V3AB', 'V4'
	def per_trial_glm_history_regression(self, rois=['V1', 'V2', 'V3', 'V4', 'TE', 'TEO'], mask_type='visual_Z'):
		"""docstring for trial_history_from_per_trial_glm_results_roi"""

		for roi in rois:
			results = {}
			for which_beta in ['stim', 'reward']:
				for which_trials in ['all', 'all_reward', 'all_stims', 'all_fix', 'fix_norewards', 'fix_rewards', 'stim_norewards', 'stim_rewards']:
					self.per_trial_glm_history_regression_roi(
						roi, mask_type=mask_type, mask_direction='pos', which_betas=which_beta, which_trials=which_trials)
					self.per_trial_glm_history_regression_roi(
						roi, mask_type=mask_type, mask_direction='neg', which_betas=which_beta, which_trials=which_trials)
					self.per_trial_glm_history_regression_roi(
						roi, mask_type=mask_type, mask_direction='excl', which_betas=which_beta, which_trials=which_trials)
					self.per_trial_glm_history_regression_roi(
						roi, mask_type=mask_type, mask_direction='all', which_betas=which_beta, which_trials=which_trials)

		
