#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from ..Sessions import * 
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from ..circularTools import *
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle

# taking the dicoms to nifti with MRIConvert, causes a whole directory tree to be made, containing .nii files. 
# to gzip and move out of their separate folders:
# for f in *; do echo $f; cd $f; gzip *; mv *.nii.gz ../ ; cd ..; done;

class OrientationDecisionSession(RetinotopicMappingSession):
	"""
	Class for orientation decision decoding sessions analysis.
	Forks from retinotopic mapping session primarily because of phase-encoded mapping runs. 
	Involves trial- and run-based support vector regression/decoding of orientation around =/- 45.
	"""
	def analyze_one_run_behavior(self, run, decision_type = 'binary', output_fsl_files = True):
		"""
		Takes a pickle file - the run's behavioral file (ending in .dat) - opens and analyzes it. 
		It saves an fsl-style text file with regressors, and adds parameters, events and response times to the session's hdf5 file.
		the orientation file is a .dat pickle file, that has to be opened in unicode mode.
		"""
		run_data = pickle.load(file(self.runFile(stage = 'processed/behavior', run = run, extension = '.dat'), 'U' ))
		
		trial_phase_transitions = [np.array(run_data['trial_phase_transition_array'][i]) for i in run_data['run_order']]
		run.trial_phase_transitions = np.concatenate([np.vstack([trial_phase_transitions[i].T, i * np.ones(trial_phase_transitions[i].shape[0])]).T for i in range(len(trial_phase_transitions))])
		run.trial_phase_transitions_dtype = np.dtype([('phase_number', '<f8'), ('trial_time', '<f8'), ('run_time', '<f8'), ('trial_number', '<f8')])
		
		run.per_trial_parameters = [run_data['trial_parameter_array'][i] for i in run_data['run_order']]
		run.parameter_dtype_dictionary = []
		for key in run.per_trial_parameters[0].keys():
			if key in ('confidence_range', 'orientation_range'):
				run.parameter_dtype_dictionary.append((key, np.float64, run.per_trial_parameters[0][key].shape))
			else:
				run.parameter_dtype_dictionary.append((key, np.float64))
		if 'confidence' not in [tp[0] for tp in run.parameter_dtype_dictionary]:
			run.parameter_dtype_dictionary.append(('confidence', np.float64))
		run.parameter_dtype_dictionary = np.dtype(run.parameter_dtype_dictionary)
		
		run.per_trial_events = [np.array(run_data['trial_event_array'][i], dtype = float) for i in run_data['run_order']]
		run.per_trial_events_full_array = np.concatenate([np.vstack([run.per_trial_events[i].T, i * np.ones(run.per_trial_events[i].shape[0])]).T for i in range(len(run.per_trial_events))])
		run.per_trial_noTR_events = [pte[pte[:,0] != 5.] for pte in run.per_trial_events]
		
		# separate in effector, and introduce regressor for only-left and only-right handed response trials
		left_hand_trials = [pte for pte in run.per_trial_noTR_events if ((pte[:,0]>5.0).sum() == 0) and (pte.shape[0] > 0)]
		right_hand_trials = [pte for pte in run.per_trial_noTR_events if ((pte[:,0]<5.0).sum() == 0) and (pte.shape[0] > 0)]
		# for all responses, instead of trials
		all_nonTR_events = np.concatenate(run.per_trial_noTR_events)
		run.left_hand_responses = all_nonTR_events[all_nonTR_events[:,0]<5]
		run.right_hand_responses = all_nonTR_events[all_nonTR_events[:,0]>5]
				
		separate_answers = []
		for (i, pte) in enumerate(run.per_trial_noTR_events):
			if pte.shape[0] == 0:
				self.logger.info('trial %i in run %i does not contain button presses, as per run.per_trial_events: %s'%(i, run.ID, str(run.per_trial_events[i])))
				continue
			elif pte.shape[0] > 2:
				self.logger.info('trial %i in run %i contains more button presses than required, as per run.per_trial_events: %s'%(i, run.ID, str(pte)))
				if pte[0,0] not in (4,9) and pte[1,0] in (4,9): # drop first, and this is actually the only thing to do ALERT - only for binary decisions
					pte = pte[1:]
				else:
					pte = np.array([pte[0], pte[2]])
			elif pte.shape[0] == 1:
				if pte[0,0] in (3,2,7,8): # only a confidence rating was given ALERT - only for binary decisions
					pte = np.array([[0,pte[0,1] - 2.0, pte[0,2] - 2.0], pte[0]])
				else:	# we assume, foolishly, that the answer given was a task-answer and not a confidence -answer.
					pte = np.array([pte[0], [0,pte[0,1] + 2.0, pte[0,2] + 2.0]])
			elif pte.shape[0] == 2:
				if pte[0,0] not in (4,9): # wrong answer for decision.
					self.logger.info('trial %i in run %i does not contain approptiate answer button presses, as per run.per_trial_events: %s'%(i, run.ID, str(pte)))
					pte[0,0] = 0
			# if all is good,
			# normalise the response against the instructed response direction
			pte[0,0] = ([4,0,9].index(pte[0,0])-1) * run.per_trial_parameters[i]['cw_ccw_response_direction']
			pte[0,1] = run.per_trial_parameters[i]['orientation'] - run.per_trial_parameters[i]['reference_orientation']
			# find the actual confidence level similarly
			pte[1,0] = run.per_trial_parameters[i]['confidence_range'][[2,3,4,9,8,7].index(pte[1,0])]
			
			separate_answers.append(pte) # [:,[0,2]]
		
		run.distilled_answers = np.array(separate_answers)
		
		run.stimulus_regressor_times = run.trial_phase_transitions[run.trial_phase_transitions[:,0] == 1, 2]
		stim_regs = np.ones((run.stimulus_regressor_times.shape[0], 3))
		stim_regs[:,0] = run.stimulus_regressor_times; stim_regs[:,1] = 0.5;
		if output_fsl_files:
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['stimulus']), stim_regs, fmt = '%3.2f', delimiter = '\t')
		
		separate_answers = np.array(separate_answers)
		run.answer_regressor_times = separate_answers[:,0,2]
		ans_regs =  np.ones((run.answer_regressor_times.shape[0], 3))
		ans_regs[:,0] = run.answer_regressor_times; ans_regs[:,1] = 0.5;
		if output_fsl_files:
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['answer']), ans_regs, fmt = '%3.2f', delimiter = '\t')
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['all']), np.vstack([stim_regs, ans_regs]), fmt = '%3.2f', delimiter = '\t')
		
		stimpos = np.array([ptp['stim_position'] for ptp in run.per_trial_parameters])
		
		# separate regressors for the different stimulus positions
		run.left_stim_times = run.stimulus_regressor_times[stimpos < 0]
		left_stims_regs = np.ones((run.left_stim_times.shape[0], 3))
		left_stims_regs[:,0] = run.left_stim_times; left_stims_regs[:,1] = 0.5;
		if output_fsl_files:
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['stimulus_left']), left_stims_regs, fmt = '%3.2f', delimiter = '\t')
		
		run.right_stim_times = run.stimulus_regressor_times[stimpos > 0]
		right_stims_regs = np.ones((run.right_stim_times.shape[0], 3))
		right_stims_regs[:,0] = run.right_stim_times; right_stims_regs[:,1] = 0.5;
		if output_fsl_files:
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['stimulus_right']), right_stims_regs, fmt = '%3.2f', delimiter = '\t')
		
		# create separate regressors for the two handed response extremities
		lh_regs =  np.ones((run.left_hand_responses.shape[0], 3))
		lh_regs[:,0] = run.left_hand_responses[:,2]; lh_regs[:,1] = 0.5; # np.array([lha[1,2] - lha[0,2] for lha in left_hand_trials]);
		if output_fsl_files:
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['LH_answer']), lh_regs, fmt = '%3.2f', delimiter = '\t')
		
		rh_regs =  np.ones((run.right_hand_responses.shape[0], 3))
		rh_regs[:,0] = run.right_hand_responses[:,2]; rh_regs[:,1] = 0.5; # np.array([rha[1,2] - rha[0,2] for rha in right_hand_trials]);
		if output_fsl_files:
			np.savetxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['RH_answer']), rh_regs, fmt = '%3.2f', delimiter = '\t')
		
		
		# keys for confidence are [3,2,1,6,7,8] (left-right order of fingers), keys for answers are [1,6] (the index fingers of both hands)
		
		
	def analyze_runs_for_regressors(self, postFix = ['mcf','tf'], per_run_feat = True, apply_reg = True, all_feat = True, run_behavior = True):
		"""docstring for analyze_runs_for_regressors"""
		for r in [self.runList[i] for i in self.conditionDict['decision']]:
			if run_behavior:
				self.analyze_one_run_behavior(r)
			
			if per_run_feat:
				try:
					self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.fsf'))
				except OSError:
					pass
			
				# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
				# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
				if 'sara' in os.uname()[1]:
					thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_more_contrasts.fsf'
				else:
					thisFeatFile = '/Volumes/HDD/research/projects/decision_fMRI/analysis/fsl/first_decision_glm_2.fsf'
				
				REDict = {
				'---NII_FILE---': 			self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
				'---NR_TRS---':				str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
				'---STIM_LEFT_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['stimulus_left']), 	
				'---STIM_RIGHT_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['stimulus_right']), 	
				'---ANSWER_LEFT_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['LH_answer']), 	
				'---ANSWER_RIGHT_FILE---': 		self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['RH_answer']), 	
				}
				featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.fsf')
				featOp = FEATOperator(inputObject = thisFeatFile)
				# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
				if r == [self.runList[i] for i in self.conditionDict['decision']][-1]:
					featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
				else:
					featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
				self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
				# run feat
				featOp.execute()
		
		if apply_reg:
			# register all runs to the standard
			for r in [self.runList[i] for i in self.conditionDict['decision']]:
				self.setupRegistrationForFeat(self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
		
		# combine the runs in a gfeat
		if all_feat:
			try:
				self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
				os.system('rm -rf ' + self.stageFolder(stage = 'processed/mri/decision/all'))
			except OSError:
				pass
			
			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
			if 'sara' in os.uname()[1]:
				thisFeatFile = '/home/knapen/projects/reward/man/analysis/reward_more_contrasts.fsf'
			else:
				thisFeatFile = '/Volumes/HDD/research/projects/decision_fMRI/analysis/fsl/dec_runs_combined_general_2.fsf'
				
			REDict = {
			'---OUTPUT_DIR---': 		self.stageFolder(stage = 'processed/mri/decision/all'),
			'---NR_FEATS---': 			str(len(self.conditionDict['decision'])),
			'---FEAT_DIRS---':			'\n'.join(['set feat_files(%d) "%s"'%(i+1, self.runFile(stage = 'processed/mri', run = self.runList[rfeat], postFix = postFix, extension = '.feat')) for (i, rfeat) in enumerate(self.conditionDict['decision'])] ),
			'---EVG---':				'\n'.join(['set fmri(evg%d.1) 1' % (i+1) for i in range(len(self.conditionDict['decision']))]),
			'---GRP_MEM---':			'\n'.join(['set fmri(groupmem.%d) 1' % (i+1) for i in range(len(self.conditionDict['decision']))]),
			}
			print REDict
			
			featOp = FEATOperator(inputObject = thisFeatFile)
			featFileName = os.path.join(self.stageFolder(stage = 'processed/mri/decision/'), 'all_combined.fsf')
			featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
			self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
			# run feat
			featOp.execute()
		
	
	def stats_to_surf(self, which_file = 'zstat'):
		# for r in [self.runList[i] for i in self.conditionDict['decision']]:
		this_feat = os.path.join(self.stageFolder(stage = 'processed/mri/decision'), 'all.gfeat')
		
		left_stim_cope_file = os.path.join(this_feat, 'cope1.feat','stats', which_file + '1.nii.gz')
		right_stim_cope_file = os.path.join(this_feat, 'cope2.feat','stats', which_file + '1.nii.gz')
		answer_cope_file = os.path.join(this_feat, 'cope3.feat','stats', which_file + '1.nii.gz')
		RL_stim_file = os.path.join(this_feat, 'cope4.feat','stats', which_file + '1.nii.gz')
		RL_answer_file = os.path.join(this_feat, 'cope5.feat','stats', which_file + '1.nii.gz')
		stim_answer_file = os.path.join(this_feat, 'cope6.feat','stats', which_file + '1.nii.gz')
			
		for (label, f) in zip(
								['left_stim_cope', 'right_stim_cope', 'answer_cope', 'R>L_stimulus', 'R>L_response', 'stim_answer'], 
								[left_stim_cope_file, right_stim_cope_file, answer_cope_file, RL_stim_file, RL_answer_file, stim_answer_file]
								):
			vsO = VolToSurfOperator(inputObject = f)
			ofn = os.path.join(self.stageFolder(stage = 'processed/mri/decision'), 'surf', 'all_' + label + '.w')
			vsO.configure(frames = {which_file:0}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
			vsO.execute()
	
	def mask_stats_to_hdf(self, run_type = 'decision', postFix = ['mcf','tf']):
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
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
		
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
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			cope_dict = {'left_stim_cope':1, 'right_stim_cope':2, 'answer_cope':3, 'R>L_stimulus':4, 'R>L_response':5, 'stim_answer':6}
			stat_files = {}
			if run_type == 'decision':
				for (i, name) in enumerate(cope_dict):
					stat_files.update({
					name + '_tstat':os.path.join(this_feat, 'stats', 'tstat'+ str(i+1) +'.nii.gz'),
					name + '_zstat':os.path.join(this_feat, 'stats', 'zstat'+ str(i+1) +'.nii.gz'),
					name + '_cope':os.path.join(this_feat, 'stats', 'cope'+ str(i+1) +'.nii.gz'),
					})
				
				
			# general info we want in all hdf files
			stat_files.update({
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'tf', 'psc']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								# for these final two, we need to pre-setup the retinotopic mapping data
								'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
								'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz'),
								'periodic_stimulus_area_mapper': os.path.join(self.stageFolder(stage = 'processed/mri/polar/'), 'polar.nii.gz'),
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
					
			
			# add parameters and behavioral things
			self.analyze_one_run_behavior(run = r, output_fsl_files = False)
			try:
				thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = 'parameters', classname='Group')
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
				thisRunGroup = h5file.createGroup("/" + this_run_group_name, 'parameters', 'Run ' + str(r.ID) +' behavior and parameters')
			
			# shell()
			# create a table for the parameters of this run's trials
			thisRunParameterTable = h5file.createTable(thisRunGroup, 'trial_parameters', r.parameter_dtype_dictionary , 'Parameters for trials in run ' + os.path.split(self.runFile(stage = 'processed/mri', run = r ))[0])
			# fill up the table
			trial = thisRunParameterTable.row
			for tr in r.per_trial_parameters:
				for par in tr.keys():
					trial[par] = tr[par]
				trial.append()
			thisRunParameterTable.flush()
			
			h5file.createArray(thisRunGroup, 'trial_phase_transitions', r.trial_phase_transitions.astype(np.float64), 'trial phase data encoded as ' + str(r.trial_phase_transitions_dtype))
			h5file.createArray(thisRunGroup, 'trial_events', r.per_trial_events_full_array.astype(np.float64), 'event data encoded identical to phase transitions' )
			h5file.createArray(thisRunGroup, 'distilled_answers', r.distilled_answers.astype(np.float64), 'distilled answers' )
		h5file.close()
	
	def run_glm_on_hdf5(self, run_list = None, hdf5_file = None, data_type = 'hpf_data', analysis_type = 'per_trial', post_fix_for_text_file = ['all']):
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['decision'][0]]), 'decision.hdf5')
		h5file = openFile(self.hdf5_filename, mode = "r+")
		super(OrientationDecisionSession, self).run_glm_on_hdf5(run_list = [self.runList[i] for i in self.conditionDict['decision']], hdf5_file = h5file, data_type = data_type, analysis_type = analysis_type, post_fix_for_text_file = post_fix_for_text_file, functionalPostFix = ['mcf','tf'])
		h5file.close()
	
	def per_trial_data_from_run(self, run, h5file, roi, data_type = 'betas', postFix = ['mcf','tf']):
		this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]
		try:
			thisParameterRunGroup = h5file.getNode(where = '/' + this_run_group_name, name = 'parameters', classname='Group')
			self.logger.info(self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' opened for analysis')
		except NoSuchNodeError:
			# import actual data
			self.logger.info('No group ' + this_run_group_name + '.parameters in this file')
		
		parameters = thisParameterRunGroup.trial_parameters.read()
		distilled_answers = thisParameterRunGroup.distilled_answers.read()
		roi_data = self.roi_data_from_hdf(h5file, run, roi, data_type, postFix = postFix)[parameters.shape[0]]	# just the ones coding for the trials, not the nuisance regressor values
		
		return [parameters, distilled_answers, roi_data]
	
	def per_trial_data(self, h5file, roi, data_type = 'betas', postFix = ['mcf','tf']):
		parameter_list = [];		roi_data_list = [];		distilled_answers_list = []
		for  r in [self.runList[i] for i in self.conditionDict['decision']]:
			temp = self.per_trial_data_from_run(r, h5file, roi, data_type = data_type, postFix = postFix)
			parameter_list.append(temp[0])
			distilled_answers_list.append(temp[1])
			roi_data_list.append(temp[2])
		parameter_list = np.vstack(parameter_list);		distilled_answers_list = np.vstack(distilled_answers_list); 		roi_data_list = np.vstack(roi_data_list);
		
		return (parameter_list, distilled_answers_list, roi_data_list)
	
	def svr_roi(self, h5file, roi, data_type = 'betas', mask_data_type = 'mapper_logp', mask_threshold = 3.0, mask_function = '__gt__', stim_position = 1, postFix = ['mcf','tf']):
		"""
		svr_roi conducts an svr analysis on 'data_type' data from a certain 'roi'. 
		this roi is masked using a 'mask_data_type', which is masked at a 'mask_threshold' with a 'mask_function'.
		only data from trials with a certain stimulus position (L vs R) are used for this analysis. so, this function has to be run twice, 
		once for when an roi is empty and once for when it contained the stimulus.
		
		"""
		parameters, distilled_answers, roi_data = self.per_trial_data(h5file, roi, data_type = data_type, postFix = postFix)
		orientations = distilled_answers[:,0,1]
		decisions = distilled_answers[:,0,0]
		definite_decisions = (decisions != 0.0)	# decision that were actually signed - no lapses or out of bounds answers.
		# have to use the stimulus position information at some point was an roi stimulated or not?
		stim_position_indices = (parameters['stim_position'] == stim_position)
		
		# implement masking of ROI data here. preferably based on the mapper or other glm results.
		mask_data = roi_data_from_hdf(h5file, self.runList[self.conditionDict['decision'][0]], roi, mask_data_type, postFix = postFix)
		thresholded_mask_data = eval('mask_data.' + maskFunction + '(' + str(mask_threshold) + ')')
		
		from sklearn.svm import SVR
		from scipy.stats import spearmanr
		svr_lin = SVR(kernel='linear', C=1e2)
		
		# trial based leave-one-out svm regression.
		res = np.zeros((roi_data.shape[0], 2))
		for i in range(roi_data.shape[0]):
			train_set_indices = np.arange(roi_data.shape[0])[np.arange(roi_data.shape[0]) != i]
			res[i,0] = svr_lin.fit(roi_data[train_set_indices], orientations[train_set_indices]).predict(roi_data[i])
			res[i,1] = orientations[i]
		
		print spearmanr(res[:,0], res[:,1])
		return res