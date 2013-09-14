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
from ..Operators.PhysioOperator import *
from pylab import *
from nifti import *


class PopulationReceptiveFieldMappingSession(Session):
	"""
	Class for population receptive field mapping sessions analysis.
	"""
	def create_dilated_cortical_mask(self, dilation_sd = 3.0):
		"""create_dilated_cortical_mask takes the rh and lh cortex files and joins them to one cortex.nii.gz file.
		it then smoothes this mask with fslmaths, using a gaussian kernel. 
		This is then thresholded at > 0.0, in order to create an enlarged cortex mask in binary format.
		"""
		# take rh and lh files and join them.
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'rh.cortex.nii.gz'))
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex.nii.gz'), **{'-add': os.path.join(self.stageFolder('processed/mri/masks/anat'), 'lh.cortex.nii.gz')})
		fmO.execute()
		
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex.nii.gz'))
		fmO.configureSmooth(smoothing_sd = dilation_sd)
		fmO.execute()
		
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_smooth.nii.gz'))
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz'), **{'-bin': ''})
		fmO.execute()
	
	def stimulus_timings(self):
		"""docstring for stimulus_timings"""
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			bO = PopulationReceptiveFieldBehaviorOperator(self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' ))
			bO.trial_times()
			r.trial_times = bO.trial_times
			r.all_button_times = bO.all_button_times
			tasks = list(np.unique(np.array([tt[0] for tt in r.trial_times])))
			for task in tasks:
				these_trials = np.array([[tt[1], tt[2] - tt[1], 1.0] for tt in r.trial_times if tt[0] == task])
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]), these_trials, fmt = '%3.2f', delimiter = '\t')
				these_buttons = np.array([[float(bt[1]), 0.5, 1.0] for bt in r.all_button_times if bt[0] == task])
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['button', task]), these_buttons, fmt = '%3.2f', delimiter = '\t')
		
	def physio(self):
		"""docstring for physio"""
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			pO = PhysioOperator(self.runFile(stage = 'processed/hr', run = r, extension = '.log' ))
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] ))
			pO.preprocess_to_continuous_signals(TR = nii_file.rtime, nr_TRs = nii_file.timepoints)
		
		
		
		
	
	def zscore_timecourse_per_condition(self):
		"""docstring for fit_voxel_timecourse"""
		pass
	
		