#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime
from Session import * 
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from ..circularTools import *
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle

class OrientationDecisionSession(RetinotopicMappingSession):
	"""
	Class for orientation decision decoding sessions analysis.
	Forks from retinotopic mapping session primarily because of phase-encoded mapping runs. 
	Involves trial- and run-based support vector regression/decoding of orientation around =/- 45.
	"""
	def analyze_one_run_behavior(self, run):
		"""
		Takes a pickle file - the run's behavioral file - opens and analyzes it. 
		It saves an fsl-style text file with regressors, and adds parameters, events and response times to the session's hdf5 file.
		the orientation file is a .dat pickle file, that has to be opened in unicode mode.
		"""
		run_data = pickle.load(file(self.runFile(stage = 'processed/behavior', run = run, extension = '.dat', 'U' )))
		trial_phase_transitions = [np.array(run_data['trial_phase_transition_array'][i]) for i in run_data['run_order']]
		trial_phase_transitions = np.concatenate([np.vstack([trial_phase_transitions[i].T, i * np.ones(trial_phase_transitions[i].shape[0])]).T for i in range(len(trial_phase_transitions))])
		trial_phase_transitions_dtype = np.dtype([('phase_number', '<f8'), ('trial_time', '<f8'), ('run_time', '<f8'), ('trial_number', '<f8')])
		
		per_trial_events = [np.array(run_data['trial_event_array'][i], dtype = float) for i in run_data['run_order']]
		per_trial_noTR_events = [pte[pte[:,0] != 5.] for pte in per_trial_events]
		# need to ascertain into what trial phase these responses fall, and what answer these responses represent. 
		# too lame to do that now. on to parameters
		
		per_trial_parameters = [run_data['trial_parameter_array'][i] for i in run_data['run_order']]
		# keys for confidence are [3,2,1,6,7,8] (left-right order of fingers), keys for answers are [1,6] (the index fingers of both hands)
		
		
		
		