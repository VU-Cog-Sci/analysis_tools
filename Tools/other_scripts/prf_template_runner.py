#!/usr/bin/env python
# encoding: utf-8
"""
analyze_7T_S1.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

import os, sys, datetime
import subprocess, logging

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

this_raw_folder = '/home/raw_data/PRF_1/'
this_project_folder = os.path.expanduser('~/projects/PRF/')

sys.path.append( os.environ['ANALYSIS_HOME'] )

from Tools.Sessions import *
from Tools.Subjects.Subject import *
from Tools.Run import *
from Tools.Projects.Project import *

which_subject = 'NA'

def runWholeSession( rA, session ):
	for r in rA:
#		if r['scanType'] == 'epi_bold':
#			r.update(epiRunParameters)
		thisRun = Run( **r )
		session.addRun(thisRun)
	session.parcelateConditions()
	session.parallelize = True
	# session.setupFiles(rawBase = presentSubject.initials, process_eyelink_file = False)
	
	# check whether the inplane_anat has a t2 or t1 - like contrast. t2 is standard. else add contrast = 't1'
	# session.registerSession() # deskull = False, bb = True, flirt = True, MNI = True
	# # after registration of the entire run comes motion correction
	# session.motionCorrectFunctionals()
	# session.resample_epis()
	# # 
	# session.createMasksFromFreeSurferLabels(annot = True, annotFile = 'aparc.a2009s')
	# session.create_dilated_cortical_mask(dilation_sd = 0.25, label = 'cortex')
	# # # 
	# session.rescaleFunctionals(operations = ['sgtf'], filterFreqs = {'highpass':48, 'lowpass': 0}, funcPostFix = ['mcf'], mask_file = os.path.join(session.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz'))
	# session.rescaleFunctionals(operations = ['percentsignalchange'], filterFreqs = {'highpass':48, 'lowpass': 0}, funcPostFix = ['mcf', 'sgtf'])
		# # 
	# session.stimulus_timings()
	# session.physio()
	# session.zscore_timecourse_per_condition()
	
	# session.design_matrix()
	# session.fit_PRF(n_pixel_elements = 40, mask_file_name = 'cortex', n_jobs = -2)

	# session.GLM_for_nuisances()
	# session.fit_PRF(n_pixel_elements = 42, mask_file_name = 'cortex_dilated_mask', n_jobs = 28)
	# session.results_to_surface(res_name = 'corrs_cortex_dilated_mask')
	# session.RF_fit(mask_file = 'cortex_dilated_mask', stat_threshold = -50.0, n_jobs = 28, run_fits = False)
	# session.results_to_surface(res_name = 'ecc', frames = {'rad':0})
	# session.results_to_surface(res_name = 'polar', frames = {'rad':0})
	# session.makeTiffsFromCondition(condition = 'PRF', y_rotation = 90.0, exit_when_ready = 0)

# for testing;
if __name__ == '__main__':	
	if which_subject == 'WK':
		# first subject; WK
		#########################################################################
		# subject information
		initials = 'WK'
		firstName = 'Wouter'
		standardFSID = 'WK_200813_12'
		birthdate = datetime.date( 1988, 11, 26 )
		labelFolderOfPreference = ''
		presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
	
		presentProject = Project( 'Population Receptive Field Mapping', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
	
		sessionDate = datetime.date(2013, 8, 20)
		sessionID = 'PRF_' + presentSubject.initials
		sj_init_data_code = 'WK_200813'
	
		WK_Session = PopulationReceptiveFieldMappingSession(sessionID, sessionDate, presentProject, presentSubject)
	
		try:
			os.mkdir(os.path.join(this_project_folder, 'data', initials))
			os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
		except OSError:
			WK_Session.logger.debug('output folders already exist')
	
	
		WK_run_array = [
		
			{'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2461_WIP_T2W_RetMap_1.25_CLEAR_6_1.nii.gz' ), 
			},
			{'ID' : 3, 'scanType': 'epi_bold', 'condition': 'PRF', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2461_WIP_RetMap_2.5_1.5_SENSE_3_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/eye/', 'wk_1.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/behavior/', 'wk_1_2013-08-20_15.23.50_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/hr/', 'SCANPHYSLOG20130820152650.log' ), 
				},
			{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'PRF', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2461_WIP_RetMap_2.5_1.5_SENSE_5_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/eye/', 'wk_2.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/behavior/', 'wk_2_2013-08-20_15.55.21_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/hr/', 'SCANPHYSLOG20130820155543.log' ), 
				},
			{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'PRF', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2461_WIP_RetMap_2.5_1.5_SENSE_7_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/eye/', 'wk_3.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/behavior/', 'wk_3_2013-08-20_16.24.12_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/hr/', 'SCANPHYSLOG20130820162406.log' ), 
				},
			{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'PRF', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2461_WIP_RetMap_2.5_1.5_SENSE_9_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/eye/', 'wk_4.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/behavior/', 'wk_4_2013-08-20_16.52.32_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/hr/', 'SCANPHYSLOG20130820165315.log' ), 
				},
		
		]
	
		runWholeSession(WK_run_array, WK_Session)
	elif which_subject == 'NA':
		# for testing 2nd Subject;
		# 2nd subject; NA
		#########################################################################
		# subject information
		initials = 'NA'
		firstName = 'Nicki'
		standardFSID = 'NA_220813_12' 
		birthdate = datetime.date( 1985, 04, 05 )
		labelFolderOfPreference = ''
		presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		
		presentProject = Project( 'Population Receptive Field Mapping', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
		
		sessionDate = datetime.date(2013, 8, 22)
		sessionID = 'PRF_' + presentSubject.initials
		sj_init_data_code = 'NA_220813'
		
		NA_Session = PopulationReceptiveFieldMappingSession(sessionID, sessionDate, presentProject, presentSubject)
		
		try:
			os.mkdir(os.path.join(this_project_folder, 'data', initials))
			os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
		except OSError:
			NA_Session.logger.debug('output folders already exist')
			
			
		NA_run_array = [

		{'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
			'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2470_T2W_RetMap_1.25_CLEAR_6_1.nii.gz' ), # Why no WIP in name, Wouter has WIP 
		},
		{'ID' : 3, 'scanType': 'epi_bold', 'condition': 'PRF', 
			'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2470_WIP_RetMap_2.5_1.5_SENSE_3_1.nii.gz' ), 
			'eyeLinkFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/eye/', 'na_1.edf' ), 
			'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/behavior/', 'na_1_2013-08-22_15.13.59_outputDict.pickle' ), 
			'physiologyFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/hr/', 'SCANPHYSLOG20130822151606.log' ), 
			},
		{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'PRF', 
			'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2470_WIP_RetMap_2.5_1.5_SENSE_5_1.nii.gz' ), 
			'eyeLinkFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/eye/', 'na_2.edf' ), 
			'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/behavior/', 'na_2_2013-08-22_15.38.26_outputDict.pickle' ), 
			'physiologyFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/hr/', 'SCANPHYSLOG20130822154516.log' ), 
			},
		{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'PRF', 
			'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2470_WIP_RetMap_2.5_1.5_SENSE_7_1.nii.gz' ), 
			'eyeLinkFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/eye/', 'na_3.edf' ), 
			'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/behavior/', 'na_3_2013-08-22_16.13.18_outputDict.pickle' ), 
			'physiologyFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/hr/', 'SCANPHYSLOG20130822161257.log' ), 
			},
		{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'PRF', 
			'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/mri/', 'sc2470_WIP_RetMap_2.5_1.5_SENSE_9_1.nii.gz' ), 
			'eyeLinkFilePath': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/eye/', 'na_4.edf' ), 
			'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/behavior/', 'na_4_2013-08-22_16.35.18_outputDict.pickle' ), 
			'physiologyFile': os.path.join(this_raw_folder, 'data/' + initials + '/' + sj_init_data_code +  '/raw/hr/', 'SCANPHYSLOG20130822164214.log' ), 
			},

		]
	
		runWholeSession(NA_run_array, NA_Session)
	