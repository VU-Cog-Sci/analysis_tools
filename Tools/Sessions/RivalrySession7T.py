#!/usr/bin/env python
# encoding: utf-8

"""RivalrySessionJW.py"""	

import os, sys, subprocess, datetime
import tempfile, logging, pickle
import numpy as np
import scipy as sp
import matplotlib.pylab as pl
from scipy.stats import *
from scipy.stats import norm
import random
from random import *
import bottleneck as bn
from matplotlib.backends.backend_pdf import PdfPages
from itertools import *
from IPython import embed as shell

import mne
import nitime
import sklearn
from nifti import *
from pypsignifit import *
from nitime import fmri
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.decomposition import *
from sklearn import svm

sys.path.append( os.environ['ANALYSIS_HOME'] )

from Tools.Sessions import *
import Tools.Project as Project
from Tools.Run import *
from Tools.plotting_tools import *
from Tools.circularTools import *
from Tools.Operators import *

class RivalrySession7T(RivalryReplaySession):
	
	####################################################################################
	## Household functions: ############################################################
	def createFolderHierarchy(self):
		"""docstring for fname"""
		rawFolders = ['raw/mri', 'raw/behavior', 'raw/hr']
		self.processedFolders = ['processed/mri', 'processed/behavior', 'processed/hr']
		conditionFolders = np.concatenate((self.conditionList, ['log','figs','masks','masks/stat','masks/anat','reg','surf','scripts']))
		
		self.makeBaseFolder()
		# assuming baseDir/raw/ exists, we must make processed
		if not os.path.isdir(os.path.join(self.baseFolder(), 'processed') ):
			os.mkdir(os.path.join(self.baseFolder(), 'processed'))
		if not os.path.isdir(os.path.join(self.baseFolder(), 'raw') ):
			os.mkdir(os.path.join(self.baseFolder(), 'raw'))
		
		# create folders for processed data
		for pf in self.processedFolders:
			if not os.path.isdir(self.stageFolder(pf)):
				os.mkdir(self.stageFolder(pf))
			# create condition folders in each of the processed data folders and also their surfs
			for c in conditionFolders:
				 if not os.path.isdir(self.stageFolder(pf+'/'+c)):
					os.mkdir(self.stageFolder(pf+'/'+c))
					if pf == 'processed/mri':
						if not os.path.isdir(os.path.join(self.stageFolder(pf+'/'+c), 'surf')):
							os.mkdir(os.path.join(self.stageFolder(pf+'/'+c), 'surf'))
			# create folders for each of the runs in the session and their surfs
			for rl in self.runList:
				if not os.path.isdir(self.runFolder(pf, run = rl)):
					os.mkdir(self.runFolder(pf, run = rl))
					if pf == 'processed/mri':
						if not os.path.isdir(os.path.join(self.runFolder(pf, run = rl), 'surf')):
							os.mkdir(os.path.join(self.runFolder(pf, run = rl), 'surf'))
						if not os.path.isdir(os.path.join(self.runFolder(pf, run = rl), 'masked')):
							os.mkdir(os.path.join(self.runFolder(pf, run = rl), 'masked'))	
	
	
	

	def registerfeats(self, run_type = 'sphere_presto', postFix = ['mcf']):
		"""run featregapply for all feat direcories in this session."""
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:	
				
				this_feat = self.runFile(stage = 'processed/mri', run = self.runList[run], base = 'rivalry', postFix = postFix, extension = '.feat')
				self.setupRegistrationForFeat(this_feat)

	def gfeat_analysis(self, run_separate = True, run_combination = True):
		
		try:	# create folder
			os.mkdir(self.stageFolder('processed/mri/masks/stat/gfeat'))
			os.mkdir(self.stageFolder('processed/mri/masks/stat/gfeat/surf'))
		except OSError:
			pass
		for i in range(1,9): # all stats
			for stat in ['z','t']:
				
				afo = FlirtOperator( 	os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]), 'combined/combined.gfeat', 'cope' + str(i) + '.feat', 'stats', stat + 'stat1.nii.gz'), 
										referenceFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] )
										)
				# here I assume that the feat registration directory has been created. it's the files that have been used to create the gfeat, so we should be cool.
				afo.configureApply(		transformMatrixFileName = os.path.join(self.stageFolder('processed/mri/reg/feat/'), 'standard2example_func.mat'), 
										outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat/gfeat'), 'cope' + str(i) + '_' + os.path.split(afo.inputFileName)[1]))
				afo.execute()
				# to surface
				stso = VolToSurfOperator(inputObject = afo.outputFileName)
				stso.configure(		frames = {'stat': 0} , 
									register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), 
									outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat/gfeat/surf'), os.path.split(afo.outputFileName)[1]))
				stso.execute()
	
	
	def mask_stats_to_hdf(self, run_type = 'rivalry', postFix = ['mcf'], polar_eccen=False):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results.
		- PER RUN: feat data, residuals, hpf_data, tf_data, and tf_psc_date.
		- COMBINED OVER RUNS: gfeat data, (polar and eccen data if polar_eccen==True)
			eccen data has to be in .../processed/mri/masks/eccen/eccen.nii.gz
			polar data has to be in .../processed/mri/masks/polar/polar.nii.gz
		"""
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
			
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		# if os.path.isfile(self.hdf5_filename):
		# 			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = openFile(self.hdf5_filename, mode = 'a', title = run_type + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")

		######################################################################################################
		# ADD STATS PER RUN:
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
			# this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			this_feat = self.runFile(stage = 'processed/mri', run = r, base = 'rivalry', postFix = ['mcf'], extension = '.feat')
			
			stat_files = {
							'STIM_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
							'STIM_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
							'STIM_cope' : os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
								
							'BR_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
							'BR_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
							'BR_cope' : os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
							
							'SFM_T': os.path.join(this_feat, 'stats', 'tstat3.nii.gz'),
							'SFM_Z': os.path.join(this_feat, 'stats', 'zstat3.nii.gz'),
							'SFM_cope' : os.path.join(this_feat, 'stats', 'cope3.nii.gz'),
							
							'BR_SFM_T': os.path.join(this_feat, 'stats', 'tstat4.nii.gz'),
							'BR_SFM_Z': os.path.join(this_feat, 'stats', 'zstat4.nii.gz'),
							'BR_SFM_cope' : os.path.join(this_feat, 'stats', 'cope4.nii.gz'),
							
							'TRANS_T': os.path.join(this_feat, 'stats', 'tstat5.nii.gz'),
							'TRANS_Z': os.path.join(this_feat, 'stats', 'zstat5.nii.gz'),
							'TRANS_cope' : os.path.join(this_feat, 'stats', 'cope5.nii.gz'),
							
							'BR_TRANS_T': os.path.join(this_feat, 'stats', 'tstat6.nii.gz'),
							'BR_TRANS_Z': os.path.join(this_feat, 'stats', 'zstat6.nii.gz'),
							'BR_TRANS_cope' : os.path.join(this_feat, 'stats', 'cope6.nii.gz'),
							
							'SFM_TRANS_T': os.path.join(this_feat, 'stats', 'tstat7.nii.gz'),
							'SFM_TRANS_Z': os.path.join(this_feat, 'stats', 'zstat7.nii.gz'),
							'SFM_TRANS_cope' : os.path.join(this_feat, 'stats', 'cope7.nii.gz'),
							
							'BR_TRANS_SFM_TRANS_T': os.path.join(this_feat, 'stats', 'tstat8.nii.gz'),
							'BR_TRANS_SFM_TRANS_Z': os.path.join(this_feat, 'stats', 'zstat8.nii.gz'),
							'BR_TRANS_SFM_TRANS_cope' : os.path.join(this_feat, 'stats', 'cope8.nii.gz'),
							}
				
			# general info we want in all hdf files
			stat_files.update({
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								# 'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'psc', 'tf']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								# for these final two, we need to pre-setup the retinotopic mapping data
								# 'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
								# 'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz')
								'tf_data': self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '_tf'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'tf_psc_data': self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '_tf_psc'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
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
					try:
						imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
						these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
						h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
					except ZeroDivisionError:
						pass
		
		######################################################################################################
		# ADD COMBINED OVER RUNS STUFF
		
		this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri/', extension = '_combined'))[1]
		
		try:
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + this_run_group_name + ' already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = h5file.createGroup("/", this_run_group_name, ' imported from ' + self.runFile(stage = 'processed/mri/rivalry/combined/combined.gfeat', postFix = postFix))
		
		"""
		Now, take different stat masks based on the run_type
		"""
		if polar_eccen == True:
			stat_files = {
				'STIM_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope1_tstat1.nii.gz'),
				'STIM_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope1_zstat1.nii.gz'),
				'BR_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope2_tstat1.nii.gz'),
				'BR_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope2_zstat1.nii.gz'),
				'SFM_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope3_tstat1.nii.gz'),
				'SFM_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope3_zstat1.nii.gz'),
				'BR>SFM_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope4_tstat1.nii.gz'),
				'BR>SFM_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope4_zstat1.nii.gz'),
				'TRANS_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope5_tstat1.nii.gz'),
				'TRANS_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope5_zstat1.nii.gz'),
				'BR_TRANS_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope6_tstat1.nii.gz'),
				'BR_TRANS_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope6_zstat1.nii.gz'),
				'SFM_TRANS_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope7_tstat1.nii.gz'),
				'SFM_TRANS_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope7_zstat1.nii.gz'),
				'BR_TRANS>SFM_TRANS_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope8_tstat1.nii.gz'),
				'BR_TRANS>SFM_TRANS_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope8_zstat1.nii.gz'),
				'ECCEN': os.path.join(self.stageFolder(stage = 'processed/mri/masks/eccen'), 'eccen.nii.gz'),
				'POLAR': os.path.join(self.stageFolder(stage = 'processed/mri/masks/polar'), 'polar.nii.gz'),
			}
		else:
			stat_files = {
				'STIM_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope1_tstat1.nii.gz'),
				'STIM_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope1_zstat1.nii.gz'),
				'BR_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope2_tstat1.nii.gz'),
				'BR_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope2_zstat1.nii.gz'),
				'SFM_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope3_tstat1.nii.gz'),
				'SFM_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope3_zstat1.nii.gz'),
				'BR>SFM_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope4_tstat1.nii.gz'),
				'BR>SFM_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope4_zstat1.nii.gz'),
				'TRANS_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope5_tstat1.nii.gz'),
				'TRANS_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope5_zstat1.nii.gz'),
				'BR_TRANS_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope6_tstat1.nii.gz'),
				'BR_TRANS_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope6_zstat1.nii.gz'),
				'SFM_TRANS_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope7_tstat1.nii.gz'),
				'SFM_TRANS_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope7_zstat1.nii.gz'),
				'BR_TRANS>SFM_TRANS_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope8_tstat1.nii.gz'),
				'BR_TRANS>SFM_TRANS_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope8_zstat1.nii.gz'),
			}
		
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
				try:
					imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
					these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
					h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
				except ZeroDivisionError:
					pass	
		
		h5file.close()
	
	
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


	def roi_data_from_hdf_runsCombined(self, h5file, this_run_group_name, roi_wildcard, data_type):
		"""
		drags data from an already opened hdf file into a numpy array, concatenating the data_type data across voxels in the different rois that correspond to the roi_wildcard
		"""
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


	
	
	
	def setup_all_data_for_decoding(self, runArray, decoding, roi, run_type = 'rivalry', postFix = ['mcf'], mask_type='STIM_Z', mask_direction='pos', threshold=None, number_voxels=None, split_by_relative_time=None, polar_eccen=False):
		
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		h5file = openFile(self.hdf5_filename, mode = 'r+', title = run_type + " file")
		
		# Load all functional data (per run):
		roi_data_per_roi = []
		mask_data_per_roi = []
		for j in range(len(roi)):
			roi_data = []
			mask_data = []
			for r in [self.runList[i] for i in self.conditionDict['rivalry']]:
				roi_data.append(self.roi_data_from_hdf(h5file, r, roi[j], 'residuals', postFix = ['mcf'])) # tf_psc_data
				mask_data.append(self.roi_data_from_hdf(h5file, r, roi[j], mask_type))
			roi_data_per_roi.append(np.hstack(roi_data))
			mask_data_per_roi.append(np.hstack(mask_data))
			
		roi_data_c = np.vstack(roi_data_per_roi).T
		mask_data_c = np.vstack(mask_data_per_roi).T
		
		mask_data_mean = mask_data_c.mean(axis = 0)
		
		# Load all decoding indices (per run):
		decoding_indices_BR = []
		decoding_indices_SFM = []
		relative_times_BR = []
		relative_times_SFM = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['rivalry']]:
			nr_runs += 1
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			search_name = 'decoding_indices'
			for node in h5file.iterNodes(where = thisRunGroup, classname = 'Group'):
				if search_name == node._v_name:
					decoding_node = node
					decoding_indices_BR.append( decoding_node.decoding_BR.read()[:,0] )
					decoding_indices_SFM.append( decoding_node.decoding_SFM.read()[:,0] )
					relative_times_BR.append( decoding_node.decoding_BR.read()[:,4] )
					relative_times_SFM.append( decoding_node.decoding_SFM.read()[:,4] )
					break
		decod_BR_c = np.concatenate(decoding_indices_BR, axis=1).T
		decod_SFM_c = np.concatenate(decoding_indices_SFM, axis=1).T
		relative_times_BR_c = np.concatenate(relative_times_BR, axis=1).T
		relative_times_SFM_c = np.concatenate(relative_times_SFM, axis=1).T
		
		# Load eccen and polar data (combined over runs):
		if polar_eccen == True:
			polar_data = []
			eccen_data = []
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri/', extension = '_combined'))[1]
			for j in range(len(roi)):
				eccen_data.append( self.roi_data_from_hdf_runsCombined(h5file, this_run_group_name, roi[j], 'ECCEN') )
				polar_data.append( self.roi_data_from_hdf_runsCombined(h5file, this_run_group_name, roi[j], 'POLAR') )
			eccen_data_c = np.vstack(eccen_data).T
			polar_data_c = np.vstack(polar_data).T
		
		# DATA FOR DECODING:
		self.X_BR = roi_data_c[decod_BR_c!=0,:]
		self.X_SFM = roi_data_c[decod_SFM_c!=0,:]
		self.y_BR = decod_BR_c[decod_BR_c!=0]
		self.y_SFM = decod_SFM_c[decod_SFM_c!=0]
		
		# DATA FOR MASKING:
		self.mask_data_mean = mask_data_mean
		if polar_eccen==True:
			self.eccen_data_c = eccen_data_c
			self.polar_data_c = polar_data_c
		self.relative_times_BR = relative_times_BR_c[relative_times_BR_c!=0]
		self.relative_times_SFM = relative_times_SFM_c[relative_times_SFM_c!=0]
		
		# GET MASKING INDICES:
		# masking indices features (voxels) based on gfeat data (thresholded!):
		mask_data_indices_thresh = np.ones(self.X_BR.shape[1], dtype = bool)
		if threshold != None:
			if mask_direction == 'pos':
				mask_data_indices_thresh = self.mask_data_mean >= threshold
			if mask_direction == 'neg':
				mask_data_indices_thresh = self.mask_data_mean < threshold
		
		# masking indices features (voxels) based on gfeat data (top or bottom # of voxels!):
		mask_data_indices_max_voxels = np.ones(self.X_BR.shape[1], dtype = bool)
		if number_voxels != None:
			if mask_direction == 'pos':
				mask_data_indices_max_voxels = self.mask_data_mean >= min(bn.partsort(self.mask_data_mean, self.mask_data_mean.size-number_voxels)[-number_voxels:])
			if mask_direction == 'neg':
				mask_data_indices_max_voxels = self.mask_data_mean <= max(bn.partsort(self.mask_data_mean, number_voxels)[:number_voxels])
		
		# masking indices samples (Trs) based on relative time:
		if decoding == 'BR':
			mask_data_indices_time = np.ones(self.y_BR.shape[0], dtype = bool)
			if split_by_relative_time != None:
				if split_by_relative_time == 'early':
					mask_data_indices_time = (self.relative_times_BR < np.median(self.relative_times_BR))
				if split_by_relative_time == 'late':
					mask_data_indices_time = (self.relative_times_BR > np.median(self.relative_times_BR))
		if decoding == 'SFM':
			mask_data_indices_time = np.ones(self.y_SFM.shape[0], dtype = bool)
			if split_by_relative_time != None:
				if split_by_relative_time == 'early':
					mask_data_indices_time = (self.relative_times_SFM < np.median(self.relative_times_SFM))
				if split_by_relative_time == 'late':
					mask_data_indices_time = (self.relative_times_SFM > np.median(self.relative_times_SFM))
		
		# MASK AND ZSCORE:
		mask_data_indices_features = mask_data_indices_thresh * mask_data_indices_max_voxels
		mask_data_indices_samples = mask_data_indices_time
		if decoding == 'BR':
			X_BR_masked = self.X_BR[mask_data_indices_samples,:]
			X_BR_masked = X_BR_masked[:,mask_data_indices_features]
			y_BR_masked = self.y_BR[mask_data_indices_samples]
			# z-score X_BR_masked:
			X_BR_masked_z = sp.stats.zscore(X_BR_masked, axis=1)
			X_BR_masked_z = sp.stats.zscore(X_BR_masked_z, axis=0)
			self.X = X_BR_masked_z
			self.y = y_BR_masked
		if decoding == 'SFM':
			X_SFM_masked = self.X_SFM[mask_data_indices_samples,:]
			X_SFM_masked = X_SFM_masked[:,mask_data_indices_features]
			y_SFM_masked = self.y_SFM[mask_data_indices_samples]
			# z-score X_SFM_masked:
			X_SFM_masked_z = sp.stats.zscore(X_SFM_masked, axis=1)
			X_SFM_masked_z = sp.stats.zscore(X_SFM_masked_z, axis=0)
			# b = X_SFM_masked
			# b1 = ((b - bn.nanmean(b, axis = 0)) / bn.nanstd(b, axis=0)).T
			# b2 = ((b1 - bn.nanmean(b1, axis = 0)) / bn.nanstd(b1, axis=0)).T
			# X_SFM_masked = b2
			self.X = X_SFM_masked_z
			self.y = y_SFM_masked
	
	####################################################################################
	## General functions: ##############################################################
	def createRegressors(self, inputObject, len_run, subSamplingRatio):
		
		inputObject = inputObject
		len_run = len_run
		subSamplingRatio = subSamplingRatio
		
		HRF = nitime.fmri.hrf.gamma_hrf(15, Fs = subSamplingRatio)
		
		regr = np.zeros((len_run,inputObject.shape[0]))
		for i in range(inputObject.shape[0]):
			start = round(inputObject[i,0]*subSamplingRatio, 0)
			dur = round(inputObject[i,1]*subSamplingRatio, 0)
			regr[start:start+dur,i] = 1.0
		regr_collapsed = np.sum(regr, axis=1)
		
		regr_convolved = np.zeros((len_run,inputObject.shape[0]))
		for i in range(inputObject.shape[0]):
			regr_convolved[:,i] = (sp.convolve(regr[:,i], HRF, mode = 'full'))[204:len_run+204]
		regr_convolved_collapsed = np.sum(regr_convolved, axis=1)
		
		return(regr_collapsed, regr_convolved_collapsed)
	
	
	
	####################################################################################
	## Analysis functions: #############################################################
	def createRegressors1(self, runArray):
		
		"""
		Create behaviour files for first session. In this sessions timestamps we're based on TRs.
		"""
		
		# data = np.loadtxt('/Research/7T_RIVALRY/data/TK/TK_250113/raw/behavior/tk_5_ongoing_7T.txt')
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:
				
				this_run = runArray[run]['ID']
				
				# print(self.subject.initials)
				# print(self.date)
				# print(run)
				# print(runArray[run]['behaviorFile'])
				
				stim_dur1 = 142.0
				stim_dur2 = 150.0
				break_between_trials = 16.0
				time_specificity = 0.65/2.0
				
				file_to_open = str(runArray[run]['behaviorFile'])
				data = np.loadtxt(file_to_open)
				
				# Get percept and transition indices:
				percept1 = (data[:,0] == 66)
				percept2 = (data[:,0] == 67)
				transition = (data[:,0] == 65)
		
				# Correct associated time stamps:
				for i in range(data.shape[0]):
					if percept1[i] == True:
						if data[i-1,1] != 255:
							if data[i+1,1] != 255:
								data[i,1] = (data[i-1,1] + data[i+1,1]) / 2
						if data[i-1,1] == 255:
							if data[i+1,1] != 255:
								data[i,1] = data[i+1,1]
						if data[i+1,1] == 255:
							if data[i-1,1] != 255:
								data[i,1] = data[i-1,1]
					if percept2[i] == True:
						if data[i-1,1] != 255:
							if data[i+1,1] != 255:
								data[i,1] = (data[i-1,1] + data[i+1,1]) / 2
							if data[i-1,1] == 255:
								if data[i+1,1] != 255:
									data[i,1] = data[i+1,1]
							if data[i+1,1] == 255:
								if data[i-1,1] != 255:
									data[i,1] = data[i-1,1]
					if transition[i] == True:
						if data[i-1,1] != 255:
							if data[i+1,1] != 255:
								data[i,1] = (data[i-1,1] + data[i+1,1]) / 2
							if data[i-1,1] == 255:
								if data[i+1,1] != 255:
									data[i,1] = data[i+1,1]
							if data[i+1,1] == 255:
								if data[i-1,1] != 255:
									data[i,1] = data[i-1,1]
				
				# Remove uninteresting time stamps:
				data = data[-(data[:,0]==49),:]
				
				# Remove dubble occurences:
				dubble_occurences_indices = zeros(data.shape[0], dtype=bool)
				for i in range(data.shape[0]):
					if data[i-1,0] == data[i,0]:
						dubble_occurences_indices[i] = True
				data = data[-dubble_occurences_indices,:]
				
				# Indices:
				percept1 = (data[:,0] == 66)
				percept2 = (data[:,0] == 67)
				transition = (data[:,0] == 65)
				
				# Check for transitions without duration (no button press for transition)
				transition_no_dur = np.zeros(data.shape[0], dtype = bool)
				for i in range(data.shape[0]):
					try:
						if percept1[i-1] == True:
							if percept2[i] == True:
								transition_no_dur[i] = True
						if percept2[i-1] == True:
							if percept1[i] == True:
								transition_no_dur[i] = True
					except IndexError:
						pass
				
				# Times:
				times = data[:,1] - data[0,1]
				
				# get last time stamp of run:
				# PLUS 350ms!!
				first_time1 = times[np.where((data[:,0] == 1))[0][0]]
				first_time2 = times[np.where((data[:,0] == 1))[0][1]]
				last_time1 = first_time1 + stim_dur1 + time_specificity
				last_time2 = first_time2 + stim_dur2 + time_specificity
				
				# paradigm indicies:
				paradigm1_indices = (times >= first_time1) * (times <= last_time1)
				paradigm2_indices = (times >= first_time2) * (times <= last_time2)
				if this_run % 2 != 0:
					BR_indices = paradigm1_indices
					SFM_indices = paradigm2_indices
				else:
					SFM_indices = paradigm1_indices
					BR_indices = paradigm2_indices
				
				# durations:
				durs = np.diff(times)
				durs = np.concatenate((durs, np.diff([times[-1],last_time2])))
				
				# check that durs do not extend into 'break between trials':
				durs2 = durs[paradigm2_indices]
				durs1 = durs[-paradigm2_indices]
				for i in range(durs1.shape[0]):
					if times[-paradigm2_indices][i] + durs1[i] >= last_time1:
						durs1[i] = (last_time1 - times[-paradigm2_indices][i])
				for i in range(times[paradigm2_indices].shape[0]):
					if times[paradigm2_indices][i] + durs2[i] >= last_time2:
						durs2[i] = last_time2 - times[paradigm2_indices][i]
				durs3 = np.concatenate((durs1,durs2))
				durs = durs3
				
				############################################################
				############ MAKE TEXT FILES REGRESSORS ####################
				# BR start:
				BR_start = np.zeros((1,3))
				BR_start[:,0] = times[BR_indices][0]
				if this_run % 2 != 0:
					BR_start[:,1] = stim_dur1
				else:
					BR_start[:,1] = stim_dur2
				BR_start[:,2] = np.ones(BR_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', extension = '.txt'), BR_start, fmt='%3.2f', delimiter = '\t')
				# BR transitions:
				BR_transition = np.zeros((sum(transition[BR_indices])+sum(transition_no_dur[BR_indices]),3))
				BR_transition[:sum(transition[BR_indices]),0] = times[transition*BR_indices]
				BR_transition[sum(transition[BR_indices]):,0] = times[transition_no_dur*BR_indices]
				BR_transition[:sum(transition[BR_indices]),1] = durs[transition*BR_indices]
				BR_transition[sum(transition[BR_indices]):,1] = 1.00
				BR_transition[:,2] = np.ones(BR_transition.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', postFix = ['trans'], extension = '.txt'), BR_transition, fmt='%3.2f', delimiter = '\t')
				# SFM start:
				SFM_start = np.zeros((1,3))
				SFM_start[:,0] = times[SFM_indices][0]
				if this_run % 2 != 0:
					SFM_start[:,1] = stim_dur2
				else:
					SFM_start[:,1] = stim_dur1
				SFM_start[:,2] = np.ones(SFM_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', extension = '.txt'), SFM_start, fmt='%3.2f', delimiter = '\t')
				# SFM transitions:
				SFM_transition = np.zeros((sum(transition[SFM_indices])+sum(transition_no_dur[SFM_indices]),3))
				SFM_transition[:sum(transition[SFM_indices]),0] = times[transition*SFM_indices]
				SFM_transition[sum(transition[SFM_indices]):,0] = times[transition_no_dur*SFM_indices]
				SFM_transition[:sum(transition[SFM_indices]),1] = durs[transition*SFM_indices]
				SFM_transition[sum(transition[SFM_indices]):,1] = 1.00
				SFM_transition[:,2] = np.ones(SFM_transition.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', postFix = ['trans'], extension = '.txt'), SFM_transition, fmt='%3.2f', delimiter = '\t')
				
				##### for decoding: #####
				####################
				BR_percept1_indices = BR_indices*(data[:,0]==66)
				BR_percept2_indices = BR_indices*(data[:,0]==67)
				SFM_percept1_indices = SFM_indices*(data[:,0]==66)
				SFM_percept2_indices = SFM_indices*(data[:,0]==67)
				
				BR1_start = np.zeros((BR_percept1_indices.sum(),3))
				BR1_start[:,0] = times[BR_percept1_indices]
				BR1_start[:,1] = durs[BR_percept1_indices]
				BR1_start[:,2] = np.ones(BR1_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_BR1', extension = '.txt'), BR1_start, fmt='%3.2f', delimiter = '\t')
				BR2_start = np.zeros((BR_percept2_indices.sum(),3))
				BR2_start[:,0] = times[BR_percept2_indices]
				BR2_start[:,1] = durs[BR_percept2_indices]
				BR2_start[:,2] = np.ones(BR2_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_BR2', extension = '.txt'), BR2_start, fmt='%3.2f', delimiter = '\t')
				SFM1_start = np.zeros((SFM_percept1_indices.sum(),3))
				SFM1_start[:,0] = times[SFM_percept1_indices]
				SFM1_start[:,1] = durs[SFM_percept1_indices]
				SFM1_start[:,2] = np.ones(SFM1_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_SFM1', extension = '.txt'), SFM1_start, fmt='%3.2f', delimiter = '\t')
				SFM2_start = np.zeros((SFM_percept2_indices.sum(),3))
				SFM2_start[:,0] = times[SFM_percept2_indices]
				SFM2_start[:,1] = durs[SFM_percept2_indices]
				SFM2_start[:,2] = np.ones(SFM2_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_SFM2', extension = '.txt'), SFM2_start, fmt='%3.2f', delimiter = '\t')
				
				############################################################
				# Barplot of means:
				
				MEANS = (mean(durs[percept1*BR_indices]), mean(durs[percept2*BR_indices]), mean(durs[percept1*SFM_indices]), mean(durs[percept2*SFM_indices]))
				SEMS = (stats.sem(durs[percept1*BR_indices]), stats.sem(durs[percept2*BR_indices]), stats.sem(durs[percept1*SFM_indices]), stats.sem(durs[percept2*SFM_indices]))
				
				my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'align': 'center'}
				
				N = 4
				ind = np.arange(N)  # the x locations for the groups
				width = 0.45       # the width of the bars
				
				# FIGURE 1
				fig = plt.figure(figsize=(10,6))
				ax = fig.add_subplot(111)
				rects1 = ax.bar(ind[0]+width, MEANS[0], width, yerr=SEMS[0], color='r', alpha = 0.75, **my_dict)
				rects2 = ax.bar(ind[1], MEANS[1], width, yerr=SEMS[1], color='r', alpha = 0.75, **my_dict)
				rects3 = ax.bar(ind[2], MEANS[2], width, yerr=SEMS[2], color='b', alpha = 0.75, **my_dict)
				rects4 = ax.bar(ind[3]-width, MEANS[3], width, yerr=SEMS[3], color='b', alpha = 0.75, **my_dict)
				# ax.set_ylim( (0.5) )
				simpleaxis(ax)
				spine_shift(ax)
				ax.set_ylabel('Duration', size = '16')
				ax.set_title(str(self.subject.initials) + ' - TIMES - run' + str(run) , size = '16')
				ax.set_xticks(( (ind[0]+width+ind[1])/2, (ind[2]+ind[3]-width)/2 ))
				ax.set_xticklabels( ('BR', 'SFM') )
				ax.tick_params(axis='x', which='major', labelsize=16)
				ax.tick_params(axis='y', which='major', labelsize=16)
				ax.set_ylim(ymax = round((plt.axis()[3]+0.5)*2.0, 0)/2 )
				
				pp = PdfPages(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'figure_PERCEPT_TIMES', extension = '.pdf'))
				fig.savefig(pp, format='pdf')
				pp.close()
	
	
	def createRegressors2(self, runArray, buttons):
		
		"""
		Create behaviour files for rest of sessions. Timestamps are based on frames.
		"""
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:
				
				this_run = runArray[run]['ID']
				print(runArray[run]['behaviorFile'])
				
				br_indices = runArray[run]['BR_indices']
				
				stim_dur1 = 150.0
				stim_dur2 = 150.0
				break_between_trials = 16.0
				
				file_to_open = str(runArray[run]['behaviorFile'])
				data = np.loadtxt(file_to_open)
				
				# Create time stamps associated to buttons:
				for i in range(data.shape[0]):
					if data[i,0] == 65:
						data[i,1] = (data[i-1,1] + data[i+1,1]) / 2.0
					if data[i,0] == 66:
						data[i,1] = (data[i-1,1] + data[i+1,1]) / 2.0
					if data[i,0] == 67:
						data[i,1] = (data[i-1,1] + data[i+1,1]) / 2.0
					if data[i,0] == 68:
						data[i,1] = (data[i-1,1] + data[i+1,1]) / 2.0
				
				# Remove frame time stamps:
				data = data[-(data[:,0]==10),:]
				
				# Remove TR time stamps:
				data = data[-(data[:,0]==49),:]
				
				# Remove dubble occurences (same two buttons sequentially):
				dubble_occurences_indices = zeros(data.shape[0], dtype=bool)
				for i in range(data.shape[0]):
					if data[i-1,0] == data[i,0]:
						dubble_occurences_indices[i] = True
				data = data[-dubble_occurences_indices,:]
				
				# Times:
				times = data[:,1] - data[0,1]
				
				# get last time stamp of run:
				# PLUS 350ms!!
				first_time1 = times[np.where((data[:,0] == 1))[0][0]]
				last_time1 = first_time1 + stim_dur1
				first_time2 = times[np.where((data[:,0] == 1))[0][1]]
				last_time2 = first_time2 + stim_dur2
				
				# durations:
				durs = np.diff(times)
				durs = np.concatenate((durs, np.diff([times[-1],last_time2])))
				
				# paradigm indicies:
				paradigm1_indices = (times >= first_time1) * (times <= last_time1)
				paradigm2_indices = (times >= first_time2) * (times <= last_time2)
				
				BR_indices = np.zeros(paradigm2_indices.shape[0], dtype = bool)
				SFM_indices = np.zeros(paradigm2_indices.shape[0], dtype = bool)
				
				if br_indices[0] == True:
					BR_indices = BR_indices + paradigm1_indices
				if br_indices[0] == False:
					SFM_indices = SFM_indices + paradigm1_indices
				if br_indices[1] == True:
					BR_indices = BR_indices + paradigm2_indices
				if br_indices[1] == False:
					SFM_indices = SFM_indices + paradigm2_indices
				
				# check that durs do not extend into 'break between trials:
				for i in range(sum(paradigm1_indices)):
					if times[paradigm1_indices][i] + durs[paradigm1_indices][i] >= last_time1:
						new_dur1 = (last_time1 - times[paradigm1_indices][i])
						index_new_dur1 = i
					else:
						new_dur1 = durs[paradigm1_indices][i]
						index_new_dur1 = i
				durs[1+index_new_dur1] = new_dur1
				
				# for i in range(sum(paradigm2_indices)):
				# 	if times[paradigm2_indices][i] + durs[paradigm2_indices][i] >= last_time2:
				# 		new_dur2 = (last_time2 - times[paradigm2_indices][i])
				# 		index_new_dur2 = i
				# 	else:
				# 		new_dur2 = durs[paradigm2_indices][i]
				# 		index_new_dur2 = i
				# durs[1+durs[paradigm1_indices].shape[0]+index_new_dur2] = new_dur2 
				
				# Print mean durations
				button_durations = np.zeros((4,2))
				button_durations[:,0] = np.array([65,66,67,68])
				button_durations[:,1] = np.array([mean(durs[data[:,0] == 65]), mean(durs[data[:,0] == 66]), mean(durs[data[:,0] == 67]), mean(durs[data[:,0] == 68])])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'button_durations', extension = '.txt'), button_durations, fmt='%3.2f', delimiter = '\t')
				
				# Get percept indices:
				percept1 = (data[:,0] == buttons['percept1'])
				percept2 = (data[:,0] == buttons['percept2'])
				transition = (data[:,0] == buttons['transition'])
				
				# Check for transitions without duration (no button press for transition)
				transition_no_dur = np.zeros(data.shape[0], dtype = bool)
				for i in range(data.shape[0]):
					try:
						if percept1[i-1] == True:
							if percept2[i] == True:
								transition_no_dur[i] = True
						if percept2[i-1] == True:
							if percept1[i] == True:
								transition_no_dur[i] = True
					except IndexError:
						pass
				
				############################################################
				############ MAKE TEXT FILES REGRESSORS ####################
				
				if br_indices.sum() == 2:
					# BR start:
					BR_start = np.zeros((2,3))
					BR_start[0,0] = times[BR_indices*paradigm1_indices][0]
					BR_start[1,0] = times[BR_indices*paradigm2_indices][0]
					BR_start[0,1] = stim_dur1
					BR_start[1,1] = stim_dur2
					BR_start[:,2] = np.ones(BR_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', extension = '.txt'), BR_start, fmt='%3.2f', delimiter = '\t')
					# BR transitions:
					BR_transition = np.zeros((sum(transition[BR_indices])+sum(transition_no_dur[BR_indices]),3))
					BR_transition[:sum(transition[BR_indices]),0] = times[transition*BR_indices]
					BR_transition[sum(transition[BR_indices]):,0] = times[transition_no_dur*BR_indices]
					BR_transition[:sum(transition[BR_indices]),1] = durs[transition*BR_indices]
					BR_transition[sum(transition[BR_indices]):,1] = 1.00
					BR_transition[:,2] = np.ones(BR_transition.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', postFix = ['trans'], extension = '.txt'), BR_transition, fmt='%3.2f', delimiter = '\t')
					# SFM start:
					SFM_start = np.zeros((1,3))
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', extension = '.txt'), SFM_start, fmt='%3.2f', delimiter = '\t')
					# SFM transitions:
					SFM_transition = np.zeros((1,3))
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', postFix = ['trans'], extension = '.txt'), SFM_transition, fmt='%3.2f', delimiter = '\t')
					
					##### for decoding: #####
					####################
					BR_percept1_indices = BR_indices*percept1
					BR_percept2_indices = BR_indices*percept2
					
					BR1_start = np.zeros((BR_percept1_indices.sum(),3))
					BR1_start[:,0] = times[BR_percept1_indices]
					BR1_start[:,1] = durs[BR_percept1_indices]
					BR1_start[:,2] = np.ones(BR1_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_BR1', extension = '.txt'), BR1_start, fmt='%3.2f', delimiter = '\t')
					BR2_start = np.zeros((BR_percept2_indices.sum(),3))
					BR2_start[:,0] = times[BR_percept2_indices]
					BR2_start[:,1] = durs[BR_percept2_indices]
					BR2_start[:,2] = np.ones(BR2_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_BR2', extension = '.txt'), BR2_start, fmt='%3.2f', delimiter = '\t')
					
					# Empty SFM texts:
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_SFM1', extension = '.txt'), np.zeros((1,3)), fmt='%3.2f', delimiter = '\t')
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_SFM2', extension = '.txt'), np.zeros((1,3)), fmt='%3.2f', delimiter = '\t')
					
					############################################################
					# Barplot of means:
					
					MEANS = (mean(durs[percept1*paradigm1_indices]), mean(durs[percept2*paradigm1_indices]), mean(durs[percept1*paradigm2_indices]), mean(durs[percept2*paradigm2_indices]))
					SEMS = (stats.sem(durs[percept1*paradigm1_indices]), stats.sem(durs[percept2*paradigm1_indices]), stats.sem(durs[percept1*paradigm2_indices]), stats.sem(durs[percept2*paradigm2_indices]))
					
					my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'align': 'center'}
					
					N = 4
					ind = np.arange(N)  # the x locations for the groups
					width = 0.45       # the width of the bars
					
					# FIGURE 1
					fig = plt.figure(figsize=(10,6))
					ax = fig.add_subplot(111)
					rects1 = ax.bar(ind[0]+width, MEANS[0], width, yerr=SEMS[0], color='r', alpha = 0.75, **my_dict)
					rects2 = ax.bar(ind[1], MEANS[1], width, yerr=SEMS[1], color='r', alpha = 0.75, **my_dict)
					rects3 = ax.bar(ind[2], MEANS[2], width, yerr=SEMS[2], color='r', alpha = 0.75, **my_dict)
					rects4 = ax.bar(ind[3]-width, MEANS[3], width, yerr=SEMS[3], color='r', alpha = 0.75, **my_dict)
					# ax.set_ylim( (0.5) )
					simpleaxis(ax)
					spine_shift(ax)
					ax.set_ylabel('Duration', size = '16')
					ax.set_title(str(self.subject.initials) + ' - TIMES - run' + str(run) , size = '16')
					ax.set_xticks(( (ind[0]+width+ind[1])/2, (ind[2]+ind[3]-width)/2 ))
					ax.set_xticklabels( ('BR', 'BR') )
					ax.tick_params(axis='x', which='major', labelsize=16)
					ax.tick_params(axis='y', which='major', labelsize=16)
					ax.set_ylim(ymax = round((plt.axis()[3]+0.5)*2.0, 0)/2 )
					
					pp = PdfPages(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'figure_PERCEPT_TIMES', extension = '.pdf'))
					fig.savefig(pp, format='pdf')
					pp.close()
				
				else:
					# BR start:
					BR_start = np.zeros((1,3))
					BR_start[:,0] = times[BR_indices][0]
					if this_run % 2 != 0:
						BR_start[:,1] = stim_dur1
					else:
						BR_start[:,1] = stim_dur2
					BR_start[:,2] = np.ones(BR_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', extension = '.txt'), BR_start, fmt='%3.2f', delimiter = '\t')
					# BR transitions:
					BR_transition = np.zeros((sum(transition[BR_indices])+sum(transition_no_dur[BR_indices]),3))
					BR_transition[:sum(transition[BR_indices]),0] = times[transition*BR_indices]
					BR_transition[sum(transition[BR_indices]):,0] = times[transition_no_dur*BR_indices]
					BR_transition[:sum(transition[BR_indices]),1] = durs[transition*BR_indices]
					BR_transition[sum(transition[BR_indices]):,1] = 1.00
					BR_transition[:,2] = np.ones(BR_transition.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', postFix = ['trans'], extension = '.txt'), BR_transition, fmt='%3.2f', delimiter = '\t')
					# SFM start:
					SFM_start = np.zeros((1,3))
					SFM_start[:,0] = times[SFM_indices][0]
					if this_run % 2 != 0:
						SFM_start[:,1] = stim_dur2
					else:
						SFM_start[:,1] = stim_dur1
					SFM_start[:,2] = np.ones(SFM_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', extension = '.txt'), SFM_start, fmt='%3.2f', delimiter = '\t')
					# SFM transitions:
					SFM_transition = np.zeros((sum(transition[SFM_indices])+sum(transition_no_dur[SFM_indices]),3))
					SFM_transition[:sum(transition[SFM_indices]),0] = times[transition*SFM_indices]
					SFM_transition[sum(transition[SFM_indices]):,0] = times[transition_no_dur*SFM_indices]
					SFM_transition[:sum(transition[SFM_indices]),1] = durs[transition*SFM_indices]
					SFM_transition[sum(transition[SFM_indices]):,1] = 1.00
					SFM_transition[:,2] = np.ones(SFM_transition.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', postFix = ['trans'], extension = '.txt'), SFM_transition, fmt='%3.2f', delimiter = '\t')
					
					##### for decoding: #####
					####################
					BR_percept1_indices = BR_indices*percept1
					BR_percept2_indices = BR_indices*percept2
					SFM_percept1_indices = SFM_indices*percept1
					SFM_percept2_indices = SFM_indices*percept2
					
					BR1_start = np.zeros((BR_percept1_indices.sum(),3))
					BR1_start[:,0] = times[BR_percept1_indices]
					BR1_start[:,1] = durs[BR_percept1_indices]
					BR1_start[:,2] = np.ones(BR1_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_BR1', extension = '.txt'), BR1_start, fmt='%3.2f', delimiter = '\t')
					BR2_start = np.zeros((BR_percept2_indices.sum(),3))
					BR2_start[:,0] = times[BR_percept2_indices]
					BR2_start[:,1] = durs[BR_percept2_indices]
					BR2_start[:,2] = np.ones(BR2_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_BR2', extension = '.txt'), BR2_start, fmt='%3.2f', delimiter = '\t')
					SFM1_start = np.zeros((SFM_percept1_indices.sum(),3))
					SFM1_start[:,0] = times[SFM_percept1_indices]
					SFM1_start[:,1] = durs[SFM_percept1_indices]
					SFM1_start[:,2] = np.ones(SFM1_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_SFM1', extension = '.txt'), SFM1_start, fmt='%3.2f', delimiter = '\t')
					SFM2_start = np.zeros((SFM_percept2_indices.sum(),3))
					SFM2_start[:,0] = times[SFM_percept2_indices]
					SFM2_start[:,1] = durs[SFM_percept2_indices]
					SFM2_start[:,2] = np.ones(SFM2_start.shape[0])
					np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_SFM2', extension = '.txt'), SFM2_start, fmt='%3.2f', delimiter = '\t')
					
					############################################################
					# Barplot of means:
					
					MEANS = (mean(durs[percept1*BR_indices]), mean(durs[percept2*BR_indices]), mean(durs[percept1*SFM_indices]), mean(durs[percept2*SFM_indices]))
					SEMS = (stats.sem(durs[percept1*BR_indices]), stats.sem(durs[percept2*BR_indices]), stats.sem(durs[percept1*SFM_indices]), stats.sem(durs[percept2*SFM_indices]))
					
					my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'align': 'center'}
					
					N = 4
					ind = np.arange(N)  # the x locations for the groups
					width = 0.45       # the width of the bars
					
					# FIGURE 1
					fig = plt.figure(figsize=(10,6))
					ax = fig.add_subplot(111)
					rects1 = ax.bar(ind[0]+width, MEANS[0], width, yerr=SEMS[0], color='r', alpha = 0.75, **my_dict)
					rects2 = ax.bar(ind[1], MEANS[1], width, yerr=SEMS[1], color='r', alpha = 0.75, **my_dict)
					rects3 = ax.bar(ind[2], MEANS[2], width, yerr=SEMS[2], color='b', alpha = 0.75, **my_dict)
					rects4 = ax.bar(ind[3]-width, MEANS[3], width, yerr=SEMS[3], color='b', alpha = 0.75, **my_dict)
					# ax.set_ylim( (0.5) )
					simpleaxis(ax)
					spine_shift(ax)
					ax.set_ylabel('Duration', size = '16')
					ax.set_title(str(self.subject.initials) + ' - TIMES - run' + str(run) , size = '16')
					ax.set_xticks(( (ind[0]+width+ind[1])/2, (ind[2]+ind[3]-width)/2 ))
					ax.set_xticklabels( ('BR', 'SFM') )
					ax.tick_params(axis='x', which='major', labelsize=16)
					ax.tick_params(axis='y', which='major', labelsize=16)
					ax.set_ylim(ymax = round((plt.axis()[3]+0.5)*2.0, 0)/2 )
					
					pp = PdfPages(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'figure_PERCEPT_TIMES', extension = '.pdf'))
					fig.savefig(pp, format='pdf')
					pp.close()
		
	
	
	def createRegressorsBlinks(self, runArray):
		
		stim_dur = 150.0
		pauze_between = 16.0
		pauze_end = 8.0
		sample_freq = 500.0
		downsample_freq = 10.0
		blinks_z_threshold = 2.5
		
		# rd = np.loadtxt('/Research/7T_RIVALRY/data/TK/TK_250113/raw/hr/SCANPHYSLOG20130125130453.log')
		
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:
				file_to_open = str(runArray[run]['hrFile'])
				rd = np.loadtxt(file_to_open) 
				srd = rd[:,[2,3,4]] / rd[:,[2,3,4]].std(axis=0)
				
				rivalry1_on = srd.shape[0] - (pauze_end*sample_freq) - (stim_dur*sample_freq) - (pauze_between*sample_freq) - (stim_dur*sample_freq)
				rivalry1_off = srd.shape[0] - (pauze_end*sample_freq) - (stim_dur*sample_freq) - (pauze_between*sample_freq)
				rivalry2_on = srd.shape[0] - (pauze_end*sample_freq) - (stim_dur*sample_freq)
				rivalry2_off = srd.shape[0] - (pauze_end*sample_freq)
				
				# Do ICA on EOG signal in pauze between stimuli. We expect many blinks here!
				srd_for_ica = srd[rivalry1_off:rivalry2_on,:]
				ica = FastICA()
				S_ = ica.fit(srd_for_ica).transform(srd_for_ica)
				A_ = ica.get_mixing_matrix()
				
				# Multiply mixing matrix (as based on pauze between stimuli) with the original signal: 
				ICA_signal = np.dot(srd,A_)
				
				# Select the right component:
				ICA_blink_comp = ICA_signal[:,2]
				
				# Z-score
				ICA_blink_comp_z = (ICA_blink_comp-np.mean(ICA_blink_comp)) / np.std(ICA_blink_comp)
				
				fig = figure(figsize=(12,6))
				ax = fig.add_subplot(111)
				plot(ICA_blink_comp_z[::100])
				axhline(2.5, alpha = 0.25)
				simpleaxis(ax)
				spine_shift(ax)
				
				# Detect blinks
				blink_indices = np.array((ICA_blink_comp_z > blinks_z_threshold), dtype = bool)
				
				plot(blink_indices[::100], color = 'r')
				
				blink_times = np.where(np.diff(blink_indices) == 1)[0]
				blink_start_times = blink_times[::2]
				blink_end_times = blink_times[1::2]
				
				pp = PdfPages(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'figure_BLINKS', extension = '.pdf'))
				fig.savefig(pp, format='pdf')
				pp.close()
				
				
				
				############################################################
				############ MAKE TEXT FILES REGRESSORS ####################
				# Blink start:
				blink_start = np.zeros((blink_start_times.shape[0],3))
				blink_start[:,0] = blink_start_times
				blink_start[:,1] = 0.1
				blink_start[:,2] = np.ones(blink_start_times.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_blinks', extension = '.txt'), blink_start, fmt='%3.2f', delimiter = '\t')
	
	
	def runAllGLMS(self):
		"""
		Take all transition events and use them as event regressors
		Run FSL on this
		"""
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:
				
				# # self.runTransitionGLM(run=r)
				# 						# create the event files
				# 						for eventType in ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray']:
				# 							eventData = self.gatherBehavioralData( whichEvents = [eventType], whichRuns = [run] )
				# 							eventName = eventType.split('EventsAsArray')[0]
				# 							dfFSL = np.ones((eventData[eventType].shape[0],3)) * [1.0,0.1,1]
				# 							dfFSL[:,0] = eventData[eventType][:,0]
				# 							np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = eventName, extension = '.evt'), dfFSL, fmt='%4.2f')
				# 							# also make files for the end of each event.
				# 							dfFSL[:,0] = eventData[eventType][:,0] + eventData[eventType][:,1]
				# 							np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = eventName, postFix = ['end'], extension = '.evt'), dfFSL, fmt='%4.2f')
				
				# remove previous feat directories
				try:
					self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.fsf'))
				except OSError:
					pass
				
				# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
				thisFeatFile = '/Research/7T_RIVALRY/analysis/fsf/7T_RIVALRY.fsf'
				REDict = {
				'---OUTPUT_DIR---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'rivalry', postFix = ['mcf']),
				'---NR_FRAMES---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'])).timepoints),
				'---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf']), 
				'---ANAT_FILE---':os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'bet', 'T1_bet' ), 
				'---BR_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', extension = '.txt'),
				'---BR_TRANS_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', postFix = ['trans'], extension = '.txt'),
				'---SFM_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', extension = '.txt'),
				'---SFM_TRANS_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', postFix = ['trans'], extension = '.txt')
				}
				featFileName = self.runFile(stage = 'processed/mri', run = self.runList[run], extension = '.fsf')
				featOp = FEATOperator(inputObject = thisFeatFile)
				
				# # In serial:
				# featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
				# # In parallel:
				featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
				
				# run feat
				featOp.execute()
		
	
	
	def create_decoding_indices_BR(self, runArray, trans_weight, run_type = 'rivalry', postFix = ['mcf'], subSamplingRatio = 100.0):
		
		# stim_dur1 = 142.0
		stim_dur1 = 150.0
		stim_dur2 = 150.0
		pauze1 = 8.0
		break_between_trials = 16.0
		pauze2 = 8.0
		
		for r in [self.runList[i] for i in self.conditionDict[run_type]]:
			
			this_run = r.ID
			
			# check out the duration of this run:
			niiFile = NiftiImage( self.runFile(stage = 'processed/mri', run = r) )
			tr, nr_trs = niiFile.rtime, niiFile.timepoints
			tr = 0.6499999761581421
			run_duration = tr * nr_trs
			
			self.subSamplingRatio = subSamplingRatio
			len_run = round(run_duration * self.subSamplingRatio,0)
			
			print(this_run)
			print(len_run)
			
			try:
				del(deco_BR1); del(deco_BR2); del(deco_trans_BR)
			except NameError:
				pass
			
			# Load behavior file:
			behaviorFile = np.loadtxt(r.behaviorFile)
			
			# Load text files:
			deco_BR1 = np.loadtxt(self.runFile( stage='processed/mri', run=r, base='deco_BR1', extension='.txt') )
			deco_BR2 = np.loadtxt(self.runFile( stage='processed/mri', run=r, base='deco_BR2', extension='.txt') )
			deco_trans_BR = np.loadtxt(self.runFile( stage='processed/mri', run=r, base='txt_BR', postFix=['trans'], extension='.txt'))
			
			# Convolve with HRF:
			regr_BR1_collapsed, regr_BR1_convolved_collapsed = self.createRegressors(deco_BR1, len_run, self.subSamplingRatio)
			regr_BR2_collapsed, regr_BR2_convolved_collapsed = self.createRegressors(deco_BR2, len_run, self.subSamplingRatio)
			regr_BR2_convolved_collapsed = regr_BR2_convolved_collapsed*-1
			regr_trans_BR_collapsed, regr_trans_BR_convolved_collapsed = self.createRegressors(deco_trans_BR, len_run, self.subSamplingRatio)
			
			# Total activity:
			total_activity = abs(regr_BR1_convolved_collapsed) + abs(regr_BR2_convolved_collapsed)
			# Total activity transitions:
			total_trans_activity = (regr_trans_BR_convolved_collapsed) * trans_weight
			# Total activity percept1 minus percept2:
			total_minus_activity = (abs(regr_BR1_convolved_collapsed) - abs(regr_BR2_convolved_collapsed))
			
			#### DECODING INDICES ########
			
			# # decoding indices based on transitions:
			decoding_indices = (total_activity > total_trans_activity) 
			# # decoding indices based on difference BOLD activity percept 1 and percept 2:
			decoding_indices = (((abs(regr_BR1_convolved_collapsed) - abs(regr_BR2_convolved_collapsed)) > 200) + ((abs(regr_BR2_convolved_collapsed) - abs(regr_BR1_convolved_collapsed)) > 200)) #* decoding_indices
			decoding_indices[(pauze1+stim_dur1)*self.subSamplingRatio:(pauze1+stim_dur1+break_between_trials)*self.subSamplingRatio] = False
			decoding_percept1_indices_BR = (abs(regr_BR1_convolved_collapsed) > abs(regr_BR2_convolved_collapsed)) * decoding_indices
			decoding_percept2_indices_BR = (abs(regr_BR2_convolved_collapsed) > abs(regr_BR1_convolved_collapsed)) * decoding_indices
			
			###
			decoding_indices2 = np.array(decoding_indices, dtype = int)
			decoding_on = np.where(np.diff(decoding_indices2)==1)[0]
			decoding_off = np.where(np.diff(decoding_indices2)==-1)[0]
			if len(decoding_on) != len(decoding_off):
				extra = np.array([len(decoding_indices)-1])
				decoding_off = np.concatenate((decoding_off, extra))
			
			decoding_indices_perc1 = np.array(decoding_percept1_indices_BR, dtype = int)
			decoding_perc1_on = np.where(np.diff(decoding_indices_perc1)==1)[0]
			decoding_perc1_off = np.where(np.diff(decoding_indices_perc1)==-1)[0]
			if len(decoding_perc1_on) != len(decoding_perc1_off):
				extra = np.array([len(decoding_percept1_indices_BR)-1])
				decoding_perc1_off = np.concatenate((decoding_perc1_off, extra))
				
			decoding_indices_perc2 = np.array(decoding_percept2_indices_BR, dtype = int)
			decoding_perc2_on = np.where(np.diff(decoding_indices_perc2)==1)[0]
			decoding_perc2_off = np.where(np.diff(decoding_indices_perc2)==-1)[0]
			if len(decoding_perc2_on) != len(decoding_perc2_off):
				extra = np.array([len(decoding_percept2_indices_BR)-1])
				decoding_perc2_off = np.concatenate((decoding_perc2_off, extra))
			
			percept1 = np.array((regr_BR1_collapsed), dtype = int)
			percept1_on = np.where(np.diff(percept1)==1)[0]
			percept1_off = np.where(np.diff(percept1)==-1)[0]
			if len(percept1_on) != len(percept1_off):
				extra = np.array([len(decoding_indices)-1])
				percept1_off = np.concatenate((percept1_off, extra))
			
			percept2 = np.array((regr_BR2_collapsed), dtype = int)
			percept2_on = np.where(np.diff(percept2)==1)[0]
			percept2_off = np.where(np.diff(percept2)==-1)[0]
			if len(percept2_on) != len(percept2_off):
				extra = np.array([len(decoding_indices)-1])
				percept2_off = np.concatenate((percept2_off, extra))
			
			##############################################
			
			# FIGURE
			fig = figure(figsize=(30,6))
			ax = subplot(311)
			axvspan((pauze1+stim_dur1)*self.subSamplingRatio,(pauze1+stim_dur1+break_between_trials)*self.subSamplingRatio, facecolor='k', alpha=0.75)
			for i in range(len(percept1_on)):
				axvspan(percept1_on[i], percept1_off[i], facecolor='g', alpha=0.25)
			for i in range(len(percept2_on)):
				axvspan(percept2_on[i], percept2_off[i], facecolor='r', alpha=0.25)
			plot(regr_BR1_convolved_collapsed, color = 'g')
			plot(regr_BR2_convolved_collapsed, color = 'r')
			axhline(0, alpha = 0.5)
			simpleaxis(ax)
			spine_shift(ax)
			title('Bincolar Rivalry - run ' + str(this_run), size=16)
			legend(('percept1', 'percept2'))
			ylabel('predicted activity')
			ax = subplot(312)
			axvspan((pauze1+stim_dur1)*self.subSamplingRatio,(pauze1+stim_dur1+break_between_trials)*self.subSamplingRatio, facecolor='k', alpha=0.75)
			for i in range(len(decoding_on)):
				axvspan(decoding_on[i], decoding_off[i], facecolor='k', alpha=0.25)
			plot(total_activity, color = 'b')
			plot(total_trans_activity, color = 'b', linestyle = '--', alpha = 0.5)
			ylim((0,400))
			simpleaxis(ax)
			spine_shift(ax)
			xlabel('time')
			ylabel('predicted activity')
			legend(('total percept activity', 'transition activity'))
			ax = subplot(313)
			axvspan((pauze1+stim_dur1)*self.subSamplingRatio,(pauze1+stim_dur1+break_between_trials)*self.subSamplingRatio, facecolor='k', alpha=0.75)
			for i in range(len(decoding_perc1_on)):
				axvspan(decoding_perc1_on[i], decoding_perc1_off[i], facecolor='g', alpha=0.25)
			for i in range(len(decoding_perc2_on)):
				axvspan(decoding_perc2_on[i], decoding_perc2_off[i], facecolor='r', alpha=0.25)
			plot(abs(total_minus_activity), color = 'k')
			# plot(total_trans_activity, color = 'b', linestyle = '--', alpha = 0.5)
			simpleaxis(ax)
			spine_shift(ax)
			xlabel('time')
			ylabel('predicted activity')
			legend(('abs(percept 1 minus percept 2)',))
			
			pp = PdfPages(self.runFile( stage = 'processed/mri', run = r, base = 'figure_PERCEPTS_BR', extension = '.pdf'))
			fig.savefig(pp, format='pdf')
			pp.close()
			
			#############################################
			
			# get timestamps TR's
			number_trs_in_behaviorFile = sum(behaviorFile[:,0] == 49)
			number_trs_before_first_tr = round( (behaviorFile[1,1] - behaviorFile[0,1]) / tr )
			timestamps_TRs = np.zeros(nr_trs)
			timestamps_TRs[-number_trs_in_behaviorFile:] = behaviorFile[behaviorFile[:,0] == 49,1] - behaviorFile[0,1]
			# timestamps_TRs[number_trs_before_first_tr-1:number_trs_before_first_tr-1+number_trs_in_behaviorFile] = behaviorFile[behaviorFile[:,0] == 49,1] - behaviorFile[0,1]
			
			# resample decoding times to 'real time':
			decoding_perc1_on = decoding_perc1_on / subSamplingRatio
			decoding_perc1_off = decoding_perc1_off / subSamplingRatio
			decoding_perc2_on = decoding_perc2_on / subSamplingRatio
			decoding_perc2_off = decoding_perc2_off / subSamplingRatio
			
			# DECODING INDICES:
			decoding_BR = np.zeros(nr_trs)
			for i in range(decoding_perc1_on.shape[0]):
				decoding_BR[((timestamps_TRs >= decoding_perc1_on[i]) * (timestamps_TRs <= decoding_perc1_off[i]))] = -1
			for i in range(decoding_perc2_on.shape[0]):
				decoding_BR[((timestamps_TRs >= decoding_perc2_on[i]) * (timestamps_TRs <= decoding_perc2_off[i]))] = 1
			
			# get on and offset percepts, based on decoding indices (as in figure above):
			timestamps_percept_decoding_on = sort(np.concatenate((decoding_perc1_on, decoding_perc2_on)))
			timestamps_percept_decoding_off = sort(np.concatenate((decoding_perc1_off, decoding_perc2_off)))
			
			# get timestamps Tr's in percept
			decoding_BR_times = np.zeros((nr_trs, 5))
			decoding_BR_times[:,0] = decoding_BR
			decoding_BR_times[:,1] = timestamps_TRs
			for i in range(timestamps_percept_decoding_on.shape[0]):
				these_TR_indices = (timestamps_TRs >= timestamps_percept_decoding_on[i]) * (timestamps_TRs <= timestamps_percept_decoding_off[i])
				for j in range(these_TR_indices.shape[0]):
					if these_TR_indices[j] == True:
						decoding_BR_times[j,2] = timestamps_TRs[j] - timestamps_percept_decoding_on[i]
						decoding_BR_times[j,3] = timestamps_percept_decoding_off[i] - timestamps_TRs[j]
						decoding_BR_times[j,4] = (timestamps_TRs[j] - timestamps_percept_decoding_on[i]) / (timestamps_percept_decoding_off[i] - timestamps_percept_decoding_on[i])
					else:
						pass
			
			# add to HDF5:
			self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
			h5file = openFile(self.hdf5_filename, mode = 'r+', title = run_type + " file")
			
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
			
			decoding_group_name = 'decoding_indices'
			
			try:
				decodingGroup = h5file.getNode(where = thisRunGroup, name = decoding_group_name, classname='Group')
				self.logger.info('data file ' + decoding_group_name + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + decoding_group_name + ' to this file')
				decodingGroup = h5file.createGroup(thisRunGroup, decoding_group_name, 'indices for decoding')
			
			try:
				decodingGroup.decoding_BR.remove()
			except NodeError, LeafError:
				pass
			h5file.createArray(decodingGroup, 'decoding_BR', decoding_BR_times, 'decoding indices BR, plus times: (1) timestamp TR, (2) time since percept onset, (3) time since percept offset, (4) relative time')
			h5file.close()
	
	
	def create_decoding_indices_SFM(self, runArray, trans_weight, run_type = 'rivalry', postFix = ['mcf'], subSamplingRatio = 100.0):
		
		# stim_dur1 = 142.0
		stim_dur1 = 150.0
		stim_dur2 = 150.0
		pauze1 = 8.0
		break_between_trials = 16.0
		pauze2 = 8.0
		
		for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
			
			this_run = r.ID
			
			# check out the duration of this run:
			niiFile = NiftiImage( self.runFile(stage = 'processed/mri', run = r) )
			tr, nr_trs = niiFile.rtime, niiFile.timepoints
			tr = 0.6499999761581421
			run_duration = tr * nr_trs
			
			self.subSamplingRatio = subSamplingRatio
			len_run = round(run_duration * self.subSamplingRatio,0)
			
			print(this_run)
			print(len_run)
			
			try:
				del(deco_BR1); del(deco_BR2); del(deco_trans_BR)
			except NameError:
				pass
			
			# Load behavior file:
			behaviorFile = np.loadtxt(r.behaviorFile)
			
			# Load text files:
			deco_SFM1 = np.loadtxt(self.runFile( stage='processed/mri', run=r, base='deco_SFM1', extension='.txt') )
			deco_SFM2 = np.loadtxt(self.runFile( stage='processed/mri', run=r, base='deco_SFM2', extension='.txt') )
			deco_trans_SFM = np.loadtxt(self.runFile( stage='processed/mri', run=r, base='txt_SFM', postFix=['trans'], extension='.txt'))
				
			if deco_SFM1.sum() > 0:
				# Convolve with HRF:
				regr_SFM1_collapsed, regr_SFM1_convolved_collapsed = self.createRegressors(deco_SFM1, len_run, self.subSamplingRatio)
				regr_SFM2_collapsed, regr_SFM2_convolved_collapsed = self.createRegressors(deco_SFM2, len_run, self.subSamplingRatio)
				regr_SFM2_convolved_collapsed = regr_SFM2_convolved_collapsed*-1
				regr_trans_SFM_collapsed, regr_trans_SFM_convolved_collapsed = self.createRegressors(deco_trans_SFM, len_run, self.subSamplingRatio)
				
				# Total activity:
				total_activity = abs(regr_SFM1_convolved_collapsed) + abs(regr_SFM2_convolved_collapsed)
				# Total activity transitions:
				total_trans_activity = (regr_trans_SFM_convolved_collapsed) * trans_weight
				# Total activity percept1 minus percept2:
				total_minus_activity = (abs(regr_SFM1_convolved_collapsed) - abs(regr_SFM2_convolved_collapsed))
				
				#### DECODING INDICES ########
				
				# # decoding indices based on transitions:
				# decoding_indices = (total_activity > total_trans_activity) 
				# # decoding indices based on difference BOLD activity percept 1 and percept 2:
				decoding_indices = ((abs(regr_SFM1_convolved_collapsed) - abs(regr_SFM2_convolved_collapsed)) > 200) + ((abs(regr_SFM2_convolved_collapsed) - abs(regr_SFM1_convolved_collapsed)) > 200)
				decoding_indices[(pauze1+stim_dur1)*self.subSamplingRatio:(pauze1+stim_dur1+break_between_trials)*self.subSamplingRatio] = False
				decoding_percept1_indices_SFM = (abs(regr_SFM1_convolved_collapsed) > abs(regr_SFM2_convolved_collapsed)) * decoding_indices
				decoding_percept2_indices_SFM = (abs(regr_SFM2_convolved_collapsed) > abs(regr_SFM1_convolved_collapsed)) * decoding_indices
				
				###
				decoding_indices2 = np.array(decoding_indices, dtype = int)
				decoding_on = np.where(np.diff(decoding_indices2)==1)[0]
				decoding_off = np.where(np.diff(decoding_indices2)==-1)[0]
				if len(decoding_on) != len(decoding_off):
					extra = np.array([len(decoding_indices)-1])
					decoding_off = np.concatenate((decoding_off, extra))
				
				decoding_indices_perc1 = np.array(decoding_percept1_indices_SFM, dtype = int)
				decoding_perc1_on = np.where(np.diff(decoding_indices_perc1)==1)[0]
				decoding_perc1_off = np.where(np.diff(decoding_indices_perc1)==-1)[0]
				if len(decoding_perc1_on) != len(decoding_perc1_off):
					extra = np.array([len(decoding_percept1_indices_SFM)-1])
					decoding_perc1_off = np.concatenate((decoding_perc1_off, extra))
				
				decoding_indices_perc2 = np.array(decoding_percept2_indices_SFM, dtype = int)
				decoding_perc2_on = np.where(np.diff(decoding_indices_perc2)==1)[0]
				decoding_perc2_off = np.where(np.diff(decoding_indices_perc2)==-1)[0]
				if len(decoding_perc2_on) != len(decoding_perc2_off):
					extra = np.array([len(decoding_percept2_indices_SFM)-1])
					decoding_perc2_off = np.concatenate((decoding_perc2_off, extra))
				
				percept1 = np.array((regr_SFM1_collapsed), dtype = int)
				percept1_on = np.where(np.diff(percept1)==1)[0]
				percept1_off = np.where(np.diff(percept1)==-1)[0]
				if len(percept1_on) != len(percept1_off):
					extra = np.array([len(decoding_indices)-1])
					percept1_off = np.concatenate((percept1_off, extra))
				
				percept2 = np.array((regr_SFM2_collapsed), dtype = int)
				percept2_on = np.where(np.diff(percept2)==1)[0]
				percept2_off = np.where(np.diff(percept2)==-1)[0]
				if len(percept2_on) != len(percept2_off):
					extra = np.array([len(decoding_indices)-1])
					percept2_off = np.concatenate((percept2_off, extra))
				
				##############################################
				
				# FIGURE
				fig = figure(figsize=(30,6))
				ax = subplot(311)
				axvspan((pauze1+stim_dur1)*self.subSamplingRatio,(pauze1+stim_dur1+break_between_trials)*self.subSamplingRatio, facecolor='k', alpha=0.75)
				for i in range(len(percept1_on)):
					axvspan(percept1_on[i], percept1_off[i], facecolor='g', alpha=0.25)
				for i in range(len(percept2_on)):
					axvspan(percept2_on[i], percept2_off[i], facecolor='r', alpha=0.25)
				plot(regr_SFM1_convolved_collapsed, color = 'g')
				plot(regr_SFM2_convolved_collapsed, color = 'r')
				axhline(0, alpha = 0.5)
				simpleaxis(ax)
				spine_shift(ax)
				title('Structure from Motion - run ' + str(this_run), size=16)
				legend(('percept1', 'percept2'))
				ylabel('predicted activity')
				ax = subplot(312)
				axvspan((pauze1+stim_dur1)*self.subSamplingRatio,(pauze1+stim_dur1+break_between_trials)*self.subSamplingRatio, facecolor='k', alpha=0.75)
				for i in range(len(decoding_on)):
					axvspan(decoding_on[i], decoding_off[i], facecolor='k', alpha=0.25)
				plot(total_activity, color = 'b')
				plot(total_trans_activity, color = 'b', linestyle = '--', alpha = 0.5)
				ylim((0,400))
				simpleaxis(ax)
				spine_shift(ax)
				xlabel('time')
				ylabel('predicted activity')
				legend(('total percept activity', 'transition activity'))
				ax = subplot(313)
				axvspan((pauze1+stim_dur1)*self.subSamplingRatio,(pauze1+stim_dur1+break_between_trials)*self.subSamplingRatio, facecolor='k', alpha=0.75)
				for i in range(len(decoding_perc1_on)):
					axvspan(decoding_perc1_on[i], decoding_perc1_off[i], facecolor='g', alpha=0.25)
				for i in range(len(decoding_perc2_on)):
					axvspan(decoding_perc2_on[i], decoding_perc2_off[i], facecolor='r', alpha=0.25)
				plot(abs(total_minus_activity), color = 'k')
				# plot(total_trans_activity, color = 'b', linestyle = '--', alpha = 0.5)
				simpleaxis(ax)
				spine_shift(ax)
				xlabel('time')
				ylabel('predicted activity')
				legend(('abs(percept 1 minus percept 2)',))
				
				pp = PdfPages(self.runFile( stage = 'processed/mri', run = r, base = 'figure_PERCEPTS_SFM', extension = '.pdf'))
				fig.savefig(pp, format='pdf')
				pp.close()
			
			#############################################
			
			# get timestamps TR's
			number_trs_in_behaviorFile = sum(behaviorFile[:,0] == 49)
			number_trs_before_first_tr = round( (behaviorFile[1,1] - behaviorFile[0,1]) / tr )
			timestamps_TRs = np.zeros(nr_trs)
			timestamps_TRs[-number_trs_in_behaviorFile:] = behaviorFile[behaviorFile[:,0] == 49,1] - behaviorFile[0,1]
			# timestamps_TRs[number_trs_before_first_tr-1:number_trs_before_first_tr-1+number_trs_in_behaviorFile] = behaviorFile[behaviorFile[:,0] == 49,1] - behaviorFile[0,1]
			
			# resample decoding times to 'real time':
			decoding_perc1_on = decoding_perc1_on / subSamplingRatio
			decoding_perc1_off = decoding_perc1_off / subSamplingRatio
			decoding_perc2_on = decoding_perc2_on / subSamplingRatio
			decoding_perc2_off = decoding_perc2_off / subSamplingRatio
			
			# DECODING INDICES:
			decoding_SFM = np.zeros(nr_trs)
			for i in range(decoding_perc1_on.shape[0]):
				decoding_SFM[((timestamps_TRs >= decoding_perc1_on[i]) * (timestamps_TRs <= decoding_perc1_off[i]))] = -1
			for i in range(decoding_perc2_on.shape[0]):
				decoding_SFM[((timestamps_TRs >= decoding_perc2_on[i]) * (timestamps_TRs <= decoding_perc2_off[i]))] = 1
			
			# get on and offset percepts, based on decoding indices (as in figure above):
			timestamps_percept_decoding_on = sort(np.concatenate((decoding_perc1_on, decoding_perc2_on)))
			timestamps_percept_decoding_off = sort(np.concatenate((decoding_perc1_off, decoding_perc2_off)))
			
			# get timestamps Tr's in percept
			decoding_SFM_times = np.zeros((nr_trs, 5))
			decoding_SFM_times[:,0] = decoding_SFM
			decoding_SFM_times[:,1] = timestamps_TRs
			for i in range(timestamps_percept_decoding_on.shape[0]):
				these_TR_indices = (timestamps_TRs >= timestamps_percept_decoding_on[i]) * (timestamps_TRs <= timestamps_percept_decoding_off[i])
				for j in range(these_TR_indices.shape[0]):
					if these_TR_indices[j] == True:
						decoding_SFM_times[j,2] = timestamps_TRs[j] - timestamps_percept_decoding_on[i]
						decoding_SFM_times[j,3] = timestamps_percept_decoding_off[i] - timestamps_TRs[j]
						decoding_SFM_times[j,4] = (timestamps_TRs[j] - timestamps_percept_decoding_on[i]) / (timestamps_percept_decoding_off[i] - timestamps_percept_decoding_on[i])
					else:
						pass
			
			self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
			h5file = openFile(self.hdf5_filename, mode = 'r+', title = run_type + " file")
			
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
			
			decoding_group_name = 'decoding_indices'
			
			try:
				decodingGroup = h5file.getNode(where = thisRunGroup, name = decoding_group_name, classname='Group')
				self.logger.info('data file ' + decoding_group_name + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + decoding_group_name + ' to this file')
				decodingGroup = h5file.createGroup(thisRunGroup, decoding_group_name, 'indices for decoding')
				
			try:
				decodingGroup.decoding_SFM.remove()
			except NodeError, LeafError:
				pass
			h5file.createArray(decodingGroup, 'decoding_SFM', decoding_SFM_times, 'decoding indices SFM, plus times: (1) timestamp TR, (2) time since percept onset, (3) time since percept offset, (4) relative time')
			h5file.close()
	
	
	def decoding_plain(self, runArray, decoding, roi, mask_type='STIM_Z', mask_direction='pos', threshold=None, number_voxels=None, split_by_relative_time=None, polar_eccen=False):
		
		self.setup_all_data_for_decoding(runArray, decoding, roi, mask_type=mask_type, mask_direction=mask_direction, threshold=threshold, number_voxels=number_voxels, split_by_relative_time=split_by_relative_time, polar_eccen=polar_eccen)
		
		if decoding == 'BR':
			print('BR decoding!' + ' --- ' + str(split_by_relative_time) + ' in percept')
		if decoding == 'SFM':
			print('SFM decoding!' + ' --- ' + str(split_by_relative_time) + ' in percept')
		print(str(roi) + ', number of features (voxels) = ' + str(self.X.shape[1]) )
		print('number of training samples (TRs) = ' + str(round(self.X.shape[0]*0.8)))
		print('number of test samples (TRs) = ' + str(round(self.X.shape[0]*0.2)))
		
		accuracy = []
		for i in range(50):
			# Balance classes:
			
			Xa = self.X[self.y == 1,:]
			Xb = self.X[self.y == -1,:]
			ya = self.y[self.y == 1]
			yb = self.y[self.y == -1]
		
			import random
			
			random_indices = random.sample(range(len(Xa)), Xb.shape[0])
			Xa = Xa[random_indices,:]
			ya = ya[random_indices]
		
			X = np.vstack((Xa,Xb))
			y = np.concatenate((ya,yb))
			
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
			# print('sample_weight = ' + str(sample_weight))
			clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
					gamma=0.001, kernel='linear', max_iter=-1, probability=False, shrinking=True,
					tol=0.001, verbose=False)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
			accuracy.append( sum(predictions == y_test) / float(len(predictions)) )
			
			print('accuracy = ' + str(accuracy[i]))
		
		print('')
		print('')
		print('mean accuracy = ' + str(mean(accuracy)))
	
	
	def decoding_complex(self, runArray, decoding, roi, mask_type='STIM_Z', mask_direction='pos', threshold=None, number_voxels=None, split_by_relative_time=None, polar_eccen=False, lag=0):
		
		self.setup_all_data_for_decoding(runArray, decoding, roi, mask_type=mask_type, mask_direction=mask_direction, threshold=threshold, number_voxels=number_voxels, split_by_relative_time=split_by_relative_time, polar_eccen=polar_eccen)
		
		lag = lag
		phases = np.concatenate([positivePhases(self.eccen_data_c[9,:]-lag)-2*pi, positivePhases(self.eccen_data_c[9,:]-lag), positivePhases(self.eccen_data_c[9,:]-lag)+2*pi])
		zs = np.concatenate([self.mask_data_mean,self.mask_data_mean,self.mask_data_mean])
		phase_order = np.argsort(phases)
		kern =  norm.pdf( np.linspace(-2.25,2.25,200) )
		sm_phases = np.convolve( phases[phase_order], kern / kern.sum(), 'valid' )
		sm_zs = np.convolve( zs[phase_order], kern / kern.sum(), 'valid' )
		
		fig = figure()
		ax = fig.add_subplot(111)
		ax.scatter(phases, zs )
		plot(sm_phases, sm_zs, 'k', linewidth = 3.0)
		axhline(0)
		ylabel("stim contrast (Z)")
		xlabel("eccentricity")
		ax.set_xlim([0,2*pi])
		
		
		############################################
		
		shell()
		
		phases_sorted = phases[phase_order]
		zs_sorted = zs[phase_order]
		XX = np.hstack((self.X,self.X,self.X))
		X_sorted = XX[:,phase_order]
		y = self.y
		
		accuracy = []
		number_voxels = phases.shape[0]/3
		for j in np.arange(0,number_voxels,10):
			voxels_for_decoding = np.arange(j+number_voxels-200, j+number_voxels+200)
			xx = X_sorted[:,voxels_for_decoding]
			
			
			for i in range(3):
				# Balance classes:
			
				Xa = xx[y == 1,:]
				Xb = xx[y == -1,:]
				ya = y[y == 1]
				yb = y[y == -1]
		
				import random
			
				random_indices = random.sample(range(len(Xa)), Xb.shape[0])
				Xa = Xa[random_indices,:]
				ya = ya[random_indices]
		
				X = np.vstack((Xa,Xb))
				y = np.concatenate((ya,yb))
			
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
				# print('sample_weight = ' + str(sample_weight))
				clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
						gamma=0.001, kernel='linear', max_iter=-1, probability=False, shrinking=True,
						tol=0.001, verbose=False)
				clf.fit(X_train, y_train)
				predictions = clf.predict(X_test)
				accuracy.append( sum(predictions == y_test) / float(len(predictions)) )
			
				print('accuracy = ' + str(accuracy[i]))
		
		print('')
		print('')
		print('mean accuracy = ' + str(mean(accuracy)))
		
		
		
		
		
		ac = []
		number_voxels = phases.shape[0]/3
		for i in range(number_voxels):
			voxels_for_decoding = np.arange(i+number_voxels-100, i+number_voxels+100)
			xx = X_sorted[:,voxels_for_decoding]
			y = self.y
			
			accuracy = []
			for i in range(1):
				
				X_train, X_test, y_train, y_test = train_test_split(xx, y, test_size=0.25)
				
				clf = svm.SVC(C=100.)
				clf.fit(X_train, y_train)
				SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
					gamma=0.001, kernel='linear', max_iter=-1, probability=False, shrinking=True,
					tol=0.001, verbose=False)
				predictions = clf.predict(X_test)
				accuracy.append( sum(predictions == y_test) / float(len(predictions)) )
			
			ac.append(mean(accuracy))
			# print('accuracy = ' + str(accuracy))
		
		
		accc = np.array(ac)
		zs_sorted_small = zs_sorted[number_voxels:2*number_voxels]
		phases_sorted_small = phases_sorted[number_voxels:2*number_voxels]
		
		
		figure()
		plot(accc)
		
		fig = figure()
		ppd = accc
		bpd = zs_sorted_small
		slope, intercept, r_value, p_value, std_err = stats.linregress(bpd,ppd)
		(m,b) = sp.polyfit(bpd, ppd, 1)
		phasic_pupil_diameter_p = sp.polyval([m,b], bpd)
		ax2 = fig.add_subplot(111)
		ax2.plot(bpd,phasic_pupil_diameter_p, color = 'k', linewidth = 1.5)
		ax2.scatter(bpd, ppd, color='#808080', alpha = 0.75)
		ax2.set_title('Correlation', size = 12)
		plt.tick_params(axis='both', which='major', labelsize=10)
		if round(p_value,5) < 0.005:
			ax2.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), ((plt.axis()[2]+plt.axis()[3])*0.5),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
		else:	
			ax2.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), ((plt.axis()[2]+plt.axis()[3])*0.5),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
		simpleaxis(ax2)
		spine_shift(ax2)
		ax2.set_xlabel('Stimulus contrast (Z)', size = 10)
		ax2.set_ylabel('Decoding performance', size = 10)
		
		fig = figure()
		ppd = accc
		bpd = phases_sorted_small
		slope, intercept, r_value, p_value, std_err = stats.linregress(bpd,ppd)
		(m,b) = sp.polyfit(bpd, ppd, 1)
		phasic_pupil_diameter_p = sp.polyval([m,b], bpd)
		ax2 = fig.add_subplot(111)
		ax2.plot(bpd,phasic_pupil_diameter_p, color = 'k', linewidth = 1.5)
		ax2.scatter(bpd, ppd, color='#808080', alpha = 0.75)
		ax2.set_title('Correlation', size = 12)
		plt.tick_params(axis='both', which='major', labelsize=10)
		if round(p_value,5) < 0.005:
			ax2.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), ((plt.axis()[2]+plt.axis()[3])*0.5),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
		else:	
			ax2.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), ((plt.axis()[2]+plt.axis()[3])*0.5),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
		simpleaxis(ax2)
		spine_shift(ax2)
		ax2.set_xlabel('Eccentricity', size = 10)
		ax2.set_ylabel('Decoding performance', size = 10)
		
		
	
	
	def decoding(self, runArray, run_type='rivalry', subSamplingRatio=100.0, postFix=['mcf'], mask_type='STIM_Z', mask_direction='pos', threshold=4, number_voxels = 200):
		
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		h5file = openFile(self.hdf5_filename, mode = 'r+', title = run_type + " file")	
		
		# roi = ['V3']
		roi = ['V1']
		
		roi_data = []
		mask_data = []
		for j in range(len(roi)):
			for r in [self.runList[i] for i in self.conditionDict['rivalry']]:
				# get all functional data:
				roi_data.append(self.roi_data_from_hdf(h5file, r, roi[j], 'tf_psc_data', postFix = ['mcf']))
				mask_data.append(self.roi_data_from_hdf(h5file, r, roi[j], mask_type))
		
		decoding_indices_BR = []
		decoding_indices_SFM = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['rivalry']]:
			nr_runs += 1
			# get decoding indices: 
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
			search_name = 'decoding_indices'
			for node in h5file.iterNodes(where = thisRunGroup, classname = 'Group'):
				if search_name == node._v_name:
					decoding_node = node
					decoding_indices_BR.append( decoding_node.decoding_BR.read() )
					decoding_indices_SFM.append( decoding_node.decoding_SFM.read() )
					break
			
		# check out the TR, assuming that's the same across runs:
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]))
		tr = niiFile.rtime
		
		# resample decoding indices to TR's
		for i in range(nr_runs):
			decoding_indices_BR[i] = decoding_indices_BR[i][::(round(subSamplingRatio*tr,0))]
			decoding_indices_SFM[i] = decoding_indices_SFM[i][::(round(subSamplingRatio*tr,0))]
		
		# concatenate runs
		
		if len(roi) > 1:
			roi_dataV1 = np.concatenate(roi_data[0:nr_runs], axis=1).T
			roi_dataV2 = np.concatenate(roi_data[nr_runs:nr_runs*2], axis=1).T
			roi_dataV3 = np.concatenate(roi_data[nr_runs*2:nr_runs*3], axis=1).T
			roi_data_c = np.hstack((roi_dataV1, roi_dataV2, roi_dataV3))
		else:
			roi_data_c = np.concatenate(roi_data, axis=1).T
		
		mask_data = np.array(mask_data).mean(axis = 0)
		
		if threshold != None:
			# thresholding of mapping data stat values:
			if mask_direction == 'pos':
				mask_data_indices = mask_data[:,0] > threshold
			else:
				mask_data_indices = mask_data[:,0] < threshold
			roi_data_c = roi_data_c[:,mask_data_indices]
		
		if number_voxels != None:
			# selecting top 100 voxels:
			if mask_direction == 'pos':
				mask_data_indices = (np.argsort(mask_data[:,0])>mask_data[:,0].shape[0]-(number_voxels+1))
			else:
				mask_data_indices = (np.argsort(mask_data[:,0])<number_voxels)
			roi_data_c = roi_data_c[:,mask_data_indices]
		
		decod_BR_c = np.concatenate(decoding_indices_BR, axis=1).T
		decod_SFM_c = np.concatenate(decoding_indices_SFM, axis=1).T
		
		X = roi_data_c[decod_BR_c!=0]
		y = decod_BR_c[decod_BR_c!=0]
		
		# Z-score X:
		b = X
		b1 = ((b - b.mean(axis = 0)) / b.std(axis=0)).T
		b2 = ((b1 - b1.mean(axis = 0)) / b1.std(axis=0)).T
		X = b2
		
		##############################
		##### PCA
		
		pca = PCA(n_components=100)
		X_pca = pca.fit(X).transform(X)
		
		##############################
		
		print('Number of voxels = ' + str(X.shape[1]))
		
		accuracy = []
		for i in range(10):
			
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
			
			clf = svm.SVC(C=100.)
			clf.fit(X_train, y_train)
			SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
				gamma=0.001, kernel='linear', max_iter=-1, probability=False, shrinking=True,
				tol=0.001, verbose=False)
			predictions = clf.predict(X_test)
			accuracy.append( sum(predictions == y_test) / float(len(predictions)) )
		
		accuracy = mean(accuracy)
		print('accuracy = ' + str(accuracy))
		
		# decoding_kind = 1
		# 
		# # Split into train / test set with a particular fraction:
		# if decoding_kind == 1:
		# 	
		# 	auc_precision = []
		# 	auc_recall = []
		# 	for i in range(2):
		# 		# Split the dataset in train and test set
		# 		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		# 
		# 		# Set the parameters by cross-validation
		# 		tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
		# 		# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
		# 		
		# 		scores = [('precision', precision_score),('recall', recall_score),]
		# 		
		# 		for score_name, score_func in scores:
		# 			print "# Tuning hyper-parameters for %s" % score_name
		# 			print
		# 	
		# 			clf = GridSearchCV(SVC(C=1), tuned_parameters, score_func=score_func, n_jobs=-1)
		# 			clf.fit(X_train, y_train, cv=5)
		# 	
		# 			print "Best parameters set found on development set:"
		# 			print
		# 			print clf.best_estimator_
		# 			print
		# 			print "Grid scores on development set:"
		# 			print
		# 			for params, mean_score, scores in clf.grid_scores_:
		# 				print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)
		# 			print
		# 	
		# 			print "Detailed classification report:"
		# 			print
		# 			print "The model is trained on the full development set."
		# 			print "The scores are computed on the full evaluation set."
		# 			print
		# 			y_true, y_pred = y_test, clf.predict(X_test)
		# 			print classification_report(y_true, y_pred)
		# 			print
		# 		
		# 			if score_name == 'precision':
		# 				auc_precision.append(auc_score(y_true, y_pred))
		# 			if score_name == 'recall':
		# 				auc_recall.append(auc_score(y_true, y_pred))
		# 		
		# 		
		# 		
		# 
		# # Split into train / test based on runs:
		# if decoding_kind == 2:
		# 	
		# 	auc_precision = []
		# 	auc_recall = []
		# 	for i in range(nr_runs):
		# 		
		# 		train_indices = np.ones(nr_runs * nr_trs, dtype = bool)
		# 		train_indices[(i*nr_trs):(i*nr_trs)+nr_trs] = 0
		# 		train_indices = train_indices[decod_BR_c!=0]
		# 		
		# 		test_indices = np.zeros(nr_runs * nr_trs, dtype = bool)
		# 		test_indices[(i*nr_trs):(i*nr_trs)+nr_trs] = 1
		# 		test_indices = test_indices[decod_BR_c!=0]
		# 		
		# 		# Split the dataset in train and test set
		# 		X_train = X[train_indices]
		# 		X_test = X[test_indices]
		# 		y_train = y[train_indices]
		# 		y_test = y[test_indices]
		# 
		# 		# Set the parameters by cross-validation
		# 		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
		# 
		# 		scores = [('precision', precision_score),('recall', recall_score),]
		# 		
		# 		for score_name, score_func in scores:
		# 			print "# Tuning hyper-parameters for %s" % score_name
		# 			print
		# 	
		# 			clf = GridSearchCV(SVC(C=1), tuned_parameters, score_func=score_func)
		# 			clf.fit(X_train, y_train, cv=5)
		# 	
		# 			print "Best parameters set found on development set:"
		# 			print
		# 			print clf.best_estimator_
		# 			print
		# 			print "Grid scores on development set:"
		# 			print
		# 			for params, mean_score, scores in clf.grid_scores_:
		# 				print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)
		# 			print
		# 	
		# 			print "Detailed classification report:"
		# 			print
		# 			print "The model is trained on the full development set."
		# 			print "The scores are computed on the full evaluation set."
		# 			print
		# 			y_true, y_pred = y_test, clf.predict(X_test)
		# 			print classification_report(y_true, y_pred)
		# 			print
		# 
		# 			if score_name == 'precision':
		# 				auc_precision.append(auc_score(y_true, y_pred))
		# 			if score_name == 'recall':
		# 				auc_recall.append(auc_score(y_true, y_pred))
		# 
		# 		
		# def decode_patterns_per_trial_for_roi(self, roi, classification_data_type = 'per_trial_hpf_data_zscore', data_type_mask = 'Z', mask_threshold = 3.5, mask_direction = 'pos', postFix = ['mcf','tf']):
		# 	reward_h5file = self.hdf5_file('reward')
		# 	mapper_h5file = self.hdf5_file('mapper')
		# 
		# 	conditions_data_types = ['left_CW_Z', 'left_CCW_Z', 'right_CW_Z', 'right_CCW_Z']
		# 	condition_labels = ['left_CW', 'left_CCW', 'right_CW', 'right_CCW']
		# 
		# 	# check out the duration of these runs, assuming they're all the same length.
		# 	niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['reward'][0]]))
		# 	tr, nr_trs = niiFile.rtime, niiFile.timepoints
		# 	run_duration = tr * nr_trs
		# 
		# 	event_data = []
		# 	roi_data = []
		# 	nr_runs = 0
		# 	for r in [self.runList[i] for i in self.conditionDict['reward']]:
		# 		roi_data.append(self.roi_data_from_hdf(reward_h5file, r, roi, classification_data_type, postFix = ['mcf','tf']))
		# 		this_run_events = []
		# 		for cond in condition_labels:
		# 			this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [cond]))[:-1,0])	# toss out last trial of each type to make sure there are no strange spill-over effects
		# 		this_run_events = np.array(this_run_events) + nr_runs * run_duration
		# 		event_data.append(this_run_events)
		# 		self.rewarded_stimulus_run(r, postFix = postFix)
		# 		nr_runs += 1
		# 
		# 	# zscore the functional data
		# 	for roid in roi_data:
		# 		roid = ((roid.T - roid.mean(axis = 1)) / roid.std(axis = 1)).T
		# 		roid = (roid - roid.mean(axis = 0)) / roid.std(axis = 0)
		# 
		# 	event_data_per_run = event_data
		# 	event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		# 
		# 	# mapping data
		# 	mapping_data_L = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'left_' + data_type_mask, postFix = ['mcf','tf'])
		# 	mapping_data_R = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][0]], roi, 'right_' + data_type_mask, postFix = ['mcf','tf'])
		# 
		# 	# thresholding of mapping data stat values
		# 	if mask_direction == 'pos':
		# 		mapping_mask_L = mapping_data_L[:,0] > mask_threshold
		# 		mapping_mask_R = mapping_data_R[:,0] > mask_threshold
		# 	else:
		# 		mapping_mask_L = mapping_data_L[:,0] < mask_threshold
		# 		mapping_mask_R = mapping_data_R[:,0] < mask_threshold
		# 
		# 	which_orientation_rewarded = self.which_stimulus_rewarded % 2
		# 	reward_run_list = [self.runList[j] for j in self.conditionDict['reward']]
		# 	L_data = np.hstack([[r[:,k].T for k in reward_run_list[i].all_stimulus_trials] for (i, r) in enumerate(roi_data)])[...,mapping_mask_L]
		# 	R_data = np.hstack([[r[:,k].T for k in reward_run_list[i].all_stimulus_trials] for (i, r) in enumerate(roi_data)])[...,mapping_mask_R]
		# 
		# 	ormd = []
		# 	mcd = []
		# 	for mf in [0,-1]:
		# 		mapper_run_events = []
		# 		for cond in condition_labels:
		# 			mapper_run_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['mapper'][mf]], extension = '.txt', postFix = [cond]))[:,0])	
		# 		mapper_condition_order = np.array([np.array([np.ones(m.shape) * i, m]).T for (i, m) in enumerate(mapper_run_events)]).reshape(-1,2)
		# 		mapper_condition_order = mapper_condition_order[np.argsort(mapper_condition_order[:,1]), 0]
		# 		mapper_condition_order_indices = np.array([mapper_condition_order == i for i in range(len(condition_labels))], dtype = bool)
		# 		mcd.append(mapper_condition_order_indices)
		# 		orientation_mapper_data = self.roi_data_from_hdf(mapper_h5file, self.runList[self.conditionDict['mapper'][mf]], roi_wildcard = roi, data_type = classification_data_type, postFix = ['mcf','tf'])
		# 		orientation_mapper_data = ((orientation_mapper_data.T - orientation_mapper_data.mean(axis = 1)) / orientation_mapper_data.std(axis = 1)).T
		# 		orientation_mapper_data = (orientation_mapper_data - orientation_mapper_data.mean(axis = 0)) / orientation_mapper_data.std(axis = 0)
		# 		ormd.append(orientation_mapper_data)
		# 
		# 	mapper_condition_order_indices = np.hstack(mcd)
		# 	orientation_mapper_data = np.hstack(ormd)
		# 
		# 	all_ori_data = np.array([orientation_mapper_data[:,moi].T for moi in mapper_condition_order_indices])
		# 	L_ori_data = all_ori_data[...,mapping_mask_L]
		# 	R_ori_data = all_ori_data[...,mapping_mask_R]
		# 
		# 	train_data_L, train_labels_L = [np.vstack((L_ori_data[i],L_ori_data[i+1])) for i in [0,2]], [np.concatenate((-np.ones(L_ori_data.shape[1]),np.ones(L_ori_data.shape[1])))  for i in [0,2]]
		# 	train_data_R, train_labels_R = [np.vstack((R_ori_data[i],R_ori_data[i+1])) for i in [0,2]], [np.concatenate((-np.ones(R_ori_data.shape[1]),np.ones(R_ori_data.shape[1])))  for i in [0,2]]
		# 
		# 	test_data_L, test_labels_L = [np.vstack((L_data[i],L_data[i+1])) for i in [0,2]], [np.concatenate((-np.ones(L_data.shape[1]),np.ones(L_data.shape[1])))  for i in [0,2]]
		# 	test_data_R, test_labels_R = [np.vstack((R_data[i],R_data[i+1])) for i in [0,2]], [np.concatenate((-np.ones(R_data.shape[1]),np.ones(R_data.shape[1])))  for i in [0,2]]
		# 
		# 	# shell()
		# 	from sklearn import neighbors, datasets, linear_model, svm, lda, qda
		# 	# kern = svm.SVC(probability=True, kernel = 'linear', C=1e4) # , C=1e3), NuSVC , C = 1.0
		# 	# kern = svm.LinearSVC(C=1e5, loss='l1') # , C=1e3), NuSVC , C = 1.0
		# 	# kern = svm.SVC(probability=True, kernel='rbf', degree=2) # , C=1e3), NuSVC , C = 1.0
		# 	kern = lda.LDA()
		# 	# kern = qda.QDA()
		# 	# kern = neighbors.KNeighborsClassifier()
		# 	# kern = linear_model.LogisticRegression(C=1e5)
		# 
		# 
		# 	corrects_L = [kern.fit(train_data_L[i], train_labels_L[i]).predict(test_data_L[i]) * test_labels_L[i] for i in [0,1]]
		# 	corrects_R = [kern.fit(train_data_R[i], train_labels_R[i]).predict(test_data_R[i]) * test_labels_R[i] for i in [0,1]]
		# 
		# 	corrects_per_cond_L = np.array([[cl[:cl.shape[0]/2], cl[cl.shape[0]/2:]] for cl in corrects_L]).reshape(4,-1)
		# 	corrects_per_cond_R = np.array([[cr[:cr.shape[0]/2], cr[cr.shape[0]/2:]] for cr in corrects_R]).reshape(4,-1)
		# 
		# 	probs_L = [kern.fit(train_data_L[i], train_labels_L[i]).predict_proba(test_data_L[i])[:,0] for i in [0,1]]
		# 	probs_R = [kern.fit(train_data_R[i], train_labels_R[i]).predict_proba(test_data_R[i])[:,0] for i in [0,1]]
		# 
		# 	probs_per_cond_L = np.array([[cl[:cl.shape[0]/2], cl[cl.shape[0]/2:]] for cl in probs_L]).reshape(4,-1)
		# 	probs_per_cond_R = np.array([[cr[:cr.shape[0]/2], cr[cr.shape[0]/2:]] for cr in probs_R]).reshape(4,-1)
		# 
		# 	print roi
		# 	print 'left: ' + str(((corrects_per_cond_L + 1) / 2).mean(axis = 1))
		# 	print 'right: ' + str(((corrects_per_cond_R + 1) / 2).mean(axis = 1))
		# 
		# 	# now, plotting
		# 	alphas = np.ones(4) * 0.45
		# 	alphas[self.which_stimulus_rewarded] = 1.0
		# 	colors = ['r', 'r--', 'k', 'k--']
		# 	if self.which_stimulus_rewarded % 2 == 0:
		# 		diff_color = 'b'
		# 	else:
		# 		diff_color = 'b--'
		# 	if self.which_stimulus_rewarded < 2:
		# 		diff_alpha = [0.75, 0, 0.25]
		# 	else:
		# 		diff_alpha = [0.25, 0, 0.75]
		# 
		# 	f = pl.figure(figsize = (12,6))
		# 	s = f.add_subplot(2,2,1)
		# 	s.axhline(y = 0.5, c = 'k', linewidth = 0.5)
		# 	s.set_title('left stimulus ROI in ' + roi)
		# 	s.set_xlabel('time [trials]')
		# 	s.set_ylabel('percentage CW')
		# 	s.set_xlim([0,probs_per_cond_L[0].shape[0]])
		# 	s.set_ylim([0,1])
		# 	[s.axvspan(i * 12, (i+1) * 12, facecolor='k', alpha=0.05, edgecolor = 'w') for i in [0,2,4]]
		# 	for i in range(4):
		# 		plot(probs_per_cond_L[i], colors[i], alpha = alphas[i], label = condition_labels[i], linewidth = alphas[i] * 2.0)
		# 	# s.axis([0,rew_corr[i,:,0].shape[0],-1.0,1.0])
		# 	s = f.add_subplot(2,2,3)
		# 	s.axhline(y = 0.5, c = 'k', linewidth = 0.5)
		# 	s.set_title('right stimulus ROI in ' + roi)
		# 	[s.axvspan(i * 12, (i+1) * 12, facecolor='k', alpha=0.05, edgecolor = 'w') for i in [0,2,4]]
		# 	for i in range(4):
		# 		plot(probs_per_cond_R[i], colors[i], alpha = alphas[i], label = condition_labels[i], linewidth = alphas[i] * 2.0)
		# 	s.set_xlabel('time [trials]')
		# 	s.set_ylabel('percentage CW')
		# 	s.set_xlim([0,probs_per_cond_L[0].shape[0]])
		# 	s.set_ylim([0,1])
		# 	# s.axis([0,rew_corr[i,:,0].shape[0],-1.0,1.0])
		# 	leg = s.legend(fancybox = True)
		# 	leg.get_frame().set_alpha(0.5)
		# 	if leg:
		# 		for t in leg.get_texts():
		# 		    t.set_fontsize('small')    # the legend text fontsize
		# 		for l in leg.get_lines():
		# 		    l.set_linewidth(3.5)  # the legend line width
		# 	s = f.add_subplot(2,2,2)
		# 	s.set_title('histogram left ROI in ' + roi)
		# 	s.axhline(y = 0.5, c = 'k', linewidth = 0.5)
		# 	for i in range(4):
		# 		s.axhline(y = probs_per_cond_L[i].mean(), c = colors[i][0], linewidth = 2.5, linestyle = '--')
		# 		pl.hist(probs_per_cond_L[i], color=colors[i][0], alpha = alphas[i], normed = True, bins = 20, rwidth = 0.5, histtype = 'step', linewidth = 2.5, orientation = 'horizontal' )
		# 	pl.text(0.5, 0.5, str(((corrects_per_cond_L + 1) / 2).mean(axis = 1)))
		# 	s.set_ylim([0,1])
		# 	s = f.add_subplot(2,2,4)
		# 	s.set_title('histogram right ROI in ' + roi)
		# 	s.axhline(y = 0.5, c = 'k', linewidth = 0.5)
		# 	for i in range(4):
		# 		s.axhline(y = probs_per_cond_R[i].mean(), c = colors[i][0], linewidth = 2.5, linestyle = '--')
		# 		pl.hist(probs_per_cond_R[i], color=colors[i][0], alpha = alphas[i], normed = True, bins = 20, rwidth = 0.5, histtype = 'stepfilled', orientation = 'horizontal' )
		# 	pl.text(0.5, 0.5, str(((corrects_per_cond_R + 1) / 2).mean(axis = 1)))
		# 	s.set_ylim([0,1])
		# 
		# 	# shell()
		# 
		# 	return [((corrects_per_cond_L + 1) / 2).mean(axis = 1), ((corrects_per_cond_R + 1) / 2).mean(axis = 1)]
		
	
	
	


