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
from ...other_scripts.circularTools import *
from ...Operators import ArrayOperator, EDFOperator, HDFEyeOperator, EyeSignalOperator
from pylab import *
from nifti import *
from IPython import embed as shell
from tables import *
import pickle
from scipy.stats import *
from ...other_scripts.plotting_tools import *

class RewardSession(Session):
	"""docstring for RewardSession"""
	def __init__(self, ID, date, project, subject, session_label, parallelize = True, loggingLevel = logging.DEBUG):
		self.session_label = session_label
		super(RewardSession, self).__init__(ID, date, project, subject, parallelize = parallelize, loggingLevel = loggingLevel)

		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/eye'), self.subject.initials + '.hdf5')
		self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)

	
	# these functions here to create the appropriate hierarchy for this project.
	def base_dir(self):
		return os.path.join(self.project.base_dir, self.session_label, self.subject.initials, self.dateCode)
	
	def makeBaseFolder(self):
		if not os.path.isdir(self.base_dir()):
			try:
				os.mkdir(os.path.join(self.project.base_dir, self.session_label, self.subject.initials))
			except OSError:
				pass
			try:
				os.mkdir(self.base_dir())
			except OSError:
				pass
	
	def registerSession(self, execute = True):
		"""registration based on the standard of the reward sessions. 
		just involves copying a bunch of standard registration files to the session's folder."""
		self.logger.info('register files')
		# setup what to register to
		self.FSsubject = self.subject.standardFSID
		
		# copy all the project files to the present session's hierarchy
		self.project_reg_folder = os.path.join(self.project.registration_dir, self.subject.initials)
		self.project_feat_folder = os.path.join(self.project.registration_dir, self.subject.initials, 'feat/')
		self.project_masks_folder = os.path.join(self.project.registration_dir, self.subject.initials, 'masks/')
		self.project_register_file = os.path.join(self.project_reg_folder, 'register.dat')
		self.project_register_flirt_file = os.path.join(self.project_reg_folder, 'register_flirt_BB.mtx')
		self.reg_to_project_flirt_file = os.path.join(self.project_reg_folder, self.session_label, 'reg_to_project.mtx')
		self.reg_to_project_bb_file = os.path.join(self.project_reg_folder, self.session_label, 'reg_to_project.dat')
		
		
		self.registerfile = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' )
		self.register_flirt_file = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'flirt', 'BB'], extension = '.dat' )
		self.session_reg_folder = self.stageFolder(stage = 'processed/mri/reg')
		self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] )
		self.bet_mask_file_name = self.runFile(stage = 'processed/mri/reg', base = 'BET_mask', postFix = [self.ID] )
		self.masks_folder = self.stageFolder(stage = 'processed/mri/masks/anat')
		self.feat_folder = self.stageFolder(stage = 'processed/mri/reg/feat')
		
		
		if execute:
			os.system('cp ' + self.project_register_file + ' ' + self.registerfile)
			os.system('cp ' + self.project_register_flirt_file + ' ' + self.register_flirt_file)
			os.system('cp ' + self.reg_to_project_flirt_file + ' ' + os.path.join(self.session_reg_folder, 'reg_to_project.mtx'))
			os.system('cp ' + self.reg_to_project_bb_file + ' ' + os.path.join(self.session_reg_folder, 'reg_to_project.dat'))
			os.system('cp ' + os.path.join(self.project_reg_folder, 'standard_EPI_BET.nii.gz') + ' ' + self.referenceFunctionalFileName)
			os.system('cp ' + os.path.join(self.project_reg_folder, 'standard_EPI_BET_mask.nii.gz') + ' ' + self.bet_mask_file_name)
			os.system('cp ' + os.path.join(self.project_reg_folder, 't2_BET.nii.gz') + ' ' + os.path.join(self.session_reg_folder, 'T2.nii.gz') )
			try:
				os.system('mkdir ' + self.feat_folder)
				os.system('mkdir ' + self.masks_folder)
			except OSError:
				pass
			os.system('cp -rf ' + self.project_feat_folder + ' ' + self.feat_folder)
			os.system('cp -rf ' + self.project_masks_folder + ' ' + self.masks_folder)
				
	
	def bet_all_nii_files(self):
		"""bet_all_nii_files does exactly what the name implies. """
		mask_file = self.runFile(stage = 'processed/mri/reg', base = 'BET_mask', postFix = [self.ID] )
		for er in self.scanTypeDict['epi_bold']:
			for postFix in [['reg'], ['mcf'], ['mcf','tf'], ['mcf','tf','psc']]:
				this_file = self.runFile(stage = 'processed/mri', run = self.runList[er], postFix = postFix )
				mO = FSLMathsOperator(this_file)
				mO.configureMask(
					mask_file = mask_file, 
					outputFileName = this_file)
				mO.execute()
		if os.path.isfile(os.path.join(self.stageFolder('/processed/mri/reward/deco'), 'residuals.nii.gz')):
			mO = FSLMathsOperator(os.path.join(self.stageFolder('/processed/mri/reward/deco'), 'residuals.nii.gz'))
			mO.configureMask(
				mask_file = mask_file, 
				outputFileName = os.path.join(self.stageFolder('/processed/mri/reward/deco'), 'residuals.nii.gz'))
			mO.execute()
		
	
	def motionCorrectFunctionals(self):
		"""
		motionCorrectFunctionals corrects all functionals in a given session.
		how we do this depends on whether we have parallel processing turned on or not
		"""
		self.logger.info('run motion correction')
		self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID]  )
		# set up a list of motion correction operator objects for the runs
		mcOperatorList = [];	
		for er in self.scanTypeDict['epi_bold']:
			# run non-moco registration first. 
			f1 = FlirtOperator(inputObject = self.runFile(stage = 'processed/mri', run = self.runList[er] ), referenceFileName = self.referenceFunctionalFileName)
			f1.configureApply( transformMatrixFileName = os.path.join(self.stageFolder('processed/mri/reg/'), 'reg_to_project.mtx'), outputFileName = self.runFile(stage = 'processed/mri', run = self.runList[er], postFix = ['reg'] ) )
			mcOperatorList.append(f1)
			
			mcf = MCFlirtOperator( self.runFile(stage = 'processed/mri', run = self.runList[er], postFix = ['reg'], extension = '' ), target = self.referenceFunctionalFileName )
		 	mcf.configure( outputFileName = self.runFile(stage = 'processed/mri', run = self.runList[er], postFix = ['mcf'] ) )
			mcOperatorList.append(mcf)
	
		if not self.parallelize:
			# first, code for serial implementation
			self.logger.info("run serial moco")
			for mcf in mcOperatorList:
				mcf.execute()
	
		if self.parallelize:
			# tryout parallel implementation - later, this should be abstracted out of course. 
			ppservers = ()
			job_server = pp.Server(ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
#			ppResults = [job_server.submit(mcf.execute,(), (), ("Tools","Tools.Operators","Tools.Sessions.MCFlirtOperator","subprocess",)) for mcf in mcOperatorList]
			ppResults = [job_server.submit(ExecCommandLine,(mcf.runcmd,),(),('subprocess','tempfile',)) for mcf in mcOperatorList]
			for fMcf in ppResults:
				fMcf()
		
			job_server.print_stats()
	
	def add_first_session_reward_responses_to_hdf5(self):
		"""docstring for add_first_session_reward_responses_to_hdf5"""
		pass


	def combine_motion_parameters_one_file(self, run):
		files = [
			self.runFile(
				stage='processed/mri', run=run, extension='.par', postFix=['mcf.nii.gz']),
			self.runFile(
				stage='processed/mri', run=run, extension='.par', postFix=['mcf.nii.gz', 'dt']),
			self.runFile(
				stage='processed/mri', run=run, extension='.par', postFix=['mcf.nii.gz', 'ddt'])
		]
		np.savetxt(self.runFile(stage='processed/mri', run=run, extension='.par', postFix=[
				   'mcf', 'all']), np.hstack([np.loadtxt(f) for f in files]), fmt='%f', delimiter='\t')

	def combine_motion_parameters(self):
		for run in self.runList:
			self.combine_motion_parameters_one_file(run)
		