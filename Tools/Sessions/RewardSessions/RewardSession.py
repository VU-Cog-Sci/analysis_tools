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
from ...plotting_tools import *

class RewardSession(Session):
	"""docstring for RewardSession"""
	def __init__(self, ID, date, project, subject, session_label, parallelize = True, loggingLevel = logging.DEBUG):
		self.session_label = session_label
		super(RewardSession, self).__init__(ID, date, project, subject, parallelize = parallelize, loggingLevel = loggingLevel)
	
	# these functions here to create the appropriate hierarchy for this project.
	def baseFolder(self):
		return os.path.join(self.project.base_dir, self.session_label, self.subject.initials, self.dateCode)
	
	def makeBaseFolder(self):
		if not os.path.isdir(self.baseFolder()):
			try:
				os.mkdir(os.path.join(self.project.base_dir, self.session_label, self.subject.initials))
			except OSError:
				pass
			try:
				os.mkdir(self.baseFolder())
			except OSError:
				pass
	
	def registerSession(self, execute = True):
		"""registration based on the standard of the reward sessions. 
		just involves copying a bunch of standard registration files to the session's folder."""
		self.logger.info('register files')
		# setup what to register to
		self.FSsubject = self.subject.standardFSID
		
		# copy all the project files to the present session's hierarchy
		self.project_reg_folder = os.path.join(self.project.base_dir, self.subject.initials)
		self.project_feat_folder = os.path.join(self.project.base_dir, self.subject.initials, 'feat')
		self.project_masks_folder = os.path.join(self.project.base_dir, self.subject.initials, 'masks')
		self.project_register_file = os.path.join(self.project_reg_folder, self.session_label, 'register.dat')
		self.project_register_flirt_file = os.path.join(self.project_reg_folder, self.session_label, 'register_flirt_BB.mtx')
		self.reg_to_project_flirt_file = os.path.join(self.project_reg_folder, self.session_label, 'reg_to_project.mtx')
		self.reg_to_project_bb_file = os.path.join(self.project_reg_folder, self.session_label, 'reg_to_project.dat')
		
		
		self.registerfile = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' )
		self.register_flirt_file = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'flirt', 'BB'], extension = '.dat' )
		self.session_reg_folder = self.stageFolder(stage = 'processed/mri/reg')
		self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] )
		self.masks_folder = self.stageFolder(stage = 'processed/mri/masks/anat')
		self.feat_folder = self.stageFolder(stage = 'processed/mri/reg/feat')
		
		if execute:
			os.system('cp ' + self.project_register_file + ' ' + self.registerfile)
			os.system('cp ' + self.project_register_flirt_file + ' ' + self.register_flirt_file)
			os.system('cp ' + self.reg_to_project_flirt_file + ' ' + os.path.join(self.session_reg_folder, 'reg_to_project.mtx'))
			os.system('cp ' + self.reg_to_project_bb_file + ' ' + os.path.join(self.session_reg_folder, 'reg_to_project.dat'))
			os.system('cp ' + self.project.standard_EPI_file + ' ' + self.referenceFunctionalFileName)
			os.system('cp ' + self.project.standard_T2_file + ' ' + os.path.join(self.session_reg_folder, 'T2.nii.gz') )
			os.system('cp -rf ' + self.project_feat_folder + ' ' + self.feat_folder)
			os.system('cp -rf ' + self.project_masks_folder + ' ' + self.masks_folder)
				
	
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
			mcf = MCFlirtOperator( self.runFile(stage = 'processed/mri', run = self.runList[er] ), target = self.referenceFunctionalFileName )
		 	mcf.configure( further_args = ' -init ' + os.path.join(os.path.join(os.path.join(self.project.base_dir, self.subject.initials, self.runList[er].session_label), 'reg_to_project.mtx') )
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
	