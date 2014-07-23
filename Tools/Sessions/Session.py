#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

import os, sys, pickle, math
from subprocess import *
import datetime

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
from nifti import *

import pp
import logging, logging.handlers, logging.config

from ..log import *
from ..Run import *
from ..Subjects.Subject import *
from ..Operators.Operator import *
from ..Operators.CommandLineOperator import *
from ..Operators.ImageOperator import *
from ..Operators.BehaviorOperator import *
# from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from IPython import embed as shell
from joblib import Parallel, delayed

from ..Operators.HDFEyeOperator import HDFEyeOperator

from joblib import Parallel, delayed

from ..Operators.HDFEyeOperator import HDFEyeOperator

class PathConstructor(object):
	"""
	FilePathConstructor is an abstract superclass for sessions.
	It constructs the file and folder naming hierarchy for a given session.
	All file naming and calling runs through this class.
	"""
	def __init__(self):
		self.fileNameBaseString = self.dateCode
	
	def base_dir(self):
		"""docstring for baseFolder"""
		return os.path.join(self.project.base_dir, self.subject.initials, self.dateCode)
	
	def make_base_dir(self):
		if not os.path.isdir(self.base_dir()):
			try:
				os.mkdir(os.path.join(self.project.base_dir, self.subject.initials))
			except OSError:
				pass
			try:
				os.mkdir(self.base_dir())
			except OSError:
				pass
	
	def stageFolder(self, stage):
		"""folder for a certain stage - such as 'raw/mri' or 'processed/eyelink', or something like that. """
		return os.path.join(self.base_dir(), stage)
	
	def runFolder(self, stage, run):
		"""docstring for runFolder"""
		return os.path.join(self.stageFolder(stage), run.condition, str(run.ID))
	
	def conditionFolder(self, stage, run):
		"""docstring for runFolder"""
		return os.path.join(self.stageFolder(stage), run.condition)
	
	def runFile(self, stage, run = None, postFix = [], extension = standardMRIExtension, base = None):
		"""
		runFile, returns runFileName in file hierarchy. usage:
		raw mri file X = self.runFile('raw/mri', postFix = [str(self.runList[X])], base = self.subject.initials)
		motion corrected file X = self.runFile('processed/mri', run = self.runList[X], postFix = ['mcf'])
		"""
		fn = ''
		if not base:
			fn += self.fileNameBaseString
		else:
			fn += base
		if run:
			fn += '_' + str(run.ID)
		for pf in postFix:
			fn += '_' + pf
		fn += extension
		
		if run and stage.split('/')[0] == 'processed':
			return os.path.join(self.runFolder(stage, run), fn)
		else:
			return os.path.join(self.stageFolder(stage), fn)
	
	def createFolderHierarchy(self):
		"""docstring for fname"""
		rawFolders = ['raw/mri', 'raw/behavior', 'raw/eye', 'raw/hr']
		self.processedFolders = ['processed/mri', 'processed/behavior', 'processed/eye', 'processed/hr']
		conditionFolders = np.concatenate((self.conditionList, ['log','figs','masks','masks/stat','masks/anat','reg','surf','scripts']))
		
		
		self.make_base_dir()
		# assuming baseDir/raw/ exists, we must make processed
		if not os.path.isdir(os.path.join(self.base_dir(), 'processed') ):
			os.mkdir(os.path.join(self.base_dir(), 'processed'))
		if not os.path.isdir(os.path.join(self.base_dir(), 'raw') ):
			os.mkdir(os.path.join(self.base_dir(), 'raw'))
		
		
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
	

class Session(PathConstructor):
	"""
	Session is an object that contains all the analysis steps 
	for analyzing an entire session. The basic class contains the base level analyis and preparations, 
	such as putting the files in place and setting up the runs. 
	Often-used analysis steps include registration with an anatomical, and motion correction of the functionals.
	"""
	def __init__(self, ID, date, project, subject, parallelize = False, loggingLevel = logging.DEBUG, name_appendix = '', **kwargs):
		self.ID = ID
		self.date = date
		self.project = project
		self.subject = subject
		self.runList = []
		self.name_appendix = name_appendix
		self.dateCode = subject.initials + '_' + ('0'+str(self.date.day))[-2:] + ('0'+str(self.date.month))[-2:] + str(self.date.year)[-2:] + self.name_appendix
		self.parallelize = parallelize
		self.loggingLevel = loggingLevel
		for k,v in kwargs.items():
			setattr(self, k, v)
		
		super(Session, self).__init__()
		
		# add logging for this session
		# sessions create their own logging file handler
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(self.loggingLevel)
		if os.path.isdir(self.stageFolder(stage = 'processed/mri/log')):
			addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.stageFolder(stage = 'processed/mri/log'), 'sessionLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		else:
			addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.stageFolder(stage = 'raw'), 'firstSessionSetupLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		# self.logger.info('starting analysis of session ' + str(self.ID))
	
	def addRun(self, run):
		"""addRun adds a run to a session's run list"""
		run.indexInSession = len(self.runList)
		self.runList.append(run)
		# recreate conditionList
		self.conditionList = list(np.unique(np.array([r.condition for r in self.runList])))
		self.scanTypeList = list(np.unique(np.array([r.scanType for r in self.runList])))
		
		self.parcelateConditions()
	
	def parcelateConditions(self):
		# conditions will vary across experiments - this one will be amenable in subclasses,
		# but these are the principal types of runs. For EPI runs conditions will depend on the experiment.
		self.scanTypeDict = {}
		if 'epi_bold' in self.scanTypeList:
			self.scanTypeDict.update({'epi_bold': [hit.indexInSession for hit in filter(lambda x: x.scanType == 'epi_bold', [r for r in self.runList])]})
		if 'inplane_anat' in self.scanTypeList:
			self.scanTypeDict.update({'inplane_anat': [hit.indexInSession for hit in filter(lambda x: x.scanType == 'inplane_anat', [r for r in self.runList])]})
		if '3d_anat' in self.scanTypeList:
			self.scanTypeDict.update({'3d_anat': [hit.indexInSession for hit in filter(lambda x: x.scanType == '3d_anat', [r for r in self.runList])]})
		if 'dti' in self.scanTypeList:
			self.scanTypeDict.update({'dti': [hit.indexInSession for hit in filter(lambda x: x.scanType == 'dti', [r for r in self.runList])]})
		if 'spectro' in self.scanTypeList:
			self.scanTypeDict.update({'spectro': [hit.indexInSession for hit in filter(lambda x: x.scanType == 'spectro', [r for r in self.runList])]})
		if 'field_map' in self.scanTypeList:
			self.scanTypeDict.update({'field_map': [hit.indexInSession for hit in filter(lambda x: x.scanType == 'field_map', [r for r in self.runList])]})
		
#		print self.scanTypeDict
		self.conditions = np.unique(np.array([r.condition for r in self.runList]))
		self.conditionDict = {}
		for c in self.conditions:
			if c != '':
				self.conditionDict.update({c: [hit.indexInSession for hit in filter(lambda x: x.condition == c, [r for r in self.runList])]})
		
			
	def import_all_edf_data(self, aliases):
		"""import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
		for r in self.runList:
			if r.indexInSession in self.scanTypeDict['epi_bold']:
				run_name = os.path.split(self.runFile(stage = 'processed/eye', run = r, extension = ''))[-1]
				ho = HDFEyeOperator(self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'))
				edf_file = subprocess.Popen('ls ' + self.runFolder(stage = 'processed/eye', run = r) + '/*.edf', shell=True, stdout=PIPE).communicate()[0].split('\n')[0]
				ho.add_edf_file(edf_file)
				ho.edf_message_data_to_hdf(alias = run_name)
				ho.edf_gaze_data_to_hdf(alias = run_name)
	
	def setupFiles(self, rawBase, process_eyelink_file = True, date_format = None):
		"""
		When all runs are listed in the session, 
		the session will be able to distill what conditions are there 
		and setup the folder hierarchy and copy the raw image files into position
		"""
		if not os.path.isfile(self.runFile(stage = 'processed/behavior', run = self.runList[0] )):
			self.logger.info('creating folder hierarchy')
			self.createFolderHierarchy()
		
		for r in self.runList:
			if hasattr(r, 'rawDataFilePath'):
				if os.path.splitext(r.rawDataFilePath)[-1] == '.PAR':
					self.logger.info('converting par/rec file %s into nii.gz file', r.rawDataFilePath)
					prc = ParRecConversionOperator( self.runFile(stage = 'raw/mri', postFix = [str(r.ID)], base = rawBase, extension = '.PAR' ) )
					prc.configure()
					prc.execute()
					if r.indexInSession in self.scanTypeDict['epi_bold']:
						# address slope and intercept issues
						f = open(self.runFile(stage = 'raw/mri', postFix = [str(r.ID)], base = rawBase, extension = '.PAR' ), 'r')
						fr = f.readlines()
						# column 13 of PAR file is slope - assuming it is identical for the whole file.
						slope = fr[100].split()[12]
						# dcm2nii creates weird file name additions like the f in front of 
						niiFile = NiftiImage(self.runFile(stage = 'raw/mri', postFix = [str(r.ID)], base = rawBase ))
						niiFile.setSlope(slope)
						niiFile.save()
				self.logger.info('place nii files in hierarchy')
				# copy raw files
				ExecCommandLine('cp ' + r.rawDataFilePath + ' ' + self.runFile(stage = 'processed/mri', run = r ) )
			# behavioral files will be copied during analysis
			if hasattr(r, 'eyeLinkFilePath'):
				elO = EyelinkOperator(r.eyeLinkFilePath, date_format = date_format)
				ExecCommandLine('cp ' + os.path.splitext(r.eyeLinkFilePath)[0] + '.* ' + self.runFolder(stage = 'processed/eye', run = r ) )
				if process_eyelink_file:
					elO.processIntoTable(hdf5_filename = self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'), compute_velocities = False, check_answers = False)
			if hasattr(r, 'rawBehaviorFile'):
				ExecCommandLine('cp ' + r.rawBehaviorFile.replace('|', '\|') + ' ' + self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' ) )
			if hasattr(r, 'physiologyFile'):
				ExecCommandLine('cp ' + r.physiologyFile.replace('|', '\|') + ' ' + self.runFile(stage = 'processed/hr', run = r, extension = '.log' ) )
	
	def registerSession(self, contrast = 't2', FSsubject = None, prepare_register = True, which_epi_target = 0, deskull = True, bb = True, makeMasks = False, maskList = ['cortex','V1','V2','V3','V3A','V3B','V4'], labelFolder = 'label', MNI = True, run_flirt = True):
		"""
		before we run motion correction we register with the freesurfer segmented version of this subject's brain. 
		For this we use either the inplane anatomical (if present), or we take the first epi_bold of the session,
		motion correct it and mean the motion corrected first epi_bold to serve as the target for the registration.
		the contrast argument indicates the contrast of the reference image in epi_bold space that is to be registered.
		Then we may choose to run flirt on the epi-bold (recommended against) or run flirt on the fs segmented brain file in the
		subject's fs folder, which we recommend. 
		"""
		self.logger.info('register files')
		# setup what to register to
		if not FSsubject:
			self.FSsubject = self.subject.standardFSID
		else:
			self.FSsubject = FSsubject
		
		self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] )
		
		if prepare_register:
			if 'inplane_anat' in self.scanTypeList:
				# we have one or more inplane anatomicals - we take the last of these as a reference.
				# first, we need to strip the skull though
				self.logger.info('using inplane_anat as a registration target')
				if deskull:
					better = BETOperator( inputObject = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][-1]]) )
					better.configure( outputFileName = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][-1]], postFix = ['NB']) )
					better.execute()
					self.originalReferenceFunctionalVolume = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][-1]], postFix = ['NB'])
				else:
					self.originalReferenceFunctionalVolume = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][-1]])
				self.logger.info('registration target is inplane_anat, ' + self.originalReferenceFunctionalVolume)
			
			else:
				# we have to make do with epi volumes. so, we motion correct the first epi_bold run
				# prepare the motion corrected first, or selected functional	
				mcFirst = MCFlirtOperator( self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][which_epi_target]] ) )
				mcFirst.configure(outputFileName = self.runFile(stage = 'processed/mri/reg', postFix = ['mcf'], base = 'firstFunc' ))
				mcFirst.execute()
				#  and average it over time. 
				fslM = FSLMathsOperator( self.runFile(stage = 'processed/mri/reg', postFix = ['mcf'], base = 'firstFunc' ) )
				# in principle taking the temporal mean is superfluous (done by mcflirt too) but oh well
				fslM.configureTMean()
				fslM.execute()		
				
				self.logger.info('using firstFunc as a registration target')
				self.originalReferenceFunctionalVolume = self.runFile(stage = 'processed/mri/reg', postFix = ['mcf', 'meanvol'], base = 'firstFunc' )
				self.logger.info('registration target is firstFunc, ' + self.originalReferenceFunctionalVolume)
			ExecCommandLine('cp ' + self.originalReferenceFunctionalVolume + ' ' + self.referenceFunctionalFileName )
		
		
		if bb:
			# register to both freesurfer anatomical and fsl MNI template
			# actual registration - BBRegister to freesurfer subject
			bbR = BBRegisterOperator( self.referenceFunctionalFileName, FSsubject = self.FSsubject, contrast = contrast )
			bbR.configure( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), flirtOutputFile = True, init_fsl = prepare_register )
			bbR.execute()
				
			
		if MNI:
			self.logger.info('running registration to standard brain for this session to be applied to feat directories.')
			# Flirt the freesurfer segmented brain to MNI brain with full search
			os.system('mri_convert ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain.mgz') + ' ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain.nii.gz'))
			flRT1 = FlirtOperator( os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain.nii.gz')  )
			flRT1.configureRun( transformMatrixFileName = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain_MNI.mat'), outputFileName = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain_MNI.nii.gz'), extra_args = ' -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12 ' )
			if run_flirt:
				flRT1.execute()
				
			# this bbregisteroperatory does not actually execute.
			bbR = BBRegisterOperator( self.referenceFunctionalFileName, FSsubject = self.FSsubject, contrast = contrast )
			bbR.configure( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), flirtOutputFile = True )
			cfO = ConcatFlirtOperator(bbR.flirtOutputFileName)
			cfO.configure(secondInputFile = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain_MNI.mat'), outputFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'MNI'], extension = '.mat' ))
			cfO.execute()
			# invert func to highres 
			invFl_Session_HR = InvertFlirtOperator(bbR.flirtOutputFileName)
			invFl_Session_HR.configure()
			invFl_Session_HR.execute()
			# and func to standard
			invFl_Session_Standard = InvertFlirtOperator(cfO.outputFileName)
			invFl_Session_Standard.configure()
			invFl_Session_Standard.execute()
			# and highres to standard
			invFl_HR_Standard = InvertFlirtOperator(flRT1.transformMatrixFileName)
			invFl_HR_Standard.configure()
			invFl_HR_Standard.execute()
			
			
			# make feat dir and put registration files in it
			try:
				os.mkdir(self.stageFolder(stage = 'processed/mri/reg/feat'))
			except OSError:
				pass
			# copy registration files to reg/feat directory
			subprocess.Popen('cp ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain_MNI.mat') + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'highres2standard.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + invFl_HR_Standard.outputFileName + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'standard2highres.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + bbR.flirtOutputFileName + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'example_func2highres.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + cfO.outputFileName + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'example_func2standard.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + invFl_Session_HR.outputFileName + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'highres2example_func.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + invFl_Session_Standard.outputFileName + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'standard2example_func.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain.nii.gz') + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'highres.nii.gz' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + flRT1.referenceFileName + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'standard.nii.gz' ), shell=True, stdout=PIPE).communicate()[0]
			
			# now take the firstFunc and create an example_func for it. 
			if not 'inplane_anat' in self.scanTypeList:
				subprocess.Popen('cp ' + self.runFile(stage = 'processed/mri/reg', postFix = ['mcf','meanvol'], base = 'firstFunc' ) + ' ' + os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'example_func.nii.gz' ), shell=True, stdout=PIPE).communicate()[0]
		
		# having registered everything (AND ONLY AFTER MOTION CORRECTION....) we now construct masks in the functional volume
		if makeMasks:
			# set up the first mean functional in the registration folder anyway for making masks. note that it's necessary to have run the moco first...
			ExecCommandLine('cp ' + self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf', 'meanvol'] ) + ' ' + self.runFile(stage = 'processed/mri/reg', postFix = ['mcf', 'meanvol'], base = 'firstFunc' ) )
			
			for mask in maskList:
				for hemi in ['lh','rh']:
					maskFileName = os.path.join(os.path.expandvars('${SUBJECTS_DIR}'), self.FSsubject, labelFolder, hemi + '.' + mask + '.' + 'label')
					stV = LabelToVolOperator(maskFileName)
					stV.configure( 	templateFileName = self.runFile(stage = 'processed/mri/reg', postFix = ['mcf', 'meanvol'], base = 'firstFunc' ), 
									hemispheres = [hemi], 
									register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), 
									fsSubject = self.FSsubject,
									outputFileName = self.runFile(stage = 'processed/mri/masks', base = hemi + '.' + mask ), 
									threshold = 0.001 )
					stV.execute()
	
	def setupRegistrationForFeat(self, feat_directory, wait_for_execute = True):
		"""apply the freesurfer/flirt registration for this session to a feat directory. This ensures that the feat results can be combined across runs and subjects without running flirt all the time."""
		try:
			os.mkdir(os.path.join(feat_directory,'reg'))
		except OSError:
			pass
		
		if not os.path.isdir(self.stageFolder(stage = 'processed/mri/reg/feat/')):
			self.registerSession(prepare_register = True, bb = False, MNI = True)
		
		os.system('cp ' + self.stageFolder(stage = 'processed/mri/reg/feat/') + '* ' + os.path.join(feat_directory,'reg/') )
		if wait_for_execute:
			os.system('featregapply ' + feat_directory )
		else:
			os.system('featregapply ' + feat_directory + ' & ' )
			
	def motionCorrectFunctionals(self, registerNoMC = False):
		"""
		motionCorrectFunctionals corrects all functionals in a given session.
		how we do this depends on whether we have parallel processing turned on or not
		"""
		self.logger.info('run motion correction')
		self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID]  )
		# set up a list of motion correction operator objects for the runs
		mcOperatorList = [];	stdOperatorList = [];
		for er in self.scanTypeDict['epi_bold']:
			mcf = MCFlirtOperator( self.runFile(stage = 'processed/mri', run = self.runList[er] ), target = self.referenceFunctionalFileName )
		 	mcf.configure()
			mcOperatorList.append(mcf)
			# add registration of non-motion corrected functionals to the forRegistration file
			# to be run together with the motion correction runs
			if registerNoMC:
				fO = FlirtOperator(inputObject = self.runFile(stage = 'processed/mri', run = self.runList[er] ), referenceFileName = self.referenceFunctionalFileName)
				fO.configure(resample = False)
				mcOperatorList.append(fO)
		
		if not self.parallelize:
			# first, code for serial implementation
			self.logger.info("run serial moco")
			for mcf in mcOperatorList:
				mcf.execute()
		
		if self.parallelize:
			# tryout parallel implementation - later, this should be abstracted out of course. 
			ppservers = ()
			job_server = pp.Server(ppservers=ppservers, secret='mc')
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
#			ppResults = [job_server.submit(mcf.execute,(), (), ("Tools","Tools.Operators","Tools.Sessions.MCFlirtOperator","subprocess",)) for mcf in mcOperatorList]
			ppResults = [job_server.submit(ExecCommandLine,(mcf.runcmd,),(),('subprocess','tempfile',)) for mcf in mcOperatorList]
			for fMcf in ppResults:
				fMcf()
			
			job_server.print_stats()
	
	def rescaleFunctionals(self, operations = ['bandpass', 'zscore'], filterFreqs = {'highpass': 30.0, 'lowpass': -1.0}, funcPostFix = ['mcf'], mask_file = None):#, 'percentsignalchange'
		"""
		rescaleFunctionals operates on motion corrected functionals
		and does high/low pass filtering, percent signal change or zscoring of the data
		"""
		self.logger.info('rescaling functionals with options %s', str(operations))
		for r in self.scanTypeDict['epi_bold']:	# now this is a for loop we would love to run in parallel
			funcFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = funcPostFix ))
			for op in operations:	# using this for loop will ensure that the order of operations as defined in the argument is adhered to
				if op[-4:] == 'pass':
					ifO = FSLMathsOperator(funcFile)
					ifO.configureBPF(nr_samples_hp = filterFreqs['highpass'], nr_samples_lp = filterFreqs['lowpass'] )
					if not self.parallelize:
						ifO.execute()
						funcFile = NiftiImage(ifO.outputFileName)
					else:
						if r == self.scanTypeDict['epi_bold'][0]:
							ifs = []
						ifs.append(ifO)
					# funcFile = NiftiImage(ifO.outputFileName)
				if op == 'percentsignalchange':
					pscO = PercentSignalChangeOperator(funcFile)
					pscO.execute()
					funcFile = NiftiImage(pscO.outputFileName)
				if op == 'zscore':
					zscO = ZScoreOperator(funcFile)
					# zscO.execute()
					funcFile = NiftiImage(zscO.outputFileName)
				if op == 'sgtf':
					sgtfO = SavitzkyGolayHighpassFilterOperator(funcFile)
					sgtfO.configure(mask_file = mask_file, TR = funcFile.rtime, width = 240, order = 3)
					sgtfO.execute()
				if op =='mbs':
					mbsO = MeanBrainSubtractionOperator(funcFile)
					mbsO.execute()
					funcFile = NiftiImage(mbsO.outputFileName)
					
					
		if self.parallelize and operations[0][-4:] == 'pass':
			# tryout parallel implementation - later, this should be abstracted out of course. 
			ppservers = ()
			job_server = pp.Server(ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = [job_server.submit(ExecCommandLine,(ifO.runcmd,),(),('subprocess','tempfile',)) for ifO in ifs]
			for ifOf in ppResults:
				ifOf()
				
			job_server.print_stats()
		
	
	def createMasksFromFreeSurferLabels(self, labelFolders = [], annot = True, annotFile = 'aparc.a2009s', template_condition = None, cortex = True):
		"""createMasksFromFreeSurferLabels looks in the subject's freesurfer subject folder and reads label files out of the subject's label folder of preference. (empty string if none given).
		Annotations in the freesurfer directory will also be used to generate roi files in the functional volume. The annotFile argument dictates the file to be used for this. 
		"""
		if labelFolders == []:
			labelFolders.append(self.subject.labelFolderOfPreference)
			
		if annot:
			self.logger.info('create labels based on anatomical parcelation as in %s.annot', annotFile)
			# convert designated annotation to labels in an identically named directory
			anlo = AnnotationToLabelOperator(inputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', 'rh' + '.' + annotFile + '.annot'))
			anlo.configure(subjectID = self.subject.standardFSID )
			anlo.execute()
			labelFolders.append(annotFile)
		
		if cortex:
			self.logger.info('create rois based on anatomical cortex definition', annotFile)
			for lf in [os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', 'lh.cortex.label'),
				os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', 'rh.cortex.label')]:
				lfx = os.path.split(lf)[-1]
				if 'lh' in lfx:
					hemi = 'lh'
				elif 'rh' in lfx:
					hemi = 'rh'
				lvo = LabelToVolOperator(lf)
				# we convert the label files to the space of the first EPI run of the session, moving them into masks/anat.
				if template_condition == None:
					template_file = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]])
				else:
					template_file = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[template_condition][0]])
				
				lvo.configure(templateFileName = template_file, hemispheres = [hemi], register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), fsSubject = self.subject.standardFSID, outputFileName = self.runFile(stage = 'processed/mri/masks/anat/', base = lfx[:-6] ), threshold = 0.05, surfType = 'label')
				lvo.execute()

		# go through the label folders and make some anatomical masks
		for lf in labelFolders:
			self.logger.info('creating masks from labels folder %s', os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', lf))
			labelFiles = subprocess.Popen('ls ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', lf) + '/*.label', shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
			for lf in labelFiles:
				lfx = os.path.split(lf)[-1]
				if 'lh' in lfx:
					hemi = 'lh'
				elif 'rh' in lfx:
					hemi = 'rh'
				lvo = LabelToVolOperator(lf)
				# we convert the label files to the space of the first EPI run of the session, moving them into masks/anat.
				if template_condition == None:
					template_file = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]])
				else:
					template_file = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[template_condition][0]])
				
				lvo.configure(templateFileName = template_file, hemispheres = [hemi], register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), fsSubject = self.subject.standardFSID, outputFileName = self.runFile(stage = 'processed/mri/masks/anat/', base = lfx[:-6] ), threshold = 0.05, surfType = 'label')
				lvo.execute()
		
	
	def createMasksFromFreeSurferAseg(self, asegFile = 'aparc.a2009s+aseg', which_regions = ['Putamen', 'Caudate', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens', 'Cerebellum_Cortex', 'Thalamus_Proper', 'Thalamus', 'VentralDC']):
		"""createMasksFromFreeSurferLabels looks in the subject's freesurfer subject folder and reads label files out of the subject's label folder of preference. (empty string if none given).
		Annotations in the freesurfer directory will also be used to generate roi files in the functional volume. The annotFile argument dictates the file to be used for this. 
		"""
		# convert aseg file to nii
		os.system('mri_convert ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', asegFile + '.mgz') + ' ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', asegFile + '.nii.gz')) 
		
		flO = FlirtOperator(inputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', asegFile + '.nii.gz'), referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf']))
		flO.configureApply(os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat/'), 'highres2example_func.mat'), outputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/masks'), asegFile + '.nii.gz'))
		if not os.path.isfile(flO.outputFileName):
			flO.execute()
		
		# find the subcortical labels in the converted brain from aseg.auto_noCCseg.label_intensities file
		with open(os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'aseg.auto_noCCseg.label_intensities.txt')) as f:
			a = f.readlines()
		list_of_areas = [[al.split(' ')[0], al.split(' ')[1]] for al in a]
		
		# open file
		aseg_image = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks'), asegFile + '.nii.gz'))
		for wanted_region in which_regions:
			for area in list_of_areas:
				if len(area[1].split('_')) > 1:
					if area[1].split('_')[1] == wanted_region:
						prefix = ''
						if area[1].split('_')[0] == 'Left':
							prefix = 'lh.'
						elif area[1].split('_')[0] == 'Right':
							prefix = 'rh.'
						# shell()
						label_opf_name = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'),  prefix + wanted_region + '.nii.gz')
						this_label_data = np.array(aseg_image.data == int(area[0]), dtype = int)
						# create new image
						label_opf = NiftiImage(this_label_data)
						label_opf.header = aseg_image.header
						label_opf.save(label_opf_name)
						self.logger.info( area[1] + ' outputted to ' + label_opf_name )
					
					
	
	
	def masksWithStatMask(self, originalMaskFolder = 'anat', statMasks = None, statMaskNr = 0, absolute = False, toSurf = False, thresholds = [2.0], maskFunction = '__gt__', delete_older_files = False):
		# now take those newly constructed anatomical masks and use them to mask the statMasks, if any, or just copy them to the lower level for wholesale use.
		roiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/' + originalMaskFolder ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(roiFileNames))
		rois = []
		for roi in roiFileNames:
			rois.append(NiftiImage(roi))
		
		# at this point, we're going to replenish the masks in the masks folder. We delete all .nii.gz files in that folder
		if delete_older_files:
			os.system('rm ' + self.runFile(stage = 'processed/mri/masks/', base = '*' ) )
		
		if statMasks:
			for statMask in statMasks:
				if absolute:
					if statMaskNr != None:
						statMaskData = np.abs(NiftiImage(self.runFile(stage = 'processed/mri/masks/stat/', base = statMask )).data[statMaskNr])
					else:
						statMaskData = np.abs(NiftiImage(self.runFile(stage = 'processed/mri/masks/stat/', base = statMask )).data)
				else:
					if statMaskNr != None:
						statMaskData = NiftiImage(self.runFile(stage = 'processed/mri/masks/stat/', base = statMask )).data[statMaskNr]
					else:
						statMaskData = NiftiImage(self.runFile(stage = 'processed/mri/masks/stat/', base = statMask )).data
				for rn in range(len(rois)):
					imo = ImageMaskingOperator(rois[rn], maskObject = statMaskData, thresholds = thresholds, outputFileName = self.runFile(stage = 'processed/mri/masks/', base = os.path.split(rois[rn].filename)[1][:-7] + '_' + statMask, extension = '' ))
					imo.applyAllMasks( maskFunction = maskFunction )
					
				if toSurf:
					# convert the statistical masks to surfaces
					vtsO = VolToSurfOperator(self.runFile(stage = 'processed/mri/masks/stat/', base = statMask ))
					vtsO.configure(frames = {statMask:0}, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = self.runFile(stage = 'processed/mri/masks/surf/', base = '' ), threshold = 0.5, surfSmoothingFWHM = 2.0)
					vtsO.execute()
		else:	# in this case copy the anatomical masks to the masks folder where they'll be used for the following extraction of functional data
			os.system('cp ' + self.runFile(stage = 'processed/mri/masks/anat/', base = '*' ) + ' ' + self.stageFolder(stage = 'processed/mri/masks/') )
			
				
		
	def maskFunctionalData(self, maskThreshold = 0.0, postFixFunctional = ['mcf'], timeSlices = [0,-1], delete_older_files = True):
		"""
		maskFunctionalData will mask each bold file with the masks present in the masks folder.
		"""
		roiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('masking functional data from files %s', str([os.path.split(f)[1] for f in roiFileNames]))
		rois = []
		for roi in roiFileNames:
			rois.append(NiftiImage(roi))
		
		for r in self.scanTypeDict['epi_bold']:
			if delete_older_files:
				maskedFiles = subprocess.Popen('ls ' + os.path.join( self.runFolder(stage = 'processed/mri/', run = self.runList[r]), 'masked/*.nii.gz' ), shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
				if len(maskedFiles) > 0:
					# delete older masked data
					self.logger.info("removing older masked data: %s", 'rm ' + os.path.join( self.runFolder(stage = 'processed/mri/', run = self.runList[r]), 'masked/*' )) 
					print os.system('rm ' + os.path.join( self.runFolder(stage = 'processed/mri/', run = self.runList[r]), 'masked/*' ) )
			funcFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = postFixFunctional ))
			for rn in range(len(rois)):
				if timeSlices[1] != -1:
					imo = ImageMaskingOperator(funcFile.data[timeSlices[0]:timeSlices[1]], maskObject = rois[rn], thresholds = [maskThreshold], outputFileName = self.runFile(stage = 'processed/mri', run = self.runList[r], base = 'masked/' + os.path.split(rois[rn].filename)[1][:-7], extension = '' ))
				else:
					imo = ImageMaskingOperator(funcFile.data[timeSlices[0]:], maskObject = rois[rn], thresholds = [maskThreshold], outputFileName = self.runFile(stage = 'processed/mri', run = self.runList[r], base = 'masked/' + os.path.split(rois[rn].filename)[1][:-7], extension = '' ))
				imo.applyAllMasks(save = True, maskFunction = '__gt__', flat = True)
		
	def gatherRIOData(self, roi, whichRuns, whichMask = '_thresh_z_stat', timeSlices = [0,-1] ):
		data = []
		for r in whichRuns:
			# roi is either a list or a string. if it's a list, we iterate across different rois. if it's a string, we make it into a 1-member list first before entering the for loop.
			if roi.__class__.__name__ == 'str':
				roi = [roi]
			runData = []
			for thisRoi in roi:
				# get ROI
				if thisRoi[:2] in ['lh','rh']:	# single - hemisphere roi
					if os.path.isfile(self.runFile(stage = 'processed/mri', run = self.runList[r], base = 'masked/' + thisRoi + whichMask, extension = '.npy')):
						thisRoiData = np.load(self.runFile(stage = 'processed/mri', run = self.runList[r], base = 'masked/' + thisRoi + whichMask, extension = '.npy'))[0]
					else:
						thisRoiData = np.array([])
				else: # combine both hemispheres in one roi
					if os.path.isfile(self.runFile(stage = 'processed/mri', run = self.runList[r], base = 'masked/lh.' + thisRoi + whichMask, extension = '.npy')):
						ld = np.load(self.runFile(stage = 'processed/mri', run = self.runList[r], base = 'masked/lh.' + thisRoi + whichMask, extension = '.npy'))[0]
						rd = np.load(self.runFile(stage = 'processed/mri', run = self.runList[r], base = 'masked/rh.' + thisRoi + whichMask, extension = '.npy'))[0]
						thisRoiData = np.hstack((ld,rd))
					else:
						thisRoiData = np.array([])
				if thisRoiData.shape[0] > 0:
					if timeSlices[1] == -1:
						runData.append(thisRoiData[timeSlices[0]:])
					else:
						runData.append(thisRoiData[timeSlices[0]:timeSlices[1]])
			data.append(np.hstack(runData))
		return np.vstack(data)
	
	def importStatMask(self, statMaskFile, registrationEPIFile, reregisterSession = True, force_operations = False, use_subject_anat = True, reg_method = 'bbregister', force_final = True):
		"""
		statmask is to be converted to anatomical space for this subject, 
		after that from anatomical to present session epi format.
		"""
		# create some of the names involved
		statMaskName = os.path.split(os.path.splitext(statMaskFile)[0])[1]
		concatenatedRegistrationFileName = os.path.join(self.stageFolder(stage = 'processed/mri/reg/'), statMaskName + '.mat')
		registeredToAnatStatMaskName = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), statMaskName + '_anat_brain.nii.gz')
		registeredToAnatSessionFileName = os.path.join(self.stageFolder(stage = 'processed/mri/reg/'), 'session_to_anat_brain.nii.gz')
		registeredToSessionStatMaskName = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), statMaskName + '.nii.gz')
		
		self.logger.info('starting import of mask file ' + statMaskFile + ' named ' + statMaskName)
		
		if use_subject_anat:
			# convert subject's anatomical to nii.gz.
			convO = MRIConvertOperator(inputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain.mgz'))
			convO.configure()
			if not os.path.isfile(convO.outputFileName) or force_operations:
				convO.execute()
			reg_target = convO.outputFileName
		else:
			reg_target = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
		
		if reg_method == 'flirt':
			# register the statmask
			statMask_flO = FlirtOperator(inputObject = registrationEPIFile, referenceFileName = reg_target)
			statMask_flO.configureRun(outputFileName = registeredToAnatStatMaskName) # , extra_args = ' -searchrx -180 180 -searchry -180 180 -searchrz -180 180 '
			if not os.path.isfile(statMask_flO.outputFileName) or force_operations:
				statMask_flO.execute()
			maskRegFile = statMask_flO.transformMatrixFileName
		elif reg_method == 'bbregister':
			bbR1 = BBRegisterOperator( registrationEPIFile, FSsubject = self.subject.standardFSID )
			bbR1.configure( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register_' + os.path.splitext(os.path.splitext(os.path.split(statMaskFile)[1])[0])[0], postFix = [self.ID], extension = '.dat' ), flirtOutputFile = True )
			if not os.path.isfile(bbR1.transformMatrixFileName) or force_operations:
				bbR1.execute()
			maskRegFile = bbR1.flirtOutputFileName
		
		if reregisterSession:
			# register the session's epi data
			sessionEPI_flO = FlirtOperator(inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf','meanvol']), referenceFileName = reg_target)
			sessionEPI_flO.configureRun(outputFileName = registeredToAnatSessionFileName, extra_args = ' -searchrx -180 180 -searchry -180 180 -searchrz -180 180 ') #, extra_args = ' -searchrx -180 180 -searchry -180 180 -searchrz -180 180 '
			if not os.path.isfile(sessionEPI_flO.outputFileName) or force_operations:
				sessionEPI_flO.execute()
			
			bbR2 = BBRegisterOperator( self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf','meanvol']), FSsubject = self.subject.standardFSID )
			bbR2.configure( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register_session_for_' + os.path.splitext(os.path.splitext(os.path.split(statMaskFile)[1])[0])[0], postFix = [self.ID], extension = '.dat' ), flirtOutputFile = True )
			if not os.path.isfile(bbR2.transformMatrixFileName) or force_operations:
				bbR2.execute()
			
			# invert the session's epi registration
			if reg_method == 'flirt':
				sessionEPI_inv_flO = InvertFlirtOperator(inputObject = sessionEPI_flO.transformMatrixFileName)
			elif reg_method == 'bbregister':
				sessionEPI_inv_flO = InvertFlirtOperator(inputObject = bbR2.flirtOutputFileName)
			sessionEPI_inv_flO.configure()
			if not os.path.isfile(sessionEPI_inv_flO.outputFileName) or force_operations:
				sessionEPI_inv_flO.execute()
			
			inverseTransformFile = sessionEPI_inv_flO.outputFileName
		else:
			# these two steps above this one have already been done in registerSession, so:
			inverseTransformFile =  self.runFile(stage = 'processed/mri/reg', base = 'flirt', postFix = [self.ID], extension = '.mat' )
		
		# concatenate registrations
		ccO = ConcatFlirtOperator(inputObject = maskRegFile)
		ccO.configure(secondInputFile = inverseTransformFile, outputFileName = concatenatedRegistrationFileName)
		if not os.path.isfile(ccO.outputFileName) or force_final:
			ccO.execute()
		
		# apply the transform
		final_flO = FlirtOperator(inputObject = statMaskFile, referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf']))
		final_flO.configureApply(concatenatedRegistrationFileName, outputFileName = registeredToSessionStatMaskName)
		if not os.path.isfile(final_flO.outputFileName) or force_final:
			final_flO.execute()
	
	def takePhaseSurfacesToFuncSpace(self, folder = '', fn = 'eccen', template_condition = None):
		for hemi in ['lh','rh']:
			stvO = SurfToVolOperator(os.path.join(folder, 'phase-' + hemi + '.w'))
			if template_condition == None:
				# template_file = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf','meanvol'])
				template_file = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID]  )
			else:
				template_file = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[template_condition][0]], postFix = ['mcf','meanvol'])
			stvO.configure(
							templateFileName = template_file, 
							register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), 
							fsSubject = self.subject.standardFSID, 
							outputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , fn + '.nii.gz'),
							hemispheres = [hemi]
							)
			# make sure it doesn't continue before there are results as nii files
			stvO.runcmd = stvO.runcmd[:-2]
			stvO.execute()
		# join eccen files
		phaseData = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , fn + '-lh.nii.gz')).data + NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , fn + '-rh.nii.gz')).data
		newImage = NiftiImage(phaseData)
		newImage.header = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , fn + '-lh.nii.gz')).header
		newImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , fn + '.nii.gz')
		newImage.save()
	
	def run_data_from_hdf(self, h5file, run, data_type, postFix = ['mcf']):
		"""docstring for parameter_data_from_hdf"""
		this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]
		try:
			thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
		except NoSuchNodeError:
			# import actual data
			self.logger.info('No group ' + this_run_group_name + ' in this file')
			return None
		
		this_data_array = eval('thisRunGroup.' + data_type + '.read()') 
		return this_data_array
		
	
	def roi_data_from_hdf(self, h5file, run = '', roi_wildcard = 'v1', data_type = 'tf_psc_data', postFix = ['mcf']):
		"""
		drags data from an already opened hdf file into a numpy array, concatenating the data_type data across voxels in the different rois that correspond to the roi_wildcard
		"""
		if type(run) == str:
			this_run_group_name = run
		# elif type(run) == Tools.Run:
		else:
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]

		try:
			thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			roi_names = []
			for roi_name in h5file.iter_nodes(where = '/' + this_run_group_name, classname = 'Group'):
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
			thisRoi = h5file.get_node(where = '/' + this_run_group_name, name = roi_name, classname='Group')
			all_roi_data.append( eval('thisRoi.' + data_type + '.read()') )
		all_roi_data_np = np.hstack(all_roi_data).T
		return all_roi_data_np
	
	def run_glm_on_hdf5(self, run_list = None, hdf5_file = None, data_type = 'hpf_data', analysis_type = 'per_trial', post_fix_for_text_file = ['all_trials'], functionalPostFix = ['mcf'], design = None, contrast_matrix = []):
		"""
		run_glm_on_hdf5 takes an open (r+) hdf5 file, a list of run objects and runs glms on all roi subregions from a run.
		it assumes:
			1. an hdf5 file that has groups for each of the _mcf.nii.gz files in the runlist
			2. that each of these groups has a sequence of roi data arrays
			3. a nii.gz file that can be found with the session's runFile function
			4. a fsl-event text file in the run folders from which to take the separate-trial regressors.
		"""
		# reward_h5file = self.hdf5_file('reward', mode = 'r+')
		for run in run_list:
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = functionalPostFix))
			tr, nr_trs = round(niiFile.rtime * 100) / 100.0, niiFile.timepoints	# needed to do this thing with the trs or else it would create new TRs in the end of the designMatrix.
			
			# everyone shares the same design matrix.
			if analysis_type == 'per_trial':
				event_data = np.loadtxt(self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = post_fix_for_text_file))[:] 
				design = Design(nrTimePoints = nr_trs, rtime = tr)
				for i in range(event_data.shape[0]):
					design.addRegressor([event_data[i]])
				design.convolveWithHRF()
				designMatrix = design.designMatrix
			elif analysis_type == 'from_design':
				if design != None:
					designMatrix = design[run_list.index(run)]
				else:
					print 'no designMatrix specified'
					return
			my_glm = nipy.labs.glm.glm.glm()
			
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = functionalPostFix))[1]
			try:
				thisRunGroup = hdf5_file.get_node(where = '/', name = this_run_group_name, classname='Group')
				for roi_name in hdf5_file.list_nodes(where = '/' + this_run_group_name, classname = 'Group'):
					if roi_name._v_name.split('.')[0] in ('rh', 'lh'):
						roi_data = eval('roi_name.' + data_type + '.read()')
						roi_data = roi_data.T - roi_data.mean(axis = 1)
						glm = my_glm.fit(roi_data.T, designMatrix, method="kalman", model="ar1")
						try: 
							hdf5_file.remove_node(where = roi_name, name = analysis_type + '_' + data_type + '_' + 'betas')
						except NoSuchNodeError:
							pass
						hdf5_file.create_array(roi_name, analysis_type + '_' + data_type + '_' + 'betas', my_glm.beta, 'beta weights for per-trial glm analysis on region ' + str(roi_name) + ' conducted at ' + datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
						stat_matrix = []
						zscore_matrix = []
						for i in range(designMatrix.shape[-1]):
							this_contrast = np.zeros(designMatrix.shape[-1])
							this_contrast[i] = 1.0
							stat_matrix.append(my_glm.contrast(this_contrast).stat())
							zscore_matrix.append(my_glm.contrast(this_contrast).zscore())
						if np.array(contrast_matrix).shape[0] > 0:
							self.logger.info('calculating extra contrasts for ' + this_run_group_name)
							for i in range(np.array(contrast_matrix).shape[0]):
								this_contrast = np.zeros(designMatrix.shape[-1])
								contrast_factors = contrast_matrix[i] != 0
								this_contrast[contrast_factors] = contrast_matrix[i][contrast_factors]
								stat_matrix.append(my_glm.contrast(this_contrast).stat())
								zscore_matrix.append(my_glm.contrast(this_contrast).zscore())
						try: 
							hdf5_file.remove_node(where = roi_name, name = analysis_type + '_' + data_type + '_' + 'stat')
							hdf5_file.remove_node(where = roi_name, name = analysis_type + '_' + data_type + '_' + 'zscore')
						except NoSuchNodeError:
							pass
						hdf5_file.create_array(roi_name, analysis_type + '_' + data_type + '_' + 'stat', np.array(stat_matrix), 'stats for per-trial glm analysis on region ' + str(roi_name) + ' conducted at ' + datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
						hdf5_file.create_array(roi_name, analysis_type + '_' + data_type + '_' + 'zscore', np.array(zscore_matrix), 'zscores for per-trial glm analysis on region ' + str(roi_name) + ' conducted at ' + datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
						self.logger.info('beta weights and stats for per-trial glm analysis on region ' + str(roi_name) + ' conducted')
			except NoSuchNodeError:
				# import actual data
				self.logger.info('No group ' + this_run_group_name + ' in this file')
				# return None
	
	def resample_epis(self, conditions = ['PRF']):
		"""resample_epi resamples the mc'd epi files back to their functional space."""
		# create identity matrix
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg'), 'eye.mtx'), np.eye(4), fmt = '%1.1f')
		
		cmds = []
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				fO = FlirtOperator(inputObject = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'] ),  referenceFileName = self.runFile(stage = 'processed/mri', run = r ))
				# fO.configureApply( transformMatrixFileName = os.path.join(self.stageFolder(stage = 'processed/mri/reg'), 'eye.mtx'), outputFileName = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res'] ) ) 
				fO.configureApply( transformMatrixFileName = None, outputFileName = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res'] ) ) 
				cmds.append(fO.runcmd)
		
		# run all of these resampling commands in parallel
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers)
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(fo,),(),('subprocess','tempfile',)) for fo in cmds]
		for fo in ppResults:
			fo()
		
		# now put stuff back in the right places
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hr']) )
				os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res']) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) )
			
	
	def resample_epis2(self, conditions=['PRF'], postFix=['mcf']):
		"""resample_epi resamples the mc'd epi files back to their functional space."""
		
		postFix_hr = [pf for pf in postFix]
		postFix_hr.append('hr')
		
		# rename motion corrected nifti to nifti_hr (for high res):
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix_hr) )
		
		# resample:
		cmds = []
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				inputObject = self.runFile(stage = 'processed/mri', run = r, postFix = postFix_hr)
				outputObject = self.runFile(stage = 'processed/mri', run = r, postFix = postFix)
				fmO = FSLMathsOperator(inputObject=inputObject)
				fmO.configure(outputFileName=outputObject, **{'-subsamp2offc': ''})
				cmds.append(fmO.runcmd)
		
		# run all of these commands in parallel
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers)
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(fo,),(),('subprocess','tempfile',)) for fo in cmds]
		for fo in ppResults:
			fo()
			
		# fix the 4th dimension, that is TR:
		cmds = []
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				pixdim1 = str(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).pixdim[0])
				pixdim2 = str(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).pixdim[1])
				pixdim3 = str(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).pixdim[2])
				pixdim4 = str(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).pixdim[3])
				cmds.append('fslchpixdim ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' ' + pixdim1 + ' ' + pixdim2 + ' ' + pixdim3 + ' ' + pixdim4)
			
		# run all of these commands in parallel
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers)
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(fo,),(),('subprocess','tempfile',)) for fo in cmds]
		for fo in ppResults:
			fo()
	
	def create_dilated_cortical_mask(self, dilation_sd = 0.5, label = 'cortex'):
		"""create_dilated_cortical_mask takes the rh and lh cortex files and joins them to one cortex.nii.gz file.
		it then smoothes this mask with fslmaths, using a gaussian kernel. 
		This is then thresholded at > 0.0, in order to create an enlarged cortex mask in binary format.
		"""
		# take rh and lh files and join them.
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'rh.' + label + '.nii.gz'))
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), '' + label + '.nii.gz'), **{'-add': os.path.join(self.stageFolder('processed/mri/masks/anat'), 'lh.' + label + '.nii.gz')})
		fmO.execute()
		
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), '' + label + '.nii.gz'))
		fmO.configureSmooth(smoothing_sd = dilation_sd)
		fmO.execute()
		
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), label + '_s%.2f.nii.gz'%dilation_sd))
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), label + '_dilated_mask.nii.gz'), **{'-bin': ''})
		fmO.execute()

	def retroicor_run(self, r, retroicor_script_file = None, nr_dummies = 6, onset_slice = 1, gradient_direction = 'x', waitForExecute = True, execute = True):
		"""retroicor_run takes a run as input argument and runs its physiology through the retroicor operator"""
		# find out the parameters of the nifti file
		run_nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r) )
		# shell()
		if hasattr(r, 'MB_factor'):
			slice_count = str(int(run_nii_file.data.shape[1] / r.MB_factor))
		else:
			slice_count = str(run_nii_file.data.shape[1])
		REDict = {
			'---LOG_FILE---': self.runFile(stage = 'processed/hr', run = r, extension = '.log'),
			'---NR_TRS---': str(run_nii_file.data.shape[0]),
			'---NR_SLICES---': slice_count,
			'---NR_SLICES_PER_BEAT---': slice_count,
			'---TR_SECS---': str(run_nii_file.rtime),
			'---NR_DUMMIES---': str(nr_dummies),
			'---ONSET_SLICE---': str(onset_slice),
			'---GRADIENT_DIRECTION---': gradient_direction,
			'---OUTPUT_FILE_NAME---': self.runFile(stage = 'processed/hr', run = r, postFix = ['regressors'], extension = '.txt'),
		}
		
		# get template script
		if retroicor_script_file == None:
			retroicor_script_file = os.path.join(os.environ['ANALYSIS_HOME'], 'Tools', 'other_scripts', 'RETROICOR_PPU_template.m')
		# start retroicor operator with said file
		rO = RETROICOROperator(retroicor_script_file)
		rO.configure(REDict = REDict, retroicor_m_filename = self.runFile(stage = 'processed/hr', run = r, extension = '.m'), waitForExecute = waitForExecute)
		if execute:
			rO.execute()
			
		return rO.runcmd
	
	def retroicor_check(self, r):
		"""docstring for retroicor_check"""
		# some diagnostics
		if not os.path.isfile(self.runFile(stage = 'processed/hr', run = r, postFix = ['regressors'], extension = '.txt')):
			self.logger.error('retroicor did not return regressors')
		else:
			regressors = np.loadtxt(self.runFile(stage = 'processed/hr', run = r, postFix = ['regressors'], extension = '.txt'))
			pl.figure()
			pl.imshow(regressors)
			pl.savefig(self.runFile(stage = 'processed/hr', run = r, postFix = ['regressors'], extension = '.pdf'))
			self.logger.debug('please verify regressors in %s' % self.runFile(stage = 'processed/hr', run = r, postFix = ['regressors'], extension = '.pdf'))
		
	def physio_retroicor(self, condition = '', retroicor_script_file = None, nr_dummies = 6, onset_slice = 1, gradient_direction = 'x'):
		"""physio loops across runs to analyze their physio data"""
		cmd_list = []
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			if not self.parallelize:
				self.retroicor_run(r, retroicor_script_file = retroicor_script_file, nr_dummies = nr_dummies, onset_slice = onset_slice, gradient_direction = gradient_direction, waitForExecute = True)
				self.retroicor_check(r)
			else:
				cmd_list.append(self.retroicor_run(r, retroicor_script_file = retroicor_script_file, nr_dummies = nr_dummies, onset_slice = onset_slice, gradient_direction = gradient_direction, execute = False))
	
		if self.parallelize:
			# tryout parallel implementation - later, this should be abstracted out of course. 
			ppservers = ()
			job_server = pp.Server(ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
#			ppResults = [job_server.submit(mcf.execute,(), (), ("Tools","Tools.Operators","Tools.Sessions.MCFlirtOperator","subprocess",)) for mcf in mcOperatorList]
			ppResults = [job_server.submit(ExecCommandLine,(cmd,),(),('subprocess','tempfile',)) for cmd in cmd_list]
			for fRICf in ppResults:
				fRICf()
	
			job_server.print_stats()
			
			for r in [self.runList[i] for i in self.conditionDict[condition]]:
				self.retroicor_check(r)
		
		
	def fRMI_quality(self):
		
		# ----------------------------------------
		# Run fMRI data quality check:           -
		# ----------------------------------------

		matlab_script = os.path.join(os.environ['ANALYSIS_HOME'], 'Tools/other_scripts/fMRI_quality/fMRI_quality_template.m')
		for er in self.scanTypeDict['epi_bold']:
			REDict = {
			'---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = self.runList[er]),
			'---FUNC_FILE_DIR---':os.path.split(self.runFile(stage = 'processed/mri', run = self.runList[er]))[0],
			}
			qcheck = RETROICOROperator(inputObject = matlab_script)
			if self.runList[er] == [self.runList[i] for i in self.scanTypeDict['epi_bold']][-1]:
				qcheck.configure(REDict = REDict, retroicor_m_filename = os.path.split(self.runFile(stage = 'processed/mri', run = self.runList[er]))[0] + '/fMRI_quality_filled_in.m', waitForExecute = True)
			else:
				qcheck.configure(REDict = REDict, retroicor_m_filename = os.path.split(self.runFile(stage = 'processed/mri', run = self.runList[er]))[0] + '/fMRI_quality_filled_in.m', waitForExecute = False)
			qcheck.execute()
		
		# ----------------------------------------
		# Move into figures folder:              -
		# ----------------------------------------
		
		import time as time
		time.sleep(10)
		
		# makedir:
		fig_dir = self.stageFolder(stage = 'processed/mri/figs/') + 'quality_check/'
		try:
			os.system('rm -rf ' + fig_dir)
		except OSError:
			pass
		subprocess.Popen('mkdir ' +  fig_dir, shell=True, stdout=PIPE).communicate()[0]
		
		# copy
		files = ['h_m_bkg', 'h_m_gho', 'h_m_obj_gho', 'h_m_obj', 'h_roi_loc', 'h_spikes']
		for er in self.scanTypeDict['epi_bold']:
			for file in files:
				paths = os.path.split(self.runFile(stage = 'processed/mri', run = self.runList[er]))
				copy_in = paths[0] + '/qReport_' + paths[1].split('.')[0] + '/figures/' + file + '.png'
				copy_out = fig_dir + '/' + file + str(self.runList[er].ID) + '.png'
				subprocess.Popen('cp ' + copy_in + ' ' + copy_out, shell=True, stdout=PIPE).communicate()[0]
	