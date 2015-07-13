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

import glob

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
import xml.etree.ElementTree as ET

from ..Operators.HDFEyeOperator import HDFEyeOperator

from joblib import Parallel, delayed

from ..Operators.HDFEyeOperator import HDFEyeOperator

from IPython import embed as shell

def numerical_derivative(data):
	"""
	This function computes the numerical derivative of the input where its output is the 
	same length as the input. It does so by using both interpolation and extrapolation.

	For a tutorial on this function see shared/tutorials
	"""

	diff_data = np.diff(data)
	diff_x = np.arange(0.5,len(data)-1)

	interp_diff = sp.interpolate.interp1d(diff_x,diff_data,kind='cubic')

	def extrap1d(interpolator):
		xs = interpolator.x
		ys = interpolator.y

		def pointwise(x):
			if x < xs[0]:
				return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
			elif x > xs[-1]:
				return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
			else:
				return interpolator(x)

		def ufunclike(xs):
			return np.array(map(pointwise, np.array(xs)))

		return ufunclike

	return extrap1d(interp_diff)(np.arange(len(data)))

def run_moco_dt_ddt_parallel(run,mcf_filename):
	"""
	For python parallel to work, this function cannot be part of the Session class.
	"""

	mcf = np.loadtxt(mcf_filename)

	mcf_dt = []; mcf_ddt = []
	for par in np.array(mcf).T:
		mcf_dt.append(numerical_derivative(par))
		mcf_ddt.append(numerical_derivative(numerical_derivative(par)))

	mcf_dt = np.array(mcf_dt).T
	mcf_ddt = np.array(mcf_ddt).T

	np.savetxt(mcf_filename[:-4]+'_dt.par', mcf_dt, delimiter="\t",fmt='%.7f')
	np.savetxt(mcf_filename[:-4]+'_ddt.par', mcf_ddt, delimiter="\t",fmt='%.7f')

class PathConstructor(object):
	"""
	FilePathConstructor is an abstract superclass for sessions.
	It constructs the file and folder naming hierarchy for a given session.
	All file naming and calling runs through this class.
	"""
	def __init__(self):
		self.fileNameBaseString = self.dateCode
	
	def base_dir(self):
		"""
		base_dir returns the path in which all data of this session
		is located. this path is located within this observer's
		folder which, in turn, is located within this project's folder
		"""
		return os.path.join(self.project.base_dir, self.subject.initials, self.dateCode)
	
	def make_base_dir(self):
		"""
		make_base_dir creates the path in which  all data of this session
		will be located. it will attempt to build the folder tree towards
		this path, but will run into trouble if even the folder containing
		self.project.base_dir doesn't exist
		"""
		if not os.path.isdir(self.base_dir()):
			try:
				os.mkdir(self.project.base_dir)
			except OSError:
				pass
			try:
				os.mkdir(os.path.join(self.project.base_dir, self.subject.initials))
			except OSError:
				pass
			try:
				os.mkdir(self.base_dir())
			except OSError:
				pass
	
	def stageFolder(self, stage):
		"""
		stageFolder returns the path associated with a certain analysis stage
		such as the subfolders 'raw/mri' or 'processed/eyelink' within the session's base_dir.
		"""
		return os.path.join(self.base_dir(), stage)
	
	def runFolder(self, stage, run):
		"""
		runFolder returns the folder path, within a certain analysis stage (e.g. 'raw/mri/'),
		associated with a particular run (e.g. the run whose ID is 1) that belongs to a certain condition (e.g. 'mapper').
		the return object of that example would be 'raw/mri/mapper/1'
		"""
		return os.path.join(self.stageFolder(stage), run.condition, str(run.ID))
	
	def conditionFolder(self, stage, run):
		"""docstring for runFolder"""
		return os.path.join(self.stageFolder(stage), run.condition)
	
	def runFile(self, stage = None, run = None, postFix = [], extension = standardMRIExtension, base = None):
		"""
		runFile returns the file path, within a certain analysis stage (e.g. 'raw/mri'),
		associated with a particular run (e.g. 1) that belongs to a certain condition (e.g. 'mapper').
		If 'stage' is left blank, then only the filename, rather than the full path, is returned.
		postFix is typically something like ['mcf'], for getting the motion corrected path, but
		can also be an array of strings, which will then be concatenated with '_' separators.
		
		runFile is more flexible than runFolder, because the run
		can have many different relevant files, for instance a motion corrected file and a non-motion
		corrected file, all within the same folder
	
		example of usage 1 (JB, dec 22 2014: I don't quite understand this first one):
		the raw mri file for run index X can be obtained with
		self.runFile('raw/mri', postFix = [str(self.runList[X])], base = self.subject.initials)
		
		example of usage 2:
		the motion corrected file of the run at index X in runList can be obtained with
		self.runFile('processed/mri', run = self.runList[X], postFix = ['mcf'])
		if self.fileNameBaseString for this session is 'QQ_101214', this run's ID and 
		condition are 'mapper' and 1, respectively, and standardMRIExtension (defined in Operator.py)
		is '.nii.gz', then the result, relative to self.base_dir(),
		is: 'processed/mri/mapper/1/QQ_101214_1_mcf.nii.gz'
		
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
		
		if stage is None:
			return fn
		elif run and stage.split('/')[0] == 'processed':
			return os.path.join(self.runFolder(stage, run), fn)
		else:
			return os.path.join(self.stageFolder(stage), fn)
	
	def create_folder_hierarchy(self):
		"""
		create_folder_hierarchy creates the folder tree for a session. It needs for at least
		the folder enclosing the project folder self.project.base_dir to exist,
		and will build from there: project/observer/session/[further subfolders].
		"""
		rawFolders = ['raw/%s'%modality for modality in self.modality_list]
		self.processed_folders = ['processed/%s'%modality for modality in self.modality_list]
		# rawFolders = ['raw/mri', 'raw/behavior', 'raw/eye', 'raw/hr']
		# self.processed_folders = ['processed/mri', 'processed/behavior', 'processed/eye', 'processed/hr']
		mri_condition_folders = np.concatenate((self.conditionList, ['log','figs','scripts','masks','masks/stat','masks/anat','reg','surf']))
		condition_folders = np.concatenate((self.conditionList, ['log','figs','scripts']))
		
		
		self.make_base_dir()
		# assuming baseDir/raw/ exists, we must make processed
		if not os.path.isdir(os.path.join(self.base_dir(), 'processed') ):
			os.mkdir(os.path.join(self.base_dir(), 'processed'))
		if not os.path.isdir(os.path.join(self.base_dir(), 'raw') ):
			os.mkdir(os.path.join(self.base_dir(), 'raw'))
		
		
		# create folders for processed data
		for pf in self.processed_folders:
			if not os.path.isdir(self.stageFolder(pf)):
				os.mkdir(self.stageFolder(pf))
			# create condition folders in each of the processed data folders and also their surfs
			if pf.split('/')[-1] == 'mri':
				cf = mri_condition_folders
			else:
				cf = condition_folders
			for c in cf:
				 if not os.path.isdir(self.stageFolder(pf+'/'+c)):
					os.mkdir(self.stageFolder(pf+'/'+c))
					if pf == 'processed/mri':
						if not os.path.isdir(os.path.join(self.stageFolder(pf+'/'+c), 'surf')):
							os.mkdir(os.path.join(self.stageFolder(pf+'/'+c), 'surf'))
			# create folders for each of the runs in the session and their surfs. NB:
			# these will only be created within the condition folder indicated by 
			# each Run object (i.e. associated with the condition that the run belongs to).
			for rl in self.runList:
				if not os.path.isdir(self.runFolder(pf, run = rl)):
					os.mkdir(self.runFolder(pf, run = rl))
				if not os.path.isdir(os.path.join(self.runFolder(pf, run = rl), 'events')):
					os.mkdir(os.path.join(self.runFolder(pf, run = rl), 'events'))
					if pf == 'processed/mri':
						if not os.path.isdir(os.path.join(self.runFolder(pf, run = rl), 'surf')):
							os.mkdir(os.path.join(self.runFolder(pf, run = rl), 'surf'))
						# if not os.path.isdir(os.path.join(self.runFolder(pf, run = rl), 'masked')):
						# 	os.mkdir(os.path.join(self.runFolder(pf, run = rl), 'masked'))
	

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
		"""
		addRun adds a run to a session's run list
		It creates/updates the following attributes of self:
		the list runList, containing all Run objects associated with self;
		the list conditionList, containing all unique condition attributes among those Run objects;
		the list scanTypeList, containing all unique scanType attributes among those Run objects;
		and various other attributes via the parcelateConditions() method (see there)
		"""
		
		run.indexInSession = len(self.runList)
		self.runList.append(run)
		# recreate conditionList
		self.conditionList = list(np.unique(np.array([r.condition for r in self.runList])))
		self.scanTypeList = list(np.unique(np.array([r.scanType for r in self.runList])))
		
		self.parcelateConditions()
	
	def parcelateConditions(self):
		"""
		parcelateConditions creates/updates the following attributes of self:
		the dict scanTypeDict, of which each key-value pair indicates a possible value for the scanType attribute
		of the Run objects of self's runList, and the values of the indexInSession arguments of those Run objects 
		that are associated with that scanType value;
		the list conditions, containing all unique condition attributes among the Run objects in runList. This appears identical to self.conditionList;
		the dict conditionDict, analogous to scanTypeDict but for condition attributes scanType attributes
		"""
		
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
		
		self.conditions = np.unique(np.array([r.condition for r in self.runList]))
		self.conditionDict = {}
		for c in self.conditions:
			if c != '':
				self.conditionDict.update({c: [hit.indexInSession for hit in filter(lambda x: x.condition == c, [r for r in self.runList])]})

		self.modality_list = []
		for r in self.runList:
			if hasattr(r, 'rawBehaviorFile') and 'behavior' not in self.modality_list:
				self.modality_list.append('behavior')
			if hasattr(r, 'rawDataFilePath') and 'mri' not in self.modality_list:
				self.modality_list.append('mri')
			if hasattr(r, 'eyeLinkFilePath') and 'eye' not in self.modality_list:
				self.modality_list.append('eye')
			if hasattr(r, 'physiologyFile') and 'hr' not in self.modality_list:
				self.modality_list.append('hr')
			if hasattr(r, 'starstim_eeg_file') and 'eeg' not in self.modality_list:
				self.modality_list.append('eeg')

			
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
		and setup the folder hierarchy and copy the raw image files into position.
		The folder within which the hierarchy will be built, is indicated by the
		base_dir attribute of self.project.
		
		Depending on settings, setupFiles will also process files: 
		nii.gz from par/rec and hdf5 from edf. NB: the par/rec->nii.gz
		conversion is not currently supported; one is supposed to have done 
		that conversion beforehand using e.g. Parrec2nii or r2agui,
		and use the .nii.gz path as r.rawDataFilePath
		"""
		
		if not os.path.isfile(self.runFile(stage = 'processed/behavior', run = self.runList[0] )):
			self.logger.info('creating folder hierarchy')
			self.create_folder_hierarchy()
		
		for r in self.runList:
			if hasattr(r, 'rawDataFilePath'):
				if os.path.splitext(r.rawDataFilePath)[-1] == '.PAR':	#if the rawDataFilePath of the Run object indicates a .par file, then convert to nifti
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
					elO.processIntoTable(hdf5_filename = self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'), compute_velocities = False, check_answers = False) #a lot happens in this line: the edf file is parsed into .hdf5 on the basis of messages sent to the eyelink during the experiment
			if hasattr(r, 'rawBehaviorFile'):
				ExecCommandLine('cp ' + r.rawBehaviorFile.replace('|', '\|') + ' ' + self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' ) )
			if hasattr(r, 'physiologyFile'):
				ExecCommandLine('cp ' + r.physiologyFile.replace('|', '\|') + ' ' + self.runFile(stage = 'processed/hr', run = r, extension = '.log' ) )
			if hasattr(r, 'starstim_eeg_file'):
				ExecCommandLine('cp ' + r.starstim_eeg_file.replace('|', '\|') + ' ' + self.runFile(stage = 'processed/eeg', run = r, extension = '.log' ) )

	
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
			bbR.configure( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), flirtOutputFile = True, init_fsl = True )
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
		"""
		setupRegistrationForFeat applies the freesurfer/flirt registration for this session to a feat directory, thereby
		making available the feat results in other spaces than the one in which feat was originally run. 
		This ensures that the feat results can be combined across runs and subjects without running flirt all the time,
		and populating the /reg subfolders of the feat folder is necessary for higher-level feat analyses, 
		which run on the standard (MNI) space data. It is not uncommon to then transform the outcome of those higher-level analyses
		back to e.g. mean func space using a FlirtOperator, because things like masks are often in that space.
		
		The procedure of setupRegistrationForFeat consists of creating (if not yet present) transformation info
		for the entire session, like the transformation matrix between mean functional and high-res anatomical,
		in the folder /processed/mri/reg/feat, and then moving this information to the single feat folder's /reg
		folder and applying FSL's featregapply on that feat folder.
		"""
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
			
	def motionCorrectFunctionals(self, postFix=[], registerNoMC = False, further_args = ''):
		"""
		motionCorrectFunctionals corrects all functionals in a given session.
		How we do this depends on whether we have parallel processing turned on or not.
		motionCorrectFunctionals automatically uses as its target the same file that was
		used to register to freesurfer anatomical by self.registerSession() (i.e. processed/mri/reg/forRegistration[...]),
		so self.registerSession() needs to be run first.
		"""
		
		self.logger.info('run motion correction')
		# set up a list of motion correction operator objects for the runs
		mcOperatorList = [];	stdOperatorList = [];
		for er in self.scanTypeDict['epi_bold']:

			# if T2anatID has been specified in the runarray, take this T2 as the target
			if hasattr(self.runList[er],'T2anatID'):
				self.referenceFunctionalFileName = self.runFile(stage='processed/mri/T2_anat/'+str(self.runList[er].T2anatID)+'/', postFix = [str(self.runList[er].T2anatID)] )
				# as registerSession takes only the latter of the inplane_anat images for registration, some may not have been betted:
				if not os.path.isfile(self.referenceFunctionalFileName[:-7] + '_NB.nii.gz'):
					better = BETOperator( inputObject = self.referenceFunctionalFileName )
					self.referenceFunctionalFileName = self.referenceFunctionalFileName[:-7] + '_NB.nii.gz'
					better.configure( outputFileName = self.referenceFunctionalFileName )
					better.execute()
			# otherwise, take the forRegistration file in the registration dir, which is created in registerSession
			else:
				self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID]  )

			mcf = MCFlirtOperator( self.runFile(stage = 'processed/mri', run = self.runList[er], postFix=postFix ), target = self.referenceFunctionalFileName )
			
			if hasattr(self.runList[er],'transformationMatrixFile'):
				mcf.transformMatrixFileName = self.runList[er].transformationMatrixFile
			# RUN WITH 7 DEGREES OF FREEDOM:
			if further_args != '':
				mcf.configure(further_args = further_args)
		 	else:
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

	def dt_ddt_moco_pars(self,conditions):
		"""
		This function takes the motion correction parameters and computes the
		first and second derivative based on the numerical_derivative function.

		For a tutorial see shared/tutorials/numerical_derivative
		"""
		for condition in conditions:

			Parallel(n_jobs = -1, verbose = 9)(delayed(run_moco_dt_ddt_parallel)(
				run = r,
				mcf_filename = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' )
				) 
				for r in [self.runList[i] for i in self.conditionDict[condition]])

	
	def rescaleFunctionals(self, operations = ['bandpass', 'zscore'], filterFreqs = {'highpass': 30.0, 'lowpass': -1.0}, funcPostFix = ['mcf'], mask_file = None):#, 'percentsignalchange'
		"""
		rescaleFunctionals operates on motion corrected functionals
		and does high/low pass filtering, percent signal change or zscoring of the data. as such, it doesn't really rescale any functionals in most cases.
		When using the sgtf method, the highpass argument in filterFreqs represents the filter window in seconds 
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
						ifs.append(ifO.runcmd)
					# funcFile = NiftiImage(ifO.outputFileName)
				if op == 'percentsignalchange':
					pscO = PercentSignalChangeOperator(funcFile)
					pscO.execute()
					funcFile = NiftiImage(pscO.outputFileName)
				if op == 'zscore':
					# zscO = ZScoreOperator(funcFile)
					# zscO.execute()
					# funcFile = NiftiImage(zscO.outputFileName)
					# create mean, std and demeaned files
					funcFile = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = funcPostFix )
					mean_cmd = 'fslmaths %s -Tmean %s' % (funcFile, funcFile[:-7] + '_m.nii.gz')
					std_cmd = 'fslmaths %s -Tstd %s' % (funcFile, funcFile[:-7] + '_std.nii.gz')
					dm_cmd = 'fslmaths %s -Tmean -mul -1 -add %s %s' %(funcFile, funcFile, funcFile[:-7] + '_dm.nii.gz')
					z_cmd = 'fslmaths %s -div %s %s' %(funcFile[:-7] + '_dm.nii.gz', funcFile[:-7] + '_std.nii.gz', funcFile[:-7] + '_Z.nii.gz')
					rem_cmd = 'rm %s' % (funcFile[:-7] + '_dm.nii.gz')
					total_cmd = ';\n'.join([mean_cmd,std_cmd,dm_cmd,z_cmd,z_cmd])
					if not self.parallelize:
						ExecCommandLine(total_cmd)
					else:
						if r == self.scanTypeDict['epi_bold'][0]:
							zsc_cmds = []
						zsc_cmds.append(total_cmd)
					# funcFile = NiftiImage(funcFile[:-7] + '_Z.nii.gz')
				if op == 'sgtf':
					sgtfO = SavitzkyGolayHighpassFilterOperator(funcFile)
					if funcFile.rtime > 10:
						TR = funcFile.rtime / 1000.0
					else:
						TR = funcFile.rtime
					# width = filterFreqs['highpass']
					width = filterFreqs['highpass'] / TR # this ensures that a window contains filterfreqs highpass seconds
					sgtfO.configure(mask_file = mask_file, TR = TR, width = width, order = 3)
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
			ppResults = [job_server.submit(ExecCommandLine,(ifO,),(),('subprocess','tempfile',)) for ifO in ifs]
			for ifOf in ppResults:
				ifOf()
			job_server.print_stats()
		if self.parallelize and 'zscore' in operations:
			# tryout parallel implementation - later, this should be abstracted out of course.
			ppservers = ()
			job_server = pp.Server(ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = [job_server.submit(ExecCommandLine,(zsc_cmd,),(),('subprocess','tempfile',)) for zsc_cmd in zsc_cmds]
			for zsc_cmd in ppResults:
				zsc_cmd()
			job_server.print_stats()
	
	def fsaverage_labels_to_masks(self, labels = ['V1.4-1','V1.4-2','V1.4-3','V1.4-4']):
		# import fsaverage labels to own subject. 
		for hemi in ['rh', 'lh']:
			for l in labels:
				llO = Label2LabelOperator(os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage', 'label', hemi + '.' + l + '.label'))
				llO.configure(source_subjectID = 'fsaverage', target_subjectID = self.subject.standardFSID, hemisphere = hemi )
				llO.execute()

	def createMasksFromFreeSurferLabels(self, labelFolders = [], annot = True, annotFile = 'aparc.a2009s', template_condition = None, cortex = True):
		"""
		createMasksFromFreeSurferLabels creates masks in the volume out of labels on the surface, located in (subfolders of) the
		subject's freesurfer label folder. Resulting mask files are written to the session's /processed/mri/masks/anat folder.
		The argument 'labelFolders' can contain the names of subfolders of the label to be included, and/or '' for the label folder itself.
		If annot is set to True, then the annotFile argument will be used to identify a freesurfer annotation file to be converted to 
		surface labels first, and then to to volume masks along with any other labels being processed.
		template_condition identifies a condition kind (default 'epi_bold') whose first scan is used as target for converting to the volume. This target
		must have the right dimensions but its space is otherwise irrelevant: the actual transformation depends on LabelToVolOperator.configure's 'register'
		argument, which by default is set to the registration file created by Session.registerSession().
		
		For some reason 'cortex' can be set to True to specifically convert the *h.cortex.label files, but it seems those would also be included in the label folder identified by ''
		"""
		if labelFolders == []:
			labelFolders.append(self.subject.labelFolderOfPreference)
			
		if annot:
			self.logger.info('create labels based on anatomical parcelation as in %s.annot', annotFile)
			# convert designated annotation to labels in an identically named directory
			anlo = AnnotationToLabelOperator(inputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', 'rh' + '.' + annotFile + '.annot'))	#initialize with (e.g.) the rh file to avoid warning of file not existing; doesn't mean that only rh file will be used
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

	def remove_empty_masks(self, masks_folder = 'anat'):
		"""
		remove_empty_masks removes mask volumes that consist of only zeros
		"""
		mask_file_names = subprocess.Popen('ls ' + os.path.join(self.stageFolder(stage = 'processed/mri/masks/' ), masks_folder, '*.nii.gz'), shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		for m in mask_file_names:
			this_mask_file = NiftiImage(m)
			if this_mask_file.data.sum() == 0:	# this is an empty mask file...
				self.logger.info('removing mask %s, because it doesn\'t contain any voxels in this EPI image space'%m)
				os.system('rm ' + m)
	
	def createMasksFromFreeSurferAseg(self, asegFile = 'aparc.a2009s+aseg', which_regions = ['Putamen', 'Caudate', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens', 'Cerebellum_Cortex', 'Thalamus_Proper', 'Thalamus', 'VentralDC','Brain-Stem'], smoothing_width = 2.0, threshold = 0.3):
		"""
		docstring for createMasksFromFreeSurferAseg. Appears to be for the delinations resulting from subcortical segmentation.
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
		aseg_file_name = os.path.join(self.stageFolder(stage = 'processed/mri/masks'), asegFile + '.nii.gz')
		aseg_image = NiftiImage(aseg_file_name)
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
						label_opf_name_int = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'),  prefix + wanted_region + '_int.nii.gz')
						selection_cmd = 'fslmaths "%s" -uthr %i -thr %i "%s"' % (aseg_file_name, int(area[0]), int(area[0]), label_opf_name_int)
						ExecCommandLine(selection_cmd)

						# label_opf_name = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'),  prefix + wanted_region + '.nii.gz')
						# selection_cmd = 'fslmaths %s –div %i %s' % (label_opf_name_int, float(area[0]), label_opf_name)
						# ExecCommandLine(selection_cmd)

						label_opf_name_s = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'),  prefix + wanted_region + '_smooth.nii.gz')
						smooth_cmd = 'fslmaths %s -s %f %s' % (label_opf_name_int, smoothing_width, label_opf_name_s)
						ExecCommandLine(smooth_cmd)

						label_opf_name_t = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'),  prefix + wanted_region + '_thresh.nii.gz')
						thres_cmd = 'fslmaths %s -thr %f %s' % (label_opf_name_s, threshold * float(area[0]), label_opf_name_t)
						ExecCommandLine(thres_cmd)

						label_opf_name = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'),  prefix + wanted_region + '.nii.gz')
						thres_cmd = 'fslmaths %s -bin %s' % (label_opf_name_t, label_opf_name)
						ExecCommandLine(thres_cmd)

						ExecCommandLine('rm ' + label_opf_name_int)
						ExecCommandLine('rm ' + label_opf_name_s)
						ExecCommandLine('rm ' + label_opf_name_t)

						self.logger.info('converting %s from %s to %s'%(area[1],aseg_file_name,label_opf_name))

						
						# this_label_data = np.array(aseg_image.data == int(area[0]), dtype = int)
						# # create new image
						# label_opf = NiftiImage(this_label_data)
						# label_opf.header = aseg_image.header
						# label_opf.save(label_opf_name)
						# self.logger.info( area[1] + ' outputted to ' + label_opf_name )
						
	def createMasksFromFreeSurferAseg2(self, asegFile = 'aparc.a2009s+aseg', which_regions = ['Putamen', 'Caudate', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens', 'Cerebellum_Cortex', 'Thalamus_Proper', 'Thalamus', 'VentralDC','Brain-Stem']):
		"""
		docstring for createMasksFromFreeSurferAseg2. Appears to be for the delinations resulting from subcortical segmentation.
		"""
		
		# find the subcortical labels in the converted brain from aseg.auto_noCCseg.label_intensities file
		with open(os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'aseg.auto_noCCseg.label_intensities.txt')) as f:
			a = f.readlines()
		list_of_areas = [[al.split(' ')[0], al.split(' ')[1]] for al in a]
		
		# open file
		aseg_image = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', asegFile + '.mgz')
		for wanted_region in which_regions:
			for area in list_of_areas:
				if area[1] == wanted_region:
					
					# create masks:
					inputObject = aseg_image
					outputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'aseg_{}.mgz'.format(area[1]))
					os.system('mri_binarize --i {} --match {} --o {}'.format(inputObject, area[0], outputObject))

					# convert to nifti
					inputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'aseg_{}.mgz'.format(area[1]))
					outputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'aseg_{}.nii.gz'.format(area[1]))
					os.system('mri_convert {} {}'.format(inputObject, outputObject))
					
					# transform to subject's space:
					inputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'aseg_{}.nii.gz'.format(area[1]))
					outputObject = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'), '{}.nii.gz'.format(area[1]))
					target = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]])
					reg = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID, 'flirt', 'BB', 'inv'], extension = '.mat' )
					flO = FlirtOperator(inputObject = inputObject, referenceFileName = target)
					flO.configureApply(transformMatrixFileName=reg, outputFileName = outputObject, sinc=False)
					flO.execute()
		
					# treshold:
					inputObject = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'), '{}.nii.gz'.format(area[1]))
					fmo = FSLMathsOperator(inputObject)
					fmo.configure(outputFileName=inputObject, **{'-thr':str(0.5), '-bin':''})
					fmo.execute()
	
	def createMasksFromFSLAtlases(self, xml_files = ['Thalamus.xml', 'Striatum-Connectivity-7sub.xml'], files = ['Thalamus/Thalamus-prob-2mm.nii.gz','Striatum/striatum-con-prob-thr25-2mm.nii.gz'], labels = ['Thalamus','Striatum'], threshold = 0.5):
		"""createMasksFromFSLAtlases takes lists of xml files in the fsl atlas folders, the corresponding nii files, and a threshold.
		These atlases are likely connectivity-based subcortical atlases.
		createMasksFromFSLAtlases uses these inputs to create subject-session specific anatomical masks, given the threshold.
		First, the xml file is read, and then the corresponding nii is transformed and 
		taken apart on a per-timepoint basis in order to create 3d instead of 4d masks
		"""

		target_file = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]])
		mni_reg_file = os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'standard2example_func.mat' )
		output_folder = self.stageFolder(stage = 'processed/mri/masks/anat')

		for xml, nii_mask_file, label in zip(xml_files, files, labels):
			# read xml files. 
			tree = ET.parse(os.path.join(os.environ['FSL_DIR'], 'data/atlases/', xml))
			root = tree.getroot()
			name_frame_dict = {child.text: int(child.attrib['index'])  for child in list(root.iterfind('data'))[0]}

			joined_output_filename = os.path.join(output_folder, nii_mask_file.split('/')[1][:-7])

			for i, hemi in enumerate(['lh','rh']):
				# create hemispheric masks, last dimension is lateral dimension in MNI space
				orig_mni = NiftiImage(os.path.join(os.environ['FSL_DIR'], 'data/atlases/', nii_mask_file))
				hemi_mni_data = orig_mni.data.copy()
				hemi_mni_data[:,:,:,i * (orig_mni.data.shape[-1]/2) : np.min([(i+1) * (orig_mni.data.shape[-1]/2), orig_mni.data.shape[-1]])] = 0
				hemi_mni = NiftiImage(hemi_mni_data)
				hemi_mni.header = orig_mni.header
				hemi_mni_filename = os.path.join(os.path.split(joined_output_filename)[0], hemi + '.' + os.path.split(joined_output_filename)[1] + '_MNI.nii.gz')
				hemi_mni.save(hemi_mni_filename)

				hemi_filename = os.path.join(os.path.split(joined_output_filename)[0], hemi + '.' + os.path.split(joined_output_filename)[1] + '.nii.gz')

				flO = FlirtOperator(inputObject = hemi_mni_filename, referenceFileName = target_file)
				flO.configureApply(transformMatrixFileName = mni_reg_file, outputFileName = hemi_filename, sinc=True, extra_args = ' -datatype float')
				flO.execute()

				# delete mni space masks to avoid errors due to import of non-compying sized niis
				os.system('rm ' + hemi_mni_filename)

				joined_output_file = NiftiImage(hemi_filename)
				for name in name_frame_dict.keys():
					this_file = NiftiImage((joined_output_file.data[name_frame_dict[name]] > threshold).astype(np.float32))
					this_file.header = joined_output_file.header
					this_file.save(os.path.join(output_folder, hemi + '.'+ label + '_' + name + '.nii.gz'))
					


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
				if roi_wildcard == roi_name._v_name:
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
		"""resample_epi resamples the mc'd epi files back to the resolution of the functional images."""
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
		"""
		resample_epi resamples the mc'd epi files back to the resolution of the functional images.
		The resulting files will by default have the postfix 'mcf', whereas the hi res files will have
		'mcf_hr'. 
		"""
		
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
		job_server.destroy()

		# # fix the 4th dimension, that is TR:
		# cmds = []
		# for cond in conditions:
		# 	for r in [self.runList[i] for i in self.conditionDict[cond]]:
		# 		pixdim1 = str(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).pixdim[0])
		# 		pixdim2 = str(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).pixdim[1])
		# 		pixdim3 = str(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).pixdim[2])
		# 		pixdim4 = str(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).pixdim[3])
		# 		cmds.append('fslchpixdim ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' ' + pixdim1 + ' ' + pixdim2 + ' ' + pixdim3 + ' ' + pixdim4)
		#
		# # run all of these commands in parallel
		# ppservers = ()
		# job_server = pp.Server(ppservers=ppservers)
		# self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		# ppResults = [job_server.submit(ExecCommandLine,(fo,),(),('subprocess','tempfile',)) for fo in cmds]
		# for fo in ppResults:
		# 	fo()
		# job_server.destroy()
		
		# fix headers:
		cmds = []
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				nii_file_orig = NiftiImage(self.runFile(stage = 'processed/mri', run = r ))
				nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
				nii_file.header = nii_file_orig.header
				nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
	
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
	
	def retroicorFSL(self, conditions=['task'], postFix=['B0', 'mcf', 'sgtf'], threshold=1.5, nr_dummies=8, sample_rate=500, gradient_direction='y', card_order=3, resp_order=2, card_resp_order=3, resp_card_order=2, slicedir='z', sliceorder='up', thisFeatFile=None, prepare=False, run=False):
		
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				
				"""
				Prepare scanphyslogfile
				-----------------------
				- load Phillips scanphyslog
				- compute summed gradient signal
				- compute slice timings based on <threshold>
				- compute "shim indices" based on max time between slice timings (this separates the shim period from scanning period)
				- compute "dummy indices" (number of <nr_dummies> * <nr_slices>)
				- compute "scan indices" as the first of every <nr_slices>
				- append volume and slice times to Phillips scanphyslog
				- plot result for visual inspection
				
				Prepare retroicor slice wise regressors
				----------------------------------------
				
				Run GLM to regress out physiological noise
				------------------------------------------
				"""
				
				# ----------------------------------------
				# Prepare scanphyslogfile:               -
				# ----------------------------------------
				
				if prepare:
					# load nifti:
					TR = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix=postFix)).rtime
					nr_slices = NiftiImage(self.runFile(stage = 'processed/mri', run = r)).volextent[-1]
					nr_TRs = NiftiImage(self.runFile(stage = 'processed/mri', run = r)).timepoints
				
					# load physio data:
					physio = np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension='.log'), skiprows=5)
					if gradient_direction == 'x':
						gradients = [6]
					if gradient_direction == 'y':
						gradients = [7]
					if gradient_direction == 'z':
						gradients = [8]
					if gradient_direction == 'all':
						gradients = [6,7,8]
					# gradient_signal = abs(np.array([physio[:,g] for g in gradients]).sum(axis = 0))
					gradient_signal = np.array([physio[:,g] for g in gradients]).sum(axis = 0)
					gradient_signal = (gradient_signal-gradient_signal.mean()) / gradient_signal.std()
					
					# slice time indexes:
					x = np.arange(gradient_signal.shape[0])
					slice_times = x[np.array(np.diff(np.array(gradient_signal>threshold, dtype=int))==1, dtype=bool)]+1
				
					# check if we had a double (due to shape gradient signal):
					if slice_times.shape[0] > (nr_TRs*nr_slices*2):
						slice_times = slice_times[0::2]
					
					# identify gaps due to shimming:
					# gaps = max(np.diff(slice_times))
					# shim_slice_indices = np.arange(slice_times.shape[0]) < int(np.where(np.diff(slice_times)==gaps)[0])+1
					# shim_slice_indices = np.arange(slice_times.shape[0]) < int(gaps[-1])+1
					# shim_slices = np.arange(x.shape[0])[shim_slice_indices]
					# shim_volumes = shim_slices[0::nr_slices]
					
					gap = np.where(np.diff(slice_times) > ((nr_TRs / nr_slices)*10.0))[0][-1]
						
					# dummy slices and volumes:
					dummy_slice_indices = (np.arange(slice_times.shape[0]) > gap) * (np.arange(slice_times.shape[0]) < gap + (nr_dummies * nr_slices))
					dummy_slices = np.arange(x.shape[0])[dummy_slice_indices]
					dummy_volumes = dummy_slices[0::nr_slices]
				
					# scans slices and volumes:
					scan_slice_indices = (np.arange(slice_times.shape[0]) > dummy_slices[-1])
					scan_slices = np.arange(x.shape[0])[scan_slice_indices][0:(nr_TRs*nr_slices)]
					scan_volumes = scan_slices[0:(nr_TRs*nr_slices):nr_slices]
				
					# append to physio file:
					scan_slices_timecourse = np.zeros(x.shape[0])
					scan_slices_timecourse[slice_times[scan_slices]] = 1
					scan_volumes_timecourse = np.zeros(x.shape[0])
					scan_volumes_timecourse[slice_times[scan_volumes]] = 1
					dummies_volumes_timecourse = np.zeros(x.shape[0])
					dummies_volumes_timecourse[slice_times[dummy_volumes]] = 1
				
					# save new physio file:
					physio_new = np.hstack((np.asmatrix(physio[:,4]).T, np.asmatrix(physio[:,5]).T, np.asmatrix(scan_slices_timecourse).T, np.asmatrix(scan_volumes_timecourse).T))
					np.savetxt(self.runFile(stage = 'processed/hr', run = r, postFix=['new'], extension='.log'), physio_new, fmt = '%3.2f', delimiter = '\t')
				
					# # save new physio file (to be used in Matlab PhsyIO Toolbox):
					# indices = (x > slice_times[scan_slices[-1]]+50)
					# physio_new = physio[-indices,:]
					# np.savetxt(self.runFile(stage = 'processed/hr', run = r, postFix=['new2'], extension='.log'), physio_new, fmt = '%i', delimiter = '\t')
					
					# plot:
					plot_timewindow = [	np.arange(0, slice_times[dummy_volumes[-1]]+(4*sample_rate)),
										np.arange(slice_times[scan_slices[-1]]-(8*sample_rate), x.shape[0]),
										np.arange(slice_times[scan_slices[-10]], slice_times[scan_slices[-5]]),
										]
				
					for i, times in enumerate(plot_timewindow):
						f = pl.figure(figsize = (15,3))
						s = f.add_subplot(111)
						pl.plot(x[times], gradient_signal[times], label='summed gradient signal (x, y, z)')
						if i in (0,1):
							pl.plot(x[times], dummies_volumes_timecourse[times]*threshold*1.5, 'k', lw=3, label='dummies')
							pl.plot(x[times], scan_volumes_timecourse[times]*threshold*1.5, 'g', lw=3, label='triggers')
						if i == 2:
							pl.plot(x[times], scan_slices_timecourse[times]*threshold*1.5, 'g', lw=1, label='slices')
						pl.axhline(threshold, color='r', ls='--', label='threshold')
						s.set_title('summed gradient signal (x, y, z) -- nr volumes = {}'.format(sum(scan_volumes_timecourse)))
						s.set_xlabel('samples, {}Hz'.format(sample_rate))
						pl.ylim((0,threshold*1.5))
						leg = s.legend(fancybox = True)
						leg.get_frame().set_alpha(0.5)
						if leg:
							for t in leg.get_texts():
							    t.set_fontsize('small')    # the legend text fontsize
							for (j, l) in enumerate(leg.get_lines()):
								l.set_linewidth(3.5)  # the legend line width
					
						pl.tight_layout()
						f.savefig(os.path.join(self.stageFolder(stage = 'processed/hr/figs'), str(r.ID) + '_gradient_signal_{}_'.format(i+1) + ['start', 'end', 'slice'][i] + '.jpg'))
					pl.close('all')
				
				# ----------------------------------------
				# Create retroicor slise-wise regressors:-
				# ----------------------------------------
				
				if prepare:
					
					# load nifti:
					TR = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix=postFix)).rtime
					nr_slices = NiftiImage(self.runFile(stage = 'processed/mri', run = r)).volextent[-1]
					nr_TRs = NiftiImage(self.runFile(stage = 'processed/mri', run = r)).timepoints
					
					# retroicor folder:
					folder = os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'retroicor')
					try:
						os.system('rm -rf ' + folder)
					except OSError:
						pass
					subprocess.Popen('mkdir ' + folder, shell=True, stdout=PIPE).communicate()[0]
					base = os.path.join(folder, 'retroicor')
					
					# FSL fix text:
					copy_in = self.runFile(stage = 'processed/hr', run = r, postFix=['new'], extension='.log')
					copy_out = base + '_input.txt'
					subprocess.call('cp ' + copy_in + ' ' + copy_out, shell=True)
					# subprocess.call('fslFixText ' + copy_in + ' ' + copy_out, shell=True)
					
					# run two commands
					inputObject = base + '_input.txt'
					outputObject = base
					retroO = FSLRETROICOROperator(inputObject=inputObject, cmd='pnm_stage1')
					retroO.configure(outputFileName=outputObject, **{'-s':str(sample_rate), '--tr='+str(TR):' ', '--smoothcard='+str(0.1):' ', '--smoothresp='+str(0.1):' ', '--resp='+str(2):' ', '--cardiac='+str(1):' ', '--trigger='+str(4):'',})
					retroO.execute()
					retroO = FSLRETROICOROperator(inputObject=inputObject, cmd='popp')
					retroO.configure(outputFileName=outputObject, **{'-s':str(sample_rate), '--tr='+str(TR):' ', '--smoothcard='+str(0.1):' ', '--smoothresp='+str(0.1):' ', '--resp='+str(2):' ', '--cardiac='+str(1):' ', '--trigger='+str(4):'',})
					retroO.execute()
					
					# run final command:
					inputObject = self.runFile(stage = 'processed/mri', run = r, postFix=postFix)
					outputObject = base
					card = base + '_card.txt'
					resp = base + '_resp.txt'
					retroO = FSLRETROICOROperator(inputObject=inputObject, cmd='pnm_evs')
					retroO.configure(outputFileName=outputObject, **{'--tr='+str(TR):' ', '-c':card, '-r':resp, '--oc='+str(card_order):' ', '--or='+str(resp_order):' ', '--multc='+str(card_resp_order):' ', '--multr='+str(resp_card_order):' ', '--slicedir='+slicedir:' ', '--sliceorder='+sliceorder:' ', '-v':''})
					retroO.execute()
					
					# grab regressors:
					regressors = [reg for reg in np.sort(glob.glob(base + 'ev*.nii*'))]
					text_file = open(base+'_evs_list.txt', 'w')
					for reg in regressors:
						text_file.write('{}\n'.format(reg))
					text_file.close()
					
				# ----------------------------------------
				# Run GLM and de-noise!!:                -
				# ----------------------------------------
				
				if run:
					
					# retroicor folder:
					folder = os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'retroicor')
					base = os.path.join(folder, 'retroicor')
					
					# grab regressors:
					regressors = [reg for reg in np.sort(glob.glob(base + 'ev*.nii*'))]
					text_file = open(base+'_evs_list.txt', 'w')
					for reg in regressors:
						text_file.write('{}\n'.format(reg))
					text_file.close()
					
					# shell()
					
					# remove previous feat directories
					try:
						# self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf', 'sgtf'], extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.fsf'))
					except OSError:
						pass
					
					thisFeatFile = thisFeatFile
					
					REDict = {
					'---NR_TRS---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix=postFix)).timepoints),
					'---TR---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix=postFix)).rtime),
					'---NR_VOXELS---':str(np.prod(np.array(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).getExtent()))),
					'---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = r, postFix = postFix),
					}
					for i, reg in enumerate(regressors):
						REDict.update({'---EV{}---'.format(i+1):reg})
					
					featFileName = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.fsf')
					featOp = FEATOperator(inputObject = thisFeatFile)
					# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
					if r == self.runList[self.scanTypeDict['epi_bold'][-1]]:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
					else:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
					self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
					# run feat
					featOp.execute()
		
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
			'---LOG_FILE---': self.runFile(stage = 'processed/hr', run = r, postFix=['new2'], extension = '.log'),
			'---NR_TRS---': str(run_nii_file.data.shape[0]),
			'---NR_SLICES---': slice_count,
			'---NR_SLICES_PER_BEAT---': slice_count,
			'---TR_SECS---': str(run_nii_file.rtime),
			'---NR_DUMMIES---': str(nr_dummies),
			'---ONSET_SLICE---': str(onset_slice),
			'---GRADIENT_DIRECTION---': gradient_direction,
			'---OUTPUT_FILE_NAME---': self.runFile(stage = 'processed/hr', run = r, postFix = ['regressors'], extension = '.txt'),
			'---FIGS_OUTPUT---': os.path.join(self.stageFolder('processed/hr/figs'), '{}_Matlab_PhysIO.jpeg'.format(r.ID)),
		}
		
		# get template script
		if retroicor_script_file == None:
			retroicor_script_file = os.path.join(os.environ['ANALYSIS_HOME'], 'Tools', 'other_scripts', 'RETROICOR_PPU_template.m')
		# start retroicor operator with said file
		rO = RETROICOROperator(retroicor_script_file)
		rO.configure(REDict = REDict, retroicor_m_filename = self.runFile(stage = 'processed/hr', run = r, extension = '.m'), waitForExecute = waitForExecute)
		if execute:
			rO.execute()
			
		return rO
	
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
				
	def B0_unwarping(self, conditions, wfs, etl, acceleration):
		
		# ----------------------------------------
		# Set-up everything for BO unwarping:    -
		# ----------------------------------------

		# Rescale the values in the nii file to -pi and pi. Use the -odt option of fslmaths to ensure you have floats:
		# fslmaths $FUNCDIR/"$SUB"_B0_phase -div 100 -mul 3.141592653589793116 -odt float $FUNCDIR/"$SUB"_B0_phase_rescaled
		for r in self.conditionDict['B0_anat_phs']:
			inputObject = self.runFile(stage = 'processed/mri', run = self.runList[r])
			outputObject = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix=['rescaled'])
			fmO = FSLMathsOperator(inputObject=inputObject)
			fmO.configurePi(outputFileName=outputObject, div=100, mul=3.141592653589793116)
			fmO.execute()
		
		# Bet:
		# bet $FUNCDIR/"$SUB"_B0_magnitude $FUNCDIR/"$SUB"_B0_magnitude_brain -m
		# fslmaths $FUNCDIR/"$SUB"_B0_magnitude_brain_mask -dilM $FUNCDIR/"$SUB"_B0_magnitude_brain_mask
		for r in self.conditionDict['B0_anat_mag']:
			inputObject = self.runFile(stage = 'processed/mri', run = self.runList[r])
			outputObject = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix=['NB'])
			better = BETOperator( inputObject = inputObject )
			better.configure( outputFileName = outputObject )
			better.execute()

			inputObject = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix=['NB', 'mask'])
			outputObject = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix=['NB', 'mask'])
			fmO = FSLMathsOperator(inputObject=inputObject)
			fmO.configure(outputFileName=outputObject, **{'-dilM': ''})
			fmO.execute()

		# Unwrap the data using prelude:
		# prelude -p $FUNCDIR/"$SUB"_B0_phase_rescaled -a $FUNCDIR/"$SUB"_B0_magnitude -o $FUNCDIR/"$SUB"_fmri_B0_phase_rescaled_unwrapped -m $FUNCDIR/"$SUB"_B0_magnitude_brain_mask
		for i in range(len(self.conditionDict['B0_anat_mag'])):
			phasevol = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['B0_anat_phs'][i]], postFix=['rescaled'])
			inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['B0_anat_mag'][i]])
			outputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['B0_anat_phs'][i]], postFix=['rescaled', 'unwrapped'])
			mask = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['B0_anat_mag'][i]], postFix=['NB', 'mask'])

			po = PreludeOperator(inputObject=inputObject)
			po.configure(phasevol=phasevol, outputFileName=outputObject, mask=mask)
			po.execute()

		# Convert to radials-per-second by multiplying with 200 (because time difference between the two scans is 5 msec):
		# fslmaths $FUNCDIR/"$SUB"_B0_phase_rescaled_unwrapped -mul 200 $FUNCDIR/"$SUB"_B0_phase_rescaled_unwrapped
		for r in self.conditionDict['B0_anat_phs']:
			inputObject = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix=['rescaled', 'unwrapped'])
			outputObject = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix=['rescaled', 'unwrapped'])
			fmO = FSLMathsOperator(inputObject=inputObject)
			fmO.configure(outputFileName=outputObject, **{'-mul': str(200)})
			fmO.execute()

		# Reorient high res T1:
		inputObject = os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'highres.nii.gz' )
		ro = ReorientOperator(inputObject = inputObject)
		ro.configure(outputFileName = inputObject)
		ro.execute()

		# Bet al epi's:
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				better = BETOperator( inputObject = self.runFile(stage = 'processed/mri', run = r ) )
				better.configure( outputFileName = self.runFile(stage = 'processed/mri', run = r, postFix = ['NB']), **{'-F': ''} )
				better.execute()

		# ----------------------------------------
		# Formula:                               -
		# ----------------------------------------

		effective_echo_spacing = ((1000.0 * wfs)/(434.215 * (etl+1))/acceleration)
		
		# ----------------------------------------
		# Do actual B0 unwarping:                -
		# ----------------------------------------
	
		# parameters:
		effective_echo_spacing = str(effective_echo_spacing)
		EPI_TE = str(27.63) # where does this comes from?
		unwarp_direction = 'y'
		signal_loss_threshold = str(10)
	
		# for er in self.scanTypeDict['epi_bold']:
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				
				# remove previous feat directories
				try:
					self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['NB'], extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['NB'], extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['NB'], extension = '.fsf'))
				except OSError:
					pass
				
				# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
				thisFeatFile = '/home/shared/Niels_UvA/Visual_UvA/analysis/feat_B0/design.fsf'
				REDict = {
				'---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = r, postFix = ['NB']), 
			
				'---UNWARP_PHS---':self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['B0_anat_phs'][0]], postFix = ['rescaled', 'unwrapped']), 
				'---UNWARP_MAG---':self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['B0_anat_mag'][0]], postFix=['NB']), 
				'---HIGHRES_FILES---':os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'highres.nii.gz' ), 

				'---TR---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['NB'])).rtime),
				'---NR_TRS---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['NB'])).timepoints),
				'---NR_VOXELS---':str(np.prod(np.array(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['NB'])).getExtent()))),
				'---EFFECTIVE_ECHO_SPACING---':effective_echo_spacing,
				'---EPI_TE---':EPI_TE,
				'---UNWARP_DIREC---':unwarp_direction,
				'---SIGNAL_LOSS_THRESHOLD---':signal_loss_threshold,
				}
			
				featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.fsf')
				featOp = FEATOperator(inputObject = thisFeatFile)
				# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
				if r == [self.runList[i] for i in self.scanTypeDict['epi_bold']][-1]:
					featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
				else:
					featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
				self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
				# run feat
				featOp.execute()





