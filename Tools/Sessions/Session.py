#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

import os, sys, pickle, math
from subprocess import *

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
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *

class PathConstructor(object):
	"""
	FilePathConstructor is an abstract superclass for sessions.
	It constructs the file and folder naming hierarchy for a given session.
	All file naming and calling runs through this class.
	"""
	def __init__(self):
		self.fileNameBaseString = self.dateCode
	
	def baseFolder(self):
		"""docstring for baseFolder"""
		return os.path.join(self.project.baseFolder, self.subject.initials, self.dateCode)
	
	def makeBaseFolder(self):
		if not os.path.isdir(self.baseFolder()):
			try:
				os.mkdir(os.path.join(self.project.baseFolder, self.subject.initials))
			except OSError:
				pass
			try:
				os.mkdir(self.baseFolder())
			except OSError:
				pass
	
	def stageFolder(self, stage):
		"""folder for a certain stage - such as 'raw/mri' or 'processed/eyelink', or something like that. """
		return os.path.join(self.baseFolder(), stage)
	
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
		rawFolders = ['raw/mri', 'raw/behavior', 'raw/eye']
		self.processedFolders = ['processed/mri', 'processed/behavior', 'processed/eye']
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
	

class Session(PathConstructor):
	"""
	Session is an object that contains all the analysis steps 
	for analyzing an entire session. The basic class contains the base level analyis and preparations, 
	such as putting the files in place and setting up the runs. 
	Often-used analysis steps include registration with an anatomical, and motion correction of the functionals.
	"""
	def __init__(self, ID, date, project, subject, parallelize = False, loggingLevel = logging.DEBUG):
		self.ID = ID
		self.date = date
		self.project = project
		self.subject = subject
		self.runList = []
		self.dateCode = subject.initials + '_' + ('0'+str(self.date.day))[-2:] + ('0'+str(self.date.month))[-2:] + str(self.date.year)[-2:]
		self.parallelize = parallelize
		self.loggingLevel = loggingLevel
		super(Session, self).__init__()
		
		# add logging for this session
		# sessions create their own logging file handler
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(self.loggingLevel)
		if os.path.isfile(os.path.join(self.stageFolder(stage = 'processed/mri/log'), 'sessionLogFile.log')):
			addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.stageFolder(stage = 'processed/mri/log'), 'sessionLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		self.logger.info('starting analysis of session ' + str(self.ID))
	
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
		
		print self.scanTypeDict
		
		self.conditions = np.unique(np.array([r.condition for r in self.runList]))
		self.conditionDict = {}
		for c in self.conditions:
			if c != '':
				self.conditionDict.update({c: [hit.indexInSession for hit in filter(lambda x: x.condition == c, [r for r in self.runList])]})
	
	def setupFiles(self, rawBase):
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
				elO = EyelinkOperator(r.eyeLinkFilePath)
				ExecCommandLine('cp ' + os.path.splitext(r.eyeLinkFilePath)[0] + '.* ' + self.runFolder(stage = 'processed/eye', run = r ) )
				elO.processIntoTable(hdf5_filename = self.runFile(stage = 'processed/eye', run = r, extension = '.hdf5'), compute_velocities = False, check_answers = False)
	
	def registerSession(self, contrast = 't2', FSsubject = None, register = True, deskull = True, bb = True, flirt = True, makeMasks = False, maskList = ['cortex','V1','V2','V3','V3A','V3B','V4'], labelFolder = 'label'):
		"""
		before we run motion correction we register with the freesurfer segmented version of this subject's brain. 
		For this we use either the inplane anatomical (if present), or we take the first epi_bold of the session,
		motion correct it and mean the motion corrected first epi_bold to serve as the target for the registration.
		the contrast argument indicates the contrast of the reference image in epi_bold space that is to be registered.
		"""
		self.logger.info('register files')
		# setup what to register to
		if not FSsubject:
			self.FSsubject = self.subject.standardFSID
		else:
			self.FSsubject = FSsubject
		
		if register:
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
				# prepare the motion corrected first functional	
				mcFirst = MCFlirtOperator( self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]] ) )
				mcFirst.configure(outputFileName = self.runFile(stage = 'processed/mri/reg', postFix = ['mcf'], base = 'firstFunc' ))
				mcFirst.execute()
				#  and average it over time. 
				fslM = FSLMathsOperator( self.runFile(stage = 'processed/mri/reg', postFix = ['mcf'], base = 'firstFunc' ) )
				# in principle taking the temporal mean in superfluous (done by mcflirt too) but oh well
				fslM.configureTMean()
				fslM.execute()		
			
				self.logger.info('using firstFunc as a registration target')
				self.originalReferenceFunctionalVolume = self.runFile(stage = 'processed/mri/reg', postFix = ['mcf', 'meanvol'], base = 'firstFunc' )
				self.logger.info('registration target is firstFunc, ' + self.originalReferenceFunctionalVolume)
			self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] )
			ExecCommandLine('cp ' + self.originalReferenceFunctionalVolume + ' ' + self.referenceFunctionalFileName )
		
			if bb:
				# register to both freesurfer anatomical and fsl MNI template
				# actual registration - BBRegister to freesurfer subject
				bbR = BBRegisterOperator( self.referenceFunctionalFileName, FSsubject = self.FSsubject, contrast = contrast )
				bbR.configure( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), flirtOutputFile = False )
				bbR.execute()
				# after registration, see bbregister log file for reg check
					
			if flirt:
				# actual registration - Flirt to MNI brain
				flR = FlirtOperator( self.referenceFunctionalFileName )
				flR.configureRun( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'flirt', postFix = [self.ID], extension = '.mtx' ) )
				flR.execute()
				invFlR = InvertFlirtOperator(self.runFile(stage = 'processed/mri/reg', base = 'flirt', postFix = [self.ID], extension = '.mtx' ))
				invFlR.configure()
				invFlR.execute()
				
				flRInv = '/usr/local/fsl/bin/convert_xfm -omat /Users/tk/Documents/research/experiments/control/data/full_pilot/data/MC/MC_160811/processed/mri/reg/flirt_controlMC_inv.mtx -inverse /Users/tk/Documents/research/experiments/control/data/full_pilot/data/MC/MC_160811/processed/mri/reg/flirt_controlMC.mtx'
		
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
	
	def motionCorrectFunctionals(self, registerNoMC = False, diagnostics = False):
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
			if diagnostics:
				# diagnostics for temporal SNR are run automatically by mcflirt for the motion corrected volumes
				# for the non-motion corrected functionals take the temporal mean and standard deviation
				nmcO = FSLMathsOperator( self.runFile(stage = 'processed/mri', run = self.runList[er] ) )
				nmcO.configureTMean()
				nmcO.execute()
				nmcO.configureTStd()
				nmcO.execute()
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
			job_server = pp.Server(ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
#			ppResults = [job_server.submit(mcf.execute,(), (), ("Tools","Tools.Operators","Tools.Sessions.MCFlirtOperator","subprocess",)) for mcf in mcOperatorList]
			ppResults = [job_server.submit(ExecCommandLine,(mcf.runcmd,),(),('subprocess','tempfile',)) for mcf in mcOperatorList]
			for fMcf in ppResults:
				fMcf()
			
			job_server.print_stats()
			
	def rescaleFunctionals(self, operations = ['highpass', 'zscore'], filterFreqs = {'highpass': 30.0, 'lowpass': -1.0}):#, 'percentsignalchange'
		"""
		rescaleFunctionals operates on motion corrected functionals
		and does high/low pass filtering, percent signal change or zscoring of the data
		"""
		self.logger.info('rescaling functionals with options %s', str(operations))
		for r in self.scanTypeDict['epi_bold']:	# now this is a for loop we would love to run in parallel
			funcFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf'] ))
			for op in operations:	# using this for loop will ensure that the order of operations as defined in the argument is adhered to
				if op == 'highpass':
					ifO = FSLMathsOperator(funcFile)
					ifO.configureHPF(nr_samples_hp = filterFreqs['highpass'] )
		#			ifO = ImageTimeFilterOperator(funcFile, filterType = 'highpass')
		#			ifO.configure(frequency = filterFreqs['highpass'])
					ifO.execute()
					funcFile = NiftiImage(ifO.outputFileName)
				if op == 'lowpass':
					ifO = ImageTimeFilterOperator(funcFile, filterType = 'lowpass')
					ifO.configure(frequency = filterFreqs['lowpass'])
					ifO.execute()
					funcFile = NiftiImage(ifO.outputFileName)
				if op == 'percentsignalchange':
					pscO = PercentSignalChangeOperator(funcFile)
					pscO.execute()
					funcFile = NiftiImage(pscO.outputFileName)
				if op == 'zscore':
					zscO = ZScoreOperator(funcFile)
					zscO.execute()
					funcFile = NiftiImage(zscO.outputFileName)
		
	
	def createMasksFromFreeSurferLabels(self, labelFolders = [], annot = True, annotFile = 'aparc.a2005s'):
		"""createMasksFromFreeSurferLabels looks in the subject's freesurfer subject folder and reads label files out of the subject's label folder of preference. (empty string if none given).
		Annotations in the freesurfer directory will also be used to generate roi files in the functional volume. The annotFile argument dictates the file to be used for this. 
		"""
		if labelFolders == []:
			labelFolders.append(self.subject.labelFolderOfPreference)
			
		if annot:
			self.logger.info('create masks based on anatomical parcelation as in %s.annot', annotFile)
			# convert designated annotation to labels in an identically named directory
			anlo = AnnotationToLabelOperator(inputObject = os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'label', 'rh' + '.' + annotFile + '.annot'))
			anlo.configure(subjectID = self.subject.standardFSID )
			anlo.execute()
			labelFolders.append(annotFile)
		
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
				lvo.configure(templateFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf'] ), hemispheres = [hemi], register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), fsSubject = self.subject.standardFSID, outputFileName = self.runFile(stage = 'processed/mri/masks/anat/', base = lfx[:-6] ), threshold = 0.05, surfType = 'label')
				lvo.execute()
		
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
				imo = ImageMaskingOperator(funcFile.data[timeSlices[0]:timeSlices[1]], maskObject = rois[rn], thresholds = [maskThreshold], outputFileName = self.runFile(stage = 'processed/mri', run = self.runList[r], base = 'masked/' + os.path.split(rois[rn].filename)[1][:-7], extension = '' ))
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
	
		
		
		
		