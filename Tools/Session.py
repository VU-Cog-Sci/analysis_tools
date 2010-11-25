#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

import os, sys, pickle
from subprocess import *

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
from nifti import *

import pp
import logging, logging.handlers, logging.config

from log import *
from Run import *
from Subjects.Subject import *
from Operators.Operator import *
from Operators.CommandLineOperator import *
from Operators.ImageOperator import *
from Operators.BehaviorOperator import *

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
		conditionFolders = np.concatenate((self.conditionList, ['log','figs','masks','reg','surf','scripts']))
		
		# assuming baseDir/raw/ exists, we must make processed
		if not os.path.isdir(os.path.join(self.baseFolder(), 'processed') ):
			os.mkdir(os.path.join(self.baseFolder(), 'processed'))
		
		# create folders for processed data
		for pf in self.processedFolders:
			if not os.path.isdir(self.stageFolder(pf)):
				os.mkdir(self.stageFolder(pf))
			# create condition folders in each of the processed data folders and also their surfs
			for c in conditionFolders:
				 if not os.path.isdir(self.stageFolder(pf+'/'+c)):
					os.mkdir(self.stageFolder(pf+'/'+c))
					if not os.path.isdir(os.path.join(self.stageFolder(pf+'/'+c), 'surf')):
						os.mkdir(os.path.join(self.stageFolder(pf+'/'+c), 'surf'))
			# create folders for each of the runs in the session and their surfs
			for rl in self.runList:
				if not os.path.isdir(self.runFolder(pf, run = rl)):
					os.mkdir(self.runFolder(pf, run = rl))
					if not os.path.isdir(os.path.join(self.runFolder(pf, run = rl), 'surf')):
						os.mkdir(os.path.join(self.runFolder(pf, run = rl), 'surf'))
	

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
		if 'epi_bold' in self.scanTypeList:
			self.epi_runs = [hit.indexInSession for hit in filter(lambda x: x.scanType == 'epi_bold', [r for r in self.runList])]
		if 'inplane_anat' in self.scanTypeList:
			self.inplane_runs = [hit.indexInSession for hit in filter(lambda x: x.scanType == 'inplane_anat', [r for r in self.runList])]
		if '3d_anat' in self.scanTypeList:
			self.anat_runs = [hit.indexInSession for hit in filter(lambda x: x.scanType == '3d_anat', [r for r in self.runList])]
		if 'dti' in self.scanTypeList:
			self.dti_runs = [hit.indexInSession for hit in filter(lambda x: x.scanType == 'dti', [r for r in self.runList])]
		if 'spectro' in self.scanTypeList:
			self.spectro_runs = [hit.indexInSession for hit in filter(lambda x: x.scanType == 'spectro', [r for r in self.runList])]
	
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
					if r.indexInSession in self.epi_runs:
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
				for r in self.runList:
					ExecCommandLine('cp ' + self.runFile(stage = 'raw/mri', postFix = [str(r.ID)], base = rawBase ) + ' ' + self.runFile(stage = 'processed/mri', run = r ) )
			# behavioral files will be copied during analysis
	
	def registerSession(self, contrast = 't2', FSsubject = None, register = True, deskull = True, makeMasks = False, maskList = ['cortex','V1','V2','V3','V3A','V3B','V4'], labelFolder = 'label'):
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
				# we have one or more inplane anatomicals - we take the first of these as a reference.
				# first, we need to strip the skull though
				self.logger.info('using inplane_anat as a registration target')
				if deskull:
					better = BETOperator( inputObject = self.runFile( stage = 'processed/mri', run = self.runList[self.inplane_runs[0]]) )
					better.configure( outputFileName = self.runFile( stage = 'processed/mri', run = self.runList[self.inplane_runs[0]], postFix = ['NB']) )
					better.execute()
					self.originalReferenceFunctionalVolume = self.runFile( stage = 'processed/mri', run = self.runList[self.inplane_runs[0]], postFix = ['NB'])
				else:
					self.originalReferenceFunctionalVolume = self.runFile( stage = 'processed/mri', run = self.runList[self.inplane_runs[0]])
				self.logger.info('registration target is inplane_anat, ' + self.originalReferenceFunctionalVolume)
			
			else:
				# we have to make do with epi volumes. so, we motion correct the first epi_bold run
				# prepare the motion corrected first functional	
				mcFirst = MCFlirtOperator( self.runFile(stage = 'processed/mri', run = self.runList[self.epi_runs[0]] ) )
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
			self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration' )
			ExecCommandLine('cp ' + self.originalReferenceFunctionalVolume + ' ' + self.referenceFunctionalFileName )
		
			# register to both freesurfer anatomical and fsl MNI template
			# actual registration - BBRegister to freesurfer subject
			bbR = BBRegisterOperator( self.referenceFunctionalFileName, FSsubject = self.FSsubject, contrast = contrast )
			bbR.configure( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register', extension = '.dat' ), flirtOutputFile = False )
			bbR.execute()
			# after registration, see bbregister log file for reg check
					
			# actual registration - Flirt to MNI brain
			flR = FlirtOperator( self.referenceFunctionalFileName )
			flR.configure( transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'flirt', extension = '.mtx' ) )
			flR.execute()
		
		# having registered everything (AND ONLY AFTER MOTION CORRECTION....) we now construct masks in the functional volume
		if makeMasks:
			# set up the first mean functional in the registration folder anyway for making masks. note that it's necessary to have run the moco first...
			ExecCommandLine('cp ' + self.runFile(stage = 'processed/mri', run = self.runList[self.epi_runs[0]], postFix = ['mcf', 'meanvol'] ) + ' ' + self.runFile(stage = 'processed/mri/reg', postFix = ['mcf', 'meanvol'], base = 'firstFunc' ) )
			
			for mask in maskList:
				for hemi in ['lh','rh']:
					maskFileName = os.path.join(os.path.expandvars('${SUBJECTS_DIR}'), self.FSsubject, labelFolder, hemi + '.' + mask + '.' + 'label')
					stV = LabelToVolOperator(maskFileName)
					stV.configure( 	templateFileName = self.runFile(stage = 'processed/mri/reg', postFix = ['mcf', 'meanvol'], base = 'firstFunc' ), 
									hemispheres = [hemi], 
									register = self.runFile(stage = 'processed/mri/reg', base = 'register', extension = '.dat' ), 
									fsSubject = self.FSsubject,
									outputFileName = self.runFile(stage = 'processed/mri/masks', base = hemi + '.' + mask ), 
									threshold = 0.001 )
					stV.execute()
	
	def motionCorrectFunctionals(self, registerNoMC = True):
		"""
		motionCorrectFunctionals corrects all functionals in a given session.
		how we do this depends on whether we have parallel processing turned on or not
		"""
		self.logger.info('run motion correction')
		self.referenceFunctionalFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration' )
		# set up a list of motion correction operator objects for the runs
		mcOperatorList = [];	stdOperatorList = [];
		for er in self.epi_runs:
			mcf = MCFlirtOperator( self.runFile(stage = 'processed/mri', run = self.runList[er] ), target = self.referenceFunctionalFileName )
		 	mcf.configure()
			mcOperatorList.append(mcf)
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
			self.logger.info("run serial")
			for mcf in mcOperatorList:
				mcf.execute()
		
		if self.parallelize:
			# tryout parallel implementation - later, this should be abstracted out of course. 
			ppservers = ()
			job_server = pp.Server(ppservers=ppservers)
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = []
			for mcf in mcOperatorList:
				fMcf = job_server.submit(mcf.execute)
				ppResults.append(fMcf)
			for fMcf in ppResults:
				fMcf()
			
			job_server.print_stats()
	

class RetinotopicMapperSession(Session):
	def parcelateConditions(self):
		super(RetinotopicMapperSession, self).parcelateConditions()
		self.polar_runs = []
		self.eccen_runs = []
		if 'polar' in self.conditionList:
			self.polar_runs = [hit.indexInSession for hit in filter(lambda x: x.condition == 'polar', [r for r in self.runList])]
		if 'eccen'  in self.conditionList:
			self.eccen_runs = [hit.indexInSession for hit in filter(lambda x: x.condition == 'eccen', [r for r in self.runList])]
	
	def retinotopicMapping(self, useMC = True, perCondition = True, perRun = False, runMapping = True, toSurf = True):
		"""
		runs retinotopic mapping on all runs in self.polar_runs and self.eccen_runs
		"""
		self.logger.info('run retinotopic mapping')
		if len(self.polar_runs) == 0 and len(self.eccen_runs) == 0:
			self.logger.warning('no retinotopic mapping runs to be run...')
			
		presentCommand = '/Users/tk/Documents/research/experiments/retinotopy/RetMapAmsterdam/analysis/other_scripts/selfreqavg_noinfs.csh'
		rmOperatorList = []
		opfNameList = []
		postFix = []
		if useMC:
			postFix.append('mcf')
		if perCondition:
			# set up a list of retinotopic mapping operator objects for the different conditions
			if len(self.polar_runs) > 0:
				# analyze polar runs
				prOperator = RetMapOperator([self.runList[pC] for pC in self.polar_runs], cmd = presentCommand)
				inputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[pC], postFix = postFix) for pC in self.polar_runs]
				outputFileName = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.polar_runs[0]]), 'polar')
				opfNameList.append(outputFileName)
				prOperator.configure( inputFileNames = inputFileNames, outputFileName = outputFileName )
				rmOperatorList.append(prOperator)
			if len(self.eccen_runs) > 0:
				# analyze eccen runs
				erOperator = RetMapOperator([self.runList[eC] for eC in self.eccen_runs], cmd = presentCommand)
				inputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[eC], postFix = postFix) for eC in self.eccen_runs]
				outputFileName = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.eccen_runs[0]]), 'eccen')
				opfNameList.append(outputFileName)
				erOperator.configure( inputFileNames = inputFileNames, outputFileName = outputFileName )
				rmOperatorList.append(erOperator)
				
		if perRun:
			# set up a list of retinotopic mapping operator objects for the different runs
			for i in self.polar_runs:
				# analyze polar runs
				prOperator = RetMapOperator([self.runList[i]], cmd = presentCommand)
				inputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[i], postFix = postFix)]
				outputFileName = os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[i]), 'polar')
				opfNameList.append(outputFileName)
				prOperator.configure( inputFileNames = inputFileNames, outputFileName = outputFileName )
				rmOperatorList.append(prOperator)
			for i in self.eccen_runs:
				# analyze eccen runs
				erOperator = RetMapOperator([self.runList[i]], cmd = presentCommand)
				inputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[i], postFix = postFix)]
				outputFileName = os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[i]), 'eccen')
				opfNameList.append(outputFileName)
				erOperator.configure( inputFileNames = inputFileNames, outputFileName = outputFileName )
				rmOperatorList.append(erOperator)
		
		if runMapping:
			opfString = ''
			for opf in opfNameList:
				opfString += '\n' + os.path.split(opf)[-1]
			self.logger.info('retinotopic mapping to produce output files ' + opfString)
			
			if not self.parallelize:
				# first, code for serial implementation
				self.logger.info("run serial")
				for op in rmOperatorList:
					op.execute()
					
			if self.parallelize:
				# tryout parallel implementation - later, this should be abstracted out of course. 
				ppservers = ()
				job_server = pp.Server(ppservers=ppservers)
				self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
				ppResults = []
				for op in rmOperatorList:
					opex = job_server.submit(op.execute)
					ppResults.append(opex)
					
				for opex in ppResults:
					opex()
					
				job_server.print_stats()
				
		if toSurf:
			# now we need to be able to view the results on the surfaces.
			vtsList = []
			for opf in opfNameList:
				vtsOp = VolToSurfOperator(inputObject = opf + standardMRIExtension)
				vtsOp.configure(register = self.runFile(stage = 'processed/mri/reg', base = 'register', extension = '.dat' ), outputFileName = os.path.join(os.path.split(opf)[0], 'surf/') )
				vtsList.append(vtsOp)
				
			if not self.parallelize:
				self.logger.info("run serial surface projection of retinotopic mapping results")
				for vts in vtsList:
					vts.execute()
					
			if self.parallelize:
				# tryout parallel implementation - later, this should be abstracted out of course. 
				ppservers = ()
				job_server = pp.Server(ppservers=ppservers)
				self.logger.info("run parallel surface projection")
				self.logger.info("Starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
				ppResults = []
				for vts in vtsList:
					vtsex = job_server.submit(vts.execute)
					ppResults.append(vtsex)
				for vtsex in ppResults:
					vtsex()
				job_server.print_stats()
	
	def runQC(self, rois = ['V1','V2','V3']):
		"""
		Quality control for this session would mean to have a look at SNR in different areas
		runQC assumes a list of masks is in place in the processed/mri/masks folder and runs separate analyses for each of these ROIs
		"""
		fs = 10
		
		if len(self.polar_runs) > 0:
			# polar files
			rawInputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[pC], postFix = ['mcf']) for pC in self.epi_runs]
			distilledInputFileNames = [os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[pC]), self.runList[pC].condition) for pC in self.epi_runs]
			distilledInputFileNamesFull = [os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.polar_runs[0]]), 'polar'),os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.polar_runs[0]]), 'eccen')]
			
			for (pd, rd) in zip(distilledInputFileNames, rawInputFileNames):
				# setting up the data files
				rdIm = NiftiImage(rd)
				pdIm = NiftiImage(pd)
				
				pp = PdfPages(self.runFile(stage = 'processed/mri', run = self.runList[self.epi_runs[rawInputFileNames.index(rd)]], extension = '.pdf' ))
				
				for roi in rois:
					for hemi in ['lh','rh']:
						maskFileName = self.runFile(stage = 'processed/mri/masks', base = hemi + '.' + roi )
						
						# run directly on the results of retmapping, that is, the polar.nii.gz and eccen.nii.gz files 
						# mask the input files:
						imO = ImageMaskingOperator(pdIm, maskObject = maskFileName)
						roiStatData = imO.applySingleMask(flat = True)		# F value is 3, noise SD is 6, phase value is 9
						
						fig = pl.figure()
						s = fig.add_subplot(321)
						pl.hist(roiStatData[3], color='r', alpha = 0.25, normed = True, bins = np.linspace(0,10,25), rwidth = 0.5 )
						s.set_title('F values in ' + roi + ' ' + hemi + '\n' + str(rd.split('/')[-3:]), fontsize=fs)
						s.axis([0, 10, 0, 0.5])
						s = fig.add_subplot(323)
						pl.hist(roiStatData[9], color='g', alpha = 0.25, normed = True, bins = np.linspace(-pi,pi,25), rwidth = 0.5)
						s.set_title('phase values in ' + roi, fontsize=fs)
						s.axis([-pi, pi, 0, 0.5])
						s = fig.add_subplot(325)
						pl.hist(roiStatData[6], color='b', alpha = 0.25, normed = True, bins = np.linspace(50,250,25), rwidth = 0.5)
						s.set_title('noise SD values in ' + roi, fontsize=fs)
						s.axis([50, 250, 0, 0.02])
						
						# run on the raw periodic data, that is, the polar.nii.gz and eccen.nii.gz files 
						imO = ImageMaskingOperator(rdIm, maskObject = maskFileName)
						roiRawData = imO.applySingleMask(flat = True)
						self.logger.debug('masked raw data from roi %s in hemi %s is %s', roi, hemi, str(roiRawData.shape) )
						# the best voxel in this file:
						bestVoxArray = roiStatData[3] == roiStatData[3].max()
						# show timecourse of run in this best voxel
						s = fig.add_subplot(222)
						s.plot( np.arange(rdIm.data.shape[0]) * rdIm.rtime, roiRawData[:,bestVoxArray] )
						s.set_title('best voxel time course in ' + roi + ' ' + hemi + '\n' + str(rd.split('/')[-3:]), fontsize=fs)
						pl.xlabel('time [s]')
						s = fig.add_subplot(224)
						ftRoiRD = np.abs(np.fft.fft(roiRawData, axis = 0)).mean(axis = -1)
						ftticks = np.fft.fftfreq(roiRawData.shape[0], d = rdIm.rtime )
						s.set_title('power spectrum ' + roi + ' ' + hemi, fontsize=fs)
						s.plot( ftticks[1:floor(ftRoiRD.shape[0]/2.0)], ftRoiRD[1:floor(ftRoiRD.shape[0]/2.0)] )
						pl.xlabel('frequency [Hz]')
					
						pp.savefig()
					
				pp.close()
#		pl.show()
	

class RivalrySession(Session):
	def parcelateConditions(self):
		super(RivalrySession, self).parcelateConditions()
		self.rivalry_runs = []
		self.disparity_runs = []
		if 'rivalry' in self.conditionList:
			self.rivalry_runs = [hit.indexInSession for hit in filter(lambda x: x.condition == 'rivalry', [r for r in self.runList])]
		if 'disparity' in self.conditionList:
			self.disparity_runs = [hit.indexInSession for hit in filter(lambda x: x.condition == 'disparity', [r for r in self.runList])]
	
	def analyzeBehavior(self):
		"""docstring for analyzeBehaviorPerRun"""
		for r in self.epi_runs:
			# do principal analysis
			self.runList[r].behavior()
			# put in the right place
			ExecCommandLine( 'cp ' + self.runList[r].bO.inputFileName + ' ' + self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.pickle' ) )
			self.runList[r].behaviorFile = self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.pickle' )
			
		if len(self.disparity_runs) > 0:
			self.disparityPsychophysics = []
			for r in self.disparity_runs:
				self.disparityPsychophysics.append([self.runList[r].bO.disparities ,self.runList[r].bO.answersPerStimulusValue, self.runList[r].bO.meanAnswersPerStimulusValue, self.runList[r].bO.fit])
				# back up behavior analysis in pickle file
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump([self.runList[r].bO.disparities ,self.runList[r].bO.answersPerStimulusValue, self.runList[r].bO.meanAnswersPerStimulusValue,self.runList[r].bO.fit.data], f)
				f.close()
			# repeat fitting across trials
			allFitsData = np.array([d[-1].data for d in self.disparityPsychophysics])
			data = zip(allFitsData[0,:,0],allFitsData[:,:,1].sum(axis = 0),allFitsData[:,:,2].sum(axis = 0))
			pf = BootstrapInference(data, sigmoid = 'gauss', core = 'ab', nafc = 1, cuts = [0.25,0.5,0.75])
			pf.sample()
			GoodnessOfFit(pf)
		
		
		if len(self.rivalry_runs) > 0:
			self.rivalryBehavior = []
			for r in self.rivalry_runs:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.buttonEvents], f)
				f.close()
			
		
			fig = pl.figure()
			s = fig.add_subplot(1,1,1)
			# first series of EPI runs for rivalry learning
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.scatter(np.arange(6)+0.5, [self.rivalryBehavior[i][0] for i in range(6)], c = 'b', alpha = 0.85)
			pl.scatter(np.arange(6)+0.5, [self.rivalryBehavior[i][2] for i in range(6)], c = 'b', alpha = 0.75, marker = 's')

			# all percept events, plotted on top of this
	#		with (first) and without (second) taking into account the transitions that were reported.
	#		pl.plot(np.concatenate([(self.rivalryBehavior[rb][3][:,0]/150.0) + rb for rb in range(6)]), np.concatenate([self.rivalryBehavior[rb][3][:,1] for rb in range(6)]), c = 'b', alpha = 0.35)
			pl.plot(np.concatenate([(self.rivalryBehavior[rb][5][:,0]/150.0) + rb for rb in range(6)]), np.concatenate([self.rivalryBehavior[rb][5][:,1] for rb in range(6)]), c = 'b', alpha = 0.25)
			# second series of EPI runs
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.scatter(np.arange(6,12)+0.5, [self.rivalryBehavior[i][0] for i in range(6,12)], c = 'g', alpha = 0.85)
			pl.scatter(np.arange(6,12)+0.5, [self.rivalryBehavior[i][2] for i in range(6,12)], c = 'g', alpha = 0.75, marker = 's')
			# all percept events, plotted on top of this
	#		with (first) and without (second) taking into account the transitions that were reported.
	#		pl.plot(np.concatenate([(self.rivalryBehavior[rb][3][:,0]/150.0) + rb - 6 for rb in range(6,12)]), np.concatenate([self.rivalryBehavior[rb][3][:,1] for rb in range(6,12)]), c = 'g', alpha = 0.35)
			pl.plot(np.concatenate([(self.rivalryBehavior[rb][5][:,0]/150.0) + rb for rb in range(6,12)]), np.concatenate([self.rivalryBehavior[rb][5][:,1] for rb in range(6,12)]), c = 'g', alpha = 0.25)
			s.axis([-1,13,0,12])
		
	#		fig.add_subplot(2,1,2)
	#		for i in range(len(self.disparityPsychophysics)):
	#			pl.plot(self.disparityPsychophysics[i][0],self.disparityPsychophysics[i][2])
			pl.savefig(self.runFile(stage = 'processed/behavior', extension = '.pdf', base = 'duration_summary' ))
		
