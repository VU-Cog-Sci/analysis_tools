#!/usr/bin/env python
# encoding: utf-8
"""
CommandLineOperator.py

Created by Tomas Knapen on 2010-09-23.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import tempfile, logging

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from nifti import *
from Operator import *
from ..log import *

### Execute program in shell:
def ExecCommandLine(cmdline):
	tmpf = tempfile.TemporaryFile()
	try:
		retcode = subprocess.call( cmdline, shell=True, bufsize=0, stdout = tmpf, stderr = tmpf)
	finally:
		tmpf.close()
		if retcode > 0:
			raise ValueError, 'Process: '+cmdline+' returned error code: '+str(retcode)
	return retcode

class CommandLineOperator( Operator ):
	def __init__(self, inputObject, cmd, **kwargs):
		"""
		CommandLineOperator can take a Nii file as input but will use only the variable inputFileName
		"""
		super(CommandLineOperator, self).__init__(inputObject = inputObject, **kwargs)
		
		if self.inputObject.__class__.__name__ == 'NiftiImage':
			self.inputFileName = self.inputObject.filename
			self.logger.info(self.__repr__() + ' initialized with ' + os.path.split(self.inputFileName)[-1])
		elif self.inputObject.__class__.__name__ == 'str':
			self.inputFileName = self.inputObject
			self.logger.info(self.__repr__() + ' initialized with file ' + os.path.split(self.inputFileName)[-1])
			if not os.path.isfile(self.inputFileName):
				self.logger.warning('inputFileName is not a file at initialization')
		
		self.cmd = cmd
	
	def configure(self):
		"""
		placeholder for configure
		to be filled in by subclasses
		"""
		self.runcmd = self.cmd + ' ' + self.inputFileName
		
	def execute(self):
		"""
		placeholder for execute
		to be filled in by subclasses
		"""
		self.logger.info(self.__repr__() + 'executing command \n' + self.runcmd)
		# print self.runcmd
		# subprocess.call( self.runcmd, shell=True, bufsize=0, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		ExecCommandLine(self.runcmd)
	

class MCFlirtOperator( CommandLineOperator ):
	"""docstring for MCFlirtOperator"""
	def __init__(self, inputObject, costFunction = 'normmi', target = None, **kwargs):
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(MCFlirtOperator, self).__init__(inputObject = inputObject, cmd = 'mcflirt', **kwargs)
		
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		self.costFunction = costFunction
		self.target = target
	
	def configure(self, plot = True, sinc = True, report = True, outputFileName = None):
		"""
		configure will run mcflirt motion correction on file in inputObject
		as specified by parameters in __init__ arguments and here to run.
		"""
		
		runcmd = self.cmd
		runcmd += ' -in ' + self.inputFileName
		runcmd += ' -cost ' + self.costFunction
		# configure optional commands
		if self.target:
			runcmd += ' -r ' + self.target
		if outputFileName:
			self.outputFileName = outputFileName
			runcmd += ' -out ' + self.outputFileName
		if plot:
			runcmd += ' -stats'
		if report:
			runcmd += ' -mats'
			runcmd += ' -plots'
			runcmd += ' -report'
		if sinc:
			runcmd += ' -stages 4'	
			runcmd += ' -sinc_final'
		
		self.runcmd = runcmd
	

class FlirtOperator( CommandLineOperator ):
	"""docstring for FlirtOperator"""
	def __init__(self, inputObject, cmd = 'flirt', referenceFileName = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz', costFunction = 'normmi', **kwargs):
		"""
		other reasonable options for referenceFileName are this subject's freesurfer anatomical or the inplane_anat that is run in the same session
		"""
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(FlirtOperator, self).__init__(inputObject = inputObject, cmd = cmd, **kwargs)
		self.referenceFileName = referenceFileName
		self.costFunction = costFunction
	
	def configureApply(self, transformMatrixFileName, outputFileName = None, sinc = True):
		"""
		apply runs flirt's applyxfm argument. 
		It takes an input matrix and a reference file in order to use transformMatrix
		to perform the transformation - it doesn't calculate the transformation itself.
		"""
		self.transformMatrixFileName = transformMatrixFileName
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.join(os.path.splitext(os.path.splitext(self.inputFileName)[0])[0], '_trans.nii.gz')
		
		applycmd = self.cmd + ' -applyxfm'
		applycmd += ' -in ' + self.inputFileName
		applycmd += ' -ref ' + self.referenceFileName
		applycmd += ' -init ' + self.transformMatrixFileName
		if sinc:
			applycmd += ' -interp sinc '
		
		self.runcmd = applycmd
		
	
	def configureRun(self, outputFileName = None, transformMatrixFileName = None, sinc = True, resample = True):
		"""
		run runs actual transformation calculation
		"""
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_trans' + standardMRIExtension
		if transformMatrixFileName:
			self.transformMatrixFileName = transformMatrixFileName
		else:
			self.transformMatrixFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_trans.mat'
		
		runcmd = self.cmd
		runcmd += ' -in ' + self.inputFileName
		runcmd += ' -ref ' + self.referenceFileName
		runcmd += ' -out ' + self.outputFileName
		runcmd += ' -omat ' + self.transformMatrixFileName
		if not resample:
			runcmd += ' -noresample'
		
		if sinc:
			runcmd += ' -interp sinc '
		
		self.runcmd = runcmd
		
	def configure(self, outputFileName = None, transformMatrixFileName = None, sinc = True, resample = True):
		"""
		standard configure is configureRun instead of apply
		"""
		self.configureRun( outputFileName = outputFileName, transformMatrixFileName = transformMatrixFileName, sinc = sinc, resample = resample)
	
	
class BETOperator( CommandLineOperator ):
	"""
	BETOperator does something like /usr/local/fsl/bin/bet /Users/tk/Documents/research/experiments/retinotopy/RetMapAmsterdam/data/TK/TK_080910/processed/mri/inplane_anat/5/TK_080910_5 /Users/tk/Documents/research/experiments/retinotopy/RetMapAmsterdam/data/TK/TK_080910/processed/mri/inplane_anat/5/TK_080910_5_NB  -f 0.5 -g 0 -m
	"""	
	def __init__(self, inputObject, **kwargs):
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(BETOperator, self).__init__(inputObject = inputObject, cmd = 'bet', **kwargs)
	
	def configure(self, outputFileName = None):
		"""
		configure will run mcflirt motion correction on file in inputObject
		as specified by parameters in __init__ arguments and here to run.
		"""
		runcmd = self.cmd
		runcmd += ' ' + self.inputFileName
		if outputFileName:
			self.outputFileName = outputFileName
			runcmd += ' ' + self.outputFileName
		else:
			runcmd += ' ' + os.path.splitext(self.inputFileName)[0] + '_NB'
		# configure parameters - will perhaps make this amenable
		runcmd += ' -f 0.4 -g 0 -m '
		self.runcmd = runcmd
	

class BBRegisterOperator( CommandLineOperator ):
	"""
	BBRegisterOperator invokes bbregister
	"""
	def __init__(self, inputObject, FSsubject, contrast = 't2', **kwargs):
		super(BBRegisterOperator, self).__init__( inputObject = inputObject, cmd = 'bbregister', **kwargs)
		self.FSsubject = FSsubject
		self.contrast = contrast
	
	def configure(self, transformMatrixFileName, flirtOutputFile = True):
		"""
		run will run mcflirt motion correction on file in inputObject
		as specified by parameters in __init__ arguments and here to run
		output file is usually session's "processed/mri/reg/register.dat"
		"""
		self.transformMatrixFileName = transformMatrixFileName
		
		runcmd = self.cmd
		runcmd += ' --s ' + self.FSsubject
		runcmd += ' --reg ' + self.transformMatrixFileName
		runcmd += ' --mov ' + self.inputFileName
		runcmd += ' --' + self.contrast
		# specify these options regardless
		runcmd += ' --init-fsl'
		# specify these options dependent on run arguments
		if flirtOutputFile:
			self.flirtOutputFileName = os.path.join(os.path.split(transformMatrixFileName)[0], 'register_flirt_BB.mtx')
			runcmd += ' --fsl ' + self.flirtOutputFileName
		
		self.runcmd = runcmd
	

class FSLMathsOperator( CommandLineOperator ):
	"""docstring for FSLMathsOperator"""
	def __init__(self, inputObject, cmd = 'fslmaths', outputDataType = 'float', **kwargs):
		super(FSLMathsOperator, self).__init__( inputObject = inputObject, cmd = cmd, **kwargs )
		self.outputDataType = outputDataType
		self.outputFileName = None
	
	def configure(self, outputFileName = None, **kwargs):
		"""
		configure takes a dict of arguments
		they are all represented as key-values in the command line 
		except when value is empty, then only key is printed on the command line
		"""
		if outputFileName:
			self.outputFileName = outputFileName
		if not self.outputFileName:
			self.outputFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_fslmaths' + standardMRIExtension
			
		mathcmd = self.cmd
		
		mathcmd += ' ' + self.inputFileName

		for k,v in kwargs.items():
			mathcmd += ' ' + k
			if v!= '':
				mathcmd += ' ' + v
		
		mathcmd += ' ' + self.outputFileName
		mathcmd += ' -odt ' +  self.outputDataType
		self.runcmd = mathcmd
	
	def configureTMean(self, outputFileName = None):
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_meanvol' + standardMRIExtension
		
		meanArgs = {'-Tmean': ''}
		self.configure( outputFileName = self.outputFileName, **meanArgs )	
	
	def configureTStd(self, outputFileName = None):
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_sigma' + standardMRIExtension
		
		meanArgs = {'-Tstd': ''}
		self.configure( outputFileName = self.outputFileName, **meanArgs )
	

class retMapRun(object):
	def   __init__(self, ID, stimType, direction, TR, niiFilePath, delay = 4.0, nSkip = 12, nrCycles = 6):
		self.ID = ID
		self.TR = TR
		self.nrCycles = nrCycles
		self.stimType = stimType
		self.direction = direction
		self.nSkip = nSkip
		self.delay = delay
		self.niiFilePath = niiFilePath
		self.standardParFileName = 'selfreqavg.par'
		self.parFileName = os.path.join(os.path.split(self.niiFilePath)[0], self.standardParFileName)
		
	def createParFile(self):
		parFileString = 'stimtype ' + self.stimType + '\n'
		parFileString += 'direction ' + self.direction + '\n'
		parFileString += 'ncycles ' + str(self.nrCycles) + '\n'
		
		f = open(self.parFileName,'w')
		f.write(parFileString)
		f.close()
	

class RetMapOperator( CommandLineOperator ):
	"""
	RetMapOperator takes a list of runList items which specify the runs to be analyzed. 
	the runlist items, from the session which calls the retmapoperator, dictate 
	mapping parameters for the retinotopic mapping analyses.
	"""
	def __init__(self, inputObject, cmd = 'other_scripts/selfreqavg_noinfs.csh', **kwargs):
		super(RetMapOperator, self).__init__(inputObject, cmd = cmd, **kwargs)
	
	def configure(self, inputFileNames, outputFileName):
		"""configure runs and the command line command to run the analysis"""
		if len(inputFileNames) != len(self.inputList):
			self.logger.error('different numbers for input file names and runs')
		self.inputFileNames = inputFileNames
		if outputFileName[-7:] == standardMRIExtension:
			self.outputFileName = outputFileName[:-7]
		else:
			self.outputFileName = outputFileName
		
		self.allRuns = []
		for (rmRun, fp) in zip(self.inputList, inputFileNames):
			thisRun = retMapRun(	
						ID = rmRun.indexInSession,
						stimType = rmRun.mappingType,
						direction = rmRun.direction,
						nrCycles = rmRun.nrCycles,
						TR = rmRun.TR,
						delay = rmRun.delay,
						niiFilePath = fp
						)
			thisRun.createParFile()
			self.allRuns.append(thisRun)
		
		# for this analysis to work each nii file has to be in a separate directory.
		self.runcmd = self.cmd
		for rmRun in self.allRuns:
			self.runcmd += ' -i ' + rmRun.niiFilePath
		self.runcmd += ' -TR ' + str(self.allRuns[0].TR)
		self.runcmd += ' -delay ' + str(self.allRuns[0].delay)
		self.runcmd += ' -detrend'
		self.runcmd += ' -o ' + self.outputFileName
		self.runcmd += ' -parname ' + self.allRuns[0].standardParFileName
	

class VolToSurfOperator( CommandLineOperator ):
	"""docstring for VolToSurfOperator"""
	def __init__(self, inputObject, cmd = 'mri_vol2surf', **kwargs):
		super(VolToSurfOperator, self).__init__(inputObject, cmd = cmd, **kwargs)
		
	def configure(self, frames = {'sig-0':0, 'map-real':1, 'map-imag':2, 'phase':9, 'noise_sd': 6,'sigf':0, 'sig2':1, 'sig3':2}, hemispheres = None, register = None, outputFileName = None, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  ):
		"""docstring for configure"""
		# don't feel like calling splitext twice
		if outputFileName[-7:] == standardMRIExtension:
			self.outputFileName = outputFileName[:-7]
		else:
			self.outputFileName = outputFileName
		self.register = register
		self.surfType = surfType
		if hemispheres == None:
			hemispheres = ['lh','rh']
		if register == None:
			self.logger.warning('no registration file given. this negligence will not stand.')
		
		self.runcmd = ''
		for hemi in hemispheres:
			for frame in frames.keys():
				self.runcmd += self.cmd + ' --srcvol ' + self.inputFileName
				self.runcmd += ' --src_type nii.gz'
				self.runcmd += ' --srcreg ' + self.register
				self.runcmd += ' --surf smoothwm'
				self.runcmd += ' --hemi  ' + hemi
				self.runcmd += " --projfrac " + str(threshold)
				self.runcmd += ' --frame ' + str(frames[frame])
				self.runcmd += ' --out_type ' + self.surfType + ' --float2int round --mapmethod nnf '
				self.runcmd += ' --o ' + self.outputFileName + frame + '-' + hemi + '.w'
				self.runcmd += ' --surf-fwhm ' + str(surfSmoothingFWHM)
				self.runcmd += ' &\n'
		# make sure the last ampersand is not listed - else running this on many runs in one go will explode.
		self.runcmd = self.runcmd[:-2]
	

class SurfToVolOperator( CommandLineOperator ):
	"""docstring for SurfToVolOperator"""
	def __init__(self, inputObject, cmd = 'mri_surf2vol', **kwargs):
		super(SurfToVolOperator, self).__init__(inputObject, cmd = cmd, **kwargs)

	def configure(self, templateFileName, hemispheres = None, register = None, fsSubject = '', outputFileName = None, threshold = 0.5, surfType = 'paint' ):
		"""docstring for configure"""
		# don't feel like calling splitext twice
		if outputFileName[-7:] == standardMRIExtension:
			self.outputFileName = outputFileName[:-7]
		else:
			self.outputFileName = outputFileName
		self.templateFileName = templateFileName
		self.register = register
		self.surfType = surfType
		if hemispheres == None:
			hemispheres = ['lh','rh']
		if register == None:
			self.logger.warning('no registration file given. this negligence will not stand.')

		self.runcmd = ''
		for hemi in hemispheres:
			for frame in frames.keys():
				self.runcmd += self.cmd + ' --surfval ' + self.inputFileName + ' ' + self.surfType 
				self.runcmd += ' --reg ' + self.register
				self.runcmd += ' --surf smoothwm'
				self.runcmd += ' --hemi  ' + hemi
				self.runcmd += ' --template '+ self.templateFileName
#				self.runcmd += ' --subject ' + fsSubject	# no need to use this if using a correct register file
				self.runcmd += " --projfrac " + str(threshold)
				self.runcmd += ' --o ' + self.outputFileName + frame + '-' + hemi + '.w'
				self.runcmd += ' &\n'
		# make sure the last ampersand is not listed - else running this on many runs in one go will explode.
		self.runcmd = self.runcmd[:-2]
	

class LabelToVolOperator( CommandLineOperator ):
	"""docstring for LabelToVolOperator"""
	def __init__(self, inputObject, cmd = 'mri_label2vol', **kwargs):
		super(LabelToVolOperator, self).__init__(inputObject, cmd = cmd, **kwargs)
		
	
	def configure(self, templateFileName, hemispheres = None, register = None, fsSubject = '', outputFileName = None, threshold = 0.5, surfType = 'label'):
		"""
		configure sets up the command line for surf to vol translation.
		"""
		# don't feel like calling splitext twice
		if outputFileName[-7:] == standardMRIExtension:
			self.outputFileName = outputFileName[:-7]
		else:
			self.outputFileName = outputFileName 
		self.templateFileName = templateFileName
		self.register = register
		self.surfType = surfType
		if hemispheres == None:
			hemispheres = ['lh','rh']
		if register == None:
			self.logger.warning('no registration file given. this negligence will not stand.')
		
		self.runcmd = ''
		for hemi in hemispheres:
			self.runcmd += self.cmd
			if self.surfType == 'label':
				self.runcmd += ' --label '+ self.inputFileName
			elif self.surfType == 'annot':
				self.runcmd += ' --annot '+ self.inputFileName
			self.runcmd += ' --temp '+ self.templateFileName
			self.runcmd += ' --reg '+ self.register
			self.runcmd += ' --fillthresh ' + str(threshold)
			self.runcmd += ' --o ' + self.outputFileName + standardMRIExtension
			self.runcmd += ' --subject ' + fsSubject
			self.runcmd += ' --hemi ' + hemi
			self.runcmd += ' &\n'
			
		# make sure the last ampersand is not listed - else running this on many runs in one go will explode.
		self.runcmd = self.runcmd[:-2]
		
	
class AnnotationToLabelOperator( CommandLineOperator ):
	"""docstring for LabelToVolOperator"""
	def __init__(self, inputObject, cmd = 'mri_annotation2label', **kwargs):
		super(AnnotationToLabelOperator, self).__init__(inputObject, cmd = cmd, **kwargs)
	
	def configure(self, subjectID, hemispheres = None ):
		"""
		configure sets up the command line for surf to vol translation.
		"""
		# self.inputObject is an annotation file name
		annotationName = os.path.splitext(os.path.split(self.inputFileName)[1])[0].split('.')[1]
		
		if hemispheres == None:
			hemispheres = ['lh','rh']
		if subjectID == None:
			self.logger.warning('no subjectID given. this negligence will not stand.')

		self.runcmd = ''
		for hemi in hemispheres:
			self.runcmd += self.cmd
			self.runcmd += ' --subject ' + subjectID
			self.runcmd += ' --hemi ' + hemi
			self.runcmd += ' --outdir ' + os.path.join(os.environ['SUBJECTS_DIR'], subjectID, 'label', annotationName)
			self.runcmd += ' &\n'

		# make sure the last ampersand is not listed - else running this on many runs in one go will explode.
		self.runcmd = self.runcmd[:-2]

class ParRecConversionOperator( CommandLineOperator ):
	"""docstring for ParRecConversionOperator"""
	def __init__(self, inputObject, cmd = 'dcm2nii', **kwargs):
		super(ParRecConversionOperator, self).__init__( inputObject, cmd = cmd, **kwargs )
		
	def configure(self):
		self.runcmd = self.cmd
#		self.runcmd += ' -b ' + os.path.expanduser('~/.dcm2nii/dcm2nii.ini')
		self.runcmd += ' -d n'
		self.runcmd += ' -e n'
		self.runcmd += ' -p n'
		self.runcmd += ' -i n'
		self.runcmd += ' -v n'
		self.runcmd += ' -f y'
		self.runcmd += ' ' + self.inputFileName
		
		