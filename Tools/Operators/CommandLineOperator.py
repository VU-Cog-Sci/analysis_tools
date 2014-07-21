#!/usr/bin/env python
# encoding: utf-8
"""
CommandLineOperator.py

Created by Tomas Knapen on 2010-09-23.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import tempfile, logging
import re

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from nifti import *
from Operator import *
from ..log import *

from IPython import embed as shell


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
		elif self.inputObject.__class__.__name__ == 'list':
			self.inputList = self.inputObject
			self.logger.info(self.__repr__() + ' initialized with files ' + str(self.inputList))
		self.cmd = cmd

	def configure(self):
		"""
		placeholder for configure
		to be filled in by subclasses
		"""
		self.runcmd = self.cmd + ' ' + self.inputFileName

	def execute(self, wait = True):
		"""
		placeholder for execute
		to be filled in by subclasses
		"""
		self.logger.debug(self.__repr__() + 'executing command \n' + self.runcmd)
		# print self.runcmd
		# subprocess.call( self.runcmd, shell=True, bufsize=0, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		if not wait:
			self.runcmd + '&'
		ExecCommandLine(self.runcmd)


class MCFlirtOperator( CommandLineOperator ):
	"""docstring for MCFlirtOperator"""
	def __init__(self, inputObject, costFunction = 'normmi', target = None, **kwargs):
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(MCFlirtOperator, self).__init__(inputObject = inputObject, cmd = 'mcflirt', **kwargs)
		if 'sara' or 'aeneas' in os.uname()[1]:
			pass
		else:
			self.cmd = 'export PATH="/usr/local/fsl/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin"; mcflirt'
		
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		self.costFunction = costFunction
		self.target = target
		
	def configure(self, plot = True, sinc = True, report = True, outputFileName = None, further_args = ''):
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
		# else:
		# 	self.outputFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_mcf.nii.gz'
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
		
		if hasattr(self, 'transformMatrixFileName'):
			runcmd += ' -init ' + self.transformMatrixFileName
		
		self.runcmd = runcmd + further_args

class MRIConvertOperator( CommandLineOperator ):
	def __init__(self, inputObject, cmd = 'mri_convert', **kwargs): # source ~/.bash_profile_fsl ;
		"""
		"""
		super(MRIConvertOperator, self).__init__(inputObject = inputObject, cmd = cmd, **kwargs)

	def configure(self, output_type = '.nii.gz'):
		self.outputFileName = os.path.splitext(self.inputFileName)[0] + output_type
		self.runcmd = self.cmd + ' ' + self.inputFileName + ' ' + self.outputFileName


class FlirtOperator( CommandLineOperator ):
	"""docstring for FlirtOperator"""
	def __init__(self, inputObject, referenceFileName = '$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz', cmd = 'flirt', costFunction = 'normmi', **kwargs): # source ~/.bash_profile_fsl ;
		"""
		other reasonable options for referenceFileName are this subject's freesurfer anatomical or the inplane_anat that is run in the same session
		"""
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(FlirtOperator, self).__init__(inputObject = inputObject, cmd = cmd, **kwargs)
		if 'sara' or 'aeneas' in os.uname()[1]:
			pass
		else:
			self.cmd = 'export PATH="/usr/local/fsl/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin"; flirt'
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
			self.outputFileName = os.path.join(os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_trans.nii.gz')

		applycmd = self.cmd + ' -applyxfm'
		applycmd += ' -in ' + self.inputFileName
		applycmd += ' -ref ' + self.referenceFileName
		applycmd += ' -init ' + self.transformMatrixFileName
		applycmd += ' -o ' + self.outputFileName

		if sinc:
			applycmd += ' -interp sinc '

		self.runcmd = applycmd


	def configureRun(self, outputFileName = None, transformMatrixFileName = None, sinc = True, resample = True, extra_args = ''):
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
			runcmd += ' -interp sinc -sincwidth 7 -sincwindow hanning '
		
		runcmd += extra_args
		
		self.runcmd = runcmd

class InvertFlirtOperator( CommandLineOperator ):
	"""docstring for FlirtOperator"""
	def __init__(self, inputObject, cmd = 'convert_xfm', **kwargs): # source ~/.bash_profile_fsl ;
		"""
		other reasonable options for referenceFileName are this subject's freesurfer anatomical or the inplane_anat that is run in the same session
		"""
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(InvertFlirtOperator, self).__init__(inputObject = inputObject, cmd = cmd, **kwargs)
	
	def configure(self, outputFileName = None):
		"""
		standard configure is configureRun instead of apply
		"""
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.splitext(self.inputFileName)[0] + '_inv.mat'
		self.transformMatrixFileName = self.outputFileName
		runcmd = self.cmd
		runcmd += ' -omat ' + self.outputFileName
		runcmd += ' -inverse ' + self.inputFileName
		self.runcmd = runcmd
		
class ConcatFlirtOperator( CommandLineOperator ):
	"""docstring for FlirtOperator"""
	def __init__(self, inputObject, cmd = 'convert_xfm', **kwargs): # source ~/.bash_profile_fsl ;
		"""
		other reasonable options for referenceFileName are this subject's freesurfer anatomical or the inplane_anat that is run in the same session
		"""
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(ConcatFlirtOperator, self).__init__(inputObject = inputObject, cmd = cmd, **kwargs)
		if 'sara' or 'aeneas' in os.uname()[1]:
			pass
		else:
			self.cmd = 'export PATH="/usr/local/fsl/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin"; convert_xfm'
	
	def configure(self, secondInputFile, outputFileName = None):
		"""
		standard configure is configureRun instead of apply
		"""
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.splitext(self.inputFileName)[0] + '_concat.mat'
		self.transformMatrixFileName = self.outputFileName
		self.secondInputFile = secondInputFile
		runcmd = self.cmd
		runcmd += ' -omat ' + self.outputFileName
		runcmd += ' -concat ' + self.secondInputFile
		runcmd += ' ' + self.inputFileName
		self.runcmd = runcmd


class BETOperator( CommandLineOperator ):
	"""
	BETOperator does something like /usr/local/fsl/bin/bet /Users/tk/Documents/research/experiments/retinotopy/RetMapAmsterdam/data/TK/TK_080910/processed/mri/inplane_anat/5/TK_080910_5 /Users/tk/Documents/research/experiments/retinotopy/RetMapAmsterdam/data/TK/TK_080910/processed/mri/inplane_anat/5/TK_080910_5_NB -z -f 0.5 -g 0 -m
	"""
	def __init__(self, inputObject, **kwargs):
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(BETOperator, self).__init__(inputObject = inputObject, cmd = 'bet', **kwargs)
		if 'sara' or 'aeneas' in os.uname()[1]:
			pass
		else:
			self.cmd = 'export PATH="/usr/local/fsl/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin"; bet'

	def configure(self, outputFileName = None, f_value = 0.5, g_value = 0.0, Z = True, **kwargs):
		"""
		configure will run BET on file in inputObject
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
		# runcmd += ' -f 0.4 -g 0 -m '
		# standard options for limited calcarine FOV:
		if Z:
			runcmd += ' -Z '
		runcmd += ' -f ' + str(f_value) + ' -g ' + str(g_value) + ' -m '
		
		for k,v in kwargs.items():
			runcmd += ' ' + k
			if v!= '':
				runcmd += ' ' + v
		
		self.runcmd = runcmd


class BBRegisterOperator( CommandLineOperator ):
	"""
	BBRegisterOperator invokes bbregister
	"""
	def __init__(self, inputObject, FSsubject, contrast = 't2', **kwargs):
		super(BBRegisterOperator, self).__init__( inputObject = inputObject, cmd = 'bbregister', **kwargs)
		self.FSsubject = FSsubject
		self.contrast = contrast

	def configure(self, transformMatrixFileName, flirtOutputFile = True, init_fsl = True):
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
		if init_fsl:
			runcmd += ' --init-fsl'
		else:
			runcmd += ' --init-reg ' + self.transformMatrixFileName
		# specify these options dependent on run arguments
		if flirtOutputFile:
			self.flirtOutputFileName = os.path.splitext(transformMatrixFileName)[0] + '_flirt_BB.mtx'
			runcmd += ' --fslmat ' + self.flirtOutputFileName

		self.runcmd = runcmd


class FSLMathsOperator( CommandLineOperator ):
	"""docstring for FSLMathsOperator"""
	def __init__(self, inputObject, cmd = 'fslmaths', outputDataType = 'float', **kwargs):
		super(FSLMathsOperator, self).__init__( inputObject = inputObject, cmd = cmd, **kwargs )
		if 'sara' or 'aeneas' in os.uname()[1]:
			pass
		else:
			self.cmd = 'export PATH="/usr/local/fsl/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin"; fslmaths'
		
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

	def configurePi(self, outputFileName = None, div = 100, mul = 3.141592653589793116):
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_rescaled' + standardMRIExtension
		
		PiArgs = {' -div ': str(div), ' -mul ': str(mul),}
		self.configure( outputFileName = self.outputFileName, **PiArgs )


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

	def configureBPF(self, outputFileName = None, nr_samples_hp = 30, nr_samples_lp = -1.0):
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_tf' + standardMRIExtension

		meanArgs = {'-bptf ': str(nr_samples_hp) + ' ' + str(nr_samples_lp)}
		self.configure( outputFileName = self.outputFileName, **meanArgs )
	
	def configureSmooth(self, outputFileName = None, smoothing_sd = 3.0):
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.splitext(os.path.splitext(self.inputFileName)[0])[0] + '_s%1.2f'%smoothing_sd + standardMRIExtension
			
		smArgs = {' -s ': str(smoothing_sd), }
		self.configure( outputFileName = self.outputFileName, **smArgs )
	
	def configureMask(self, mask_file, outputFileName = None ):
		if outputFileName == None:
			self.outputFileName = self.inputFileName
		else:
			self.outputFileName = outputFileName
		maskArgs = {'-mas ': mask_file}
		self.configure( outputFileName = self.outputFileName, **maskArgs )
		
	def configureAdd(self, add_file, outputFileName = None ):
		if outputFileName == None:
			self.outputFileName = self.inputFileName
		else:
			self.outputFileName = outputFileName
		addArgs = {'-add ': add_file}
		self.configure( outputFileName = self.outputFileName, **addArgs )
		
		
class FEATOperator( CommandLineOperator ):
	"""FEATOperator assumes bash is the shell used, and that fsl binaries are located in /usr/local/fsl/bin/"""
	def __init__(self, inputObject, **kwargs):
		super(FEATOperator, self).__init__(inputObject = inputObject, cmd = 'feat ', **kwargs)
		self.featFile = self.inputObject
		# on lisa it doesn't pay to include fsl paths like that. only necessary after things have been mucked up by macports on the mac.
		if 'sara' or 'aeneas' in os.uname()[1]:
			pass
		else:
			self.cmd = 'export PATH="/usr/local/fsl/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin"; feat'
		

	def configure(self, REDict = {}, featFileName = '', waitForExecute = False):
		"""
		configure will run feat on file in inputObject
		as specified by parameters in __init__ arguments and here to run.
		"""

		self.featFileName = featFileName

		sf = open(self.featFile,'r')
		workingString = sf.read()
		sf.close()
		for e in REDict:
			rS = re.compile(e)
			workingString = re.sub(rS, REDict[e], workingString)

		of = open(self.featFileName, 'w')
		of.write(workingString)
		of.close()

		runcmd = self.cmd
		runcmd += self.featFileName
		if not waitForExecute:
			runcmd += ' & '
		self.runcmd = runcmd

class RETROICOROperator( CommandLineOperator ):
	"""MatlabOperator assumes bash is the shell used, and that fsl binaries are located in /usr/local/fsl/bin/"""
	def __init__(self, inputObject, **kwargs):
		super(RETROICOROperator, self).__init__(inputObject = inputObject, cmd = 'matlab -nodesktop -nosplash -c /home/shared/Niels_UvA/matlab_scripts/license.dat -r ', **kwargs)
		self.m_file = self.inputObject

	def configure(self, REDict = {}, retroicor_m_filename = '', waitForExecute = False):
		"""
		configure will run retroicor on file in inputObject
		as specified by parameters in __init__ arguments and here to run.
		"""

		self.retroicor_m_filename = retroicor_m_filename

		sf = open(self.m_file,'r')
		workingString = sf.read()
		sf.close()
		for e in REDict:
			rS = re.compile(e)
			workingString = re.sub(rS, REDict[e], workingString)

		of = open(self.retroicor_m_filename, 'w')
		of.write(workingString)
		of.close()

		runcmd = self.cmd
		runcmd += '"run ' + self.retroicor_m_filename + '"'
		if not waitForExecute:
			runcmd += ' & '
		self.runcmd = runcmd

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
	def __init__(self, inputObject, cmd = os.path.join(os.environ['ANALYSIS_HOME'], 'Tools', 'other_scripts', 'selfreqavg_noinfs.csh'), **kwargs):
		super(RetMapOperator, self).__init__(inputObject, cmd = cmd, **kwargs)

	def configure(self, inputFileNames, outputFileName):
		"""configure runs and the command line command to run the analysis"""
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
						niiFilePath = fp,
						nSkip = rmRun.nSkip
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
		self.runcmd += ' -nskip ' + str(self.allRuns[0].nSkip)
		self.runcmd += ' -o ' + self.outputFileName
		self.runcmd += ' -parname ' + self.allRuns[0].standardParFileName


class VolToSurfOperator( CommandLineOperator ):
	"""docstring for VolToSurfOperator"""
	def __init__(self, inputObject, cmd = 'mri_vol2surf', **kwargs):
		super(VolToSurfOperator, self).__init__(inputObject, cmd = cmd, **kwargs)

	def configure(self, frames = {'sig-0':0, 'map-real':1, 'map-imag':2, 'phase':9, 'noise_sd': 6,'sigf':0, 'sig2':1, 'sig3':2, 'F':3}, hemispheres = None, register = None, outputFileName = None, threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  ):
		"""docstring for configure"""
		# don't feel like calling splitext twice
		if outputFileName[-7:] == standardMRIExtension:
			self.outputFileName = outputFileName[:-7]
		else:
			self.outputFileName = outputFileName
#		self.outputFileName += '_' + str(surfSmoothingFWHM)

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
				if type(threshold) == float:
					self.runcmd += " --projfrac " + str(threshold)
				else:
					self.runcmd += "  --projfrac-max %d %d %1.2f" % tuple(threshold)
				self.runcmd += ' --frame ' + str(frames[frame])
				self.runcmd += ' --out_type ' + self.surfType + ' --float2int round --mapmethod nnf '
				self.runcmd += ' --o ' + self.outputFileName + frame + '-' + hemi + '.mgh'
				self.runcmd += ' --surf-fwhm ' + str(surfSmoothingFWHM)
				self.runcmd += ' &\n'
		# make sure the last ampersand is not listed - else running this on many runs in one go will explode.
		self.runcmd = self.runcmd[:-2]
		# self.runcmd += ' &'


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
			self.runcmd += self.cmd + ' --surfval ' + self.inputFileName + ' ' + self.surfType
			self.runcmd += ' --reg ' + self.register
			self.runcmd += ' --surf smoothwm'
			self.runcmd += ' --hemi ' + hemi
			self.runcmd += ' --template '+ self.templateFileName
#			self.runcmd += ' --subject ' + fsSubject	# no need to use this if using a correct register file
			self.runcmd += " --projfrac " + str(threshold)
			self.runcmd += ' --o ' + self.outputFileName + '-' + hemi + standardMRIExtension
			self.runcmd += ' ;\n'
		# make sure the last ampersand is not listed - else running this on many runs in one go will explode.
		# self.runcmd = self.runcmd[:-2]
		# self.runcmd += ' &'

class SurfToSurfOperator( CommandLineOperator ):
	"""docstring for SurfToVolOperator"""
	def __init__(self, inputObject, cmd = 'mri_surf2surf', **kwargs):
		super(SurfToSurfOperator, self).__init__(inputObject, cmd = cmd, **kwargs)

	def configure(self, fsSourceSubject = '', fsTargetSubject = '', hemi = None, outputFileName = None, insmooth = 0, intype = 'paint', outtype = 'paint' ):
		"""docstring for configure"""
		# mri_surf2surf --hemi rh --srcsubject ico --srcsurfval icodata-rh --src_type bfloat --trgsubject bert --trgsurfval ./bert-ico-rh.w --trg_type paint
		self.runcmd = self.cmd + ' --srcsubject ' + fsSourceSubject
		self.runcmd += ' --srcsurfval ' + self.inputFileName
		self.runcmd += ' --trgsubject ' + fsTargetSubject
		self.runcmd += ' --hemi ' + hemi
		self.runcmd += ' --trgsurfval ' + outputFileName
		self.runcmd += ' --fwhm-src ' + str(insmooth)
		self.runcmd += ' --src_type ' + intype
		self.runcmd += ' --trg_type ' + outtype
#		self.runcmd += ' ;\n'
		# make sure the last ampersand is not listed - else running this on many runs in one go will explode.
		# self.runcmd = self.runcmd[:-2]
		# self.runcmd += ' &'

class MRISConvertOperator( CommandLineOperator ):
	"""docstring for SurfToVolOperator"""
	def __init__(self, inputObject, cmd = 'mris_convert', **kwargs):
		super(MRISConvertOperator, self).__init__(inputObject, cmd = cmd, **kwargs)

	def configure(self, outputFileName = None, surfaceFile = 'inflated'):
		if outputFileName == None:
			self.outputFileName = os.path.splitext(self.inputFileName)[0] + '.asc'
		else:
			self.outputFileName = outputFileName

		self.runcmd = self.cmd + ' ' + self.inputFileName
#		self.runcmd += ' ' + surfaceFile
		self.runcmd += ' ' + self.outputFileName

class LabelToVolOperator( CommandLineOperator ):
	"""docstring for LabelToVolOperator"""
	def __init__(self, inputObject, cmd = 'mri_label2vol', **kwargs):
		super(LabelToVolOperator, self).__init__(inputObject, cmd = cmd, **kwargs)


	def configure(self, templateFileName, hemispheres = None, register = None, fsSubject = '', outputFileName = None, threshold = 0.5, surfType = 'label', proj_frac = '0 1 .1'):
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
			self.runcmd += ' --proj frac ' + proj_frac
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
		annotationName = os.path.splitext(os.path.split(self.inputFileName)[1])[0].split('.')
		annotationName.pop(0)
		annotationName = '.'.join(annotationName)

		if hemispheres == None:
			hemispheres = ['lh','rh']
		if subjectID == None:
			self.logger.warning('no subjectID given. this negligence will not stand.')

		self.runcmd = ''
		for hemi in hemispheres:
			self.runcmd += self.cmd
			self.runcmd += ' --annotation ' + annotationName
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

class RetMapReDrawOperator( CommandLineOperator ):
	"""docstring for MCFlirtOperator"""
	def __init__(self, inputObject, **kwargs):
		super(RetMapReDrawOperator, self).__init__(inputObject = inputObject, cmd = 'tksurfer ', **kwargs)
		self.redrawFile = self.inputObject

	def configure(self, REDict = {}, redrawFileName = '', waitForExecute = False):
		"""
		configure will run feat on file in inputObject
		as specified by parameters in __init__ arguments and here to run.
		"""

		self.redrawFileName = redrawFileName

		sf = open(self.redrawFile,'r')
		workingString = sf.read()
		sf.close()
		for e in REDict:
			rS = re.compile(e)
			workingString = re.sub(rS, REDict[e], workingString)

		of = open(self.redrawFileName, 'w')
		of.write(workingString)
		of.close()


		runcmd = 'cd ' + os.path.split(redrawFileName)[0] + '; '+ self.cmd

		runcmd += ' ' + REDict['---NAME---']
		runcmd += ' ' + REDict['---HEMI---'] + ' inflated -tcl '

		runcmd += self.redrawFileName
		if not waitForExecute:
			runcmd += ' & '
		self.runcmd = runcmd

class EDF2ASCOperator( CommandLineOperator ):
	"""
	EDF2ASCOperator will convert an edf file to a pair of output files, one containing the gaze samples (.gaz) and another containing all the messages/events (.msg).
	It uses edf2asc command-line executable, which is assumed to be on the $PATH.
	Missing values are imputed as 0.0001, time is represented as a floating point number for 2000Hz sampling.
	"""
	def __init__(self, inputObject, **kwargs):
		super(EDF2ASCOperator, self).__init__(inputObject = inputObject, cmd = 'edf2asc', **kwargs)

	def configure(self, gazeOutputFileName = None, messageOutputFileName = None, settings = ' -t -ftime '):
		if gazeOutputFileName == None:
			self.gazeOutputFileName = os.path.splitext(self.inputFileName)[0] + '.gaz'
		else:
			self.gazeOutputFileName = gazeOutputFileName
		if messageOutputFileName == None:
			self.messageOutputFileName = os.path.splitext(self.inputFileName)[0] + '.msg'
		else:
			self.messageOutputFileName = messageOutputFileName
		standardOutputFileName = os.path.splitext(self.inputFileName)[0] + '.asc'

		self.intermediatecmd = self.cmd
		self.intermediatecmd += settings

		self.gazcmd = self.intermediatecmd + '-y -z -v -s -miss 0.0001 -vel "'+self.inputFileName+'"; mv ' + '"' + standardOutputFileName.replace('|', '\|') + '" "' + self.gazeOutputFileName.replace('|', '\|') + '"'
		self.msgcmd = self.intermediatecmd + '-y -z -v -e "'+self.inputFileName+'"; mv ' + '"' + standardOutputFileName.replace('|', '\|') + '" "' + self.messageOutputFileName.replace('|', '\|') + '"'

		self.runcmd = self.gazcmd + '; ' + self.msgcmd

class FSLSplitOperator( CommandLineOperator ):
	"""docstring for FlirtOperator"""
	def __init__(self, inputObject, cmd = 'fslsplit'): # source ~/.bash_profile_fsl ;
		"""
		other reasonable options for referenceFileName are this subject's freesurfer anatomical or the inplane_anat that is run in the same session
		"""
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(FSLSplitOperator, self).__init__(inputObject = inputObject, cmd = cmd)
		if 'sara' or 'aeneas' in os.uname()[1]:
			pass
		else:
			self.cmd = 'export PATH="/usr/local/fsl/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin"; fslsplit'

	def configure(self, outputFileName = None):
		"""
		apply runs flirt's applyxfm argument.
		It takes an input matrix and a reference file in order to use transformMatrix
		to perform the transformation - it doesn't calculate the transformation itself.
		"""
		
		if outputFileName:
			self.outputFileName = outputFileName
		else:
			self.outputFileName = os.path.join(os.path.splitext(os.path.splitext(self.inputObject)[0])[0] + '_split')
		
		applycmd = self.cmd + ''
		applycmd += ' ' + self.inputFileName
		applycmd += ' ' + self.outputFileName
		
		self.runcmd = applycmd

class FSLMergeOperator( CommandLineOperator ):
	"""docstring for FlirtOperator"""
	def __init__(self, inputObject, cmd = 'fslmerge'): # source ~/.bash_profile_fsl ;
		"""
		other reasonable options for referenceFileName are this subject's freesurfer anatomical or the inplane_anat that is run in the same session
		"""
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(FSLMergeOperator, self).__init__(inputObject = inputObject, cmd = cmd)
		if 'sara' or 'aeneas' in os.uname()[1]:
			pass
		else:
			self.cmd = 'export PATH="/usr/local/fsl/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin"; fslsplit'

	def configure(self, outputFileName = None, TR=2):
		"""
		apply runs flirt's applyxfm argument.
		It takes an input matrix and a reference file in order to use transformMatrix
		to perform the transformation - it doesn't calculate the transformation itself.
		"""
		
		applycmd = self.cmd + ' -tr'
		applycmd += ' ' + outputFileName
		for s in self.inputObject:
			applycmd += ' ' + s
		applycmd += ' ' + str(TR)
		
		self.runcmd = applycmd
	
class PreludeOperator( CommandLineOperator ):
	"""docstring for FlirtOperator"""
	def __init__(self, inputObject, cmd = 'prelude', **kwargs): # source ~/.bash_profile_fsl ;
		"""
		other reasonable options for referenceFileName are this subject's freesurfer anatomical or the inplane_anat that is run in the same session
		"""
		# options for costFunction {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}
		super(PreludeOperator, self).__init__(inputObject = inputObject, cmd = cmd, **kwargs)

	def configure(self, outputFileName = None, phasevol = None, mask = None):
		"""
		standard configure is configureRun instead of apply
		"""
		
		self.outputFileName = outputFileName
		runcmd = self.cmd
		runcmd += ' -p ' + phasevol
		runcmd += ' -a ' + self.inputFileName
		runcmd += ' -o ' + self.outputFileName
		runcmd += ' -m ' + mask
		self.runcmd = runcmd

class ReorientOperator( CommandLineOperator ):
	def __init__(self, inputObject, cmd = 'fslreorient2std', **kwargs): # source ~/.bash_profile_fsl ;
		"""
		"""
		super(ReorientOperator, self).__init__(inputObject = inputObject, cmd = cmd, **kwargs)
	
	def configure(self, outputFileName = None):
		self.outputFileName = outputFileName
		self.runcmd = self.cmd + ' ' + self.inputFileName + ' ' + self.outputFileName