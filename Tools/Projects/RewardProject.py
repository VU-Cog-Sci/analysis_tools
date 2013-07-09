#!/usr/bin/env python
# encoding: utf-8
"""
project.py

Created by Tomas HJ Knapen on 2009-12-08.
Copyright (c) 2009 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from nifti import *
from math import *
from itertools import *

from ..Sessions.RewardSessions import *
from Project import *

class RewardProject(Project):
	"""a RewardProject has a subject, and a standard T2 anatomical and EPI run.
	This means that all subsequent sessions will have a T2 anatomical that can be co-registered with the standard project T2 anatomical,
	and that registration can be applied to all EPI's in that session while resampling to the standard EPI run. 
	"""
	
	def __init__(self, projectName, subject, base_dir, input_T2_file, input_EPI_file, **kwargs):
		super(RewardProject, self).__init__(projectName, subject, base_dir, **kwargs)
		self.input_T2_file = input_T2_file
		self.input_EPI_file = input_EPI_file
		self.standard_T2_file = os.path.join(self.base_dir, self.subject.initials, 't2_BET.nii.gz')
		self.standard_EPI_file = os.path.join(self.base_dir, self.subject.initials, 'standard_EPI.nii.gz')
		
	
	def register(self, bet_f_value = 0.5, bet_g_value = 0.0, mni_feat = True, through_MC = True, sinc = False, label_folder = 'visual_areas'):
		self.FSsubject = self.subject.standardFSID
		feat_dir = os.path.join(self.base_dir, self.subject.initials, 'feat' )
		mask_dir = os.path.join(self.base_dir, self.subject.initials, 'masks' )
		
		if not os.path.isfile(self.standard_T2_file):
			better = BETOperator( inputObject = self.input_T2_file )
			better.configure( outputFileName = self.standard_T2_file, f_value = bet_f_value, g_value = bet_g_value )
			better.execute()
					
		# motion correct to self
		mcf = MCFlirtOperator( self.input_EPI_file, target = self.standard_T2_file )
	 	mcf.configure(sinc = sinc)
		if not os.path.isfile(os.path.splitext(os.path.splitext(self.input_EPI_file)[0])[0] + '_mcf_meanvol.nii.gz'):
			mcf.execute()
		# add registration of non-motion corrected functionals to the forRegistration file
		# to be run together with the motion correction runs
		if through_MC == False:
			fO = FlirtOperator(inputObject = os.path.splitext(os.path.splitext(self.input_EPI_file)[0])[0] + '_mcf_meanvol.nii.gz', referenceFileName = self.standard_T2_file)
			fO.configureRun()
			fO.execute()
		
			f1 = FlirtOperator(inputObject = fO.outputFileName, referenceFileName = self.input_EPI_file)
		else:
			f1 = FlirtOperator(inputObject = os.path.splitext(os.path.splitext(self.input_EPI_file)[0])[0] + '_mcf_meanvol.nii.gz', referenceFileName = self.input_EPI_file)
		f1.configureApply(transformMatrixFileName = os.path.join(self.base_dir, 'eye.mtx'), sinc = sinc )
		f1.execute()
		ExecCommandLine('cp ' + f1.outputFileName + ' ' + self.standard_EPI_file)
					
		# bbregister
		bbR = BBRegisterOperator( self.standard_T2_file, FSsubject = self.FSsubject, contrast = 'T2' )
		bbR.configure( transformMatrixFileName = os.path.join(self.base_dir, self.subject.initials, 'register.dat' ), flirtOutputFile = True )
		if not os.path.isfile(os.path.join(self.base_dir, self.subject.initials, 'register.dat' )):
			bbR.execute()
		
		if mni_feat:
			# to MNI 
			os.system('mri_convert ' + os.path.join(os.environ['SUBJECTS_DIR'], self.FSsubject, 'mri', 'brain.mgz') + ' ' + os.path.join(os.environ['SUBJECTS_DIR'], self.FSsubject, 'mri', 'brain.nii.gz'))
			flRT1 = FlirtOperator( os.path.join(os.environ['SUBJECTS_DIR'], self.FSsubject, 'mri', 'brain.nii.gz')  )
			flRT1.configureRun( transformMatrixFileName = os.path.join(os.environ['SUBJECTS_DIR'], self.FSsubject, 'mri', 'brain_MNI.mat'), outputFileName = os.path.join(os.environ['SUBJECTS_DIR'], self.FSsubject, 'mri', 'brain_MNI.nii.gz'), extra_args = ' -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12 ' )
			flRT1.execute()
		
			# this bbregisteroperatory does not actually execute.
			cfO = ConcatFlirtOperator(bbR.flirtOutputFileName)
			cfO.configure(secondInputFile = os.path.join(os.environ['SUBJECTS_DIR'], self.FSsubject, 'mri', 'brain_MNI.mat'), outputFileName = os.path.join(self.base_dir, self.subject.initials, 'register_MNI.mat' ))
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
				os.mkdir(feat_dir)
			except OSError:
				pass
			# copy registration files to feat directory
			subprocess.Popen('cp ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain_MNI.mat') + ' ' + os.path.join(feat_dir,'highres2standard.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + invFl_HR_Standard.outputFileName + ' ' + os.path.join(feat_dir,'standard2highres.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + bbR.flirtOutputFileName + ' ' + os.path.join(feat_dir,'example_func2highres.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + cfO.outputFileName + ' ' + os.path.join(feat_dir,'example_func2standard.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + invFl_Session_HR.outputFileName + ' ' + os.path.join(feat_dir,'highres2example_func.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + invFl_Session_Standard.outputFileName + ' ' + os.path.join(feat_dir,'standard2example_func.mat' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'brain.nii.gz') + ' ' + os.path.join(feat_dir,'highres.nii.gz' ), shell=True, stdout=PIPE).communicate()[0]
			subprocess.Popen('cp ' + flRT1.referenceFileName + ' ' + os.path.join(feat_dir,'standard.nii.gz' ), shell=True, stdout=PIPE).communicate()[0]
		
			# now take the firstFunc and create an example_func for it. 
			subprocess.Popen('cp ' + self.standard_EPI_file + ' ' + os.path.join(feat_dir,'example_func.nii.gz' ), shell=True, stdout=PIPE).communicate()[0]
	
		# having registered everything (AND ONLY AFTER MOTION CORRECTION....) we now construct masks in the functional volume
		try:
			os.mkdir(mask_dir)
		except OSError:
			pass
	
		# set up the first mean functional in the registration folder anyway for making masks. note that it's necessary to have run the moco first...
		# ExecCommandLine('cp ' + os.path.splitext(os.path.splitext(self.input_EPI_file)[0])[0] + '_mcf_meanvol.nii.gz' + ' ' + self.runFile(stage = 'processed/mri/reg', postFix = ['mcf', 'meanvol'], base = 'firstFunc' ) )
		
		maskList = ['V1','V2','V3','V3AB','V4']
		for mask in maskList:
			for hemi in ['lh','rh']:
				maskFileName = os.path.join(os.path.expandvars('${SUBJECTS_DIR}'), self.FSsubject, 'label', label_folder, hemi + '.' + mask + '.' + 'label')
				stV = LabelToVolOperator(maskFileName)
				stV.configure( 	templateFileName = self.standard_EPI_file,
								hemispheres = [hemi], 
								register = os.path.join(self.base_dir, self.subject.initials, 'register.dat' ), 
								fsSubject = self.FSsubject,
								outputFileName = os.path.join(mask_dir, hemi + '.' + mask ), 
								threshold = 0.001 )
				stV.execute()
		
	
