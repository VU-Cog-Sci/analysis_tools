#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

import * from Session

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
	
