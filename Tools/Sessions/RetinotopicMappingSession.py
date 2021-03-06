#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import * 

class RetinotopicMappingSession(Session):
	def parcelateConditions(self):
		super(RetinotopicMappingSession, self).parcelateConditions()
		
		self.mappingTypeDict = {}
		if 'epi_bold' in self.scanTypeDict.keys():
			self.mappingTypes = np.unique(np.array([r.mappingType for r in [self.runList[e] for e in self.scanTypeDict['epi_bold']]]))
			for mt in self.mappingTypes:
				if mt != '':
					self.mappingTypeDict.update({mt: [hit.indexInSession for hit in filter(lambda x: x.mappingType == mt, [self.runList[e] for e in self.scanTypeDict['epi_bold']])]})
		
	def retinotopicMapping(self, postFix = ['mcf'], perCondition = True, perRun = False, runMapping = True, toSurf = True):
		"""
		runs retinotopic mapping on all runs in self.conditionDict['polar'] and self.conditionDict['eccen']
		"""
		self.logger.info('run retinotopic mapping')
		if not self.mappingTypeDict.has_key('polar') and not self.mappingTypeDict.has_key('eccen'):
			self.logger.warning('no retinotopic mapping runs to be run...')
			
		presentCommand = os.path.join(os.environ['ANALYSIS_HOME'], 'Tools', 'other_scripts', 'selfreqavg_noinfs.csh')
		rmOperatorList = []
		opfNameList = []
		# postFix = []
		# if useMC:
		# 	postFix.append('mcf')
		if perCondition:
			for c in self.conditionDict:
				if 'polar' in c or 'eccen' in c:
					prOperator = RetMapOperator([self.runList[pC] for pC in self.conditionDict[c]], cmd = presentCommand)
					inputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[pC], postFix = postFix) for pC in self.conditionDict[c]]
					outputFileName = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[c][0]]), self.runList[pC].mappingType)
					opfNameList.append(outputFileName)
					prOperator.configure( inputFileNames = inputFileNames, outputFileName = outputFileName )
					rmOperatorList.append(prOperator)
			
		if perRun:
			for c in self.conditionDict:
				if 'polar' in c or 'eccen' in c:
					for i in self.conditionDict[c]:
						prOperator = RetMapOperator([self.runList[i]], cmd = presentCommand)
						inputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[i], postFix = postFix)]
						outputFileName = os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[i]), self.runList[i].mappingType)
						opfNameList.append(outputFileName)
						prOperator.configure( inputFileNames = inputFileNames, outputFileName = outputFileName )
						rmOperatorList.append(prOperator)
			
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
				ppservers = ()
				job_server = pp.Server(ppservers=ppservers)
				self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
				ppResults = [job_server.submit(ExecCommandLine,(rm.runcmd,),(),('subprocess','tempfile',)) for rm in rmOperatorList]
				for rm in ppResults:
					rm()
					
				job_server.print_stats()		
		self.opfNameList = opfNameList
		if toSurf:
			self.convertVolumeToSurface()
		
	
	def convertVolumeToSurface(self, surfSmoothingFWHM = 0.0):
			# now we need to be able to view the results on the surfaces.
			vtsList = []
			for opf in self.opfNameList:
				vtsOp = VolToSurfOperator(inputObject = opf + standardMRIExtension)
				vtsOp.configure(register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = os.path.join(os.path.split(opf)[0], 'surf/'), surfSmoothingFWHM = surfSmoothingFWHM )
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
				ppResults = [job_server.submit(ExecCommandLine,(sp.runcmd,),(),('subprocess','tempfile',)) for sp in vtsList]
				for sp in ppResults:
					sp()
				job_server.print_stats()
		
	
	def convertSurfaceToVolume(self):
			# now we need to be able to view the results on the surfaces.
			vtsList = []
			for opf in self.opfNameList:
				surfaceFilesFromOpf = subprocess.Popen('ls ' + os.path.join(os.path.split(opf)[0], 'surf', '*.w'), shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
				for sf in surfaceFilesFromOpf:
					vtsOp = SurfToOperator(inputObject = opf + standardMRIExtension)
					vtsOp.configure(templateFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][0]], postFix = ['mcf'] ), register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = os.path.join(os.path.split(opf)[0], 'surf/') )
					vtsList.append(vtsOp)

			if not self.parallelize:
				self.logger.info("run serial reverse surface projection of retinotopic mapping results")
				for vts in vtsList:
					vts.execute()

			if self.parallelize:
				# tryout parallel implementation - later, this should be abstracted out of course. 
				ppservers = ()
				job_server = pp.Server(ppservers=ppservers)
				self.logger.info("run parallel reverse surface projection")
				self.logger.info("Starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
				ppResults = []
				for vts in vtsList:
					vtsex = job_server.submit(vts.execute, (), (), ("subprocess",))
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
		
		if 'polar' in self.conditionDict.keys() and len(self.conditionDict['polar']) > 0:
			# polar files
			rawInputFileNames = [self.runFile( stage = 'processed/mri', run = self.runList[pC], postFix = ['mcf']) for pC in self.scanTypeDict['epi_bold'] if self.runList[pC].condition in ('polar','eccen')]
			distilledInputFileNames = [os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[pC]), self.runList[pC].condition) for pC in self.scanTypeDict['epi_bold'] if self.runList[pC].condition in ('polar','eccen')]
			distilledInputFileNamesFull = [os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['polar'][0]]), 'polar.nii.gz'),os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['polar'][0]]), 'eccen.nii.gz')]
			
			for (pd, rd) in zip(distilledInputFileNames, rawInputFileNames):
				# setting up the data files
				rdIm = NiftiImage(rd)
				pdIm = NiftiImage(pd)
				
#				pp = PdfPages(self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][rawInputFileNames.index(rd)]], extension = '.pdf' ))
				
				for roi in rois:
					for hemi in ['lh','rh']:
						maskFileName = self.runFile(stage = 'processed/mri/masks/anat', base = hemi + '.' + roi )
						
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
					
						pl.savefig(self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][rawInputFileNames.index(rd)]], extension = '_' + hemi + '_' + roi + '.png' ))
					
#				pp.close()
		pl.draw()
	 
	def makeTiffsFromCondition(self, condition, y_rotation = 90.0, exit_when_ready = 1 ):
		thisFeatFile = os.path.join(os.environ['ANALYSIS_HOME'], 'Tools/other_scripts/redraw_retmaps.tcl' )
		for hemi in ['lh','rh']:
			REDict = {
			'---HEMI---': hemi,
			'---CONDITION---': condition, 
			'---CONDITIONFILENAME---': condition.replace('/', '_'), 
			'---FIGPATH---': os.path.join(self.stageFolder(stage = 'processed/mri/'), condition, 'surf'),
			'---NAME---': self.subject.standardFSID,
			'---BASE_Y_ROTATION---': str(y_rotation),
			'---EXIT---': str(exit_when_ready),
			}
			rmtOp = RetMapReDrawOperator(inputObject = thisFeatFile)
			redrawFileName = os.path.join(self.stageFolder(stage = 'processed/mri/scripts'), hemi + '_' + condition.replace('/', '_') + '.tcl')
			rmtOp.configure( REDict = REDict, redrawFileName = redrawFileName, waitForExecute = False )
			# run 
			rmtOp.execute()
