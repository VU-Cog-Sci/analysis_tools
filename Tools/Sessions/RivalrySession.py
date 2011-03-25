#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import *

class RivalryReplaySession(Session):
	def analyzeBehavior(self):
		"""docstring for analyzeBehaviorPerRun"""
		for r in self.scanTypeDict['epi_bold']:
			# do principal analysis, keys vary across dates but taken care of within behavior function
			self.runList[r].behavior()
			# put in the right place
			try:
				ExecCommandLine( 'cp ' + self.runList[r].bO.inputFileName + ' ' + self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.dat' ) )
			except ValueError:
				pass
			self.runList[r].behaviorFile = self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.dat' )
		
		if 'rivalry' in self.conditionDict:
			self.rivalryBehavior = []
			for r in self.conditionDict['rivalry']:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents, 'yokedEventsAsArray': np.array(self.runList[r].bO.yokedPeriods), 'halfwayTransitionsAsArray': np.array(self.runList[r].bO.halfwayTransitionsAsArray) }
			
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
			for r in self.conditionDict['replay']:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents, 'yokedEventsAsArray': np.array(self.runList[r].bO.yokedPeriods), 'halfwayTransitionsAsArray': np.array(self.runList[r].bO.halfwayTransitionsAsArray) }
			
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
			for r in self.conditionDict['replay2']:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents, 'yokedEventsAsArray': np.array(self.runList[r].bO.yokedPeriods), 'halfwayTransitionsAsArray': np.array(self.runList[r].bO.halfwayTransitionsAsArray) }
				
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
	
	def gatherBehavioralData(self, whichRuns, whichEvents = ['perceptEventsAsArray','transitionEventsAsArray'], sampleInterval = [0,0]):
		data = dict([(we, []) for we in whichEvents])
		timeOffset = 0.0
		for r in whichRuns:
			# behavior for this run, assume behavior analysis has already been run so we can load the results.
			behFile = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle'), 'r')
			behData = pickle.load(behFile)
			behFile.close()
			
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf']))
			TR = niiFile.rtime
			nrTRs = niiFile.timepoints
			
			for we in whichEvents:
				# take data from the time in which we can reliably sample the ERAs
				timeIndices = (behData[we][:,0] > -sampleInterval[0]) * (behData[we][:,0] < -sampleInterval[1] + (TR * nrTRs))
				behData[we] = behData[we][ timeIndices ]
				# implement time offset. 
				behData[we][:,0] = behData[we][:,0] + timeOffset
				data[we].append(behData[we])
				
				
			timeOffset += TR * nrTRs
			
		for we in whichEvents:
			data[we] = np.vstack(data[we])
			
		self.logger.debug('gathered behavioral data from runs %s', str(whichRuns))
		
		return data
	
	def deconvolveEvents(self, roi):
		"""deconvolution analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting deconvolution for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichMask = 'rivalry_Z' )
		
#		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichEvents = ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray','halfwayTransitionsAsArray'] )
#		eventArray = [eventData['perceptEventsAsArray'][:,0], eventData['transitionEventsAsArray'][:,0], eventData['yokedEventsAsArray'][:,0], eventData['halfwayTransitionsAsArray'][:,0]]
		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichEvents = ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray'] )
		eventArray = [eventData['perceptEventsAsArray'][:,0], eventData['transitionEventsAsArray'][:,0], eventData['yokedEventsAsArray'][:,0]]

		
		self.logger.debug('deconvolution analysis with input data shaped: %s', str(roiData.shape))
		# mean data over voxels for this analysis
		decOp = DeconvolutionOperator(inputObject = roiData.mean(axis = 1), eventObject = eventArray)
#		pl.plot(decOp.deconvolvedTimeCoursesPerEventType[0], c = 'r', alpha = 0.75)
#		pl.plot(decOp.deconvolvedTimeCoursesPerEventType[1], c = 'g', alpha = 0.75)
#		pl.plot(decOp.deconvolvedTimeCoursesPerEventType[2], c = 'b', alpha = 0.75)
		pl.plot(decOp.deconvolvedTimeCoursesPerEventType.T)
		
		return decOp.deconvolvedTimeCoursesPerEventType
		
		
	
	def deconvolveEventsFromRois(self, roiArray = ['V1','V2','MT','lingual','superiorparietal','inferiorparietal','insula'], eventType = 'perceptEventsAsArray'):
		res = []
		fig = pl.figure(figsize = (3.5,10))
		pl.subplots_adjust(hspace=0.4)
		for r in range(len(roiArray)):
			s = fig.add_subplot(len(roiArray),1,r+1)
			if r == 0:
				s.set_title(self.subject.initials + ' deconvolution', fontsize=12)
			res.append(self.deconvolveEvents(roiArray[r]))
			s.set_xlabel(roiArray[r], fontsize=9)
		return res
	
	def eventRelatedAverageEvents(self, roi, eventType = 'perceptEventsAsArray', whichRuns = None, whichMask = '_transStateGTdomState', color = 'k'):
		"""eventRelatedAverage analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting eventRelatedAverage for roi %s', roi)
		
		res = []
		
		roiData = self.gatherRIOData(roi, whichRuns = whichRuns, whichMask = whichMask )
		eventData = self.gatherBehavioralData( whichRuns = whichRuns, whichEvents = ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray'], sampleInterval = [-6,21] )
		
		# split out two types of events
#		[ones, twos] = [np.abs(eventData[eventType][:,2]) == 1, np.abs(eventData[eventType][:,2]) == 2]
#		all types of transition/percept events split up, both types and beginning/end separately
#		eventArray = [eventData[eventType][ones,0], eventData[eventType][ones,0] + eventData[eventType][ones,1], eventData[eventType][twos,0], eventData[eventType][twos,0] + eventData[eventType][twos,1]]
		
#		combine across percepts types, but separate onsets/offsets
#		eventArray = [eventData[eventType][:,0], eventData[eventType][:,0] + eventData[eventType][:,1]]
		
#		separate out different percepts - looking at onsets
# 		eventArray = [eventData[eventType][ones,0], eventData[eventType][twos,0]]
		
		# just all the onsets
		# take also the half-way transitions as events
		
		
		eventArray = [eventData[eventType][:,0]]
		
		self.logger.debug('eventRelatedAverage analysis with input data shaped: %s, and %s events of type %s', str(roiData.shape), str(eventData[eventType].shape[0]), eventType)
		
		smoothingWidth = 7
		f = np.array([pow(math.e, -(pow(x,2.0)/(smoothingWidth))) for x in np.linspace(-smoothingWidth,smoothingWidth,smoothingWidth*2.0)])
		
		# mean data over voxels for this analysis
		roiData = roiData.mean(axis = 1)
		for e in range(len(eventArray)):
			eraOp = EventRelatedAverageOperator(inputObject = np.array([roiData]), eventObject = eventArray[e], interval = [-2.0,14.0])
			d = eraOp.run(binWidth = 4.0, stepSize = 0.25)
			pl.plot(d[:,0], d[:,1], c = color, alpha = 0.25)
			pl.plot(d[:,0], np.convolve(f/f.sum(), d[:,1], 'same'), c = color, alpha = 0.6)
			res.append(d)
		return res
			
	
	def eventRelatedAverageEventsFromRois(self, roiArray = ['V1','V2','MT','lingual','superiorparietal','inferiorparietal','insula'], whichMask = '_transStateGTdomState'):
		evRes = []
		fig = pl.figure(figsize = (3.5,10))
		
		pl.subplots_adjust(hspace=0.4)
		for r in range(len(roiArray)):
			evRes.append([])
			s = fig.add_subplot(len(roiArray),1,r+1)
			if r == 0:
				s.set_title(self.subject.initials + ' averaged', fontsize=12)
			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'perceptEventsAsArray', whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichMask = whichMask, color = 'r'))
			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'transitionEventsAsArray', whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichMask = whichMask, color = 'g'))
			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'yokedEventsAsArray', whichRuns = self.conditionDict['replay'], whichMask = whichMask, color = 'b'))
			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'yokedEventsAsArray', whichRuns = self.conditionDict['replay2'], whichMask = whichMask, color = 'k'))
#			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'halfwayTransitionsAsArray', whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichMask = whichMask, color = 'k'))
			s.set_xlabel(roiArray[r], fontsize=9)
#			s.axis([-5,17,-2.1,3.8])
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'event-related.pdf'))
		return evRes
	
	def runTransitionGLM(self, perRun = False, acrossRuns = True):
		"""
		Take all transition events and use them as event regressors
		Run FSL on this
		"""
		if perRun:
			for condition in ['rivalry', 'replay', 'replay2']:
				for run in self.conditionDict[condition]:
					# create the event files
					for eventType in ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray']:
						eventData = self.gatherBehavioralData( whichEvents = [eventType], whichRuns = [run] )
						eventName = eventType.split('EventsAsArray')[0]
						dfFSL = np.ones((eventData[eventType].shape[0],3)) * [1.0,0.1,1]
						dfFSL[:,0] = eventData[eventType][:,0]
						np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = eventName, extension = '.evt'), dfFSL, fmt='%4.2f')
						# also make files for the end of each event.
						dfFSL[:,0] = eventData[eventType][:,0] + eventData[eventType][:,1]
						np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = eventName, postFix = ['end'], extension = '.evt'), dfFSL, fmt='%4.2f')
					# remove previous feat directories
					try:
						self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.fsf'))
					except OSError:
						pass
					# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
					thisFeatFile = '/Users/tk/Documents/research/analysis_tools/Tools/other_scripts/transition.fsf'
					REDict = {
					'---NR_FRAMES---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'])).timepoints),
					'---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf']), 
					'---ANAT_FILE---':os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'bet', 'T1_bet' ), 
					'---TRANS_ON_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'transition', extension = '.evt'),
					'---TRANS_OFF_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'transition', postFix = ['end'], extension = '.evt')
					}
					featFileName = self.runFile(stage = 'processed/mri', run = self.runList[run], extension = '.fsf')
					featOp = FEATOperator(inputObject = thisFeatFile)
					if run == self.conditionDict['replay2'][-1]:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
					else:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
					# run feat
					featOp.execute()
		# group GLM
		if acrossRuns:
			nrFeats = len(self.scanTypeDict['epi_bold'])
			inputFeats = [self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat') for run in self.scanTypeDict['epi_bold']]
			inputRules = '';	evRules = '';	groupRules = '';
			for i in range(nrFeats):
				inputRules += 'set feat_files(' + str(i+1) + ') "' + inputFeats[i] + '"\n' 
				evRules += 'set fmri(evg' + str(i+1) + '.1) 1\n'
				groupRules += 'set fmri(groupmem.' + str(i+1) + ') 1\n'
				
			try:
				os.mkdir(self.stageFolder(stage = 'processed/mri/feat'))
			except OSError:
				pass
			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			thisFeatFile = '/Users/tk/Documents/research/analysis_tools/Tools/other_scripts/acrossRuns.fsf'
			REDict = {
			'---NR_RUNS---':str(i+1),
			'---INPUT_RULES---':inputRules, 
			'---OUTPUT_FOLDER---': os.path.join(self.stageFolder(stage = 'processed/mri/feat'), 'acrossRuns'), 
			'---EV_RULES---':evRules,
			'---GROUP_RULES---':groupRules
			}
			featFileName = os.path.join( self.stageFolder(stage = 'processed/mri/feat'), 'acrossRuns.fsf')
			featOp = FEATOperator(inputObject = thisFeatFile)
			featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
			featOp.execute()
			
	
#			def coherenceAnalysis(self, rois = [['pericalcarine', 'lateraloccipital','lingual'],['inferiorparietal', 'superiorparietal','cuneus','precuneus','supramarginal'],['insula','superiortemporal', 'parsorbitalis','parstriangularis','parsopercularis','rostralmiddlefrontal'],['caudalmiddlefrontal','precentral', 'superiorfrontal']], labels = ['occipital','parietal','inferiorfrontal','fef']):
	def coherenceAnalysis(self, roiArray = [['pericalcarine', 'lateraloccipital','lingual'],['inferiorparietal', 'superiorparietal','cuneus','precuneus'],['supramarginal'],['superiortemporal', 'parsorbitalis','parstriangularis','parsopercularis','caudalmiddlefrontal','precentral'], ['superiorfrontal'], ['rostralmiddlefrontal']], labels = ['occ','par','tpj','inffr','fef','dlpfc'], acrossAreas = False, acrossConditions = True):
		
		#Import the time-series objects: 
		from nitime.timeseries import TimeSeries 
		#Import the analysis objects:
		from nitime.analysis import CoherenceAnalyzer
		#Import utility functions:
		import nitime.viz as viz
		from nitime.viz import drawmatrix_channels,drawgraph_channels
		
		# parameters
		TR=2.0
		f_lb = 0.1
		f_ub = 0.5
		
		self.labels = labels
		roi_names = np.array(self.labels)
		
		self.coherences = []
		self.delays = []
		self.Cs = []
		
		if acrossAreas:
		
			plotNr = 1
			for (ts, runs) in zip([[4,64],[69,129],[69,129]],[self.scanTypeDict['epi_bold'],self.conditionDict['replay'],self.conditionDict['replay2']]):
				roiData = []
				for roi in roiArray:
					thisRoiData = self.gatherRIOData(roi, runs, whichMask = '_rivalry_Z', timeSlices = ts)
					roiData.append(thisRoiData.mean(axis = 1))
				roiData = np.array(roiData)
			#	self.conditionCoherenceDict.append( np.array(roiData) )
				n_samples = roiData.shape[1]
			
				T = TimeSeries(roiData,sampling_interval=TR)
				T.metadata['roi'] = roi_names
			
				C = CoherenceAnalyzer(T, unwrap_phases = True)
				freq_idx = np.where((C.frequencies>f_lb) * (C.frequencies<f_ub))[0]
			
				self.coherences.append(np.mean(C.coherence[:,:,freq_idx],-1))
				self.delays.append(np.mean(C.delay[:,:,freq_idx],-1))
				self.Cs.append(C)
	#			drawmatrix_channels(self.delays[-1],roi_names, title = self.subject.initials + '\n Delay ' + ['rivalry','instantaneous replay','duration-matched replay'][plotNr - 1])
	#			drawmatrix_channels(self.coherences[-1],roi_names, title = self.subject.initials + '\n Coherence ' + ['rivalry','instantaneous replay','duration-matched replay'][plotNr - 1])
				plotNr += 1
		
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[0],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay rivalry')
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[1],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay instant replay')
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[2],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay duration-matched replay')
		
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[0] - self.delays[1],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay difference between rivalry and instant replay')
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[0] - self.delays[2],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay difference between rivalry and duration-matched replay')
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[2] - self.delays[1],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay difference between duration-matched and instant replay')
		
		if acrossConditions:
			self.plotData = []
			fig1 = pl.figure(figsize = (8,3))
			for (counter, runs) in zip([0,1],[self.conditionDict['replay'],self.conditionDict['replay2']]):
				roiData = []
				for roi in roiArray:
					thisRivData = self.gatherRIOData(roi, runs, whichMask = '_rivalry_Z', timeSlices = [4,64]).mean(axis = 1)
					thisRepData = self.gatherRIOData(roi, runs, whichMask = '_rivalry_Z', timeSlices = [69,129]).mean(axis = 1)
				
					roiData.append(thisRivData)
					roiData.append(thisRepData)
				roiData = np.array(roiData)
				n_samples = roiData.shape[1]
			
				T = TimeSeries(roiData,sampling_interval=TR)
				T.metadata['roi'] = np.array([[roi_names[i] + ' Riv', roi_names[i] + ' Rep'] for i in range(len(roi_names))]).ravel()
			
				C = CoherenceAnalyzer(T, unwrap_phases = True)
				freq_idx = np.where((C.frequencies>f_lb) * (C.frequencies<f_ub))[0]
				np.mean(C.coherence[:,:,freq_idx],-1)
				np.mean(C.delay[:,:,freq_idx],-1)
			
		#		f = pl.figure(figsize = (5,7))
		#		drawmatrix_channels(np.mean(C.delay[:,:,freq_idx],-1), T.metadata['roi'], size = (5,7), fig = f, title = self.subject.initials + '\n Rivalry and Replay ' + ['intant', 'dur-match'][counter])
				
				rivrepindices = np.array([np.array([0,1]) + (2 * i) for i in range(6)])
				delay = np.mean(C.delay[:,:,freq_idx],-1)
				plotData = [delay[rivrepindices[i,0],rivrepindices[i,1]] for i in range(len(rivrepindices))]
				
				self.plotData.append(plotData)
	


class RivalryLearningSession(Session):
	def analyzeBehavior(self):
		"""docstring for analyzeBehaviorPerRun"""
		for r in self.scanTypeDict['epi_bold']:
			# do principal analysis, keys vary across dates but taken care of within behavior function
			self.runList[r].behavior()
			# put in the right place
			try:
				ExecCommandLine( 'cp ' + self.runList[r].bO.inputFileName + ' ' + self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.pickle' ) )
			except ValueError:
				pass
			self.runList[r].behaviorFile = self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.pickle' )
			
		if 'disparity' in self.conditionDict:
			self.disparityPsychophysics = []
			for r in self.conditionDict['disparity']:
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
		
		
		if 'rivalry' in self.conditionDict:
			self.rivalryBehavior = []
			for r in self.conditionDict['rivalry']:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents, 'yokedEventsAsArray': np.array(self.runList[r].bO.yokedPeriods) }
				
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
			
			firstHalfLength = floor(len(self.conditionDict['rivalry']) / 2.0)
		
			fig = pl.figure()
			s = fig.add_subplot(1,1,1)
			# first series of EPI runs for rivalry learning
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.scatter(np.arange(firstHalfLength)+0.5, [self.rivalryBehavior[i][0] for i in range(firstHalfLength)], c = 'b', alpha = 0.85)
			pl.scatter(np.arange(firstHalfLength)+0.5, [self.rivalryBehavior[i][2] for i in range(firstHalfLength)], c = 'b', alpha = 0.75, marker = 's')
			
			# all percept events, plotted on top of this
			pl.plot(np.concatenate([(self.rivalryBehavior[rb][5][:,0]/150.0) + rb for rb in range(firstHalfLength)]), np.concatenate([self.rivalryBehavior[rb][5][:,1] for rb in range(firstHalfLength)]), c = 'b', alpha = 0.25)
			# second series of EPI runs
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.scatter(np.arange(firstHalfLength,len(self.conditionDict['rivalry']))+0.5, [self.rivalryBehavior[i][0] for i in range(firstHalfLength,len(self.conditionDict['rivalry']))], c = 'g', alpha = 0.85)
			pl.scatter(np.arange(firstHalfLength,len(self.conditionDict['rivalry']))+0.5, [self.rivalryBehavior[i][2] for i in range(firstHalfLength,len(self.conditionDict['rivalry']))], c = 'g', alpha = 0.75, marker = 's')
			# all percept events, plotted on top of this
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.plot(np.concatenate([(self.rivalryBehavior[rb][3][:,0]/150.0) + rb for rb in range(firstHalfLength,len(self.conditionDict['rivalry']))]), np.concatenate([self.rivalryBehavior[rb][3][:,1] for rb in range(firstHalfLength,len(self.conditionDict['rivalry']))]), c = 'g', alpha = 0.25)
			s.axis([-1,13,0,12])
		
			pl.savefig(self.runFile(stage = 'processed/behavior', extension = '.pdf', base = 'duration_summary' ))
		
	
	def gatherBehavioralData(self, whichRuns, whichEvents = ['perceptEventsAsArray','transitionEventsAsArray'], sampleInterval = [0,0]):
		data = dict([(we, []) for we in whichEvents])
		timeOffset = 0.0
		for r in whichRuns:
			# behavior for this run, assume behavior analysis has already been run so we can load the results.
			behFile = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle'), 'r')
			behData = pickle.load(behFile)
			behFile.close()
			
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf']))
			TR = niiFile.rtime
			nrTRs = niiFile.timepoints
			
			for we in whichEvents:
				# take data from the time in which we can reliably sample the ERAs
				behData[we] = behData[we][ (behData[we][:,0] > -sampleInterval[0]) * (behData[we][:,0] < -sampleInterval[1] + (TR * nrTRs)) ]
				# implement time offset. 
				behData[we][:,0] = behData[we][:,0] + timeOffset
				data[we].append(behData[we])
				
			
			timeOffset += TR * nrTRs
			
		for we in whichEvents:
			data[we] = np.vstack(data[we])
			
		self.logger.debug('gathered behavioral data from runs %s', str(whichRuns))
		
		return data
		
	
	def deconvolveEvents(self, roi, eventType = 'perceptEventsAsArray'):
		"""deconvolution analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting deconvolution for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = self.conditionDict['rivalry'] )
		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] )
		# split out two types of events
		[ones, twos] = [np.abs(eventData[eventType][:,2]) == 1, np.abs(eventData[eventType][:,2]) == 2]
#		all types of transition/percept events split up, both types and beginning/end separately
#		eventArray = [eventData[eventType][ones,0], eventData[eventType][ones,0] + eventData[eventType][ones,1], eventData[eventType][twos,0], eventData[eventType][twos,0] + eventData[eventType][twos,1]]
		
#		combine across percepts types, but separate onsets/offsets
		eventArray = [eventData[eventType][:,0], eventData[eventType][:,0] + eventData[eventType][:,1]]
		
#		separate out different percepts - looking at onsets
#		eventArray = [eventData[eventType][ones,0], eventData[eventType][twos,0]]
		
		self.logger.debug('deconvolution analysis with input data shaped: %s, and %s events of type %s', str(roiData.shape), str(eventData[eventType].shape[0]), eventType)
		# mean data over voxels for this analysis
		colors = ['k','r','g','b']
		decOp = DeconvolutionOperator(inputObject = roiData.mean(axis = 1), eventObject = eventArray)
		pl.plot(decOp.deconvolvedTimeCoursesPerEventType.T)
	
	def deconvolveEventsFromRois(self, roiArray = ['V1','V2','MT','lingual','superiorparietal','inferiorparietal','insula'], eventType = 'perceptEventsAsArray'):
		
		fig = pl.figure(figsize = (3.5,10))
		
		for r in range(len(roiArray)):
			s = fig.add_subplot(len(roiArray),1,r+1)
			self.deconvolveEvents(roiArray[r], eventType = eventType)
			s.set_xlabel(roiArray[r], fontsize=10)
	
	def eventRelatedAverageEvents(self, roi, eventType = 'perceptEventsAsArray', whichRuns = None, color = 'k'):
		"""eventRelatedAverage analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting eventRelatedAverage for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = whichRuns )
		eventData = self.gatherBehavioralData( whichRuns = whichRuns, sampleInterval = [-5,30] )
		
		# split out two types of events
		[ones, twos] = [np.abs(eventData[eventType][:,2]) == 1, np.abs(eventData[eventType][:,2]) == 2]
#		all types of transition/percept events split up, both types and beginning/end separately
#		eventArray = [eventData[eventType][ones,0], eventData[eventType][ones,0] + eventData[eventType][ones,1], eventData[eventType][twos,0], eventData[eventType][twos,0] + eventData[eventType][twos,1]]
		
#		combine across percepts types, but separate onsets/offsets
#		eventArray = [eventData[eventType][:,0], eventData[eventType][:,0] + eventData[eventType][:,1]]
		
#		separate out different percepts - looking at onsets
# 		eventArray = [eventData[eventType][ones,0], eventData[eventType][twos,0]]
		
		# just all the onsets
		eventArray = [eventData[eventType][:,0]]
		
		self.logger.debug('eventRelatedAverage analysis with input data shaped: %s, and %s events of type %s', str(roiData.shape), str(eventData[eventType].shape[0]), eventType)
		# mean data over voxels for this analysis
		roiData = roiData.mean(axis = 1)
		for e in range(len(eventArray)):
			eraOp = EventRelatedAverageOperator(inputObject = np.array([roiData]), eventObject = eventArray[e], interval = [-3.0,15.0])
			d = eraOp.run(binWidth = 3.0, stepSize = 0.25)
			pl.plot(d[:,0], d[:,1], c = color, alpha = 0.75)
	
	def eventRelatedAverageEventsFromRois(self, roiArray = ['V1','V2','MT','lingual','superiorparietal','inferiorparietal','insula'], eventType = 'transitionEventsAsArray', learningPartitions = None):
		
		fig = pl.figure(figsize = (3.5,10))
		
		for r in range(len(roiArray)):
			s = fig.add_subplot(len(roiArray),1,r+1)
			self.eventRelatedAverageEvents(roiArray[r], eventType = eventType, whichRuns = self.conditionDict['rivalry'])
			s.set_xlabel(roiArray[r], fontsize=10)
			s.axis([-5,17,-0.1,0.1])
			
		# now, for learning...
		# in learning, there were twelve rivalry runs - there were two sequences of six across which there was 'learning', or at least they were of the same type.
		# we'll make 6 separate ERAs for 
		if learningPartitions:
			colors = [(i/float(len(learningPartitions)), 1.0 - i/float(len(learningPartitions)), 0.0) for i in range(len(learningPartitions))]
			
			fig = pl.figure(figsize = (3.5,10))
			pl.subplots_adjust(hspace=0.4)
			for r in range(len(roiArray)):
				s = fig.add_subplot(len(roiArray),1,r+1)
				for (ind, lp) in zip(range(len(learningPartitions)), learningPartitions):
					self.eventRelatedAverageEvents(roiArray[r], eventType = eventType, whichRuns = [self.conditionDict['rivalry'][i] for i in lp], color = colors[ind])
				s.set_xlabel(roiArray[r], fontsize=9)
				s.axis([-5,17,-0.1,0.1])
	
	def prepareTransitionGLM(self, functionals = False):
		"""
		Take all transition events and use them as event regressors
		Make one big nii file that contains all the motion corrected and zscored rivalry data
		Run FSL on this
		"""
		
		if functionals:
			# make nii file
			niiFiles = [NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf','hp','Z'])) for r in self.conditionDict['rivalry'] ]
			allNiiFile = NiftiImage(np.concatenate([nf.data for nf in niiFiles]), header = niiFiles[0].header)
			allNiiFile.save( os.path.join( self.conditionFolder( stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]), 'all_rivalry.nii.gz') )
		
		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] )
		
		perceptDataForFSL = np.ones((eventData['perceptEventsAsArray'].shape[0],3)) * [1.0,0.1,1]
		perceptDataForFSL[:,0] = eventData['perceptEventsAsArray'][:,0]
		np.savetxt(os.path.join( self.conditionFolder( stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]), 'all_rivalry_percept' + '.evt'), perceptDataForFSL, fmt='%4.2f')
		
		transDataForFSL = np.ones((eventData['transitionEventsAsArray'].shape[0],3)) * [1.0,0.1,1]
		transDataForFSL[:,0] = eventData['transitionEventsAsArray'][:,0]
		np.savetxt(os.path.join( self.conditionFolder( stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]), 'all_rivalry_trans' + '.evt'), transDataForFSL, fmt='%4.2f' )
		
		
	


