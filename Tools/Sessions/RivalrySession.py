#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import *

class RivalrySession(Session):
	def analyzeBehavior(self):
		"""docstring for analyzeBehaviorPerRun"""
		for r in self.scanTypeDict['epi_bold']:
			# do principal analysis, keys vary across dates
			if self.date == datetime.date(2010, 11, 22):
				self.runList[r].behavior([2,3])
			else:
				self.runList[r].behavior([1,2])
			# put in the right place
			ExecCommandLine( 'cp ' + self.runList[r].bO.inputFileName + ' ' + self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.pickle' ) )
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
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents }
				
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
			
		
			fig = pl.figure()
			s = fig.add_subplot(1,1,1)
			# first series of EPI runs for rivalry learning
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.scatter(np.arange(6)+0.5, [self.rivalryBehavior[i][0] for i in range(6)], c = 'b', alpha = 0.85)
			pl.scatter(np.arange(6)+0.5, [self.rivalryBehavior[i][2] for i in range(6)], c = 'b', alpha = 0.75, marker = 's')

			# all percept events, plotted on top of this
			pl.plot(np.concatenate([(self.rivalryBehavior[rb][5][:,0]/150.0) + rb for rb in range(6)]), np.concatenate([self.rivalryBehavior[rb][5][:,1] for rb in range(6)]), c = 'b', alpha = 0.25)
			# second series of EPI runs
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.scatter(np.arange(6,12)+0.5, [self.rivalryBehavior[i][0] for i in range(6,12)], c = 'g', alpha = 0.85)
			pl.scatter(np.arange(6,12)+0.5, [self.rivalryBehavior[i][2] for i in range(6,12)], c = 'g', alpha = 0.75, marker = 's')
			# all percept events, plotted on top of this
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.plot(np.concatenate([(self.rivalryBehavior[rb][3][:,0]/150.0) + rb for rb in range(6,12)]), np.concatenate([self.rivalryBehavior[rb][3][:,1] for rb in range(6,12)]), c = 'g', alpha = 0.25)
			s.axis([-1,13,0,12])
		
			pl.savefig(self.runFile(stage = 'processed/behavior', extension = '.pdf', base = 'duration_summary' ))
		
	
	def gatherBehavioralData(self, whichRuns, whichEvents = ['perceptEventsAsArray','transitionEventsAsArray']):
		data = dict([(we, []) for we in whichEvents])
		timeOffset = 0.0
		for r in whichRuns:
			# behavior for this run, assume behavior analysis has already been run so we can load the results.
			behFile = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle'), 'r')
			behData = pickle.load(behFile)
			behFile.close()
			
			for we in whichEvents:
				# implement time offset. 
				behData[we][:,0] = behData[we][:,0] + timeOffset
				data[we].append(behData[we])
				
			
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf']))
			TR = niiFile.rtime
			nrTRs = niiFile.timepoints
			
			timeOffset += TR * nrTRs
			
		for we in whichEvents:
			data[we] = np.vstack(data[we])
			
		self.logger.debug('gathered behavioral data from runs %s', str(whichRuns))
		
		return data
		
	
	def deconvolveEvents(self, roi, eventType = 'transitionEventsAsArray'):
		"""deconvolution analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting deconvolution for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = self.conditionDict['rivalry'] )
		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] )
		# split out two types of events
		[ones, twos] = [np.abs(eventData[eventType][:,2]) == 1, np.abs(eventData[eventType][:,2]) == 2]
		eventArray = [eventData[eventType][ones,0], eventData[eventType][ones,0] + eventData[eventType][ones,1], eventData[eventType][twos,0], eventData[eventType][twos,0] + eventData[eventType][twos,1]]
		
		self.logger.debug('deconvolution analysis with input data shaped: %s, and %s events of type %s', str(roiData.shape), str(eventData[eventType].shape[0]), eventType)
		# mean data over voxels for this analysis
		decOp = DeconvolutionOperator(inputObject = roiData.mean(axis = 1), eventObject = eventArray)
		pl.plot(decOp.deconvolvedTimeCoursesPerEventType.T)
		
	def deconvolveEventsFromRois(self, roiArray = ['V1','V2','MT'], eventType = 'transitionEventsAsArray'):
		
		fig = pl.figure()
		for r in range(len(roiArray)):
			s = fig.add_subplot(len(roiArray),1,r)
			self.deconvolveEvents(roiArray[r], eventType = eventType)
			s.set_xlabel(roiArray[r], fontsize=10)
		
		pl.show()
		
		
		
		
