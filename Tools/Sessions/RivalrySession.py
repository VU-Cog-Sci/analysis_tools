#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import *

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
		
