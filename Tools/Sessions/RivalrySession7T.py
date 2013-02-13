#!/usr/bin/env python
# encoding: utf-8
	
"""RivalrySessionJW.py"""	

import os, sys, subprocess, datetime
import tempfile, logging, pickle

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from scipy.stats import *
sys.path.append( os.environ['ANALYSIS_HOME'] )

from Tools.Sessions import *
import Tools.Project as Project
# from Tools.Subject import *
from Tools.Run import *
from Tools.plotting_tools import *
# from Session import *
from itertools import *
from Tools.circularTools import *

from nifti import *
from pypsignifit import *
from Tools.Operators import *
# from ..log import *

import mne
import sklearn
from sklearn.decomposition import PCA
from scipy.stats import *
# from scipy.stats import normal
from scipy.stats import norm
import scipy as sp
# import mdp.nodes.FastICANode as ica
from sklearn.decomposition import *

from IPython import embed as shell

class RivalrySessionJW(RivalryReplaySession):
	
	
	def createFolderHierarchy(self):
		"""docstring for fname"""
		rawFolders = ['raw/mri', 'raw/behavior', 'raw/hr']
		self.processedFolders = ['processed/mri', 'processed/behavior', 'processed/hr']
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

	
	def createRegressors(self, runArray):
		
		from matplotlib.backends.backend_pdf import PdfPages
		
		# data = np.loadtxt('/Research/7T_RIVALRY/data/TK/TK_250113/raw/behavior/tk_5_ongoing_7T.txt')
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:
				
				this_run = runArray[run]['ID']
				
				# print(self.subject.initials)
				# print(self.date)
				# print(run)
				# print(runArray[run]['behaviorFile'])
				
				stim_dur1 = 142.0
				stim_dur2 = 150.0
				break_between_trials = 16.0
				time_specificity = 0.65/2.0
				
				file_to_open = str(runArray[run]['behaviorFile'])
				data = np.loadtxt(file_to_open)
				
				# Get percept and transition indices:
				percept1 = (data[:,0] == 66)
				percept2 = (data[:,0] == 67)
				transition = (data[:,0] == 65)
		
				# Correct associated time stamps:
				for i in range(data.shape[0]):
					if percept1[i] == True:
						if data[i-1,1] != 255:
							if data[i+1,1] != 255:
								data[i,1] = (data[i-1,1] + data[i+1,1]) / 2
						if data[i-1,1] == 255:
							if data[i+1,1] != 255:
								data[i,1] = data[i+1,1]
						if data[i+1,1] == 255:
							if data[i-1,1] != 255:
								data[i,1] = data[i-1,1]
					if percept2[i] == True:
						if data[i-1,1] != 255:
							if data[i+1,1] != 255:
								data[i,1] = (data[i-1,1] + data[i+1,1]) / 2
							if data[i-1,1] == 255:
								if data[i+1,1] != 255:
									data[i,1] = data[i+1,1]
							if data[i+1,1] == 255:
								if data[i-1,1] != 255:
									data[i,1] = data[i-1,1]
					if transition[i] == True:
						if data[i-1,1] != 255:
							if data[i+1,1] != 255:
								data[i,1] = (data[i-1,1] + data[i+1,1]) / 2
							if data[i-1,1] == 255:
								if data[i+1,1] != 255:
									data[i,1] = data[i+1,1]
							if data[i+1,1] == 255:
								if data[i-1,1] != 255:
									data[i,1] = data[i-1,1]
				
				# Remove uninteresting time stamps:
				data = data[-(data[:,0]==49),:]
				
				# Remove dubble occurences:
				dubble_occurences_indices = zeros(data.shape[0], dtype=bool)
				for i in range(data.shape[0]):
					if data[i-1,0] == data[i,0]:
						dubble_occurences_indices[i] = True
				data = data[-dubble_occurences_indices,:]
				
				# Indices:
				percept1 = (data[:,0] == 66)
				percept2 = (data[:,0] == 67)
				transition = (data[:,0] == 65)
				
				# Check for transitions without duration (no button press for transition)
				transition_no_dur = np.zeros(data.shape[0], dtype = bool)
				for i in range(data.shape[0]):
					try:
						if percept1[i-1] == True:
							if percept2[i] == True:
								transition_no_dur[i] = True
						if percept2[i-1] == True:
							if percept1[i] == True:
								transition_no_dur[i] = True
					except IndexError:
						pass
				
				# Times:
				times = data[:,1] - data[0,1]
				
				# get last time stamp of run:
				# PLUS 350ms!!
				first_time1 = times[np.where((data[:,0] == 1))[0][0]]
				first_time2 = times[np.where((data[:,0] == 1))[0][1]]
				last_time1 = first_time1 + stim_dur1 + time_specificity
				last_time2 = first_time2 + stim_dur2 + time_specificity
				
				# paradigm indicies:
				paradigm1_indices = (times >= first_time1) * (times <= last_time1)
				paradigm2_indices = (times >= first_time2) * (times <= last_time2)
				if this_run % 2 != 0:
					BR_indices = paradigm1_indices
					SFM_indices = paradigm2_indices
				else:
					SFM_indices = paradigm1_indices
					BR_indices = paradigm2_indices
				
				# durations:
				durs = np.diff(times)
				durs = np.concatenate((durs, np.diff([times[-1],last_time2])))
				
				# check that durs do not extend into 'break between trials':
				durs2 = durs[paradigm2_indices]
				durs1 = durs[-paradigm2_indices]
				for i in range(durs1.shape[0]):
					if times[-paradigm2_indices][i] + durs1[i] >= last_time1:
						durs1[i] = (last_time1 - times[-paradigm2_indices][i])
				for i in range(times[paradigm2_indices].shape[0]):	
					if times[paradigm2_indices][i] + durs2[i] >= last_time2:
						durs2[i] = last_time2 - times[paradigm2_indices][i]
				durs3 = np.concatenate((durs1,durs2))
				durs = durs3
				
				############################################################
				############ MAKE TEXT FILES REGRESSORS ####################
				# BR start:
				BR_start = np.zeros((1,3))
				BR_start[:,0] = times[BR_indices][0]
				if this_run % 2 != 0:
					BR_start[:,1] = stim_dur1
				else:
					BR_start[:,1] = stim_dur2
				BR_start[:,2] = np.ones(BR_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', extension = '.txt'), BR_start, fmt='%3.2f', delimiter = '\t')
				# BR transitions:
				BR_transition = np.zeros((sum(transition[BR_indices])+sum(transition_no_dur[BR_indices]),3))
				BR_transition[:sum(transition[BR_indices]),0] = times[transition*BR_indices]
				BR_transition[sum(transition[BR_indices]):,0] = times[transition_no_dur*BR_indices]
				BR_transition[:sum(transition[BR_indices]),1] = durs[transition*BR_indices]
				BR_transition[sum(transition[BR_indices]):,1] = 1.00
				BR_transition[:,2] = np.ones(BR_transition.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', postFix = ['trans'], extension = '.txt'), BR_transition, fmt='%3.2f', delimiter = '\t')
				# SFM start:
				SFM_start = np.zeros((1,3))
				SFM_start[:,0] = times[SFM_indices][0]
				if this_run % 2 != 0:
					SFM_start[:,1] = stim_dur2
				else:
					SFM_start[:,1] = stim_dur1
				SFM_start[:,2] = np.ones(SFM_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', extension = '.txt'), SFM_start, fmt='%3.2f', delimiter = '\t')
				# SFM transitions:
				SFM_transition = np.zeros((sum(transition[SFM_indices])+sum(transition_no_dur[SFM_indices]),3))
				SFM_transition[:sum(transition[SFM_indices]),0] = times[transition*SFM_indices]
				SFM_transition[sum(transition[SFM_indices]):,0] = times[transition_no_dur*SFM_indices]
				SFM_transition[:sum(transition[SFM_indices]),1] = durs[transition*SFM_indices]
				SFM_transition[sum(transition[SFM_indices]):,1] = 1.00
				SFM_transition[:,2] = np.ones(SFM_transition.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', postFix = ['trans'], extension = '.txt'), SFM_transition, fmt='%3.2f', delimiter = '\t')
				
				##### for decoding: #####
				####################
				BR_percept1_indices = BR_indices*(data[:,0]==66)
				BR_percept2_indices = BR_indices*(data[:,0]==67)
				SFM_percept1_indices = SFM_indices*(data[:,0]==66)
				SFM_percept2_indices = SFM_indices*(data[:,0]==67)
				
				BR1_start = np.zeros((BR_percept1_indices.sum(),3))
				BR1_start[:,0] = times[BR_percept1_indices]
				BR1_start[:,1] = durs[BR_percept1_indices]
				BR1_start[:,2] = np.ones(BR1_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_BR1', extension = '.txt'), BR1_start, fmt='%3.2f', delimiter = '\t')
				BR2_start = np.zeros((BR_percept2_indices.sum(),3))
				BR2_start[:,0] = times[BR_percept2_indices]
				BR2_start[:,1] = durs[BR_percept2_indices]
				BR2_start[:,2] = np.ones(BR2_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_BR2', extension = '.txt'), BR2_start, fmt='%3.2f', delimiter = '\t')
				SFM1_start = np.zeros((SFM_percept1_indices.sum(),3))
				SFM1_start[:,0] = times[SFM_percept1_indices]
				SFM1_start[:,1] = durs[SFM_percept1_indices]
				SFM1_start[:,2] = np.ones(SFM1_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_SFM1', extension = '.txt'), SFM1_start, fmt='%3.2f', delimiter = '\t')
				SFM2_start = np.zeros((SFM_percept2_indices.sum(),3))
				SFM2_start[:,0] = times[SFM_percept2_indices]
				SFM2_start[:,1] = durs[SFM_percept2_indices]
				SFM2_start[:,2] = np.ones(SFM2_start.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'deco_SFM2', extension = '.txt'), SFM2_start, fmt='%3.2f', delimiter = '\t')
				
				############################################################
				# Barplot of means:
				
				MEANS = (mean(durs[percept1*BR_indices]), mean(durs[percept2*BR_indices]), mean(durs[percept1*SFM_indices]), mean(durs[percept2*SFM_indices]))
				SEMS = (stats.sem(durs[percept1*BR_indices]), stats.sem(durs[percept2*BR_indices]), stats.sem(durs[percept1*SFM_indices]), stats.sem(durs[percept2*SFM_indices]))
				
				my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'align': 'center'}
				
				N = 4
				ind = np.arange(N)  # the x locations for the groups
				width = 0.45       # the width of the bars
				
				# FIGURE 1
				fig = plt.figure(figsize=(10,6))
				ax = fig.add_subplot(111)
				rects1 = ax.bar(ind[0]+width, MEANS[0], width, yerr=SEMS[0], color='r', alpha = 0.75, **my_dict)
				rects2 = ax.bar(ind[1], MEANS[1], width, yerr=SEMS[1], color='r', alpha = 0.75, **my_dict)
				rects3 = ax.bar(ind[2], MEANS[2], width, yerr=SEMS[2], color='b', alpha = 0.75, **my_dict)
				rects4 = ax.bar(ind[3]-width, MEANS[3], width, yerr=SEMS[3], color='b', alpha = 0.75, **my_dict)
				# ax.set_ylim( (0.5) )
				simpleaxis(ax)
				spine_shift(ax)
				ax.set_ylabel('Duration', size = '16')
				ax.set_title(str(self.subject.initials) + ' - TIMES - run' + str(run) , size = '16')
				ax.set_xticks(( (ind[0]+width+ind[1])/2, (ind[2]+ind[3]-width)/2 ))
				ax.set_xticklabels( ('BR', 'SFM') )
				ax.tick_params(axis='x', which='major', labelsize=16)
				ax.tick_params(axis='y', which='major', labelsize=16)
				ax.set_ylim(ymax = round((plt.axis()[3]+0.5)*2.0, 0)/2 )
				
				pp = PdfPages(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'figure_PERCEPT_TIMES', extension = '.pdf'))
				fig.savefig(pp, format='pdf')
				pp.close()
	

	def createRegressorsBlinks(self, runArray):
		
		from matplotlib.backends.backend_pdf import PdfPages
		
		stim_dur = 150.0
		pauze_between = 16.0
		pauze_end = 8.0
		sample_freq = 500.0
		downsample_freq = 10.0
		blinks_z_threshold = 2.5
		
		# rd = np.loadtxt('/Research/7T_RIVALRY/data/TK/TK_250113/raw/hr/SCANPHYSLOG20130125130453.log')
		
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:
				file_to_open = str(runArray[run]['hrFile'])
				rd = np.loadtxt(file_to_open) 
				srd = rd[:,[2,3,4]] / rd[:,[2,3,4]].std(axis=0)
				
				rivalry1_on = srd.shape[0] - (pauze_end*sample_freq) - (stim_dur*sample_freq) - (pauze_between*sample_freq) - (stim_dur*sample_freq)
				rivalry1_off = srd.shape[0] - (pauze_end*sample_freq) - (stim_dur*sample_freq) - (pauze_between*sample_freq)
				rivalry2_on = srd.shape[0] - (pauze_end*sample_freq) - (stim_dur*sample_freq)
				rivalry2_off = srd.shape[0] - (pauze_end*sample_freq)
				
				# Do ICA on EOG signal in pauze between stimuli. We expect many blinks here!
				srd_for_ica = srd[rivalry1_off:rivalry2_on,:]
				ica = FastICA()
				S_ = ica.fit(srd_for_ica).transform(srd_for_ica)
				A_ = ica.get_mixing_matrix()
				
				# Multiply mixing matrix (as based on pauze between stimuli) with the original signal: 
				ICA_signal = np.dot(srd,A_)
				
				# Select the right component:
				ICA_blink_comp = ICA_signal[:,2]
				
				# Z-score
				ICA_blink_comp_z = (ICA_blink_comp-np.mean(ICA_blink_comp)) / np.std(ICA_blink_comp)
				
				fig = figure(figsize=(12,6))
				ax = fig.add_subplot(111)
				plot(ICA_blink_comp_z[::100])
				axhline(2.5, alpha = 0.25)
				simpleaxis(ax)
				spine_shift(ax)
				
				# Detect blinks
				blink_indices = np.array((ICA_blink_comp_z > blinks_z_threshold), dtype = bool)
				
				plot(blink_indices[::100], color = 'r')
				
				blink_times = np.where(np.diff(blink_indices) == 1)[0]
				blink_start_times = blink_times[::2]
				blink_end_times = blink_times[1::2]
				
				pp = PdfPages(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'figure_BLINKS', extension = '.pdf'))
				fig.savefig(pp, format='pdf')
				pp.close()
				
				
				
				############################################################
				############ MAKE TEXT FILES REGRESSORS ####################
				# Blink start:
				blink_start = np.zeros((blink_start_times.shape[0],3))
				blink_start[:,0] = blink_start_times
				blink_start[:,1] = 0.1
				blink_start[:,2] = np.ones(blink_start_times.shape[0])
				np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_blinks', extension = '.txt'), blink_start, fmt='%3.2f', delimiter = '\t')
	

	def runAllGLMS(self):
		"""
		Take all transition events and use them as event regressors
		Run FSL on this
		"""
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:
				
				# # self.runTransitionGLM(run=r)
				# 						# create the event files
				# 						for eventType in ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray']:
				# 							eventData = self.gatherBehavioralData( whichEvents = [eventType], whichRuns = [run] )
				# 							eventName = eventType.split('EventsAsArray')[0]
				# 							dfFSL = np.ones((eventData[eventType].shape[0],3)) * [1.0,0.1,1]
				# 							dfFSL[:,0] = eventData[eventType][:,0]
				# 							np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = eventName, extension = '.evt'), dfFSL, fmt='%4.2f')
				# 							# also make files for the end of each event.
				# 							dfFSL[:,0] = eventData[eventType][:,0] + eventData[eventType][:,1]
				# 							np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = eventName, postFix = ['end'], extension = '.evt'), dfFSL, fmt='%4.2f')
				
				# remove previous feat directories
				try:
					self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.fsf'))
				except OSError:
					pass
				
				# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
				thisFeatFile = '/Research/7T_RIVALRY/analysis/fsf/7T_RIVALRY.fsf'
				REDict = {
				'---OUTPUT_DIR---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'rivalry', postFix = ['mcf']),
				'---NR_FRAMES---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'])).timepoints),
				'---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf']), 
				'---ANAT_FILE---':os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'bet', 'T1_bet' ), 
				'---BR_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', extension = '.txt'),
				'---BR_TRANS_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_BR', postFix = ['trans'], extension = '.txt'),
				'---SFM_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', extension = '.txt'),
				'---SFM_TRANS_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'txt_SFM', postFix = ['trans'], extension = '.txt')
				}
				featFileName = self.runFile(stage = 'processed/mri', run = self.runList[run], extension = '.fsf')
				featOp = FEATOperator(inputObject = thisFeatFile)
				
				# # In serial:
				# featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
				# In parallel:
				featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
				
				# run feat
				featOp.execute()
		
	

	def registerfeats(self, run_type = 'sphere_presto', postFix = ['mcf']):
		"""run featregapply for all feat direcories in this session."""
		for condition in ['rivalry']:
			for run in self.conditionDict[condition]:	
				this_feat = self.runFile(stage = 'processed/mri', run = self.runList[run], base = 'rivalry', postFix = postFix, extension = '.feat')
				self.setupRegistrationForFeat(this_feat)
	

	def mapper_feat_analysis(self, run_separate = True, run_combination = True):
		
		try:	# create folder
			os.mkdir(self.stageFolder('processed/mri/masks/stat/gfeat_mapper'))
			os.mkdir(self.stageFolder('processed/mri/masks/stat/gfeat_mapper/surf'))
		except OSError:
			pass
		for i in range(1,9): # all stats
			for stat in ['z','t']:
				afo = FlirtOperator( 	os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]), 'combined/combined.gfeat', 'cope' + str(i) + '.feat', 'stats', stat + 'stat1.nii.gz'), 
										referenceFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] )
										)
				# here I assume that the feat registration directory has been created. it's the files that have been used to create the gfeat, so we should be cool.
				afo.configureApply(		transformMatrixFileName = os.path.join(self.stageFolder('processed/mri/reg/feat/'), 'standard2example_func.mat'), 
										outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat/gfeat_mapper'), 'cope' + str(i) + '_' + os.path.split(afo.inputFileName)[1]))
				afo.execute()
				# to surface
				stso = VolToSurfOperator(inputObject = afo.outputFileName)
				stso.configure(		frames = {'stat': 0} , 
									register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), 
									outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat/gfeat_mapper/surf'), os.path.split(afo.outputFileName)[1]))
				stso.execute()
	

	def mask_stats_to_hdf(self, run_type = 'rivalry', postFix = ['mcf']):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""
		
		shell()
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
			
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = openFile(self.hdf5_filename, mode = "w", title = run_type + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
			"""loop over runs, and try to open a group for this run's data"""
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			
			try:
				thisRunGroup = h5file.getNode(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
			"""
			Now, take different stat masks based on the run_type
			"""
			# this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			this_feat = self.runFile(stage = 'processed/mri', run = r, base = 'rivalry', postFix = ['mcf'], extension = '.feat')
			
			stat_files = {
							'STIM_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
							'STIM_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
							'STIM_cope' : os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
							
							'BR_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
							'BR_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
							'BR_cope' : os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
							
							'SFM_T': os.path.join(this_feat, 'stats', 'tstat3.nii.gz'),
							'SFM_Z': os.path.join(this_feat, 'stats', 'zstat3.nii.gz'),
							'SFM_cope' : os.path.join(this_feat, 'stats', 'cope3.nii.gz'),
							
							'BR_SFM_T': os.path.join(this_feat, 'stats', 'tstat4.nii.gz'),
							'BR_SFM_Z': os.path.join(this_feat, 'stats', 'zstat4.nii.gz'),
							'BR_SFM_cope' : os.path.join(this_feat, 'stats', 'cope4.nii.gz'),
							
							'TRANS_T': os.path.join(this_feat, 'stats', 'tstat5.nii.gz'),
							'TRANS_Z': os.path.join(this_feat, 'stats', 'zstat5.nii.gz'),
							'TRANS_cope' : os.path.join(this_feat, 'stats', 'cope5.nii.gz'),
							
							'BR_TRANS_T': os.path.join(this_feat, 'stats', 'tstat6.nii.gz'),
							'BR_TRANS_Z': os.path.join(this_feat, 'stats', 'zstat6.nii.gz'),
							'BR_TRANS_cope' : os.path.join(this_feat, 'stats', 'cope6.nii.gz'),
							
							'SFM_TRANS_T': os.path.join(this_feat, 'stats', 'tstat7.nii.gz'),
							'SFM_TRANS_Z': os.path.join(this_feat, 'stats', 'zstat7.nii.gz'),
							'SFM_TRANS_cope' : os.path.join(this_feat, 'stats', 'cope7.nii.gz'),
							
							'BR_TRANS_SFM_TRANS_T': os.path.join(this_feat, 'stats', 'tstat8.nii.gz'),
							'BR_TRANS_SFM_TRANS_Z': os.path.join(this_feat, 'stats', 'zstat8.nii.gz'),
							'BR_TRANS_SFM_TRANS_cope' : os.path.join(this_feat, 'stats', 'cope8.nii.gz'),
							}
				
			# general info we want in all hdf files
			stat_files.update({
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								# 'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'psc', 'tf']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								# for these final two, we need to pre-setup the retinotopic mapping data
								# 'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz'),
								# 'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz')
			})
			
			stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.getNode(where = "/" + this_run_group_name, name = roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
				for (i, sf) in enumerate(stat_files.keys()):
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					try:
						imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
						these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
						h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
					except ZeroDivisionError:
						pass
				
		h5file.close()
		
	
	
	
