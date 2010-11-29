#!/usr/bin/env python
# encoding: utf-8
"""
analyzeTK.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

import os, sys, datetime
import subprocess, logging

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

thisFolder = '/Users/tk/Documents/research/experiments/rivalry_fMRI/Learning'
sys.path.append( os.environ['ANALYSIS_HOME'] )

from Tools.Sessions import *
import Tools.Project as Project

# importing this file creates the presentSubject variable
from Tools.Subjects.JB import * 
from Tools.Run import *

# project information - not really informative but still, useful for creating paths and the like
presentProject = Project.Project( 'rivalry learning', baseFolder = os.path.join(thisFolder, 'data') )

sessions = []

# session information 
sessionDate = datetime.date(2010, 11, 5)
sessionID = 'rivalry_learning_' + presentSubject.initials
presentSession = RivalrySession(sessionID, sessionDate, presentProject, presentSubject)

runArray = [
	{'ID' : 3, 'scanType': '3d_anat', 'condition': '3d_anat', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['3'], base = presentSubject.initials)},
	{'ID' : 4, 'scanType': 'dti', 'condition': 'dti', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['4'], base = presentSubject.initials)},
	
	{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'disparity', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['5'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['ST_all'], base = presentSubject.firstName + '_1_*', extension = '.pickle')},
	
	{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['6'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_0_*', extension = '.pickle')},
	{'ID' : 7, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['7'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_1_*', extension = '.pickle')},
	{'ID' : 8, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['8'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_2_*', extension = '.pickle')},
	{'ID' : 9, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['9'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_3_*', extension = '.pickle')},
	{'ID' : 10, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['10'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_4_*', extension = '.pickle')},
	{'ID' : 11, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['11'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_5_*', extension = '.pickle')},
	
	{'ID' : 12, 'scanType': 'inplane_anat', 'condition': 'inplane_anat', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['12'], base = presentSubject.initials)},
	
	{'ID' : 13, 'scanType': 'epi_bold', 'condition': 'disparity', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['13'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['ST_all'], base = presentSubject.firstName + '_2_*', extension = '.pickle')},
	
	{'ID' : 14, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['14'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_6_*', extension = '.pickle')},
	{'ID' : 15, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['15'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_7_*', extension = '.pickle')},
	{'ID' : 16, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['16'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_8_*', extension = '.pickle')},
	{'ID' : 17, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['17'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_9_*', extension = '.pickle')},
	{'ID' : 18, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['18'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_10_*', extension = '.pickle')},
	{'ID' : 19, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['19'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_11_*', extension = '.pickle')},
	
	{'ID' : 20, 'scanType': 'epi_bold', 'condition': 'disparity', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['20'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['ST_all'], base = presentSubject.firstName + '_3_*', extension = '.pickle')},
	
	{'ID' : 22, 'scanType': 'dti', 'condition': 'dti', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['22'], base = presentSubject.initials)},
	{'ID' : 23, 'scanType': '3d_anat', 'condition': '3d_anat', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['23'], base = presentSubject.initials)}
	
	]   


for r in runArray:
	thisRun = Run( **r )
	presentSession.addRun(thisRun)

# presentSession.setupFiles(rawBase = presentSubject.initials)

# check whether the inplane_anat has a t2 or t1 - like contrast. t2 is standard. else add contrast = 't1'
# presentSession.registerSession()

# after registration of the entire run comes motion correction
# presentSession.motionCorrectFunctionals()

# functional analysis must happen next.
# first we need to analyze the behavior - most important...
presentSession.analyzeBehavior( )

sessions.append(presentSession)

#________________________________________________________________________
# second (behavioral) session information 
#________________________________________________________________________
sessionDate = datetime.date(2010, 11, 10)
sessionID = 'rivalry_learning_' + presentSubject.initials
presentSession = RivalrySession(sessionID, sessionDate, presentProject, presentSubject)
runArray = [
	{'ID' : 0, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_0_*', extension = '.pickle')},
	{'ID' : 1, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_1_*', extension = '.pickle')},
	{'ID' : 2, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_2_*', extension = '.pickle')},
	{'ID' : 3, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_3_*', extension = '.pickle')},
	{'ID' : 4, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_4_*', extension = '.pickle')},
	{'ID' : 5, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_5_*', extension = '.pickle')},
	{'ID' : 6, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_6_*', extension = '.pickle')},
	{'ID' : 7, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_7_*', extension = '.pickle')},
	{'ID' : 8, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_8_*', extension = '.pickle')},
	{'ID' : 9, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_9_*', extension = '.pickle')},
	{'ID' : 10, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_10_*', extension = '.pickle')},
	{'ID' : 11, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_11_*', extension = '.pickle')}
]
for r in runArray:
	thisRun = Run( **r )
	presentSession.addRun(thisRun)
presentSession.setupFiles(rawBase = presentSubject.initials)
presentSession.analyzeBehavior( )
#________________________________________________________________________
sessions.append(presentSession)


#________________________________________________________________________
# third (behavioral) session information 
#________________________________________________________________________
sessionDate = datetime.date(2010, 11, 11)
sessionID = 'rivalry_learning_' + presentSubject.initials
presentSession = RivalrySession(sessionID, sessionDate, presentProject, presentSubject)
runArray = [
	{'ID' : 0, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_0_*', extension = '.pickle')},
	{'ID' : 1, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_1_*', extension = '.pickle')},
	{'ID' : 2, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_2_*', extension = '.pickle')},
	{'ID' : 3, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_3_*', extension = '.pickle')},
	{'ID' : 4, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_4_*', extension = '.pickle')},
	{'ID' : 5, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_5_*', extension = '.pickle')},
	{'ID' : 6, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_6_*', extension = '.pickle')},
	{'ID' : 7, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_7_*', extension = '.pickle')},
	{'ID' : 8, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_8_*', extension = '.pickle')},
	{'ID' : 9, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_9_*', extension = '.pickle')},
	{'ID' : 10, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_10_*', extension = '.pickle')},
	{'ID' : 11, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_11_*', extension = '.pickle')}
]
for r in runArray:
	thisRun = Run( **r )
	presentSession.addRun(thisRun)
presentSession.setupFiles(rawBase = presentSubject.initials)
presentSession.analyzeBehavior( )
#________________________________________________________________________
sessions.append(presentSession)


#________________________________________________________________________
# fourth (behavioral) session information 
#________________________________________________________________________
sessionDate = datetime.date(2010, 11, 12)
sessionID = 'rivalry_learning_' + presentSubject.initials
presentSession = RivalrySession(sessionID, sessionDate, presentProject, presentSubject)
runArray = [
	{'ID' : 0, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_0_*', extension = '.pickle')},
	{'ID' : 1, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_1_*', extension = '.pickle')},
	{'ID' : 2, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_2_*', extension = '.pickle')},
	{'ID' : 3, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_3_*', extension = '.pickle')},
	{'ID' : 4, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_4_*', extension = '.pickle')},
	{'ID' : 5, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_5_*', extension = '.pickle')},
	{'ID' : 6, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_6_*', extension = '.pickle')},
	{'ID' : 7, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_7_*', extension = '.pickle')},
	{'ID' : 8, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_8_*', extension = '.pickle')},
	{'ID' : 9, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_9_*', extension = '.pickle')},
	{'ID' : 10, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_10_*', extension = '.pickle')},
	{'ID' : 11, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_11_*', extension = '.pickle')}
]
for r in runArray:
	thisRun = Run( **r )
	presentSession.addRun(thisRun)
presentSession.setupFiles(rawBase = presentSubject.initials)
presentSession.analyzeBehavior( )
#________________________________________________________________________
sessions.append(presentSession)


#________________________________________________________________________
# fifth (behavioral) session information 
#________________________________________________________________________
sessionDate = datetime.date(2010, 11, 16)
sessionID = 'rivalry_learning_' + presentSubject.initials
presentSession = RivalrySession(sessionID, sessionDate, presentProject, presentSubject)
runArray = [
	{'ID' : 0, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_0_*', extension = '.pickle')},
	{'ID' : 1, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_1_*', extension = '.pickle')},
	{'ID' : 2, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_2_*', extension = '.pickle')},
	{'ID' : 3, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_3_*', extension = '.pickle')},
	{'ID' : 4, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_4_*', extension = '.pickle')},
	{'ID' : 5, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_5_*', extension = '.pickle')},
	{'ID' : 6, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_6_*', extension = '.pickle')},
	{'ID' : 7, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_7_*', extension = '.pickle')},
	{'ID' : 8, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_8_*', extension = '.pickle')},
	{'ID' : 9, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_9_*', extension = '.pickle')},
	{'ID' : 10, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_10_*', extension = '.pickle')},
	{'ID' : 11, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_11_*', extension = '.pickle')}
]
for r in runArray:
	thisRun = Run( **r )
	presentSession.addRun(thisRun)
presentSession.setupFiles(rawBase = presentSubject.initials)
presentSession.analyzeBehavior( )
#________________________________________________________________________
sessions.append(presentSession)


#________________________________________________________________________
# fifth (behavioral) session information 
#________________________________________________________________________
sessionDate = datetime.date(2010, 11, 17)
sessionID = 'rivalry_learning_' + presentSubject.initials
presentSession = RivalrySession(sessionID, sessionDate, presentProject, presentSubject)
runArray = [
	{'ID' : 0, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_0_*', extension = '.pickle')},
	{'ID' : 1, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_1_*', extension = '.pickle')},
	{'ID' : 2, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_2_*', extension = '.pickle')},
	{'ID' : 3, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_3_*', extension = '.pickle')},
	{'ID' : 4, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_4_*', extension = '.pickle')},
	{'ID' : 5, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_5_*', extension = '.pickle')},
	{'ID' : 6, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_6_*', extension = '.pickle')},
	{'ID' : 7, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_7_*', extension = '.pickle')},
	{'ID' : 8, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_8_*', extension = '.pickle')},
	{'ID' : 9, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_9_*', extension = '.pickle')},
	{'ID' : 10, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_10_*', extension = '.pickle')},
	{'ID' : 11, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_11_*', extension = '.pickle')}
]
for r in runArray:
	thisRun = Run( **r )
	presentSession.addRun(thisRun)
presentSession.setupFiles(rawBase = presentSubject.initials)
presentSession.analyzeBehavior( )
#________________________________________________________________________
sessions.append(presentSession)


#________________________________________________________________________
# sixth (behavioral) session information 
#________________________________________________________________________
sessionDate = datetime.date(2010, 11, 18)
sessionID = 'rivalry_learning_' + presentSubject.initials
presentSession = RivalrySession(sessionID, sessionDate, presentProject, presentSubject)
runArray = [
	{'ID' : 0, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_0_*', extension = '.pickle')},
	{'ID' : 1, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_1_*', extension = '.pickle')},
	{'ID' : 2, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_2_*', extension = '.pickle')},
	{'ID' : 3, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_3_*', extension = '.pickle')},
	{'ID' : 4, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_4_*', extension = '.pickle')},
	{'ID' : 5, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_5_*', extension = '.pickle')},
	{'ID' : 6, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_6_*', extension = '.pickle')},
	{'ID' : 7, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_7_*', extension = '.pickle')},
	{'ID' : 8, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_8_*', extension = '.pickle')},
	{'ID' : 9, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_9_*', extension = '.pickle')},
	{'ID' : 10, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_10_*', extension = '.pickle')},
	{'ID' : 11, 'condition': 'rivalry', 'scanType': 'epi_bold', 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.initials + '_11_*', extension = '.pickle')}
]
for r in runArray:
	thisRun = Run( **r )
	presentSession.addRun(thisRun)
presentSession.setupFiles(rawBase = presentSubject.initials)
presentSession.analyzeBehavior( )
#________________________________________________________________________
sessions.append(presentSession)


#________________________________________________________________________
# seventh (scanning) session information 
#________________________________________________________________________
# session information 
sessionDate = datetime.date(2010, 11, 22)
sessionID = 'rivalry_learning_' + presentSubject.initials
presentSession = RivalrySession(sessionID, sessionDate, presentProject, presentSubject)

runArray = [
	{'ID' : 3, 'scanType': '3d_anat', 'condition': '3d_anat', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['3'], base = presentSubject.initials)},
	{'ID' : 4, 'scanType': 'dti', 'condition': 'dti', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['4'], base = presentSubject.initials)},
	
	{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'disparity', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['5'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['ST_all'], base = presentSubject.firstName + '_0_*', extension = '.pickle')},
	
	{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['6'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_0_*', extension = '.pickle')},
	{'ID' : 7, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['7'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_1_*', extension = '.pickle')},
	{'ID' : 8, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['8'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_2_*', extension = '.pickle')},
	{'ID' : 9, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['9'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_3_*', extension = '.pickle')},
	{'ID' : 11, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['11'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_4_*', extension = '.pickle')},
	{'ID' : 13, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['13'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_5_*', extension = '.pickle')},
	
	{'ID' : 14, 'scanType': 'inplane_anat', 'condition': 'inplane_anat', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['15'], base = presentSubject.initials)},
	# use the latter one, it has twice the in-plane resolution
	{'ID' : 15, 'scanType': 'inplane_anat', 'condition': 'inplane_anat', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['15'], base = presentSubject.initials)},
	
	{'ID' : 16, 'scanType': 'epi_bold', 'condition': 'disparity', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['16'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['ST_all'], base = presentSubject.firstName + '_1_*', extension = '.pickle')},
	
	{'ID' : 17, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['17'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_6_*', extension = '.pickle')},
	{'ID' : 18, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['18'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_7_*', extension = '.pickle')},
	{'ID' : 19, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['19'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_8_*', extension = '.pickle')},
	{'ID' : 20, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['20'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_9_*', extension = '.pickle')},
	{'ID' : 21, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['21'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_10_*', extension = '.pickle')},
	{'ID' : 22, 'scanType': 'epi_bold', 'condition': 'rivalry', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['22'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['BR_all'], base = presentSubject.firstName + '_11_*', extension = '.pickle')},
	
	{'ID' : 23, 'scanType': 'epi_bold', 'condition': 'disparity', 
	'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['23'], base = presentSubject.initials), 
	'behaviorFile':presentSession.runFile('raw/behavior', postFix = ['ST_all'], base = presentSubject.firstName + '_2_*', extension = '.pickle')},
	
	{'ID' : 24, 'scanType': 'dti', 'condition': 'dti', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['24'], base = presentSubject.initials)},
	{'ID' : 25, 'scanType': '3d_anat', 'condition': '3d_anat', 'rawDataFilePath': presentSession.runFile('raw/mri', postFix = ['25'], base = presentSubject.initials)}
	
	]   


for r in runArray:
	thisRun = Run( **r )
	presentSession.addRun(thisRun)

presentSession.setupFiles(rawBase = presentSubject.initials)

# check whether the inplane_anat has a t2 or t1 - like contrast. t2 is standard. else add contrast = 't1'
presentSession.registerSession()

# after registration of the entire run comes motion correction
presentSession.motionCorrectFunctionals()

# functional analysis must happen next.
# first we need to analyze the behavior - most important...
presentSession.analyzeBehavior( )

sessions.append(presentSession)





f = open(os.path.join(thisFolder, 'data', presentSubject.initials, 'allRivBehav.pickle') , 'w')
pickle.dump([s.rivalryBehavior for s in sessions], f)
f.close()

pl.show()
