#!/usr/bin/env python
# encoding: utf-8
"""
analyze_7T_S1.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

import os, sys, datetime
import subprocess

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

# this line adds the standard software to your path
sys.path.append( os.environ['ANALYSIS_HOME'] )

# import sessions, operators from analysis home.
from Tools.Sessions import *
from Tools.Operators.PhysioOperator import PhysioOperator


# if this script is run on its own, and not when it's imported from anywhere
if __name__ == '__main__':	
	
	# just some small variables to track who we're looking at.
	sj_init = 'TK'
	session_date = '130813'
	
	# we could make a list of files to input.
	# we could create a list of physio operators, for example, each with its own input file.
	# fileNameList = subprocess.Popen('ls ' + '/Volumes/HDD/research/projects/ORA/PRF/data/' + sj_init + '/' + sj_init + '_' + session_date + '/raw/hr/' + '*.log', shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
	# for f in fileNameList:
	po = PhysioOperator('/Volumes/HDD/research/projects/ORA/PRF/data/' + sj_init + '/' + sj_init + '_' + session_date + '/raw/hr/SCANPHYSLOG20130813092644.log')
	po.preprocess_to_continuous_signals(TR = 1.5, nr_TRs = 835)
	
