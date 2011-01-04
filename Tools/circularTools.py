#!/usr/bin/env python
# encoding: utf-8
"""
circularTools.py

Created by Tomas Knapen on 2010-12-18.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from math import *

def positivePhases( phases ):
	return fmod(phases + 2 * pi, 2 * pi)

def circularDifference( phase1, phase2 ):
	# we need to do a circular subtraction; as follows
	return np.fmod(phase1 - phase2 + 3*pi, 2*pi) - pi

def SCR( phases ):
	S,C = np.sin(phases).sum(), np.cos(phases).sum()
	R = sqrt(S**2+C**2)
	return S, C, R

def circularMean( phases ):
	S, C, R = SCR(phases)
	return np.arcsin(S/R)

def circularVariance( phases ):
	S, C, R = SCR(phases)
	return 1 - (R/phases.shape[0])

def circularStandardDeviation( phases ):
	V = circularVariance(phases)
	return sqrt( -2.0 * log(1.0 - V) )

def standardDeviationOnPhaseDifferences( sigma1, sigma2 ):
	# gaussian distributions allow us to easily get the standard deviation of the difference between two of them.
	return np.sqrt( np.power(sigma1, 2) + np.power(sigma2, 2) )

def circularCorrelation( phase1, phase2 ):
	"""
	phaseCorrelation calculates the circular correlation coefficient 
	according to Fisher and Lee '83
	equation taken from http://cnx.org/content/m22974/1.3/
	"""
	enum = 0
	for i in range(phase1.shape[0]-1):	# double for loop is ugly but I'll do this in arrays later
		for j in range(i+1,phase1.shape[0]):
			enum += sin(phase1[i]-phase1[j]) * sin(phase2[i]-phase2[j])
	denomA = denomB = 0
	for i in range(phase1.shape[0]-1):	# double for loop is ugly but I'll do this in arrays later
		for j in range(i+1,phase1.shape[0]):
			denomA += sin(phase1[i]-phase1[j])**2
			denomB += sin(phase2[i]-phase2[j])**2
	denom = sqrt(denomA * denomB)
	r = enum / denom
	return r

def circularStandardDeviationFromNoiseSD( real, imag, noiseSD ):
	"""
	# so, for each of these phaseArrays, we calculate not only the phase, but also,
	# following Gudbjartsson and Patz, 1995, eq. 8
	# the standard deviation on the phase, which is equal to 
	# the standard deviation of the noise divided by the amplitude of the signal, 
	# and is gaussian distributed.
	"""
	complexAtFundamental = np.zeros(np.shape(real), dtype = np.complex128)
	complexAtFundamental.real = real
	complexAtFundamental.imag = imag
	amplitudes = abs(complexAtFundamental)
	result = noiseSD/amplitudes
	# take out places where amplitude was zero, i.e. rule above resulted in inf.
	# these are replaced by the sdForZeroAmplitude defined above
	resultInfToZero = np.nan_to_num(result * np.isfinite(result))
	return np.clip( np.nan_to_num( np.isinf( result ) * result ), 0, sdForZeroAmplitude ) + resultInfToZero

def fitVonMises( data, initial = [0,1] ):
	from scipy.stats import vonmises
	from scipy.optimize import fmin
	
	# negative log likelihood sum is to be minimized, this maximized the likelihood
	vmL = lambda v : -np.sum(np.log( vonmises.pdf(v[0] ,v[1] , data) ))
	
	return fmin(vmL, initial)
	
	
		