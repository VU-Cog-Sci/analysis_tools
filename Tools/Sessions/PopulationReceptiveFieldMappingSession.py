#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""
import datetime, os, sys
from ..Sessions import * 
from ..Operators.ArrayOperator import *
from ..Operators.EyeOperator import *
from ..Operators.PhysioOperator import *

from pylab import *
import numpy as np
import scipy as sp
from scipy.stats import spearmanr
from scipy import ndimage
import matplotlib.mlab as mlab

from nifti import *
from math import *
import shutil

from joblib import Parallel, delayed
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge, RidgeCV, ElasticNet, ElasticNetCV
# from skimage import filter, measure
from skimage.morphology import disk

def fitARDRidge(design_matrix, timeseries, n_iter = 100, compute_score=True):
	"""fitARDRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	fitARDRidge is too time-expensive.
	"""
	br = ARDRegression(n_iter = n_iter, compute_score = compute_score)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return br.coef_, srp


def fitBayesianRidge(design_matrix, timeseries, n_iter = 50, compute_score = False, verbose = True):
	"""fitBayesianRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	if n_iter == 0:
		br = BayesianRidge(compute_score = compute_score, verbose = verbose)
	else:
		br = BayesianRidge(n_iter = n_iter, compute_score = compute_score, verbose = verbose)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return br.coef_, srp, predicted_signal.sum(axis = 1), timeseries

def fitElasticNetCV(design_matrix, timeseries, verbose = True, l1_ratio = [.1, .5, .7, .9, .95, .99, 1]):
	"""fitBayesianRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	ecv = ElasticNetCV(verbose = verbose, l1_ratio = l1_ratio, n_jobs = 28)
	ecv.fit(design_matrix, timeseries)
	
	predicted_signal = ecv.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return ecv.coef_, srp


def fitRidge(design_matrix, timeseries, alpha = 1.0):
	"""fitRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	br = Ridge(alpha = alpha)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return br.coef_, srp,predicted_signal.sum(axis = 1), timeseries#EV_spatial_profile
	
def fitRidgeCV(design_matrix, timeseries, alphas = None):
	"""fitRidgeCV fits a design matrix to a given timeseries using
	built-in cross-validation.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	if alphas == None:
		alphas = np.logspace(0.001,10,100)
	br = RidgeCV(alphas = alphas)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]
	return br.coef_, srp


def normalize_histogram(input_array, mask_array = None):
	if mask_array == None:
		mask_array = input_array != 0.0
	
	return (input_array - input_array[mask_array].min()) / (input_array[mask_array].max() - input_array[mask_array].min())


def analyze_PRF_from_spatial_profile(spatial_profile_array, stats_data=[], upscale = 5, diagnostics_plot = False, contour_level = 0.9, voxel_no = 1, cond = cond, normalize_to = [], fit_on='smoothed_betas',plotdir = []):
	"""analyze_PRF_from_spatial_profile tries to fit a PRF 
	to the spatial profile of spatial beta values from the ridge regression """
	
	## upsample and smooth PRF
	n_pixel_elements = int(sqrt(spatial_profile_array.shape[0]))
	us_spatial_profile = np.repeat(np.repeat(spatial_profile_array.reshape((n_pixel_elements, n_pixel_elements)),upscale,axis=1),upscale,axis=0)
	# us_spatial_profile = ndimage.interpolation.zoom(spatial_profile_array.reshape((n_pixel_elements, n_pixel_elements)), upscale)
	uss_spatial_profile = ndimage.gaussian_filter(us_spatial_profile, upscale/2)
	
	## compute maximum
	maximum = ndimage.measurements.maximum_position(uss_spatial_profile)
	
	if fit_on == 'smoothed_betas':
		PRF = uss_spatial_profile
	elif fit_on == 'raw_betas':
		PRF = spatial_profile_array.reshape(n_pixel_elements, n_pixel_elements)
		maximum = tuple((np.array(maximum).astype('float')/5).astype('int'))
		upscale = 1

	# ## compute surface:
	# 1. normalize PRF
	if normalize_to == 'z-score':
		PRF_circle = disk((n_pixel_elements*upscale-1)/2)
		PRF_n = (PRF-np.mean(PRF[PRF_circle==1]))/np.std(PRF[PRF_circle==1])
		# cutoffs = np.arange(1,2.5,0.5)
		cutoffs = [1]
	elif normalize_to == 'zero-one':
		PRF_n = (PRF - np.min(PRF)) / (np.max(PRF) - np.min(PRF))
		cutoffs = np.array([0.5,0.6,0.7,0.8])

	# 2. generate mask
	all_params = []
	fitimages = []
	EVS = []
	PRF_n_t = []
	# 
	for i, thresh in enumerate(cutoffs):
		labels = []; label_index = []
		PRF_n_m= zeros((n_pixel_elements*upscale,n_pixel_elements*upscale))
		PRF_n_m[PRF_n > thresh] = 1

		# 3. pick right PRF then create new windows around it using banded profile
		labels = ndimage.label(PRF_n_m)[0]
		label_index = labels[(maximum)]
		label_4_surf = copy(labels)
		label_4_surf[label_4_surf!=label_index] = 0
		PRF_n_t.append(copy(PRF_n))
		PRF_n_t[i][labels!=label_index] = 0
		
		# 4. compute surface and volume
		p, infodict, errmsg, fi = gaussfit(PRF_n_t[i],voxel_no=voxel_no)
		if not p == []: all_params.append(p)
		if not fi == []: fitimages.append(fi)
		
		EVS.append(1- np.linalg.norm(PRF_n_t[i].ravel()-fitimages[i].ravel())**2 / np.linalg.norm(PRF_n_t[i].ravel())**2)
			
	if all_params == []:
		center = [0,0]
		max_norm = 2.0 * (np.array(center) / (n_pixel_elements*upscale)) - 1.0 
		max_comp_gauss =  np.complex(max_norm[0], max_norm[1])
		max_comp_abs = np.complex(max_norm[0], max_norm[1])
		surf_gauss = 0
		surf_mask = 0.5
		vol = 0
		EV = 0

		print '>>>>>>>>>>>>>>>>>>>>>>>'
		# pl.imshow(PRF_n,interpolation='nearest')
		# ugly_plotdir = os.path.join(plotdir,'ugly_PRFS')
		# pl.savefig(os.path.join(ugly_plotdir + 'vox_'  + str(voxel_no)  + '_' + cond + '_' + fit_on + '_thresh_' + str(thresh) + '.pdf'))

	else:
		best = EVS.index(np.max(EVS))
		params = all_params[best]
		fitimage = fitimages[best]
		
		thresh = cutoffs[best]
		# calculate EV
		# ---------------
		# create time-series for this voxel by multiplying the beta values
	
		
		EV = EVS[best]#1- np.linalg.norm(PRF_n_t.ravel()-fitimage.ravel())**2 / np.linalg.norm(PRF_n_t.ravel())**2
		center_gauss = np.array([params[3],params[2]])
		center_abs = copy(maximum).astype('float32')
		max_norm_gauss = 2.0 * (np.array(center_gauss) / (n_pixel_elements*upscale)) - 1.0
		# max_norm_gauss = 2.0 * np.array(center_gauss) - 1.0
		max_comp_gauss =  np.complex(max_norm_gauss[0], max_norm_gauss[1])
		max_norm_abs = 2.0 * (np.array(center_abs) / (n_pixel_elements*upscale)) - 1.0 
		max_comp_abs =  np.complex(max_norm_abs[0], max_norm_abs[1])
		ecc_gauss = np.abs(max_comp_gauss) * 27/2.0
		ecc_abs = np.abs(max_comp_abs) * 27/2.0
		# surf_gauss = 2*np.sqrt(2*(math.log(2)))*sd
		surf_gauss = (np.pi * (params[4]) * (params[5])) / (np.pi * (n_pixel_elements*upscale/2.0) * (n_pixel_elements*upscale/2.0)) * 27
		sd_gauss =  ( ((params[4]+params[5]) / 2)  /  (n_pixel_elements*upscale)) * 27
		# sd_gauss = np.sqrt(surf_gauss / np.pi)
		fwhm = 2.0*np.sqrt(2.0*np.log(2))*sd_gauss
		surf_mask = label_4_surf.sum().astype('float32') / (np.pi * (n_pixel_elements*upscale/2.0) * (n_pixel_elements*upscale/2.0)) * 27
		sd_mask = np.sqrt(surf_mask / np.pi)
		vol = 2*surf_gauss*(params[1]-params[0])
		
		#
	# print voxel_no
	if diagnostics_plot:
		# mask = [(stats[r][:,0]  > corr_threshold) * (results[r][:,results_frames['ecc_gauss']] < 0.7*(27.0/2)) * (results[r][:,results_frames['ecc_gauss']] > 0.0) *  (results[r][:,results_frames['EV']] > 0.85) for r in range(len(end_rois))]
		# if np.all([ stats_data[0] > 0.2, ecc_gauss < 0.7*(27.0/2), EV > 0.85, fitimage != [],np.max(labels)<5]):
		f=pl.figure(figsize = (16,7))
		ax = f.add_subplot(121)
		pl.imshow(PRF)
		pl.imshow(PRF_n_t[best],alpha=0.7)
		ax.set_title('PRF spatial profile')
		pl.text(int(PRF_n_t[best].shape[0]/8),int(PRF_n_t[best].shape[0]/8*6), '# regions: %d \n EV: %.2f \nsize (sd): %.2f \necc: %.2f \np-val: %.2f \nr-val: %.2f \ncond: %s' %(np.max(labels),EV,sd_gauss,ecc_gauss,stats_data[1],stats_data[0] ,cond),fontsize=14,fontweight ='bold',bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
		
		pl.axis('off')
		# c = pl.contour(fitimage)
		# e = matplotlib.patches.Ellipse(tuple([center_gauss[1],center_gauss[0]]),params[4],params[5],angle=params[6],alpha=0.5)
		# e = matplotlib.patches.Circle(tuple([center_gauss[1],center_gauss[0]]),sd_gauss,alpha=0.5)
		# ax.add_artist(e)
		ax = f.add_subplot(122)
		pl.imshow(fitimage)
		# pl.text(int(PRF_n_t[best].shape[0]/8),int(PRF_n_t[best].shape[0]/8*6), 'EV: %.2f \nsize (sd): %.2f \necc: %.2f \nthresh: %.2f \ncond: %s' %(EV,sd_gauss,ecc_gauss,thresh ,cond),fontsize=14,fontweight ='bold',bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
		pl.axis('off')
		# pl.plot([center_gauss[1]], [center_gauss[0]], 'ko')
		# pl.plot([center_abs[1]], [center_abs[0]], 'wo')
		ax.set_title('Gauss fit')
		pl.savefig(os.path.join(plotdir + 'vox_'  + str(voxel_no)  + '_' + cond + '_' + fit_on + '_thresh_' + str(thresh) + '.pdf'))
		pl.close()
						
	return max_comp_gauss, max_comp_abs, surf_gauss, surf_mask, vol, EV, sd_gauss, sd_mask, fwhm, np.max(labels)
	
def moments(data,circle,rotate,vheight,estimator=median,voxel_no=1,**kwargs):
	"""Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
	the gaussian parameters of a 2D distribution by calculating its
	moments.  Depending on the input parameters, will only output 
	a subset of the above.

	If using masked arrays, pass estimator=np.ma.median
	"""
	total = np.abs(data).sum()
	Y, X = np.indices(data.shape) # python convention: reverse x,y np.indices
	y = np.argmax((X*np.abs(data)).sum(axis=1)/total)
	x = np.argmax((Y*np.abs(data)).sum(axis=0)/total)
	col = data[int(y),:]
	# FIRST moment, not second!
	width_x = np.sqrt(np.abs((np.arange(col.size)-y)*col).sum()/np.abs(col).sum())
	row = data[:, int(x)]
	width_y = np.sqrt(np.abs((np.arange(row.size)-x)*row).sum()/np.abs(row).sum())
	width = ( width_x + width_y ) / 2.
	height = estimator(data.ravel())
	amplitude = data.max()-height
	mylist = [amplitude,x,y]
	if np.isnan(width_y) or np.isnan(width_x) or np.isnan(height) or np.isnan(amplitude):
		mylist = []
	else:
		if vheight==1:
			mylist = [height] + mylist
		if circle==0:
			mylist = mylist + [width_x,width_y]
		if rotate==1:
			mylist = mylist + [0.] 
		else:  
			mylist = mylist + [width]
	return mylist

def twodgaussian(inpars, circle=False, rotate=True, vheight=True, shape=None):
	"""Returns a 2d gaussian function of the form:
	x' = np.cos(rota) * x - np.sin(rota) * y
	y' = np.sin(rota) * x + np.cos(rota) * y
	(rota should be in degrees)
	g = b + a * np.exp ( - ( ((x-center_x)/width_x)**2 +
	((y-center_y)/width_y)**2 ) / 2 )

	inpars = [b,a,center_x,center_y,width_x,width_y,rota]
	(b is background height, a is peak amplitude)

	where x and y are the input parameters of the returned function,
	and all other parameters are specified by this function

	However, the above values are passed by list.  The list should be:
	inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)

	You can choose to ignore / neglect some of the above input parameters 
	unp.sing the following options:
	circle=0 - default is an elliptical gaussian (different x, y
	widths), but can reduce the input by one parameter if it's a
	circular gaussian
	rotate=1 - default allows rotation of the gaussian ellipse.  Can
	remove last parameter by setting rotate=0
	vheight=1 - default allows a variable height-above-zero, i.e. an
	additive constant for the Gaussian function.  Can remove first
	parameter by setting this to 0
	shape=None - if shape is set (to a 2-parameter list) then returns
	an image with the gaussian defined by inpars
	    """
	inpars_old = inpars
	inpars = list(inpars)
	if vheight == 1:
		height = inpars.pop(0)
		height = float(height)
	else:
		height = float(0)
	amplitude, center_y, center_x = inpars.pop(0),inpars.pop(0),inpars.pop(0)
	amplitude = float(amplitude)
	center_x = float(center_x)
	center_y = float(center_y)
	if circle == 1:
		width = inpars.pop(0)
		width_x = float(width)
		width_y = float(width)
		rotate = 0
	else:
		width_x, width_y = inpars.pop(0),inpars.pop(0)
		width_x = float(width_x)
		width_y = float(width_y)
	if rotate == 1:
		rota = inpars.pop(0)
		rota = pi/180. * float(rota)
		rcen_x = center_x * np.cos(rota) - center_y * np.sin(rota)
		rcen_y = center_x * np.sin(rota) + center_y * np.cos(rota)
	else:
		rcen_x = center_x
		rcen_y = center_y
	if len(inpars) > 0:
		raise ValueError("There are still input parameters:" + str(inpars) + \
			" and you've input: " + str(inpars_old) + \
			" circle=%d, rotate=%d, vheight=%d" % (circle,rotate,vheight) )
        
	def rotgauss(x,y):
		if rotate==1:
			xp = x * np.cos(rota) - y * np.sin(rota)
			yp = x * np.sin(rota) + y * np.cos(rota)
		else:
			xp = x
			yp = y
		g = height+amplitude*np.exp(
		-(((rcen_x-xp)/width_x)**2+
		((rcen_y-yp)/width_y)**2)/2.)
		return g
	if shape is not None:
		return rotgauss(*np.indices(shape))
	else:
		return rotgauss

def gaussfit(data,err=None,params=(),return_all=False,circle=False,
	fixed=np.repeat(False,7),limitedmin=[False,False,False,False,True,True,True],
	limitedmax=[False,False,False,False,False,False,True],
	usemoment=np.array([],dtype='bool'),
	minpars=np.repeat(0,7),maxpars=[0,0,0,0,0,0,360],
	rotate=1,vheight=1,quiet=True,returnmp=False,
	returnfitimage=True,voxel_no=1,**kwargs):
	"""
	Gaussian fitter with the ability to fit a variety of different forms of
	2-dimensional gaussian.

	Input Parameters:
	data - 2-dimensional data array
	err=None - error array with same size as data array
	params=[] - initial input parameters for Gaussian function.
	(height, amplitude, x, y, width_x, width_y, rota)
	if not input, these will be determined from the moments of the system, 
	assuming no rotation
	autoderiv=1 - use the autoderiv provided in the lmder.f function (the
	alternative is to us an analytic derivative with lmdif.f: this method
	is less robust)
	return_all=0 - Default is to return only the Gaussian parameters.  
	1 - fit params, fit error
	returnfitimage - returns (best fit params,best fit image)
	returnmp - returns the full mpfit struct
	circle=0 - default is an elliptical gaussian (different x, y widths),
	but can reduce the input by one parameter if it's a circular gaussian
	rotate=1 - default allows rotation of the gaussian ellipse.  Can remove
	last parameter by setting rotate=0.  np.expects angle in DEGREES
	vheight=1 - default allows a variable height-above-zero, i.e. an
	additive constant for the Gaussian function.  Can remove first
	parameter by setting this to 0
	usemoment - can choose which parameters to use a moment estimation for.
	Other parameters will be taken from params.  Needs to be a boolean
	array.

	Output:
	Default output is a set of Gaussian parameters with the same shape as
	the input parameters

	If returnfitimage=True returns a np array of a gaussian
	contructed using the best fit parameters.

	If returnmp=True returns a `mpfit` object. This object contains
	a `covar` attribute which is the 7x7 covariance array
	generated by the mpfit class in the `mpfit_custom.py`
	module. It contains a `param` attribute that contains a
	list of the best fit parameters in the same order as the
	optional input parameter `params`.

	Warning: Does NOT necessarily output a rotation angle between 0 and 360 degrees.
"""
	usemoment=np.array(usemoment,dtype='bool')
	params=np.array(params,dtype='float')
	if usemoment.any() and len(params)==len(usemoment):
		moment = np.array(moments(data,circle,rotate,vheight,**kwargs),dtype='float')
		params[usemoment] = moment[usemoment]
	elif params == [] or len(params)==0:
		params = (moments(data,circle,rotate,vheight,voxel_no=voxel_no,**kwargs))
	if vheight==0:
		vheight=1
		params = np.concatenate([[0],params])
		fixed[0] = 1

	if params == []:
		p, infodict, errmsg, fitimage = [],[],[],[]
	else:
		# mpfit will fail if it is given a start parameter outside the allowed range:
		for i in xrange(len(params)): 
			if params[i] > maxpars[i] and limitedmax[i]: params[i] = maxpars[i]
			if params[i] < minpars[i] and limitedmin[i]: params[i] = minpars[i]

	if err is None:
		errorfunction = lambda p: np.ravel((twodgaussian(p,circle,rotate,vheight)(*np.indices(data.shape)) - data))
	else:
		errorfunction = lambda p: np.ravel((twodgaussian(p,circle,rotate,vheight)(*np.indices(data.shape)) - data)/err)
                    
	parinfo = [ 
		{'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],'parname':"AMPLITUDE",'error':0},
		{'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],'parname':"XSHIFT",'error':0},
		{'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],'parname':"YSHIFT",'error':0},
		{'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4],'parname':"XWIDTH",'error':0} 
		]
	    
	if vheight == 1:
		parinfo.insert(0,{'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],'parname':"HEIGHT",'error':0})
	if circle == 0:
		parinfo.append({'n':5,'value':params[5],'limits':[minpars[5],maxpars[5]],'limited':[limitedmin[5],limitedmax[5]],'fixed':fixed[5],'parname':"YWIDTH",'error':0})
	if rotate == 1:
		parinfo.append({'n':6,'value':params[6],'limits':[minpars[6],maxpars[6]],'limited':[limitedmin[6],limitedmax[6]],'fixed':fixed[6],'parname':"ROTATION",'error':0})

	p, cov, infodict, errmsg, success = scipy.optimize.leastsq(errorfunction, params, full_output=1)
	
	if returnfitimage:
		fitimage = twodgaussian(p)(*np.indices(data.shape))

	return p, infodict, errmsg, fitimage
	

class PRFModelTrial(object):
	"""docstring for PRFModelTrial"""
	def __init__(self, orientation, n_elements, n_samples, sample_duration, bar_width = 0.1):
		super(PRFModelTrial, self).__init__()
		self.orientation = -(orientation + np.pi/2.0)
		# self.orientation = orientation
		self.n_elements = n_elements
		self.n_samples = n_samples
		self.sample_duration = sample_duration
		self.bar_width = bar_width
		
		self.rotation_matrix = np.matrix([[cos(self.orientation), -sin(self.orientation)],[sin(self.orientation), cos(self.orientation)]])

		x, y = np.meshgrid(np.linspace(-1,1,self.n_elements), np.linspace(-1,1,self.n_elements))
		self.xy = np.matrix([x.ravel(), y.ravel()]).T  
		self.rotated_xy = np.array(self.xy * self.rotation_matrix)
		self.ecc_test = (np.array(self.xy) ** 2).sum(axis = 1) <= 1.0
	
	def in_bar(self, time = 0):
		"""in_bar, a method, not Ralph."""
		# a bar of self.bar_width width
		# position = 2.0 * ((time * (1.0 + self.bar_width / 2.0)) - (0.5 + self.bar_width / 4.0))
		position = 2.0 * ((time * (1.0 + self.bar_width)) - (0.5 + self.bar_width / 2.0))
		extent = [-self.bar_width/2.0 + position, self.bar_width/2.0 + position] 
		# rotating the xy matrix itself allows us to test only the x component 
		return ((self.rotated_xy[:,0] >= extent[0]) * (self.rotated_xy[:,0] <= extent[1]) * self.ecc_test).reshape((self.n_elements, self.n_elements))
	
	def pass_through(self):
		"""pass_through models a single pass-through of the bar, 
		with padding as in the padding list for start and end."""

		self.pass_matrix = np.array([self.in_bar(i) for i in np.linspace(0.0, 1.0, self.n_samples, endpoint = True)])

class PRFModelRun(object):
	"""docstring for PRFModelRun"""
	def __init__(self, run, n_TRs, TR, n_pixel_elements, sample_duration = 0.6, bar_width = 0.1):
		super(PRFModelRun, self).__init__()
		self.run = run
		self.n_TRs = n_TRs
		self.TR = TR
		self.n_pixel_elements = n_pixel_elements
		self.sample_duration = sample_duration
		self.bar_width = bar_width
		
		self.orientation_list = self.run.orientations
	
	def simulate_run(self, orientations,save_images_to_file = None):
		"""docstring for simulate_run"""
		self.sample_times = np.arange(0, self.n_TRs * self.TR, self.sample_duration)
		
		self.run_matrix = np.zeros((self.sample_times.shape[0], self.n_pixel_elements, self.n_pixel_elements))
		
		# for i in range(len(self.orientation_list)): # trials
		for i in range(len(self.run.trial_times)):
		
			# if self.run.ID == 3:
			# 	
			samples_in_trial = (self.sample_times >= (self.run.trial_times[i][1])) * (self.sample_times < (self.run.trial_times[i][2]))

			# print samples_in_trial.sum()
			# if self.run.trial_times[i][0] != 'fix_no_stim':
			# 
			# 


			if np.all([self.run.trial_times[i][0] != 'fix_no_stim', self.orientation_list[i] in np.radians(orientations)]):
				pt = PRFModelTrial(orientation = self.orientation_list[i], n_elements = self.n_pixel_elements, n_samples = samples_in_trial.sum(), sample_duration = self.sample_duration, bar_width = self.bar_width)
				pt.pass_through()
				self.run_matrix[samples_in_trial] = pt.pass_matrix
		
		if save_images_to_file != None:
			for i in range(self.run_matrix.shape[0]):
				if i < 200:
					f = pl.figure()
					s = f.add_subplot(111)
					pl.imshow(self.run_matrix[i])
					pl.savefig(save_images_to_file + '_' + str(i) + '.pdf')

# class PRFModelTrial(object):
# 	"""docstring for PRFModelTrial"""
# 	def __init__(self, orientation, n_elements, n_samples, sample_duration, bar_width = 0.09):
# 		super(PRFModelTrial, self).__init__()
# 		self.orientation = -(orientation + np.pi/2.0)
# 		# self.orientation = orientation
# 		self.n_elements = n_elements
# 		self.n_samples = n_samples
# 		self.sample_duration = sample_duration
# 		self.bar_width = bar_width

# 		self.rotation_matrix = np.matrix([[cos(self.orientation), -sin(self.orientation)],[sin(self.orientation), cos(self.orientation)]])

# 		x, y = np.meshgrid(np.linspace(-1,1,self.n_elements*5), np.linspace(-1,1,self.n_elements*5))
# 		self.xy = np.array(np.matrix([x.ravel(), y.ravel()]).T )
# 		# self.rotated_xy = np.array(self.xy * self.rotation_matrix)
# 		self.ecc_test = (np.array(self.xy) ** 2).sum(axis = 1) <= 1.0
	
# 	def in_bar(self, time = 0):
# 		"""in_bar, a method, not Ralph."""
# 		# a bar of self.bar_width width
# 		# position = 2.0 * ((time * (1.0 + self.bar_width / 2.0)) - (0.5 + self.bar_width / 4.0))
# 		shell()
# 		for time in np.arange(0,1,0.1):
# 		# for t in np.linspace(0,1,800):
# 			position = 2.0 * ((time * (1.0 + self.bar_width)) - (0.5 + self.bar_width / 2.0))
# 			# extent = [-self.bar_width/2.0 + position, self.bar_width/2.0 + position] 


# 			x_bar, y_bar = np.meshgrid(np.linspace(-(self.bar_width/2)+position,(self.bar_width/2)+position,self.n_elements),np.linspace(-1,1,self.n_elements*5))
# 			xy_bar = np.array(np.matrix([x_bar.ravel(), y_bar.ravel()]).T )
# 			rotated_xy_bar = xy_bar * self.rotation_matrix
# 			# rounded_xy_bar = np.round(xy_bar,3)
# 			# rounded_xy = np.round(self.xy,3)

# 			differences = np.zeros((np.shape(rotated_xy_bar)[0],np.shape(self.xy)[0]))
# 			for bi,barc in enumerate(rotated_xy_bar):
# 				for si,screenc in enumerate(self.xy):
# 					differences[bi,si] =  norm(screenc-barc)
# 			bar_on_screen = np.unique([np.argmin(differences[i,:]) for i in range((self.n_elements*5)**2)])
# 			this_dm =  np.zeros(self.n_elements*5 * self.n_elements*5)
# 			this_dm[bar_on_screen] = 1
# 			this_dm = this_dm.reshape((self.n_elements*5,self.n_elements*5))
# 			mask = disk(self.n_elements*5/2)
# 			mask1 = np.delete(mask,self.n_elements*5/2,axis=0)
# 			mask2 = np.delete(mask1,self.n_elements*5/2,axis=1)
# 			this_dm = this_dm * mask2
# 			# imshow(this_dm,interpolation='nearest')
# 			# show()

# 		return this_dm
	
	# def pass_through(self):
	# 	"""pass_through models a single pass-through of the bar, 
	# 	with padding as in the padding list for start and end."""

	# 	self.pass_matrix = np.array([self.in_bar(i) for i in np.linspace(0.0, 1.0, self.n_samples, endpoint = True)])

# class PRFModelRun(object):
# 	"""docstring for PRFModelRun"""
# 	def __init__(self, run, n_TRs, TR, n_pixel_elements, sample_duration = 0.6, bar_width = 0.05):
# 		super(PRFModelRun, self).__init__()
# 		self.run = run
# 		self.n_TRs = n_TRs
# 		self.TR = TR
# 		self.n_pixel_elements = n_pixel_elements
# 		self.sample_duration = sample_duration
# 		self.bar_width = bar_width
		
# 		self.orientation_list = self.run.orientations
	
# 	def simulate_run(self, orientations,save_images_to_file = None):
# 		"""docstring for simulate_run"""
# 		self.sample_times = np.arange(0, self.n_TRs * self.TR, self.sample_duration)
		
# 		self.run_matrix = np.zeros((self.sample_times.shape[0], self.n_pixel_elements, self.n_pixel_elements))
		
# 		# for i in range(len(self.orientation_list)): # trials
# 		for i in range(len(self.run.trial_times)):
		
# 			# if self.run.ID == 3:
# 			# 	
# 			samples_in_trial = (self.sample_times >= (self.run.trial_times[i][1])) * (self.sample_times < (self.run.trial_times[i][2]))

# 			# print samples_in_trial.sum()
# 			# if self.run.trial_times[i][0] != 'fix_no_stim':
# 			# 
# 			# 


# 			if np.all([self.run.trial_times[i][0] != 'fix_no_stim', self.orientation_list[i] in np.radians(orientations)]):
# 				pt = PRFModelTrial(orientation = self.orientation_list[i], n_elements = self.n_pixel_elements, n_samples = samples_in_trial.sum(), sample_duration = self.sample_duration, bar_width = self.bar_width)
# 				pt.pass_through()
# 				self.run_matrix[samples_in_trial] = pt.pass_matrix
		
# 		if save_images_to_file != None:
# 			for i in range(self.run_matrix.shape[0]):
# 				if i < 200:
# 					f = pl.figure()
# 					s = f.add_subplot(111)
# 					pl.imshow(self.run_matrix[i])
# 					pl.savefig(save_images_to_file + '_' + str(i) + '.pdf')	

class PopulationReceptiveFieldMappingSession(Session):
	"""
	Class for population receptive field mapping sessions analysis.
	"""

	def preprocessing_evaluation(self):

		mask = 'lh.V1'
		mask_data = np.array(NiftiImage(os.path.join(self.stageFolder( stage = 'processed/mri/masks/anat'), mask)).data, dtype = bool)
		k=0
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			raw_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = [] ))
			mcf_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'] ))
			sgtf_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf'] ))

			# psc_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc'] ))
			prZ_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','prZ'] ))
			# res_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','prZ','res'] ))

			all_files = ['raw_file','mcf_file','sgtf_file','prZ_file']	 # ,'res_file'
			f = pl.figure(figsize = ((24,24)))

			for i,p in enumerate(all_files):
				s = f.add_subplot(len(all_files),1,i+1)
				exec("pl.plot("+p+".data[:,mask_data], alpha = 0.05)")
				simpleaxis(s)
				spine_shift(s)
				pl.title(p,fontsize=14)
			k+=1
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'preprocessing_evaluation_run_%d.pdf'%k))


	def resample_epis(self, condition = 'PRF'):
		"""resample_epi resamples the mc'd epi files back to their functional space."""
		# create identity matrix
		np.savetxt(os.path.join(self.stageFolder(stage = 'processed/mri/reg'), 'eye.mtx'), np.eye(4), fmt = '%1.1f')
		self.logger.info('resampling epis back to functional space')
		cmds = []
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			fO = FlirtOperator(inputObject = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'] ),  referenceFileName = self.runFile(stage = 'processed/mri', run = r ))
			fO.configureApply( transformMatrixFileName = os.path.join(self.stageFolder(stage = 'processed/mri/reg'), 'eye.mtx'), outputFileName = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res'] ) ) 
			cmds.append(fO.runcmd)
		
		# run all of these resampling commands in parallel
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers)
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(fo,),(),('subprocess','tempfile',)) for fo in cmds]
		for fo in ppResults:
			fo()
		
		# now put stuff back in the right places
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','hr']) )
			os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res']) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf']) )
			
	def eye_informer(self, length_thresh, nr_dummy_scans = 6):
		"""
		- the times at which a blink began per run 
		- duration of blink
		Timings of the blinks are corrected for the start of the scan by the nr_dummy_scans
		"""

	
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			# 
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = r))
			tr = round(niiFile.rtime*1)/1000.0
			with open (self.runFile(stage = 'processed/eye', run = r, extension = '.msg')) as inputFileHandle:
				msg_file = inputFileHandle.read()


			sacc_re = 'ESACC\t(\S+)[\s\t]+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+.?\d+)'
			fix_re = 'EFIX\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?'
			blink_re = 'EBLINK\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d?.?\d*)?'
			start_eye = 'START\t(-?\d+\.?\d*)'

			# self.logger.info('reading eyelink events from %s', os.path.split(self.message_file)[-1])
			saccade_strings = re.findall(re.compile(sacc_re), msg_file)
			fix_strings = re.findall(re.compile(fix_re), msg_file)
			blink_strings = re.findall(re.compile(blink_re), msg_file)
			start_time_scan = float(re.findall(re.compile(start_eye),msg_file)[0])
			
			if len(saccade_strings) > 0:
				self.saccades_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3]),'start_x':float(e[4]),'start_y':float(e[5]),'end_x':float(e[6]),'end_y':float(e[7]), 'length':float(e[8]),'peak_velocity':float(e[9])} for e in saccade_strings]
				self.fixations_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3]),'x':float(e[4]),'y':float(e[5]),'pupil_size':float(e[6])} for e in fix_strings]
				self.blinks_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3])} for e in blink_strings]
			
				self.saccade_type_dictionary = np.dtype([(s , np.array(self.saccades_from_message_file[0][s]).dtype) for s in self.saccades_from_message_file[0].keys()])
				self.fixation_type_dictionary = np.dtype([(s , np.array(self.fixations_from_message_file[0][s]).dtype) for s in self.fixations_from_message_file[0].keys()])
				if len(self.blinks_from_message_file) > 0:
					self.blink_type_dictionary = np.dtype([(s , np.array(self.blinks_from_message_file[0][s]).dtype) for s in self.blinks_from_message_file[0].keys()])
			eye_blinks = [[((self.blinks_from_message_file[i]['start_timestamp']- start_time_scan)/1000) - nr_dummy_scans*tr, self.blinks_from_message_file[i]['duration']/1000,1] for i in range(len(self.blinks_from_message_file)) if (self.blinks_from_message_file[i]['start_timestamp']- start_time_scan) > (nr_dummy_scans*tr*1000)]
			saccades = [[((self.saccades_from_message_file[i]['start_timestamp']- start_time_scan)/1000) - nr_dummy_scans*tr, self.saccades_from_message_file[i]['duration']/1000,1] for i in range(len(self.saccades_from_message_file)) if np.all([(self.saccades_from_message_file[i]['start_timestamp']- start_time_scan) > (nr_dummy_scans*tr*1000), (self.saccades_from_message_file[i]['length'] > length_thresh)]) ]
		
			np.savetxt(self.runFile(stage = 'processed/eye', run = r, extension = '.txt', postFix = ['eye_blinks']), np.array(eye_blinks), fmt = '%3.2f', delimiter = '\t')
			np.savetxt(self.runFile(stage = 'processed/eye', run = r, extension = '.txt', postFix = ['saccades']), np.array(saccades), fmt = '%3.2f', delimiter = '\t')
			
		return saccades
	
	# def select_trials(self):
	#
	# 	# 1. open condition nifti
	# 	# 2. check whether
		
			
	def create_dilated_cortical_mask(self, dilation_sd = 0.5, label = 'cortex'):
		"""create_dilated_cortical_mask takes the rh and lh cortex files and joins them to one cortex.nii.gz file.
		it then smoothes this mask with fslmaths, using a gaussian kernel. 
		This is then thresholded at > 0.0, in order to create an enlarged cortex mask in binary format.
		"""
		self.logger.info('creating dilated %s mask with sd %f'%(label, dilation_sd))
		# take rh and lh files and join them.
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'rh.' + label + '.nii.gz'))
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), '' + label + '.nii.gz'), **{'-add': os.path.join(self.stageFolder('processed/mri/masks/anat'), 'lh.' + label + '.nii.gz')})
		fmO.execute()
		
		fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), '' + label + '.nii.gz'))
		fmO.configureSmooth(smoothing_sd = dilation_sd)
		fmO.execute()
		
		fmO = FSLMathsOperator(fmO.outputFileName)
		fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), label + '_dilated_mask.nii.gz'), **{'-bin': ''})
		fmO.execute()

	def create_early_visual_mask(self):

		mask_path = os.path.join(self.stageFolder('processed/mri/masks/anat'))
		all_rois = []
		rois_combined = zeros((29,96,96))
		for i in range(4):
			if os.path.isfile(os.path.join(mask_path,'lh.V%d.nii.gz'%(i+1))):
				exec("lh_v%d = np.array(NiftiImage(os.path.join(mask_path,'lh.V%d.nii.gz')).data,dtype=bool)"%((i+1),(i+1)) )
				all_rois.append('lh_v%d'%(i+1))
				rois_combined += eval('lh_v%d'%(i+1))
			if os.path.isfile(os.path.join(mask_path,'rh.V%d.nii.gz'%(i+1))):
				exec("rh_v%d = np.array(NiftiImage(os.path.join(mask_path,'rh.V%d.nii.gz')).data,dtype=bool)"%((i+1),(i+1)) )
				all_rois.append('rh_v%d'%(i+1))
				rois_combined += eval('rh_v%d'%(i+1))

		early_visual = np.zeros(lh_v1.shape)
		early_visual[rois_combined!=0] =1 

		new_nifti = NiftiImage(early_visual)
		new_nifti.header = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'lh.V1.nii.gz')).header
		new_nifti.save(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'early_visual.nii.gz'))



	def stimulus_timings(self, stim_offsets = [0.0, 0.0]):
		# 
		"""stimulus_timings uses behavior operators to distil:
		- the times at which stimulus presentation began and ended per task type
		- the times at which the task buttons were pressed. 
		"""
		
		for r in [self.runList[i] for i in self.conditionDict['PRF']]:
			bO = PopulationReceptiveFieldBehaviorOperator(self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' ))
			bO.trial_times(stim_offsets = stim_offsets) # sets up all behavior  
			r.trial_times = bO.trial_times
			saccades=self.eye_informer(length_thresh = 5)
			for ti, tb in enumerate(r.trial_times):
				# subtract 2.5 seconds from every trial onset:
				# r.trial_times[ti][1] -= 0.5
				# r.trial_times[ti][2] -= 0.5
				# change trials to fix_no_stim
				# for s in saccades:
# 					if np.all([ (s[0] > tb[1]), (s[0] < tb[2]) ]):
# 						r.trial_times[ti][0] = 'fix_no_stim'
				pass
			r.all_button_times = bO.all_button_times
			r.parameters = bO.parameters
			# if r.tasks = []
			r.tasks = [t.task for t in bO.trials]
			r.orientations = [t.parameters['orientation'] for t in bO.trials]
			tasks = list(np.unique(np.array([tt[0] for tt in r.trial_times])))
			# 
			these_trials = np.array([[tt[1], tt[2] - tt[1], 1.0] for tt in r.trial_times])
			np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['trial_times']), these_trials, fmt = '%3.2f', delimiter = '\t')
			for task in tasks:
				these_trials = np.array([[tt[1], tt[2] - tt[1], 1.0] for tt in r.trial_times if tt[0] == task])
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]), these_trials, fmt = '%3.2f', delimiter = '\t')
				these_buttons = np.array([[float(bt[1]), 0.5, 1.0] for bt in r.all_button_times if bt[0] == task])
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['button', task]), these_buttons, fmt = '%3.2f', delimiter = '\t')

	def stimulus_timings_square(self, specific_direction=True,stimulus_correction = 0):

		run_start_time = []
		run_duration = []
		orientations = [0,45,90,135,180,225,270,315]
		for ri, r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]):
			filename = self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' )
			with open(filename) as f:
				picklefile = pickle.load(f)
			run_start_time_string = [e for e in picklefile['eventArray'][0] if e[:len('trial 0 phase 1')] == 'trial 0 phase 1']
			run_start_time.append(float(run_start_time_string[0].split(' ')[-1]))

			# niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = r)) 
			# tr  = round(niiFile.rtime*1)/1000.0
			# if ri == 0:
			# 	run_duration.append(0)
			# else:
			# 	run_duration.append(round(niiFile.rtime*1)/1000.0 * niiFile.timepoints)

			# corrected_durations = np.cumsum(np.array(run_duration))

			task_per_trial = np.array([picklefile['parameterArray'][i]['task'] for i in range(len(picklefile['parameterArray'])) ])
			orientation_per_trial = np.array([picklefile['parameterArray'][i]['motion_direction'] for i in range(len(picklefile['parameterArray'])) ])

			stimulus_correction_dict = {0:6.75,45:5.5,90:5,135:5.5,180:6.75,225:5.5,270:5,315:5.5}
			stimulus_correction_per_trial = [stimulus_correction_dict[orient] for orient in orientation_per_trial]
			trial_start_times = np.concatenate([np.array([float(e.split('at ')[-1]) for e in picklefile['eventArray'][i] if e[0] == 't' and "phase 2" in e]) for i in range(len(picklefile['eventArray']))]) - run_start_time[ri]
			trial_end_times = np.concatenate([np.array([float(e.split('at ')[-1]) for e in picklefile['eventArray'][i] if e[0] == 't' and "phase 3" in e]) for i in range(len(picklefile['eventArray']))]) - run_start_time[ri]

			# trial_start_times = np.concatenate([np.array([float(e.split('at ')[-1]) for e in picklefile['eventArray'][i] if e[0] == 't' and "phase 2" in e]) for i in range(len(picklefile['eventArray']))]) - run_start_time[ri] + corrected_durations[ri]
			# trial_end_times = np.concatenate([np.array([float(e.split('at ')[-1]) for e in picklefile['eventArray'][i] if e[0] == 't' and "phase 3" in e]) for i in range(len(picklefile['eventArray']))]) - run_start_time[ri] + corrected_durations[ri]
			trial_durations = trial_end_times - trial_start_times
			r.trial_duration = np.median(trial_durations)

			fix_no_stim_times = np.dstack([ trial_start_times[task_per_trial==0], trial_durations[task_per_trial==0],np.ones(len(task_per_trial[task_per_trial==0]))])[0] 
			fix_stim_times = np.dstack([ trial_start_times[task_per_trial==1], trial_durations[task_per_trial==1],np.ones(len(task_per_trial[task_per_trial==1]))])[0] 
			for orient in unique(orientation_per_trial):
				indices = np.all([task_per_trial==1, orientation_per_trial==orient],axis=0)
				exec("fix_stim_times_%d = np.dstack([ trial_start_times[indices], trial_durations[indices],np.ones(np.sum(indices))])[0]"%(orient))

			# save to text
			np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['fix_no_stim']), fix_no_stim_times, fmt = '%3.2f', delimiter = '\t')
			np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['fix_stim']), fix_stim_times, fmt = '%3.2f', delimiter = '\t')
			for orient in unique(orientation_per_trial):
				this_param = "fix_stim_times_%d"%orient
				this_save_name = "fix_stim_%d"%orient
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [this_save_name]), eval(this_param), fmt = '%3.2f', delimiter = '\t')

			# save to run 

			trial_names = [np.tile(['fix_no_stim','fix_stim'],(40,1))[rint][task_per_trial[rint]] for rint in range(len(task_per_trial))]
			if specific_direction:
				trial_names = [trial_names[i] + '_' + str(orientation_per_trial[i]) if str(trial_names[i]) != 'fix_no_stim' else trial_names[i] for i in range(len(trial_names)) ]
 			r.trial_times = [ [trial_names[i], trial_start_times[i], trial_end_times[i]] for i in range(len(trial_names))]

			r.orientations = [np.radians(t['motion_direction']) for t in picklefile['parameterArray'] ]

	def stimulus_timings_square_2(self, specific_direction=True,stimulus_correction = 0):

		run_start_time = []
		run_duration = []
		orientations = [0,45,90,135,180,225,270,315]
		add_time_for_previous_runs = 0
		for ri, r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]):
			filename = self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' )
			with open(filename) as f:
				picklefile = pickle.load(f)
			run_start_time_string = [e for e in picklefile['eventArray'][0] if e[:len('trial 0 phase 1')] == 'trial 0 phase 1']
			run_start_time.append(float(run_start_time_string[0].split(' ')[-1]))

		
 

			# niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = r)) 
			# tr  = round(niiFile.rtime*1)/1000.0
			# if ri == 0:
			# 	run_duration.append(0)
			# else:
			# 	run_duration.append(round(niiFile.rtime*1)/1000.0 * niiFile.timepoints)

			# corrected_durations = np.cumsum(np.array(run_duration)

			orientation_per_trial=np.array([int(np.degrees(picklefile['parameterArray'][i]['orientation'])) for i in range(len(picklefile['parameterArray'])) ])
			task_per_trial = np.array([ 0 if 'fix_no_stim' in picklefile['parameterArray'][i]['task'] else 1 for i in range(len(picklefile['parameterArray'])) ])
			
			stimulus_correction_dict = {0:6.75,45:5.5,90:5,135:5.5,180:6.75,225:5.5,270:5,315:5.5}
			stimulus_correction_per_trial = [stimulus_correction_dict[int(orient)] for orient in orientation_per_trial]
			trial_start_times = np.concatenate([np.array([float(e.split('at ')[-1]) for e in picklefile['eventArray'][i] if e[0] == 't' and "phase 2" in e]) for i in range(len(picklefile['eventArray']))]) - run_start_time[ri]  + stimulus_correction
			trial_end_times = np.concatenate([np.array([float(e.split('at ')[-1]) for e in picklefile['eventArray'][i] if e[0] == 't' and "phase 3" in e]) for i in range(len(picklefile['eventArray']))]) - run_start_time[ri]  + stimulus_correction


			# trial_start_times = np.concatenate([np.array([float(e.split('at ')[-1]) for e in picklefile['eventArray'][i] if e[0] == 't' and "phase 2" in e]) for i in range(len(picklefile['eventArray']))]) - run_start_time[ri] + corrected_durations[ri]
			# trial_end_times = np.concatenate([np.array([float(e.split('at ')[-1]) for e in picklefile['eventArray'][i] if e[0] == 't' and "phase 3" in e]) for i in range(len(picklefile['eventArray']))]) - run_start_time[ri] + corrected_durations[ri]
			trial_durations = trial_end_times - trial_start_times
			r.trial_duration = np.median(trial_durations)

			fix_no_stim_times = np.dstack([ trial_start_times[task_per_trial==0], trial_durations[task_per_trial==0],np.ones(len(task_per_trial[task_per_trial==0]))])[0] 
			fix_stim_times = np.dstack([ trial_start_times[task_per_trial==1], trial_durations[task_per_trial==1],np.ones(len(task_per_trial[task_per_trial==1]))])[0] 
			for orient in unique(orientation_per_trial):
				indices = np.all([task_per_trial==1, orientation_per_trial==orient],axis=0)
				exec("fix_stim_times_%d = np.dstack([ trial_start_times[indices], trial_durations[indices],np.ones(np.sum(indices))])[0]"%(orient))

			# save to text
			np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['fix_no_stim']), fix_no_stim_times, fmt = '%3.2f', delimiter = '\t')
			np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['fix_stim']), fix_stim_times, fmt = '%3.2f', delimiter = '\t')
			for orient in unique(orientation_per_trial):
				this_param = "fix_stim_times_%d"%orient
				this_save_name = "fix_stim_%d"%orient
				np.savetxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [this_save_name]), eval(this_param), fmt = '%3.2f', delimiter = '\t')

			# save to run 

			trial_names = [np.tile(['fix_no_stim','fix_stim'],(48,1))[rint][task_per_trial[rint]] for rint in range(len(task_per_trial))]
			if specific_direction:
				trial_names = [trial_names[i] + '_' + str(orientation_per_trial[i]) if str(trial_names[i]) != 'fix_no_stim' else trial_names[i] for i in range(len(trial_names)) ]
			r.trial_times = [ [trial_names[i], trial_start_times[i], trial_end_times[i]] for i in range(len(trial_names))]

			r.orientations = [t['orientation'] for t in picklefile['parameterArray'] ]

			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','prZ'] ))
			add_time_for_previous_runs += nii_file.getRepetitionTime() * nii_file.getTimepoints()
			
	
	def physio(self, condition = 'PRF'):
		"""physio loops across runs to analyze their physio data"""
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			pO = PhysioOperator(self.runFile(stage = 'processed/hr', run = r, extension = '.log' ))
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] ))
			if nii_file.rtime > 10:
				TR = nii_file.rtime / 1000.0
			else:
				TR = nii_file.rtime
			pO.preprocess_to_continuous_signals(TR = TR, nr_TRs = nii_file.timepoints)
	
	def GLM_for_nuisances(self, condition = 'PRF', physiology_type = 'RETROICOR', postFix = ['mcf', 'sgtf', 'prZ']):
		"""GLM_for_nuisances takes a diverse set of nuisance regressors,
		runs a GLM on them in order to run further PRF analysis on the 
		residuals after GLM. It assumes physio, motion correction and 
		stimulus_timings have been run beforehand, as it uses the output
		text files of these procedures.
		"""
		# 
		if 'Square' in self.project.projectName: self.stimulus_timings_square()
		else: self.stimulus_timings()
		
		self.eye_informer(length_thresh=5)
		# physio regressors
		physio_list = []
		mcf_list = []
		trial_times_list = []
		button_times_list = []
		blink_times_list = []
		total_trs  = 0
		for j, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			# 
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			# moco and physiology regressors are per-TR regressors that need no convolution anymore.
			if physiology_type != None and physiology_type == 'RETROICOR':
				physio_list.append(np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['regressors']) ).T)
			elif physiology_type != None:
				physio_list.append(np.array([
					np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp']) ),
					np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu']) ),
					np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp', 'raw']) ),
					np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu', 'raw']) )
					]))
				
			mcf_list.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' )))
			# final regressor captures instruction-related variance that may otherwise cause strong responses in periphery
			# trial_times are single events that have to still be convolved with HRF
			# 
			trial_times_list.extend([[[(j * nii_file.rtime * nii_file.timepoints) + tt[1], 1.5, 1.0]] for tt in r.trial_times]) 

			# button_times_list.extend([[[(j * nii_file.rtime * nii_file.timepoints) + float(tt[1]), 0.5, 1.0]] for tt in r.all_button_times]) # changed the occurrence of this event to -4.5 to -1.5...
			# lateron, this will also have pupil size and the occurrence of saccades in there.
			# this_blink_events = np.loadtxt(self.runFile(stage = 'processed/eye', run = r, extension = '.txt', postFix = ['eye_blinks']))
			# blink_times_list.extend([[[(j * nii_file.rtime * nii_file.timepoints) + float(tt[0]), tt[1],tt[2]]] for tt in this_blink_events])
			
			total_trs += nii_file.timepoints
		# total_trs = 1944
		# 
		# to arrays with these regressors
		mcf_list = np.vstack(mcf_list).T
		if physiology_type!= None: physio_list = np.hstack(physio_list)
		
		# check for weird nans and throw out those columns
		if physiology_type!= None: physio_list = physio_list[-np.array(np.isnan(physio_list).sum(axis = 1), dtype = bool),:]
		# 
		# create a design matrix and convolve 
		run_design = Design(total_trs, nii_file.rtime, subSamplingRatio = 10)
		run_design.configure(trial_times_list)
		# run_design.configure(blink_times_list)
		# run_design.configure([np.array(button_times_list).squeeze()])
		if physiology_type != None: joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, mcf_list, physio_list]).T)
		else: joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, mcf_list]).T)
		# joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, physio_list]).T)
		# only using the mc and physio now
		# joined_design_matrix = np.mat(np.vstack([mcf_list, physio_list]).T)
		# joined_design_matrix = np.mat(physio_list.T)
		# 
		# 
		self.logger.info('nuisance and trial_onset design_matrix of dimensions %s'%(str(joined_design_matrix.shape)))
		# take data
		data_list = []
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			# self.logger.info('per-condition Z-score of run %s' % r)
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			data_list.append(nii_file.data[:,cortex_mask])
		data_list = np.vstack(data_list)
		# now we run the GLM
		self.logger.info('nifti data loaded from %s for nuisance/trial onset analysis'%(self.runFile(stage = 'processed/mri', run = r, postFix = postFix )))
		
		betas = ((joined_design_matrix.T * joined_design_matrix).I * joined_design_matrix.T) * np.mat(data_list.T).T
		residuals = data_list - (np.mat(joined_design_matrix) * np.mat(betas))
		
		self.logger.info('GLM finished; outputting data to %s and %s'%(os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res']))[-1], os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['betas']))[-1]))
		# and now, back to image files
		for i, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			output_data_res = np.zeros(nii_file.data.shape, dtype = np.float32)
			output_data_res[:,cortex_mask] = residuals[i*nii_file.data.shape[0]:(i+1)*nii_file.data.shape[0],:]
			
			res_nii_file = NiftiImage(output_data_res)
			res_nii_file.header = nii_file.header
			res_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res']))
			
			output_data_betas = np.zeros([betas.shape[0]]+list(cortex_mask.shape), dtype = np.float32)
			output_data_betas[:,cortex_mask] = betas
			
			betas_nii_file = NiftiImage(output_data_betas)
			betas_nii_file.header = nii_file.header
			betas_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['betas']))
			
		# 

	def GLM_for_nuisances_per_run(self, condition = 'PRF', postFix = ['mcf', 'sgtf']):
		"""GLM_for_nuisances takes a diverse set of nuisance regressors,
		runs a GLM on them in order to run further PRF analysis on the 
		residuals after GLM. It assumes physio, motion correction and 
		stimulus_timings have been run beforehand, as it uses the output
		text files of these procedures.
		"""
		if 'Square' in self.project.projectName: self.stimulus_timings_square()
		else: self.stimulus_timings()
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		for j, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			self.logger.info('nifti data loaded from %s for nuisance/trial onset analysis'%(self.runFile(stage = 'processed/mri', run = r, postFix = postFix )))
			# moco and physiology regressors are per-TR regressors that need no convolution anymore.
			physio = np.array([
				np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp']) ),
				np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu']) ), 
				np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp', 'raw']) ),
				np.loadtxt(self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu', 'raw']) )
				])
				
			mcf = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' ))
			# final regressor captures instruction-related variance that may otherwise cause strong responses in periphery
			# trial_times are single events that have to still be convolved with HRF
			instruct_times = [[[tt[1] - 3.0, 3.0, 1.0]] for tt in r.trial_times]
			trial_onset_times = [[[tt[1], 0.5, 1.0]] for tt in r.trial_times]
			# lateron, this will also have pupil size and the occurrence of saccades in there.
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/eye', run = r, extension = '.txt', postFix = ['eye_blinks']))
			blink_times_list = np.array([[[float(tt[0]), tt[1],tt[2]]] for tt in this_blink_events])
			blink_times_list = blink_times_list.reshape((1,blink_times_list.shape[0], blink_times_list.shape[-1]))

			instruct_times.extend(blink_times_list)
			# 

			run_design = Design(nii_file.timepoints, nii_file.rtime, subSamplingRatio = 10)
			# run_design.configure(np.vstack([instruct_times, blink_times_list]), hrfType = 'doubleGamma', hrfParameters = {'a1' : 6, 'a2' : 12, 'b1' : 0.9, 'b2' : 0.9, 'c' : 0.35})
			run_design.configure(instruct_times, hrfType = 'doubleGamma', hrfParameters = {'a1' : 6, 'a2' : 12, 'b1' : 0.9, 'b2' : 0.9, 'c' : 0.35})
			joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, mcf.T, physio]).T)
			
			f = pl.figure(figsize = (10, 10))
			s = f.add_subplot(111)
			pl.imshow(joined_design_matrix)
			s.set_title('nuisance design matrix')
			pl.savefig(self.runFile(stage = 'processed/mri', run = r, postFix = postFix, base = 'nuisance_design', extension = '.pdf' ))
			
			self.logger.info('nuisance and trial_onset design_matrix of dimensions %s for run %s'%(str(joined_design_matrix.shape), r))
			betas = ((joined_design_matrix.T * joined_design_matrix).I * joined_design_matrix.T) * np.mat(nii_file.data[:,cortex_mask].T).T
			residuals = nii_file.data[:,cortex_mask] - (np.mat(joined_design_matrix) * np.mat(betas))
			
			self.logger.info('nuisance GLM finished; outputting residuals to %s'%os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res']))[-1])
			output_data_res = np.zeros(nii_file.data.shape, dtype = np.float32)
			output_data_res[:,cortex_mask] = residuals
			
			res_nii_file = NiftiImage(output_data_res)
			res_nii_file.header = nii_file.header
			res_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res']))
			
			self.logger.info('nuisance GLM finished; outputting betas to %s'%os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['betas']))[-1])
			output_data_betas = np.zeros([betas.shape[0]]+list(cortex_mask.shape), dtype = np.float32)
			output_data_betas[:,cortex_mask] = betas
			
			betas_nii_file = NiftiImage(output_data_betas)
			betas_nii_file.header = nii_file.header
			betas_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['betas']))
		
		# 
	
	def zscore_timecourse_per_condition(self, dilate_width = 2, condition = 'PRF', postFix = ['mcf', 'sgtf']):
		"""fit_voxel_timecourse loops over runs and for each run:
		looks when trials of a certain type occurred, 
		and dilates these times by dilate_width TRs.
		The data in these TRs are then z-scored on a per-task basis,
		and rejoined after which they are saved.
		"""
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'cortex_dilated_mask.nii.gz')).data, dtype = bool)
		# loop over runs
		for r in [self.runList[i] for i in self.conditionDict[condition]]:
			self.logger.info('per-condition Z-score of run %s' % r)
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			tr_times = np.arange(0, nii_file.timepoints * nii_file.rtime, nii_file.rtime)
			masked_input_data = nii_file.data[:,cortex_mask]
			if not hasattr(r, 'trial_times'):
				if 'Square' in self.project.projectName: self.stimulus_timings_square()
				else: self.stimulus_timings()			
			tasks = list(np.unique(np.array([tt[0] for tt in r.trial_times])))
			tasks.pop(tasks.index('fix_no_stim'))
			output_data = np.zeros(list(masked_input_data.shape) + [len(tasks)])
			fix_no_stim_TRs = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = ['fix_no_stim']))
			which_trs_fix_no_stim = np.array([(tr_times > (t[0] - (dilate_width * nii_file.rtime))) * (tr_times < (t[1] + (dilate_width * nii_file.rtime))) for t in np.array([fix_no_stim_TRs[:,0] , fix_no_stim_TRs[:,0] + fix_no_stim_TRs[:,1]]).T]).sum(axis = 0, dtype = bool)
			# loop over tasks
			for i, task in enumerate(tasks):
				self.logger.info('Z-scoring of task %s' % task)
				trial_events = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]))
				which_trs_this_task = np.array([(tr_times > (t[0] - (dilate_width * nii_file.rtime))) * (tr_times < (t[1] + (dilate_width * nii_file.rtime))) for t in np.array([trial_events[:,0] , trial_events[:,0] + trial_events[:,1]]).T]).sum(axis = 0, dtype = bool)
				output_data[which_trs_this_task,:,i] = (masked_input_data[which_trs_this_task] - masked_input_data[which_trs_this_task + which_trs_fix_no_stim].mean(axis = 0)) / masked_input_data[which_trs_this_task + which_trs_fix_no_stim].std(axis = 0)
			output_data[which_trs_fix_no_stim,:,i] = (masked_input_data[which_trs_fix_no_stim] - masked_input_data.mean(axis = 0)) / masked_input_data.std(axis = 0)


			output_data = output_data.mean(axis = -1) * len(tasks)
			file_output_data = np.zeros(nii_file.data.shape, dtype = np.float32)
			file_output_data[:,cortex_mask] = output_data
			opf = NiftiImage(file_output_data)
			opf.header = nii_file.header
			self.logger.info('saving output file %s with dimensions %s' % (self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['prZ'] ), str(output_data.shape)))
			opf.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['prZ'] ))

	
	def design_matrix(self, method = 'hrf', gamma_hrfType = 'doubleGamma', gamma_hrfParameters = {'a1' : 6, 'a2' : 12, 'b1' : 0.9, 'b2' : 0.9, 'c' : 0.35}, fir_ratio = 6, n_pixel_elements = 40, sample_duration = 0.6, plot_diagnostics = False, ssr = 100, condition = 'PRF', save_design_matrix = True, orientations = [0,45,90,135,180,225,270,315],specific_direction=False,stimulus_correction=0):
		"""design_matrix creates a design matrix for the runs
		using the PRFModelRun and PRFTrial classes. The temporal grain
		of the model is specified by sample_duration. In our case, the 
		stimulus was refreshed every 600 ms. 
		method can be hrf or fir. when gamma, we can specify 
		the parameters of gamma and double-gamma, etc.
		FIR fitting is still to be implemented, as the shape of
		the resulting design matrix will differ from the HRF version.
		"""
		# other options: 
		# gamma_hrfType = 'singleGamma', gamma_hrfParameters = {'a': 6, 'b': 0.9}

		# get orientations and stimulus timings
		# 
		# self.stimulus_timings(stim_offsets = [-1.5, -0.5])
		# 
		if 'Square' in self.project.projectName: self.stimulus_timings_square(specific_direction=specific_direction,stimulus_correction=stimulus_correction)
		# else: self.stimulus_timings()
		else: self.stimulus_timings_square_2(specific_direction=specific_direction,stimulus_correction=stimulus_correction)
		self.logger.info('design_matrix of %d pixel elements and %1.2f s sample_duration and %1.2f s stimulus timing correction'%(n_pixel_elements, sample_duration,stimulus_correction))
		
		self.stim_matrix_list = []
		self.design_matrix_list = []
		self.sample_time_list = []
		self.tr_time_list = []
		self.trial_start_list = []
		for i, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			# 
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = [] ))
			if nii_file.rtime > 10:
				TR = nii_file.rtime / 1000.0
			else:
				TR = nii_file.rtime

			bar_width = 0.1
			# bar_width = 1/(0.64*float(n_pixel_elements))
			mr = PRFModelRun(r, n_TRs = nii_file.timepoints, TR = TR, n_pixel_elements = n_pixel_elements, sample_duration = sample_duration, bar_width = bar_width)

			mr.simulate_run( orientations )

			self.stim_matrix_list.append(mr.run_matrix)
			self.sample_time_list.append(mr.sample_times + i * nii_file.timepoints * TR)
			self.tr_time_list.append(np.arange(0, nii_file.timepoints * TR, TR) + i * nii_file.timepoints * TR)
			self.trial_start_list.append(np.array(np.array(r.trial_times)[:,1], dtype = float) + i * nii_file.timepoints * TR) 		
			
			# k = 1
			# trial_dur_samp =  int((r.trial_times[1][1]-r.trial_times[0][1])/sample_duration)
			# f=figure(figsize=(24,24))
			# for g in range(10):
			# 	for i in range(len(orientations)): 
			# 		o = where(np.array(r.trial_times)[:,0]=='fix_stim_%d'%orientations[i])[0][0]
			# 		s=f.add_subplot(10,len(orientations),k)
			# 		imshow(mr.run_matrix[int(int(r.trial_times[o][1]/sample_duration) + 2/sample_duration*g),:,:],interpolation='nearest',cmap='gray')
			# 		title('%d deg, %d s'%(orientations[i],(trial_dur_samp*o + 2/sample_duration*g)*sample_duration),fontsize=16)
			# 		pl.axis('off')
			# 		k += 1
			# pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'dm_unconvolved.pdf'))
 	
			# f=figure(figsize=(16,16))
			# half_trial = int(ceil((r.trial_duration)/sample_duration)/2)
			# cross = np.zeros((n_pixel_elements,n_pixel_elements))
			# for i in range(len(orientations)): 
			# 		o = where(np.array(r.trial_times)[:,0]=='fix_stim_%d'%orientations[i])[0][0]
			# 		cross += mr.run_matrix[half_trial+int(r.trial_times[o][1]/sample_duration),:,:]
			# imshow(cross,interpolation='nearest',cmap='gray')
			# pl.axis('off')
			# pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'dm_cross_unconvolved.pdf'))

			if method == 'hrf':
				run_design = Design(mr.run_matrix.shape[0], mr.sample_duration, subSamplingRatio = ssr)
				rdm = mr.run_matrix.reshape((mr.run_matrix.shape[0], mr.run_matrix.shape[1] * mr.run_matrix.shape[2])).T
				run_design.rawDesignMatrix = np.repeat(mr.run_matrix, ssr, axis=0).reshape((-1,n_pixel_elements*n_pixel_elements)).T
				run_design.convolveWithHRF(hrfType = gamma_hrfType, hrfParameters = gamma_hrfParameters)
				workingDesignMatrix = run_design.designMatrix

				# k = 1
				# f=figure(figsize=(24,24))
				# for g in range(10):
				# 	for i in range(len(orientations)): 
				# 		o = where(np.array(r.trial_times)[:,0]=='fix_stim_%d'%orientations[i])[0][0]				
				# 		s=f.add_subplot(10,8,k)
				# 		imshow(np.reshape(workingDesignMatrix,(n_pixel_elements,n_pixel_elements,-1))[:,:,int(int(r.trial_times[o][1]/sample_duration) + 2/sample_duration*g)],interpolation='nearest',cmap='gray')
				# 		clim(-1000,3500)
				# 		title('%s deg, %d s'%(r.trial_times[i][0][9:],2/sample_duration*g*sample_duration),fontsize=22)
				# 		pl.axis('off')
				# 		k += 1
				# pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'dm_convolved.pdf'))
	 	
				# f = figure(figsize=(16,16))
				# dm=np.reshape(workingDesignMatrix,(n_pixel_elements,n_pixel_elements,-1))
				# trial_mid = int(r.trial_duration/2/sample_duration+(5/sample_duration))# (half_trial + 100 HRF delay (5s))
				# cross = np.zeros((n_pixel_elements,n_pixel_elements))
				# for i in range(len(orientations)): 
				# 	o = where(np.array(r.trial_times)[:,0]=='fix_stim_%d'%orientations[i])[0][0]
				# 	cross += dm[:,:,trial_mid+int(r.trial_times[o][1]/sample_duration)]
				# # cross = dm[:,:,trial_mid] + dm[:,:,trial_mid+trial_dur_samp*1]+dm[:,:,trial_mid+trial_dur_samp*3]+dm[:,:,trial_mid+trial_dur_samp*4]+dm[:,:,trial_mid+trial_dur_samp*6]+dm[:,:,trial_mid+trial_dur_samp*8]+dm[:,:,trial_mid+trial_dur_samp*10]+dm[:,:,trial_mid+trial_dur_samp*14]
				# imshow(cross,interpolation='nearest',cmap='gray')
				# pl.axis('off')
				# pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'dm_cross_convolved.pdf'))
				
				# new_run_design = NewDesign(mr.run_matrix.shape[0], mr.sample_duration, sample_duration = 0.01)
				# rdm = mr.run_matrix.reshape((mr.run_matrix.shape[0], mr.run_matrix.shape[1] * mr.run_matrix.shape[2])).T
				# new_run_design.raw_design_matrix = np.repeat(rdm, int(mr.sample_duration / 0.01), axis = 1)
				# new_run_design.convolve_with_HRF(hrf_type = gamma_hrfType, hrf_parameters = gamma_hrfParameters)
				# workingDesignMatrix = new_run_design.design_matrix

				# 
			elif method == 'fir':
				new_size = list(mr.run_matrix.shape)
				new_size[0] *= int(fir_ratio)
				new_array = np.zeros(new_size)
				for i in np.arange(mr.run_matrix.shape[0]) * int(fir_ratio):
					new_array[i:i+int(fir_ratio)] = mr.run_matrix[int(floor(i/int(fir_ratio)))]
				workingDesignMatrix = new_array
			
			# 
			self.design_matrix_list.append(workingDesignMatrix)
		self.full_design_matrix = np.hstack(self.design_matrix_list).T
		self.full_design_matrix = np.array(self.full_design_matrix - self.full_design_matrix.mean(axis = 0) )# / self.full_design_matrix.std(axis = 0)
		self.tr_time_list = np.concatenate(self.tr_time_list)
		self.sample_time_list = np.concatenate(self.sample_time_list)
		self.trial_start_list = np.concatenate(self.trial_start_list)
		self.logger.info('design_matrix of shape %s created, of which %d are valid stimulus locations'%(str(self.full_design_matrix.shape), int((self.full_design_matrix.sum(axis = 0) != 0).sum())))
		
		# 
		if plot_diagnostics:
			f = pl.figure(figsize = (10, 3))
			s = f.add_subplot(111)
			pl.plot(self.full_design_matrix.sum(axis = 1))
			pl.plot(self.trial_start_list / sample_duration, np.ones(self.trial_start_list.shape), 'ko')
			s.set_title('original')
			s.axis([0,200,0,200])

		if save_design_matrix:
			with open(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'design_matrix_%1.2f_%ix%i_%s.pickle'%(sample_duration, n_pixel_elements, n_pixel_elements, method)), 'w') as f:
				pickle.dump({'tr_time_list' : self.tr_time_list, 'full_design_matrix' : self.full_design_matrix, 'sample_time_list' : self.sample_time_list, 'trial_start_list' : self.trial_start_list} , f)
		
	def stats_to_mask(self, mask_file_name, postFix = ['mcf', 'sgtf', 'prZ', 'res'], condition = 'PRF', task_condition = ['all'], threshold = 5.0):
		"""stats_to_mask takes the stats from an initial fitting and converts it to a anatomical mask, and places it in the masks/anat folder"""
		input_file = os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_condition[0] + '-' + condition + '.nii.gz')
		p_values = NiftiImage(input_file).data[1] > threshold
		self.logger.info('statistic mask created for threshold %2.2f, resulting in %i voxels' % (threshold, int(p_values.sum())))
		output_image = NiftiImage(np.array(p_values, dtype = np.int16))
		output_image.header = NiftiImage(input_file).header
		output_image.save(os.path.join(self.stageFolder('processed/mri/masks/anat'), mask_file_name + '_' + task_condition[0] + '.nii.gz'))

	def fit_PRF(self, n_pixel_elements = 30, mask_file_name = 'single_voxel', postFix = ['mcf', 'sgtf', 'prZ', 'res'], n_jobs = 15, task_conditions = ['fix'], condition = 'PRF', sample_duration = 0.15, save_all_data = True, orientations = [0,45,90,135,180,225,270,315],specific_direction=False,delve_deeper=False,method='old',stimulus_correction=0): # cortex_dilated_mask
		"""fit_PRF creates a design matrix for the full experiment, 
		with n_pixel_elements determining the amount of singular pixels in the display in each direction.
		fit_PRF uses a parallel joblib implementation of the Bayesian Ridge Regression from sklearn
		http://en.wikipedia.org/wiki/Ridge_regression
		http://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression
		mask_single_file allows the user to set a binary mask to mask the functional data
		for fitting of individual regions.
		"""
		# 
		orient_list = ''
		for i in range(len(orientations)):
			orient_list += '_' + str(orientations[i])
		# we need a design matrix (can't just load in existing one because of other global variables created in the function)
		self.design_matrix(n_pixel_elements = n_pixel_elements, condition = condition, sample_duration = sample_duration, orientations = orientations, specific_direction=specific_direction,stimulus_correction=stimulus_correction)
		valid_regressors = self.full_design_matrix.sum(axis = 0) != 0
		self.full_design_matrix = self.full_design_matrix[:,valid_regressors]
		
		mask_file = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), mask_file_name +  '.nii.gz'))
		cortex_mask = np.array(mask_file.data, dtype = bool)
		
		# numbered slices, for sub-TR timing to follow stimulus timing. 
		slices = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T[cortex_mask]
		slices_in_full = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T
		
		data_list = []
		for i, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			data_list.append(nii_file.data[:,cortex_mask])
			if nii_file.rtime > 10:
				self.TR = nii_file.rtime / 1000.0
			else:
				self.TR = nii_file.rtime

		tasks = list(np.unique(np.array([tt[0] for tt in r.trial_times])))
		z_data = np.array(np.vstack(data_list), dtype = np.float32)
		# get rid of the raw data list that will just take up memory
		del(data_list)
		self.logger.info('data for PRF model fits read')
		# do the separation based on condition
		# loop over tasks

		if method == 'old':
			task_tr_times = np.zeros((len(tasks), self.tr_time_list.shape[0]))
			task_sample_times = np.zeros((len(tasks), self.sample_time_list.shape[0]))
			dilate_width = 5.0 # in seconds
			for i, task in enumerate(tasks):
				add_time_for_previous_runs = 0.0
				trial_events = []
				for j, r in enumerate([self.runList[k] for k in self.conditionDict['PRF']]):
					this_nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
					trial_events.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]))[:,0] + add_time_for_previous_runs)
					trial_duration = np.median(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.txt', postFix = [task]))[:,1])
					add_time_for_previous_runs += self.TR * this_nii_file.timepoints
				trial_events = np.concatenate(trial_events)
				task_tr_times[i] = np.array([(self.tr_time_list > (t - dilate_width)) * (self.tr_time_list < (t + dilate_width + trial_duration)) for t in trial_events]).sum(axis = 0, dtype = bool)
				task_sample_times[i] = np.array([(self.sample_time_list > (t - dilate_width)) * (self.sample_time_list < (t + dilate_width + trial_duration)) for t in trial_events]).sum(axis = 0, dtype = bool)
			# what conditions are we asking for?
			if task_conditions == ['all']:
				selected_tr_times = task_tr_times.sum(axis = 0, dtype = bool)
				selected_sample_times = task_sample_times.sum(axis = 0, dtype = bool)
			else: # only one condition is selected, which means we must add fix_no_stim
				if specific_direction:
					all_conditions = task_conditions
				else:
					all_conditions = task_conditions + ['fix_no_stim']
				selected_tr_times = task_tr_times[[tasks.index(c) for c in all_conditions]].sum(axis = 0, dtype = bool)
				selected_sample_times = task_sample_times[[tasks.index(c) for c in all_conditions]].sum(axis = 0, dtype = bool)
		
		self.logger.info('timings for PRF model fits calculated')

		# set up empty arrays for saving the data
		all_coefs = np.zeros([int(valid_regressors.sum())] + list(cortex_mask.shape))
		all_corrs = np.zeros([2] + list(cortex_mask.shape))
		all_predicted = np.zeros([208] + list(cortex_mask.shape))
		all_data = np.zeros([208] + list(cortex_mask.shape))
		# all_predicted = np.zeros([selected_tr_times.sum()] + list(cortex_mask.shape))
		# all_data = np.zeros([selected_tr_times.sum()] + list(cortex_mask.shape))
		# all_predicted=[]
		self.logger.info('PRF model fits on %d voxels' % int(cortex_mask.sum()))
		# run through slices, each slice having a certain timing
		for sl in np.arange(cortex_mask.shape[0]):
			voxels_in_this_slice = (slices == sl)
			voxels_in_this_slice_in_full = (slices_in_full == sl)
			if voxels_in_this_slice.sum() > 0:
				these_tr_times = self.tr_time_list + sl * (self.TR / float(cortex_mask.shape[0])) # was shape[1], but didn't make sense, so changed to shape[0] (amount of slices instead of voxels)
				these_voxels = z_data[:,voxels_in_this_slice].T
				# closest sample in designmatrix
				if method == 'old':
					these_samples = np.array([np.argmin(np.abs(self.sample_time_list - t)) for t in these_tr_times[selected_tr_times]]) 
					this_design_matrix = np.array(self.full_design_matrix[these_samples,:], dtype = np.float64, order = 'F')

				self.logger.info('starting fitting of slice %d, with %d voxels ' % (sl, int((cortex_mask * voxels_in_this_slice_in_full).sum())))

				#######################################################################
				## simple smoothing method:

				# smoothed_data = np.array([savitzky_golay(these_voxels[v,selected_tr_times],15,1) for v in range(np.shape(these_voxels)[0])])
				# raw_data = these_voxels[:,selected_tr_times]
				
				# # plot(these_voxels[5,selected_tr_times],'--k',alpha=0.2)
				# # plot(smoothed_data[5],'r',linewidth=3)

				# voxno=15
				# a = fitBayesianRidge(self.full_design_matrix[these_samples,:],smoothed_data[voxno])
				# b = fitBayesianRidge(self.full_design_matrix[these_samples,:],raw_data[voxno])
				# plot(raw_data[voxno],'--r',alpha=0.2);plot(smoothed_data[voxno],'--k',alpha=0.5);plot(a[2],'r');plot(b[2],'k')
				
				##
				#######################################################################
				#######################################################################
				## SPLICE METHOD:

				# there are three types of data: (1) BOLD signal (2) TR times that go with the BOLD signals (3) sample times from design matrix
				# we want to get all tr times per condition, sort them on order, then sort the BOLD signal with same sequence and find sample times that go with them
				if method == 'new':
					# stimulus_correction = -7
					# for stimulus_correction in np.arange(-10,10,0.5):
						# for alpha in [1e9,1e11,1e13,1e15]:#np.arange(-3,0,0.25):
					print stimulus_correction
					dilate_width = 5 # trs after stimulus disappearance to include in tria				
					hdl_correction = 6
					trial_duration = r.trial_duration
					trs_in_trial = round(trial_duration/self.TR)
					add_time_for_previous_runs = 0
					trial_start_times = [];trial_names=[]
					for j, r in enumerate([self.runList[k] for k in self.conditionDict['PRF']]):
						this_nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
						trial_start_times.append(np.array(r.trial_times)[:,1].astype('float32') + add_time_for_previous_runs)
						trial_names.append(np.array(r.trial_times)[:,0])
						add_time_for_previous_runs += self.TR * this_nii_file.timepoints
					trial_names = np.ravel(np.array(trial_names))[1:-1]
					trial_start_times = np.ravel(np.array(trial_start_times))[1:-1]
					extended_period = int(trs_in_trial+dilate_width) 
					pad_trs_4_smooth=5
					filter_width = 17
					
					#initialization
					all_smoothed_data = []
					# all_spliced_data = []
					# all_corrected_sample_time_ind = []
					all_dm = []

					for i, orient in enumerate(orientations):
						
						# stimulus_correction = {0:-6.75,45:-5.5,90:-5,135:-5.5,180:-6.75,225:-5.5,270:-5,315:-5.5}[orient]

						trial_start_one_dir = trial_start_times[np.where(trial_names=='fix_stim_%s'%orient)] + hdl_correction 
						trials_this_dir = size(where(trial_names=='fix_stim_%s'%orient))

						# select BOLD timepoints:
						corrected_tr_times = np.hstack(np.array([ np.hstack([(these_tr_times - t)[(these_tr_times-t) < 0][-pad_trs_4_smooth:],(these_tr_times - t)[(these_tr_times-t) > 0][:extended_period+pad_trs_4_smooth]]) for t in trial_start_one_dir]))
						sorted_tr_times = np.sort(corrected_tr_times)
						tr_order = np.argsort(corrected_tr_times)

						# smooth and resample selected data
						orig_tr_ind = np.hstack(np.array([ np.hstack([np.arange(these_tr_times.shape[0])[(these_tr_times-t) < 0][-pad_trs_4_smooth:],np.arange(these_tr_times.shape[0])[(these_tr_times-t) > 0][:extended_period+pad_trs_4_smooth]])  for t in trial_start_one_dir]))
						concatenated_data = these_voxels[:,orig_tr_ind]
						spliced_data = concatenated_data[:,tr_order]
						trimmed_spliced_data = spliced_data[:,pad_trs_4_smooth*trials_this_dir:-pad_trs_4_smooth*trials_this_dir]
						smoothed_data = np.array( [savitzky_golay(s,filter_width,1) for s in spliced_data] )
						trimmed_smoothed_data = smoothed_data[:,pad_trs_4_smooth*trials_this_dir:-pad_trs_4_smooth*trials_this_dir]
						resampled_signal = resample(trimmed_smoothed_data,extended_period,axis=1)

						# find according median dm samples
						# orig_tr_ind = np.hstack(np.array([ np.arange(these_tr_times.shape[0])[(these_tr_times-t) > 0 ][:extended_period] for t in trial_start_one_dir]))
						sorted_orig_tr_times = these_tr_times[orig_tr_ind][tr_order][pad_trs_4_smooth*trials_this_dir:-pad_trs_4_smooth*trials_this_dir]
						corrected_sample_time_ind = np.array([np.argmin(np.abs(self.sample_time_list - (t+stimulus_correction))) for t in sorted_orig_tr_times])
						dm = np.median(np.array([ self.full_design_matrix[corrected_sample_time_ind.astype('int32')[np.arange(i,len(corrected_sample_time_ind),trials_this_dir)],:] for i in range(trials_this_dir) ]),0)

						# all_spliced_data.append(trimmed_spliced_data)
						all_smoothed_data.append(resampled_signal)
						all_dm.append(dm)

					# all_spliced_data = np.reshape(np.swapaxes(np.array(all_spliced_data),0,1),(voxels_in_this_slice.sum(),-1))
					all_smoothed_data = np.reshape(np.swapaxes(np.array(all_smoothed_data),0,1),(voxels_in_this_slice.sum(),-1))
					all_dm = np.reshape(np.array(all_dm),(-1,valid_regressors.sum()))
					# all_corrected_sample_time_ind = np.reshape(np.array(all_corrected_sample_time_ind),-1)
					# resampled_signal = resample(all_smoothed_data,all_smoothed_data.shape[1]/trials_per_dir,axis=1)
					# mean_dm = np.median(np.array([ self.full_design_matrix[all_corrected_sample_time_ind.astype('int32')[np.arange(i,len(all_corrected_sample_time_ind),trials_per_dir)],:] for i in range(trials_per_dir) ]),0)
					# mean_dm_upscaled = resample(mean_dm,mean_dm.shape[0]*trials_per_dir)
					alpha = 1e13

							# if sl == 7:
							# pl.close('all')
							# voxno = 0
							# a=fitRidge(all_dm,all_smoothed_data[voxno,:],alpha=alpha)
							# prf = np.zeros(n_pixel_elements**2)		
							# prf[valid_regressors]=a[0]
							# f=figure(figsize=(16,16))
							# imshow(np.reshape(prf,(n_pixel_elements,n_pixel_elements)),interpolation='nearest',cmap='gray')
							# pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'PRF_voxel_%d_%d_%1.1e.pdf'%(voxno,stimulus_correction,alpha)))

					# if n_pixel_elements < 10: alpha = 1e8
					# elif (n_pixel_elements > 10) * (n_pixel_elements < 20): alpha = 1e9
					# elif (n_pixel_elements > 20) * (n_pixel_elements < 30): alpha = 1e11
					# elif (n_pixel_elements > 30) * (n_pixel_elements < 40): alpha = 1e13
					# elif (n_pixel_elements > 40) * (n_pixel_elements < 50): alpha = 1e15
					

					# timepoints = np.arange(104*4,104*8)
	

					if delve_deeper:

						tr_block = all_smoothed_data.shape[1]/len(orientations)
						n_orientations = len(orientations)

						shell()
						# plot timecourse of voxel
						f = figure(figsize=(24,24,))
						for voxno in range(voxels_in_this_slice.sum()):
							s=f.add_subplot(voxels_in_this_slice.sum(),1,(voxno+1))
							pl.title('smoothed timecourse of voxel ' + str(voxno),fontsize=18)
							# pl.plot(resampled_signal[voxno,:])
							pl.plot(all_smoothed_data[voxno,:],'r')
							# pl.plot(all_spliced_data[voxno,:],'--k',alpha=0.1)
							simpleaxis(s)
							spine_shift(s) 
							s.set_xlim(-20,tr_block*n_orientations+20)
							pl.xticks(np.arange(0,n_orientations*tr_block,tr_block),orientations)
							pl.tick_params(labelsize=18)
							s.grid(axis = 'x', linestyle = '--', linewidth = 0.25)
						# pl.show()
						pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'voxel_time_courses.pdf'))
						
							# plot used design matrix:
						di, d = 0,0
						dm_screen = np.zeros((tr_block*n_orientations,n_pixel_elements**2))
						dm_screen[:,valid_regressors] = all_dm
						dmr_screen = np.reshape(dm_screen,(tr_block*n_orientations,n_pixel_elements,n_pixel_elements))
						# for di,d in enumerate(orientations):
						f=figure(figsize=(24,24))
						timepoints = np.arange(tr_block*di,tr_block*(di+1))
						for i,t in enumerate(timepoints):
							s=f.add_subplot(int(np.ceil(np.sqrt(tr_block))),int(np.ceil(np.sqrt(tr_block))),i+1)
							imshow(np.reshape(dm_screen[t,:],(n_pixel_elements,n_pixel_elements)),interpolation='nearest',cmap='gray')
							# print np.min(dm_screen[t,:]),np.max(dm_screen[t,:])
							# simpleaxis(s)
							# pl.clim(-1000,3000)
							pl.axis('off')
							pl.title('timepoint ' + str(t))
						# pl.show()
						pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'dm_%s_%s_%d.pdf'%(n_pixel_elements,d,stimulus_correction)))

						# plot cross:


						#animate design matrix
						pl.close('all')
						from matplotlib import animation
						# for di,d in enumerate(orientations):
						di = 4
						d = orientations[di]
						timepoints = np.arange(tr_block*di,tr_block*(di+1))
						ims = []
						f=pl.figure()
						for i,t in enumerate(timepoints):
							s=f.add_subplot(111)
							im=pl.imshow(np.reshape(dm_screen[t,:],(n_pixel_elements,n_pixel_elements)),interpolation='nearest',cmap='gray')
							pl.clim(np.min(dmr_screen),np.max(dmr_screen))
							ims.append([im])
							pl.title('Direction: ' + str(d))
							pl.axis('off')
						ani = animation.ArtistAnimation(f, ims, interval=5, blit=True, repeat_delay=10)
						# pl.show()
						# ani.save(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'dm_movie_%d.mp4'%d))
				
						# plot individual regressors
						# f = pl.figure(figsize=(16,8))
						# timepoints = [np.arange(tr_block*0,tr_block*1),np.arange(tr_block*4,tr_block*5)]
						# for i in range(2):
						# 	s = f.add_subplot(1,2,i+1)
						# 	pl.plot(mean_dm_upscaled[timepoints[i],45]/np.max(mean_dm_upscaled),color=['r','b'][i])
						# 	pl.plot(all_smoothed_data[2,timepoints[i]],'--k')
						# pl.show()

						# plot fit prediction and PRF
						plt.close('all')
						# for voxno in range(voxels_in_this_slice.sum()):
						voxno=12
						a=[]
						f=pl.figure(figsize=(16,24))
						for i in range(n_orientations):
							timepoints = np.arange(tr_block*i,tr_block*(i+1))
							# a.append(fitBayesianRidge(mean_dm_upscaled[timepoints,:],all_smoothed_data[voxno,timepoints]))
							a.append(fitRidge(mean_dm_upscaled[timepoints,:],all_smoothed_data[voxno,timepoints],alpha=alpha))
							s=f.add_subplot(8,2,np.arange(1,17,2)[i])
							plot(all_smoothed_data[voxno,timepoints],'--k');plot(a[i][2],'r')
							coef_array = np.zeros(n_pixel_elements**2)
							coef_array[valid_regressors] = np.array(a[i][0])
							simpleaxis(s)
							spine_shift(s)
							pl.tick_params(labelsize=16)
							s=f.add_subplot(8,4,np.arange(3,40,4)[i])
							imshow(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)),interpolation='nearest',cmap='gray')
							# print np.min(coef_array),np.max(coef_array)
							# pl.clim(-1e-5,1e-5)
							plt.axis('off')	
						pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'single_direction_PRFs_voxel_%d_%1.2f_%0.1e.pdf'%(voxno,stimulus_correction,alpha)))
						mean_prf = np.mean(np.array(a)[:,0])
						mean_prf_empty = np.zeros(n_pixel_elements**2)		
						mean_prf_empty[valid_regressors]=mean_prf
						f=figure(figsize=(16,16))
						imshow(np.reshape(mean_prf_empty,(n_pixel_elements,n_pixel_elements)),interpolation='nearest',cmap='gray')
						# pl.show()
						# print np.min(mean_prf_empty),np.max(mean_prf_empty)
						# pl.clim(-1e-5,1e-5)
						pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'PRF_voxel_%d_%1.2f_%0.1e.pdf'%(voxno,stimulus_correction,alpha)))
						
						plt.close('all')
						for voxno in range(voxels_in_this_slice.sum()):
							f=pl.figure(figsize=(16,24))
							for i in range(n_orientations):
								timepoints = np.arange(tr_block*i,tr_block*(i+1))
								a=fitRidge(all_dm[timepoints,:],all_smoothed_data[voxno,timepoints],alpha=alpha)
								s=f.add_subplot(8,2,np.arange(1,17,2)[i])
								plot(all_smoothed_data[voxno,timepoints],'--k');plot(a[i][2],'r')
								coef_array = np.zeros(n_pixel_elements**2)
								coef_array[valid_regressors] = np.array(a[i][0])
								simpleaxis(s)
								spine_shift(s)
								pl.tick_params(labelsize=16)
								s=f.add_subplot(8,4,np.arange(3,40,4)[i])
								imshow(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)),interpolation='nearest',cmap='gray')
								# print np.min(coef_array),np.max(coef_array)
								# pl.clim(-1e-5,1e-5)
								plt.axis('off')	
							pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'single_direction_PRFs_voxel_%d_%d_%1.1e.pdf'%(voxno,stimulus_correction,alpha)))
							
							mean_prf = np.mean(np.array(a)[:,0])
							mean_prf_empty = np.zeros(n_pixel_elements**2)		
							mean_prf_empty[valid_regressors]=mean_prf
							f=figure(figsize=(16,16))
							imshow(np.reshape(mean_prf_empty,(n_pixel_elements,n_pixel_elements)),interpolation='nearest',cmap='gray')
							# pl.show()
							# print np.min(mean_prf_empty),np.max(mean_prf_empty)
							# pl.clim(-1e-5,1e-5)
							pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/delve_deeper/'),'PRF_voxel_%d_%d_%1.1e.pdf'%(voxno,stimulus_correction,alpha)))

						midline=[]
						PRF=[]
						for voxno in range(voxels_in_this_slice.sum()):
							# voxno=2
							for i,orient in enumerate(orientations):
								# rotation_matrix = np.matrix([[cos(orient), -sin(orient)],[sin(orient), cos(orient)]])
								timepoints = np.arange(tr_block*i,tr_block*(i+1))
								a=fitRidge(mean_dm_upscaled[timepoints,:],all_smoothed_data[voxno,timepoints],alpha=alpha)
								coef_array = np.zeros(n_pixel_elements**2)
								coef_array[valid_regressors] = np.array(a[0])
															
								if orient == 0:
									PRF.append(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)))
									midline.append(PRF[i][:,n_pixel_elements/2.0+0.5])							
								elif orient == 45:
									PRF.append(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)))
									midline.append(np.diagonal(PRF[i])[floor(0.2*n_pixel_elements):-floor(0.2*n_pixel_elements)])
								if orient == 90:
									PRF.append(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)))
									midline.append(PRF[i][::-1][n_pixel_elements/2.0+0.5,:])
								elif orient == 135:
									PRF.append(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)))	
									midline.append(np.diagonal(PRF[i][::-1])[floor(0.2*n_pixel_elements):-floor(0.2*n_pixel_elements)])
								elif orient == 180:								
									PRF.append(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)))
									midline.append(PRF[i][:,n_pixel_elements/2.0+0.5])
								elif orient == 225:
									PRF.append(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)))
									midline.append(np.diagonal(PRF[i])[floor(0.2*n_pixel_elements):-floor(0.2*n_pixel_elements)])
								elif orient == 270:
									PRF.append(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)))
									midline.append(PRF[i][::-1][n_pixel_elements/2.0+0.5,:])
								elif orient == 315:
									PRF.append(np.reshape(coef_array,(n_pixel_elements,n_pixel_elements)))
									midline.append(np.diagonal(PRF[i][::-1])[floor(0.2*n_pixel_elements):-floor(0.2*n_pixel_elements)])
											# f=pl.figure(figsize=(18,18))
						# 
						f=pl.figure(figsize=(18,18))
						prf_subplots = [1,2,3,4,9,10,11,12]
						midline_subplots = [3,4,7,8]
						for i in range(4):
							s =f.add_subplot(4,4,prf_subplots[i+i])
							imshow(np.mean(np.array(PRF)[[0+i,8+i,16+i,24+i],:,:],axis=0),cmap='gray',interpolation='nearest')
							s =f.add_subplot(4,4,prf_subplots[i+i+1])
							imshow(np.mean(np.array(PRF)[[4+i,12+i,20+i,28+i],:,:],axis=0),cmap='gray',interpolation='nearest')
							s =f.add_subplot(4,2,midline_subplots[i])
							pl.plot(np.mean(np.array(midline)[np.array([0+i,8+i,16+i,24+i])],axis=0),color=['b','g','r','k'][i],linestyle='--')
							pl.plot(np.mean(np.array(midline)[np.array([4+i,12+i,20+i,28+i])],axis=0),color=['b','g','r','k'][i],linestyle='-')
							# pl.title()
						# show()
						pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/delve_deeper/'), 'stimulus_timings_all_directions_avg_' + str(n_pixel_elements) +'_' + str(stimulus_correction) + '_.pdf'))

						# f=pl.figure(figsize=(18,18))
						# s = f.add_subplot(2,2,1)
						# # imshow(np.mean(PRF[0],cmap='gray',interpolation='nearest')
						# imshow(np.mean(np.array(PRF)[[0,2,4],:,:],axis=0),cmap='gray',interpolation='nearest')
						# s =f.add_subplot(2,2,2)
						# # imshow(PRF[1],cmap='gray',interpolation='nearest')
						# imshow(np.mean(np.array(PRF)[[1,3,5],:,:],axis=0),cmap='gray',interpolation='nearest')
						# s =f.add_subplot(212)
						# pl.plot(np.mean(np.array(midline)[[0,2,4],:],axis=0),linestyle='--')
						# pl.plot(np.mean(np.array(midline)[[1,3,5],:],axis=0),linestyle='-')
						# 	# pl.title()
						# # show()
						# pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/delve_deeper/'), 'stimulus_timings_check_avg_' + str(n_pixel_elements) +'_' + str(stimulus_correction) + '_.pdf'))

						# f=pl.figure()
						# for voxno in range(voxels_in_this_slice.sum()):
						# 	timepoints = np.arange(tr_block*i,tr_block*(i+1))
						# 	a=fitRidge(mean_dm_upscaled,all_smoothed_data[voxno,:],alpha=1e9)
						# 	coef_array = np.zeros(n_pixel_elements**2)
						# 	coef_array[valid_regressors] = np.array(a[0])
						# 	PRF = np.reshape(coef_array,(n_pixel_elements,n_pixel_elements))
						# 	s=f.add_subplot(3,2,voxno)
						# 	imshow(PRF,cmap='gray',interpolation='nearest')
						
						# pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/'), 'figs', 'stimulus_timings_check_avg_PRFs_' + str(n_pixel_elements) +'_' + str(stimulus_correction) + '_.pdf'))


					# exit



					res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitRidge)(all_dm, vox_timeseries,alpha=alpha) for vox_timeseries in all_smoothed_data)
					# res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitRidge)(mean_dm_upscaled, vox_timeseries,alpha=alpha) for vox_timeseries in all_smoothed_data)

				##
				#######################################################################
				#######################################################################
				## UGLY AVERAGE METHOD:
				
				# trial_onsets = [np.argmin(np.abs(these_tr_times - (trial_events[j]-5))) for j in range(len(trial_events))]
				# trial_offsets = [np.argmin(np.abs(these_tr_times - (trial_events[j]+trial_duration+5))) for j in range(len(trial_events))]
				# min_trial_dur = np.min(np.abs(np.array(trial_onsets)-np.array(trial_offsets)))
				# averaged_data = np.mean([these_voxels[:,trial_onsets[t]:trial_offsets[t]][:,:min_trial_dur] for t in range(len(trial_onsets))],axis=0)
				# start_dm = np.argmin(np.abs(self.sample_time_list - these_tr_times[trial_onsets[0]]))
				# end_dm = np.argmin(np.abs(self.sample_time_list - these_tr_times[trial_offsets[0]]))
				# these_samples_avg = np.around(np.arange(start_dm,end_dm,self.TR/sample_duration)).astype('int32')
				
				## smooth with gaussian (didnt work out nicely)
				# kernel = scipy.signal.gaussian(50,5)
				# smooth_data = [fftconvolve(averaged_data[i],kernel) for i in range(len(averaged_data))]
				# plot([fftconvolve(averaged_data[i],kernel,'same') for i in range(len(averaged_data))][5])
				# plot(averaged_data[5])

				##
				#######################################################################

				# if save_all_data:
				# 	save_design_matrix = np.zeros((this_design_matrix.shape[0],n_pixel_elements * n_pixel_elements))
				# 	save_design_matrix[:,valid_regressors] = this_design_matrix
				# 	np.save(os.path.join(self.stageFolder('processed/mri/PRF/'), 'design_matrix_%ix%i_%s_%s'%(n_pixel_elements, n_pixel_elements, task_conditions[0],orient_list)), save_design_matrix)
				
				# fitBayesianRidge returns coefficients of results, and spearman correlation R and p as a 2-tuple

				# res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitRidge)(self.full_design_matrix[these_samples,:], vox_timeseries, alpha = 1e7) for vox_timeseries in these_voxels[:,selected_tr_times])
				# res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitBayesianRidge)(self.full_design_matrix[these_samples,:], vox_timeseries) for vox_timeseries in these_voxels[:,selected_tr_times])
						
				if method == 'old':
					# res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitBayesianRidge)(self.full_design_matrix[these_samples,:], vox_timeseries) for vox_timeseries in these_voxels[:,selected_tr_times])
					res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(fitRidge)(self.full_design_matrix[these_samples,:], vox_timeseries, alpha = 1e7) for vox_timeseries in these_voxels[:,selected_tr_times])
				# res = []
				# for vox_timeseries in these_voxels[:,selected_tr_times]:
					# res.append(fitBayesianRidge(self.full_design_matrix[these_samples,:], vox_timeseries))
				# res = [fitRidge(self.full_design_matrix[these_samples,:], vox_timeseries, alpha = 1e6, n_jobs = n_jobs) for vox_timeseries in these_voxels]
				self.logger.info('done fitting of slice %d, with %d voxels' % (sl, int((cortex_mask * voxels_in_this_slice_in_full).sum())))
				# if mask_file_name == 'single_voxel':
				# 	pl.figure()
				# 	pl.imshow(all_coefs[:, cortex_mask * voxels_in_this_slice_in_full].reshape((n_pixel_elements,n_pixel_elements)))
				# 	pl.show()
				all_coefs[:, cortex_mask * voxels_in_this_slice_in_full] = np.array([rs[0] for rs in res]).T
				all_corrs[:, cortex_mask * voxels_in_this_slice_in_full] = np.array([rs[1] for rs in res]).T
				# all_predicted[:,cortex_mask * voxels_in_this_slice_in_full] = np.array([rs[2] for rs in res]).T
				# all_data[:, cortex_mask * voxels_in_this_slice_in_full] = np.array([rs[3] for rs in res]).T

		# shell()
		output_coefs = np.zeros([n_pixel_elements ** 2] + list(cortex_mask.shape))
		output_coefs[valid_regressors] = all_coefs

		# # ac = np.reshape(all_corrs[:,cortex_mask],(2,-1))
		# ad = np.reshape(all_data[:,cortex_mask],(int((floor(40/self.TR)+dilate_width)*len(trial_events)),-1))
		# ap = np.reshape(all_predicted[:,cortex_mask],(int((floor(40/self.TR)+dilate_width)*len(trial_events)),-1))

		# # trial_dur = floor(selected_tr_times.sum()/4)
		# # ad_mean = np.array([np.mean(np.array([ad[0:trial_dur,i],ad[trial_dur*1:trial_dur*2,i],ad[trial_dur*2:trial_dur*3,i],ad[trial_dur*3:trial_dur*4,i]]),axis=0) for i in range(cortex_mask.sum())]).T
		# # ap_mean = np.array([np.mean(np.array([ap[0:trial_dur,i],ap[trial_dur*1:trial_dur*2,i],ap[trial_dur*2:trial_dur*3,i],ap[trial_dur*3:trial_dur*4,i]]),axis=0) for i in range(cortex_mask.sum())]).T

		# voxels_to_plot= [19,  20,  24,  32,  54,  60,  61,  63,  64,  65]
		# f=pl.figure(figsize=(24,24))
		# for e,i in enumerate(voxels_to_plot):
		# 		s=f.add_subplot(5,2,e)
		# 		pl.plot(ap[:,i],'r',label='predicted')
		# 		pl.plot(ad[:,i],'--k',label='data')
		# 		simpleaxis(s)
		# 		spine_shift(s)
		# 		pl.title('voxel ' + str(i),fontsize=14)
		# 		leg = s.legend(fancybox = True, loc = 'best')
		# 		leg.get_frame().set_alpha(0.5)
		# 		if leg:
		# 			for t in leg.get_texts():
		# 			    t.set_fontsize(14)    # the legend text fontsize
		# 			for l in leg.get_lines():
		# 			    l.set_linewidth(3.5)  # the legend line width
		# pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/'), 'figs', 'trial_average_data_and_fit_' + str(n_pixel_elements) + '_cond_' + task_conditions[0] +'.pdf'))

		# 
		self.logger.info('saving coefficients and correlations of PRF fits')
		coef_nii_file = NiftiImage(output_coefs)
		coef_nii_file.header = mask_file.header
		coef_nii_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'coefs_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '-' + condition + orient_list + '-' + str(n_pixel_elements) + '_%0.3f.nii.gz'%sample_duration))
		
		# replace infs in correlations with the maximal value of the rest of the array.
		all_corrs[np.isinf(all_corrs)] = all_corrs[-np.isinf(all_corrs)].max() + 1.0
		corr_nii_file = NiftiImage(all_corrs)
		corr_nii_file.header = mask_file.header
		corr_nii_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '-' + condition + orient_list+ '-' + str(n_pixel_elements) + '_%0.3f.nii.gz'%sample_duration))
	
		# if save_all_data:
		# 	all_data = np.zeros([selected_tr_times.sum()] + list(cortex_mask.shape))
		# 	all_data[:,cortex_mask] = z_data[selected_tr_times]
			
		# 	data_nii_file = NiftiImage(all_data)
		# 	data_nii_file.header = mask_file.header
		# 	data_nii_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'data_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '-'  + condition + orient_list + '-' + str(n_pixel_elements) + '_%0.3f.nii.gz'%sample_duration))
			
		# 	np.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'data_' + mask_file_name + '_' + '_'.join(postFix) + '_' + task_conditions[0] + '-' + condition + orient_list), z_data[selected_tr_times])
	
	
	
	def results_to_surface(self, file_name = 'corrs_cortex', output_file_name = 'polar', frames = {'_f':1}, smooth = 0.0, condition = 'PRF'):
		"""docstring for results_to_surface"""
		vsO = VolToSurfOperator(inputObject = os.path.join(self.stageFolder('processed/mri/%s/'%condition), file_name + '.nii.gz'))
		ofn = os.path.join(self.stageFolder('processed/mri/%s/surf/'%condition), output_file_name )
		vsO.configure(frames = frames, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = smooth, surfType = 'paint'  )
		vsO.execute()
	
	def mask_results_to_surface(self, stat_file = '', value_file= '', threshold = 5.0, stat_frame = 1, fill_value = -3.15):
		"""Mask the results, polar.nii.gz, for instance, with the threshold and convert to surface format for easy viewing"""
		if not os.path.isfile(os.path.join(self.stageFolder('processed/mri/%s/'%condition), stat_file + '.nii.gz')) or not os.path.isfile(os.path.join(self.stageFolder('processed/mri/%s/'%condition), value_file + '.nii.gz')):
			self.logger.error('images for mask_results_to_surface %s %s are not files' % (stat_file, value_file))
		
		val_data = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), value_file + '.nii.gz')).data
		stat_mask = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), stat_file + '.nii.gz')).data[stat_frame] < threshold
		val_data[:,stat_mask] = fill_value
		
		op_nii_file = NiftiImage(val_data)
		op_nii_file.header = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), value_file + '.nii.gz')).header
		op_nii_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), value_file + '_%2.2f.nii.gz'%threshold) )
		
		self.results_to_surface(file_name = value_file + '_%2.2f'%threshold, output_file_name = condition, frames = {'_polar':0, '_ecc':1, '_real':2, '_imag':3, 'surf': 4})
		
	
	def RF_fit(self, mask_file = 'cortex_dilated_mask', postFix = ['mcf','sgtf','prZ','res'], task_condition = 'all', anat_mask = 'cortex_dilated_mask', stat_threshold = -10.0, n_jobs = 28, run_fits = True, condition = 'PRF', fit_on = 'smoothed_betas', normalize_to = [],voxels_to_plot=[],example_plots = False,orientations=orientations,n_pixel_elements=[],sample_duration=[],convert_to_surf=False,stimulus_correction=0 ):
		"""select_voxels_for_RF_fit takes the voxels with high stat values
		and tries to fit a PRF model to their spatial selectivity profiles.
		it takes the images from the mask_file result file, and uses stat_threshold
		to select all voxels crossing a p-value (-log10(p)) threshold.
		"""
		orient_list = ''
		for i in range(len(orientations)):
			orient_list += '_' + str(orientations[i])
		anat_mask = os.path.join(self.stageFolder('processed/mri/'), 'masks', 'anat', anat_mask + '.nii.gz')
		# filename = mask_file + '_' + '_'.join(postFix + [task_condition]) + '-%s%s'%(condition,orient_list)
		filename = mask_file + '_' + '_'.join(postFix + [task_condition]) + '-%s%s-%d_%0.3f'%(condition,orient_list,n_pixel_elements,sample_duration)
		if run_fits:
			stats_data = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + filename + '.nii.gz')).data
			spatial_data = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'coefs_' + filename + '.nii.gz')).data
			anat_mask = NiftiImage(anat_mask).data > 0
			stat_mask = stats_data[1] > stat_threshold

			voxel_spatial_data_to_fit = spatial_data[:,stat_mask * anat_mask]
			stats_data_to_fit = stats_data[:,stat_mask * anat_mask]
			self.logger.info('starting fitting of prf shapes')
			plotdir = self.stageFolder('processed/mri/') + 'figs/PRF_plots_%d_%0.3f_%d/'%(n_pixel_elements,sample_duration,stimulus_correction)
			if  os.path.isdir(plotdir): shutil.rmtree(plotdir); os.mkdir(plotdir)
			else: os.mkdir(plotdir)
				# if voxels_to_plot == []: voxels_to_plot = np.array(np.random.sample(len(voxel_spatial_data_to_fit.T))*len(voxel_spatial_data_to_fit.T)).astype(int)
				
				# res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(analyze_PRF_from_spatial_profile)(voxel_spatial_data_to_fit.T[i], stats_data = stats_data_to_fit.T[i], diagnostics_plot = True, fit_on=fit_on,normalize_to = normalize_to, cond=task_condition,voxel_no=i,plotdir = plotdir) for i in voxels_to_plot)
			# for i in range(shape(voxel_spatial_data_to_fit.T)[0]):
				# analyze_PRF_from_spatial_profile(voxel_spatial_data_to_fit.T[i], stats_data = stats_data_to_fit.T[i], diagnostics_plot = example_plots, normalize_to = normalize_to, fit_on = fit_on, cond=task_condition,voxel_no=i,plotdir = plotdir)
			# else:
			res = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(analyze_PRF_from_spatial_profile)(voxel_spatial_data_to_fit.T[i], stats_data = stats_data_to_fit.T[i], diagnostics_plot = example_plots, normalize_to = normalize_to, fit_on = fit_on, cond=task_condition,voxel_no=i,plotdir = plotdir) for i in range(shape(voxel_spatial_data_to_fit.T)[0]))
			surf_gauss = np.real(res)[:,2]
			surf_mask = np.real(res)[:,3]
			vol = np.real(res)[:,4]
			EV = np.real(res)[:,5]
			sd_gauss = np.real(res)[:,6]
			sd_surf = np.real(res)[:,7]
			fwhm = np.real(res)[:,8]
			n_regions = np.real(res)[:,9]

			max_comp_gauss = np.array(res)[:,0]
			polar_gauss = np.angle(max_comp_gauss)
			ecc_gauss = np.abs(max_comp_gauss)
			real_gauss = np.real(max_comp_gauss)
			imag_gauss = np.imag(max_comp_gauss)
		
			max_comp_abs = np.array(res)[:,1]
			polar_abs = np.angle(max_comp_abs)
			ecc_abs = np.abs(max_comp_abs)
			# old code:
			# real_abs = np.real(max_comp_abs)
			# imag_abs = np.imag(max_comp_abs)

			# JWs changes:
			real_abs = np.array([math.cos(p) for p in polar_abs]) * stats_data[1,stat_mask*anat_mask].ravel()
			imag_abs = np.array([math.sin(p) for p in polar_abs]) * stats_data[1,stat_mask*anat_mask].ravel()

			# continuation old code:
			prf_res = np.vstack([polar_gauss, polar_abs, ecc_gauss, ecc_abs, real_gauss, real_abs, imag_gauss, imag_abs, surf_gauss, surf_mask, vol, EV, sd_gauss, sd_surf, fwhm,n_regions])

			empty_res = np.zeros([len(prf_res)] + [np.array(stats_data.shape[1:]).prod()])
			empty_res[:,(stat_mask * anat_mask).ravel()] = prf_res

			all_res = empty_res.reshape([len(prf_res)] + list(stats_data.shape[1:]))

			self.logger.info('saving prf parameters')

			all_res_file = NiftiImage(all_res)
			all_res_file.header = NiftiImage(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'corrs_' + filename + '.nii.gz')).header
			all_res_file.save(os.path.join(self.stageFolder('processed/mri/%s/'%condition), 'results_' + filename + '.nii.gz'))	
		
		if convert_to_surf:
			self.logger.info('converting prf values to surfaces')
			# old code
			# for sm in [0,2,5]: # different smoothing values.
			# 	# reproject the original stats
			# 	self.results_to_surface(file_name = 'corrs_' + filename, output_file_name = 'corrs_' + filename + '_' + str(sm), frames = {'_f':1}, smooth = sm, condition = condition)
			# 	# and the spatial values
			# 	self.results_to_surface(file_name = 'results_' + filename, output_file_name = 'results_' + filename + '_' + str(sm), frames = {'_polar':0, '_ecc':1, '_real':2, '_imag':3, '_surf':4}, smooth = sm, condition = condition)
			
			# 	# but now, we want to do a surf to vol for the smoothed real and imaginary numbers.
			# 	self.surface_to_polar(filename = os.path.join(self.stageFolder('processed/mri/%s/surf/'%condition), 'results_' + filename + '_' + str(sm) ))
			
			# JWs code:

			for sm in [0,2,4]: # different smoothing values.
				# reproject the original stats
				self.results_to_surface(file_name = 'corrs_' + filename, output_file_name = 'corrs_' + filename + '_' + str(sm), frames = {'_f':1}, smooth = sm, condition = condition)
				# and the spatial values
				# self.results_to_surface(file_name = 'results_' + filename, output_file_name = 'results_' + filename + '_' + str(sm), frames = {'_polar_gaus':0, '_polar_abs':1, '_ecc_gaus':2, '_ecc_abs':3,}, smooth = sm, condition = condition)
				self.results_to_surface(file_name = 'results_' + filename, output_file_name = 'results_' + filename + '_' + str(sm), frames = {'_polar_gaus':0, '_polar_abs':1, '_ecc_gaus':2, '_ecc_abs':3, '_real':5, '_imag':7, }, smooth = sm, condition = condition)
				
				# but now, we want to do a surf to vol for the smoothed real and imaginary numbers.
				self.surface_to_polar(filename = os.path.join(self.stageFolder('processed/mri/%s/surf/'%condition), 'results_' + filename + '_' + str(sm) ))

				self.makeTiffsFromCondition(condition='PRF',results_file = 'results_' + filename, exit_when_ready=1)
			
	def surface_to_polar(self, filename, condition = 'PRF'):
		"""surface_to_polar takes a (smoothed) surface file for both real and imaginary parts and re-converts it to polar and eccentricity angle."""
		self.logger.info('converting %s from (smoothed) surface to nii back to surface')
		for hemi in ['lh','rh']:
			for component in ['real', 'imag']:
				svO = SurfToVolOperator(inputObject = filename + '_' + component + '-' + hemi + '.mgh' )
				svO.configure(templateFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[condition][0]], postFix = ['mcf']), hemispheres = [hemi], register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), fsSubject = self.subject.standardFSID, outputFileName = filename + '_' + component + '.nii.gz', threshold = 0.5, surfType = 'paint')
				print svO.runcmd
				svO.execute()
				# 
				
			# now, there's a pair of imag and real nii files for this hemisphere. Let's open them and make polar and eccen phases before re-transforming to surface. 
			complex_values = NiftiImage(filename + '_real-' + hemi + '.nii.gz').data + 1j * NiftiImage(filename + '_imag-' + hemi + '.nii.gz').data
		
			comp = NiftiImage(np.array([np.angle(complex_values), np.abs(complex_values)]))
			comp.header = NiftiImage(filename + '_real-' + hemi + '.nii.gz').header
			comp.save(filename + '_polecc-' + hemi + '.nii.gz')
		
		# add the two polecc files together
		addO = FSLMathsOperator(filename + '_polecc-' + 'lh' + '.nii.gz')
		addO.configureAdd(add_file = filename + '_polecc-' + 'rh' + '.nii.gz', outputFileName = filename + '_polecc.nii.gz')
		addO.execute()
		
		# self.results_to_surface(file_name = filename + '_polecc.nii.gz', output_file_name = filename, frames = , smooth = 0)
		vsO = VolToSurfOperator(inputObject = os.path.join(self.stageFolder('processed/mri/PRF/'), filename + '_polecc.nii.gz'))
		# ofn = os.path.join(self.stageFolder('processed/mri/PRF/surf/'), output_file_name )
		vsO.configure(frames = {'_polar':0, '_ecc':1}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = filename + '_sm', threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
		vsO.execute()

		
	
	def makeTiffsFromCondition(self, condition, results_file, y_rotation = 90.0, exit_when_ready = 1 ):
		thisFeatFile = os.path.join(os.environ['ANALYSIS_HOME'], 'Tools/other_scripts/redraw_retmaps.tcl' )
		for hemi in ['lh','rh']:
			REDict = {
			'---HEMI---': hemi,
			'---CONDITION---': condition, 
			'---CONDITIONFILENAME---': results_file, 
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
	
	def mask_stats_to_hdf(self, condition = 'PRF', mask_file = 'cortex_dilated_mask_all', postFix = ['mcf','sgtf','prZ','res'], task_conditions = ['fix','all','color','sf','orient','speed'],orientations=orientations,n_pixel_elements=[],sample_duration=[]):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""
		
		# 
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		anatRoiFileNames = [anRF for anRF in anatRoiFileNames if 'cortex' not in anRF]
		# anatRoiFileNames = ['/home/shared/PRF_square/data/DVE/DVE_291014/processed/mri/masks/anat/rh.V1.nii.gz']
		# anatRoiFileNames = [os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat/%s.nii.gz'%mask_file))]


		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
		
		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri/%s'%condition),  condition  +'-'+ str(n_pixel_elements) +"_%0.3f_file.hdf5"%sample_duration)
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = open_file(self.hdf5_filename, mode = "w", title = condition  +'-'+ str(n_pixel_elements) +"_%0.3f_file"%sample_duration)
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = open_file(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		this_run_group_name = 'prf'
		try:
			thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = h5file.create_group("/", this_run_group_name, '')
			
		orient_list = ''
		for i in range(len(orientations)):
			orient_list += '_' + str(orientations[i])

		stat_files = {}
		for c in task_conditions:
		# 	"""loop over runs, and try to open a group for this run's data"""
		#
		# 	"""
		# 	Now, take different stat masks based on the run_type
		# 	"""
	
			for res_type in ['results', 'coefs', 'corrs']:
				filename = mask_file + '_' + '_'.join(postFix + [c]) + '-' + condition + orient_list
				stat_files.update({c+'_'+res_type: os.path.join(self.stageFolder('processed/mri/%s'%condition), res_type + '_' + filename +'-'+ str(n_pixel_elements)+ '_%0.3f.nii.gz'%sample_duration)})
		
		
		stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
		
		for (roi, roi_name) in zip(rois, roinames):
			try:
				thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
				thisRunGroup = h5file.create_group("/" + this_run_group_name, roi_name, 'ROI ' + roi_name +' imported' )
		
			for (i, sf) in enumerate(stat_files.keys()):
				# loop over stat_files and rois
				# to mask the stat_files with the rois:
				imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
				these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
				h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
		
		h5file.close()

	def prf_data_from_hdf(self, roi = 'v2d', condition = 'PRF', base_task_condition = 'fix', comparison_task_conditions = ['fix', 'color', 'sf', 'speed', 'orient'], corr_threshold = 0.1, ecc_thresholds = [0.025, 0.6]):
		self.logger.info('starting prf data correlations from region %s'%roi)
		results_frames = {'polar_gauss':0, 'polar_abs':1, 'ecc_gauss':2, 'ecc_abs':3, 'real_gauss':4, 'real_abs':5, 'imag_gauss':6, 'imag_abs':7, 'surf_gauss':8, 'surf_mask':9, 'vol':10, 'EV':11, 'sd_gauss':12,'sd_surf':13} 
		stats_frames = {'corr': 0, '-logp': 1}

		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri/%s'%condition), condition  +'-'+ str(n_pixel_elements) +"_%0.3f_file"%sample_duration)
		h5file = open_file(self.hdf5_filename, mode = "r", title = condition + " file")
		# 
		# data to be correlated
		
		base_task_data = self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi, data_type = base_task_condition + '_results')
		all_comparison_task_data = [self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi, data_type = c + '_results') for c in comparison_task_conditions]
		# thisRoi = h5file.get_node(where = '/' + 'prf', name = roi, classname='Group')
		# base_task_data = eval('thisRoi.' + 'fix_results' + '.read()').T
		# all_comparison_task_data = []
		# all_comparison_task_data.append(eval('thisRoi.' + 'fix_results' + '.read()').T)
		# correlations on which to base the tasks
		base_task_corr = self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi, data_type = base_task_condition + '_corrs')
		all_comparison_task_corr = [self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi, data_type = c + '_corrs') for c in comparison_task_conditions]
		# base_task_corr = eval('thisRoi.' + 'fix_corrs' + '.read()').T
		# all_comparison_task_corr = []
		# all_comparison_task_corr.append(eval('thisRoi.' + 'fix_corrs' + '.read()').T)
		h5file.close()

		# create and apply the mask. 
		mask = base_task_corr[:,0] > corr_threshold
		mask = mask * (base_task_data[:,results_frames['ecc_gauss']] > ecc_thresholds[0]) * (base_task_data[:,results_frames['ecc_abs']] > ecc_thresholds[0]) * (base_task_data[:,results_frames['ecc_gauss']] < ecc_thresholds[1]) * (base_task_data[:,results_frames['ecc_abs']] < ecc_thresholds[1]) * (base_task_data[:,results_frames['surf_gauss']] > 0.0) * (base_task_data[:,results_frames['surf_mask']] > 0.0)
		# mask = mask * (base_task_data[:,results_frames['ecc_gauss']] - base_task_data[:,results_frames['ecc_abs']] < 0.05)	
		base_task_data, all_comparison_task_data = base_task_data[mask, :], np.array([ac[mask, :] for ac in all_comparison_task_data])
		base_task_corr, all_comparison_task_corr = base_task_corr[mask, 0], np.array([ac[mask, 0] for ac in all_comparison_task_corr])
		
		if base_task_data[:,results_frames['ecc_gauss']] != []:
			order = np.argsort(base_task_data[:,results_frames['ecc_gauss']])
			kern =  stats.norm.pdf( np.linspace(-.25,.25,int(round(base_task_data.shape[0] / 10)) ))
			sm_ecc = np.convolve( base_task_data[:,results_frames['ecc_gauss']][order], kern / kern.sum(), 'valid' )  
		
			# scatter plots for results frames
			colors = [(c, 1-c, 1-c) for c in np.linspace(0.0,1.0,len(comparison_task_conditions))]
			mcs = ['o', 'v', 's', '>', '<']
			f = pl.figure(figsize = (16,8))
			for j, res_type in enumerate(['ecc_gauss','sd_gauss']): # , 'ecc'
				s = f.add_subplot(1,2,1+j)
				for i, tc in enumerate(comparison_task_conditions):
					
					# fit = polyfit(results[j][mask[j],results_frames['ecc_gauss']], results[j][mask[j],results_frames['surf_gauss']], 1)
					# fit_fn = poly1d(fit)

					# pl.plot(results[j][mask[j],results_frames['ecc_gauss']],results[j][mask[j],results_frames['surf_gauss']], c = colors[j], marker = 'o', linewidth = 0, alpha = 0.3, mec = 'w', ms = 3.5)
					# pl.plot(results[j][mask[j],results_frames['ecc_gauss']], fit_fn(results[j][mask[j],results_frames['ecc_gauss']]),linewidth = 3.5, alpha = 0.75, linestyle = '-', c = colors[j], label=roi)
					
					
					pl.plot(base_task_data[:,results_frames['ecc_gauss']], all_comparison_task_data[i][:,results_frames[res_type]], c = colors[i], marker = 'o', linewidth = 0, alpha = 0.3, mec = 'w', ms = 3.5)
					sm_signal = np.convolve( all_comparison_task_data[i][:,results_frames[res_type]][order], kern / kern.sum(), 'valid' )
					pl.plot(sm_ecc, sm_signal, c = colors[i], linewidth = 3.5, alpha = 0.75, label = comparison_task_conditions[i] )
				s.set_title(roi + ' ' + res_type)

				# if  j==1:
					# s.set_ylim([0.85,1.25])
				# else:
					# s.set_ylim([0,0.6])
				leg = s.legend(fancybox = True, loc = 'best')
				leg.get_frame().set_alpha(0.5)
				if leg:
					for t in leg.get_texts():
					    t.set_fontsize('small')    # the legend text fontsize
					for l in leg.get_lines():
					    l.set_linewidth(3.5)  # the legend line width

				s.set_xlim(ecc_thresholds)
				simpleaxis(s)
				spine_shift(s)
				s.set_xlabel('eccentricity of %s condition'%base_task_condition)
				s.set_ylabel(res_type)

			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/'), 'figs', roi + '_45*45_noGLM.pdf'))
			pl.close()
			
			# 
			# circular plot
			colors = [(c, 1-c, 1-c) for c in np.linspace(0.0,1.0,2)]
			mcs = ['o', 'v', 's', '>', '<']
			f = pl.figure(figsize = (16,8))
			median_dis = []
			sd_dis = []
			median_eccen_dif = []
			sd_eccen_dif = []
			
			for i, tc in enumerate(comparison_task_conditions):
				if i == 0 or i == 1:
					s=f.add_subplot(2,3,i+1)
				else:
					s=f.add_subplot(2,3,i+2)
				# 
				x_loc = base_task_data[:,results_frames['ecc_gauss']]*np.cos(base_task_data[:,results_frames['polar_gauss']])
				y_loc = base_task_data[:,results_frames['ecc_gauss']]*np.sin(base_task_data[:,results_frames['polar_gauss']])
				x_dif = all_comparison_task_data[i][:,results_frames['ecc_gauss']]*np.cos(all_comparison_task_data[i][:,results_frames['polar_gauss']]) - base_task_data[:,results_frames['ecc_gauss']]*np.cos(base_task_data[:,results_frames['polar_gauss']])
				y_dif = all_comparison_task_data[i][:,results_frames['ecc_gauss']]*np.sin(all_comparison_task_data[i][:,results_frames['polar_gauss']]) - base_task_data[:,results_frames['ecc_gauss']]*np.sin(base_task_data[:,results_frames['polar_gauss']])
				sd_dis.append(np.std( [np.linalg.norm([x_dif[z], y_dif[z]]) for z in range(len(x_loc))] ))
				sd_dis[i] = np.std( [np.linalg.norm([x_dif[z], y_dif[z]]) for z in range(len(x_loc)) if np.linalg.norm([x_dif[i], y_dif[i]]) < 3*sd_dis[i] ] )
				median_dis.append(np.median( [np.linalg.norm([x_dif[z], y_dif[z]]) for z in range(len(x_loc)) if np.linalg.norm([x_dif[i], y_dif[i]]) < 3*sd_dis[i] ] ))
				
				# 
				eccen_difference = all_comparison_task_data[i][:,results_frames['ecc_gauss']]/all_comparison_task_data[i][:,results_frames['surf_gauss']] - base_task_data[:,results_frames['ecc_gauss']]/base_task_data[:,results_frames['surf_gauss']]  
				sd_eccen_dif.append( np.std(eccen_difference))
				median_eccen_dif.append(np.median(eccen_difference[eccen_difference < sd_eccen_dif[i]*3 ]) )
				sd_eccen_dif[i] = np.std(eccen_difference[eccen_difference < sd_eccen_dif[i]*3 ])
				
				pl.plot(x_loc,y_loc, c = colors[0], marker = 'o', linewidth = 0, alpha = 0.3, mec = 'w', ms = 3.5)
				pl.plot(x_loc+x_dif,y_loc+y_dif, c = colors[1], marker = 'o', linewidth = 0, alpha = 0.3, mec = 'w', ms = 3.5)
				for a in range(shape(base_task_data)[0]):
					if np.linalg.norm([x_dif[a], y_dif[a]]) < 3*sd_dis[i]:
						pl.arrow(x_loc[a],y_loc[a],x_dif[a],y_dif[a],linewidth = 0.1, head_width = 0.01, color = 'k')

				s.set_xlim([-1,1])
				s.set_ylim([-1,1])
				s.set_title(roi + ' ' + tc)
				s.set_xticks([-1,0,1])
				s.set_yticks([-1,0,1])
				# s.set_xticklabels( (comparison_task_conditions) )

			s = f.add_subplot(2,3,3)
			pl.bar(np.arange(len(median_dis))-0.25,median_dis, yerr = sd_dis, width = 0.5)
			s.set_ylim([0,np.max(median_dis)+np.max(sd_dis)*1.2])
			s.set_ylabel('median displacement')
			s.set_xticks(np.arange(len(median_dis)))
			s.set_title('Displacement difference with fix')
			s.set_xticklabels( (comparison_task_conditions) )
			
			s = f.add_subplot(2,3,6)
			pl.bar(np.arange(len(median_eccen_dif))-0.25,median_eccen_dif, yerr = sd_eccen_dif, width = 0.5)
			s.set_ylim([0,np.max(median_eccen_dif)+np.max(sd_eccen_dif)*1.2])
			s.set_ylabel('median eccentricity difference / PRF size')
			s.set_xticks(np.arange(len(median_eccen_dif)))
			s.set_title('Eccentricity difference with fix')
			s.set_xticklabels( (comparison_task_conditions) )
			
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/'), 'figs/v1_45*45_noGLM_displacement' + roi + '.pdf'))
			pl.close()
	
	def ecc_surf_correlations(self, condition = 'PRF', corr_threshold = 0.3, rois = [], task_condition = [],n_pixel_elements=[],sample_duration=[],stimulus_correction=0):
		
		self.logger.info('starting eccen-surf correlations')
		results_frames = {'polar_gauss':0, 'polar_abs':1, 'ecc_gauss':2, 'ecc_abs':3, 'real_gauss':4, 'real_abs':5, 'imag_gauss':6, 'imag_abs':7, 'surf_gauss':8, 'surf_mask':9, 'vol':10, 'EV':11,'sd_gauss':12,'sd_surf':13,'fwhm':14,'n_regions':15} 
		stats_frames = {'corr': 0, '-logp': 1}

		self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri/%s'%condition), condition  +'-'+ str(n_pixel_elements) +"_%0.3f_file.hdf5"%sample_duration)
		h5file = open_file(self.hdf5_filename, mode = "r", title = condition  +'-'+ str(n_pixel_elements) +"_%0.3f_file"%sample_duration)
		
		# combine rois
		end_rois = {'v1':0,'v2':1,'v3':2,'v4':3}#,'v7':4,'LO':5,'VO':6,'TO':7,'IPS':8}
		roi_comb = {0:['V1'],1:['V2v','V2d'],2:['V3v','V3d'],3:['V4']}#,4:['v7'],5:['LO1','LO2'],6:['VO1','VO2'],7:['TO1','TO2'],8:['IPS1','IPS2']}
		
		# end_rois = {}
		# roi_comb = {}
		# for i in range(len(rois)):
		# 	end_rois[rois[i]] = i
		# 	roi_comb[i] = [rois[i]]
		# end_rois = {'V1':0,'V2':1,'V3':2,'V4':3}
		# roi_comb = {'V1':0,'V2':1,'V3':2,'V4':3}
		# roi_comb = {0:['V1'],1:['V2'],2:['V3'],3:['V4']}
		# end_rois = {rois:0}
		# roi_comb = {0: rois}

		# if np.all([size(end_rois) == 1, size(roi_comb[0]) == 1]):
		# 	results = []
		# 	stats = []
		# 	results.append(self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi_comb[0][0], data_type = task_condition[0] + '_results'))
		# 	stats.append(self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = roi_comb[0][0], data_type = task_condition[0] + '_corrs'))
		# else:
		results = [ np.concatenate([self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = rci, data_type = task_condition[0] + '_results') for rci in roi_comb[ri]]) for ri in range(len(end_rois)) ]
		stats = [ np.concatenate([self.roi_data_from_hdf(h5file, run = 'prf', roi_wildcard = rci, data_type = task_condition[0] + '_corrs') for rci in roi_comb[ri]]) for ri in range(len(end_rois)) ]

		
		for r in range(len(end_rois)):
			results[r][:,results_frames['ecc_gauss']] = results[r][:,results_frames['ecc_gauss']] * 27.0/2
			results[r][:,results_frames['ecc_abs']] = results[r][:,results_frames['ecc_abs']] * 27.0/2
			# results[r][:,results_frames['fwhm']] = results[r][:,results_frames['ecc_abs']] * 27.0/2

		# mask = [(stats[r][:,0]  > corr_threshold) *(results[r][:,results_frames['ecc_abs']] < 12 )for r in range(len(end_rois))]
		mask = [(stats[r][:,0]  > corr_threshold) * (results[r][:,results_frames['ecc_gauss']] < 0.7*(27.0/2)) *  (results[r][:,results_frames['EV']] > 0.85) * (results[r][:,results_frames['n_regions']] ==1) for r in range(len(end_rois))]
		# mask = [(stats[r][:,0]  > corr_threshold) * (results[r][:,results_frames['sd_gauss']] > 0.0) * (results[r][:,results_frames['sd_gauss']] < 27.0/2) * (results[r][:,results_frames['ecc_gauss']] < 0.7*(27.0/2)) * (results[r][:,results_frames['EV']] > 0.85) * (results[r][:,results_frames['n_regions']] <20) for r in range(len(end_rois))]

		f = pl.figure(figsize = (16,8))
		s = f.add_subplot(1,2,1)
		pl.hist(results[0][mask[0],results_frames['ecc_gauss']],20,)
		simpleaxis(s)
		spine_shift(s)

		# pl.show()
		# s.set_xlim(0,10)
		s = f.add_subplot(1,2,2)
				
		colors = [(c, 1-c, 1-c) for c in np.linspace(0.0,1,len(end_rois))]
		eccen_bins = np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9] ])
		eccen_x = np.arange(np.mean(eccen_bins[0]),np.mean(eccen_bins[-1])+1,1)
		mean_per_bin = []
		sd_per_bin = []
		for j, roi in enumerate(end_rois.keys()):
			mean_per_bin.append([])
			sd_per_bin.append([])
			for b in range(len(eccen_bins)):			
				mean_per_bin[j].append( np.mean( results[j][ mask[j] * (results[j][:,results_frames['ecc_gauss']]>eccen_bins[b][0]) * (results[j][:,results_frames['ecc_gauss']]<eccen_bins[b][1]) ,results_frames['sd_gauss']] ) )
				sd_per_bin[j].append( np.std( results[j][ mask[j] * (results[j][:,results_frames['ecc_gauss']]>eccen_bins[b][0]) * (results[j][:,results_frames['ecc_gauss']]<eccen_bins[b][1]), results_frames['sd_gauss']] ) )
		
		for j, roi in enumerate(end_rois.keys()):			
			
			# fit = polyfit(results[j][mask[j],results_frames['ecc_gauss']], results[j][mask[j],results_frames['sd_gauss']], 1)
			fit = polyfit(eccen_x, mean_per_bin[j], 1)
			fit_fn = poly1d(fit)

			# pl.plot(results[j][mask[j],results_frames['ecc_gauss']],results[j][mask[j],results_frames['sd_gauss']], c = colors[j], marker = 'o', linewidth = 0, alpha = 0.3, mec = 'w', ms = 3.5)
			pl.plot(eccen_x,mean_per_bin[j],c = colors[j], marker = 'o', markersize = 2,linewidth = 0.1, mec = 'w', ms = 3.5)
			pl.plot(results[j][mask[j],results_frames['ecc_gauss']], fit_fn(results[j][mask[j],results_frames['ecc_gauss']]),linewidth = 3.5, alpha = 0.75, linestyle = '-', c = colors[j], label=roi)

			s.set_xlim([np.min(eccen_bins),np.max(eccen_bins)])
			s.set_ylim([1,5])

			leg = s.legend(fancybox = True, loc = 'best')
			leg.get_frame().set_alpha(0.5)
			if leg:
				for t in leg.get_texts():
				    t.set_fontsize('small')    # the legend text fontsize
				for l in leg.get_lines():
				    l.set_linewidth(3.5)  # the legend line width


			simpleaxis(s)
			spine_shift(s)
			s.set_xlabel('pRF eccentricity')
			s.set_ylabel('pRF size (sd)')

		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/'), 'figs', 'eccen_surf_cor_%s_%0.3f_%d.pdf'%(n_pixel_elements,sample_duration,stimulus_correction)))
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		



