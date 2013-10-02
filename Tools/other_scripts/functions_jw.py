#!/usr/bin/env python
# encoding: utf-8

#!/usr/bin/env python
# encoding: utf-8

"""
functions_jw.py

Created by Jan Willem de Gee on 2012-06-19.
Copyright (c) 2012 Jan Willem de Gee. All rights reserved.
"""

import numpy as np
import scipy as sp
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from IPython import embed as shell

def IRF_canonical(t=0, s=1.0/(10**26), n=10.1, tmax=930):
	
	"""
	Canocial pupil impulse fucntion.
	
	"""
	
	h = ( (s) * (t**n) * (math.e**((-n*t)/930)) )
	return(h)


def createRegressors(inputObject, len_run, pupil_IRF, type_convolve = 'stick'):
	
	inputObject = inputObject
	len_run = len_run
	pupil_IRF = pupil_IRF
	
	if type_convolve == 'stick':
		regr = np.zeros((len_run,inputObject.shape[0]))
		for i in range(inputObject.shape[0]):
			start = round(inputObject[i,0], 0)
			regr[start,i] = 1.0
		regr_collapsed = np.sum(regr, axis=1)
	
	if type_convolve == 'ramp_down':
		regr = np.zeros((len_run,inputObject.shape[0]))
		for i in range(inputObject.shape[0]):
			start = round(inputObject[i,0], 0)
			dur = round(inputObject[i,1], 0)
			regr[start:start+dur,i] = np.linspace((2/dur),0,dur)
		regr_collapsed = np.sum(regr, axis=1)
	
	if type_convolve == 'ramp_up':
		regr = np.zeros((len_run,inputObject.shape[0]))
		for i in range(inputObject.shape[0]):
			start = round(inputObject[i,0], 0)
			dur = round(inputObject[i,1], 0)
			regr[start:start+dur,i] = np.linspace(0,(2/dur),dur)
		regr_collapsed = np.sum(regr, axis=1)
	
	regr_convolved = np.zeros((len_run,inputObject.shape[0]))
	for i in range(inputObject.shape[0]):
		regr_convolved[:,i] = (sp.convolve(regr[:,i], pupil_IRF, 'full'))[:-(pupil_IRF.shape[0]-1)]
	regr_convolved_collapsed = np.sum(regr_convolved, axis=1)
	
	return(regr_collapsed, regr_convolved_collapsed)


def movingaverage(interval, window_size):
	
	window = np.ones(int(window_size))/float(window_size)
	moving_average = np.repeat(np.nan, len(interval))
	moving_average[(window_size/2):-(window_size/2)] = np.convolve(interval, window, 'full')[window_size-1:-window_size]
	
	return(moving_average)



def permutationTest(group1, group2, nrand=5000, tail=0):
	
	"""
	non-parametric permutation test (Efron & Tibshirani, 1998)
	
	tail = 0 (test A~=B), 1 (test A>B), -1 (test A<B)
	
	"""
	
	import numpy as np
	import numpy.random as random
	
	a = group1
	b = group2
	ntra = len(a)
	ntrb = len(b) 
	meana = np.mean(a)
	meanb = np.mean(b)
	alldat = np.concatenate((a,b))
	
	triala = np.zeros(nrand)
	trialb = np.zeros(nrand)
	
	indices = np.arange(alldat.shape[0])
	
	for i in range(nrand):
		random.shuffle(indices)
		triala[i] = np.mean(alldat[indices[:ntra]])
		trialb[i] = np.mean(alldat[indices[ntra:]])
	
	if tail == 0:
		out = sum(abs(triala-trialb)>=abs(meana-meanb)) / float(nrand)
	else:
		out = sum((tail*(triala-trialb))>=(tail*(meana-meanb))) / float(nrand)
	
	return(out, out, out)


def permutationTest_correlation(a, b, tail=0, nrand=5000):
	"""
	test whether 2 correlations are significantly different. For permuting single corr see randtest_corr2
	function out = randtest_corr(a,b,tail,nrand, type)
	tail = 0 (test A~=B), 1 (test A>B), -1 (test A<B)
	type = 'Spearman' or 'Pearson'
	"""
	
	import numpy as np
	import numpy.random as random
	
	ntra = a.shape[0]
	ntrb = b.shape[0]
	
	truecorrdiff = sp.stats.pearsonr(a[:,0],a[:,1])[0] - sp.stats.pearsonr(b[:,0],b[:,1])[0]
	
	alldat = np.vstack((a,b))
	corrdiffrand = np.zeros(nrand)
	
	indices = np.arange(alldat.shape[0])
	
	for irand in range(nrand):
		
		random.shuffle(indices)
		
		randa = sp.stats.pearsonr(alldat[indices[:ntra],0],alldat[indices[:ntra],1])[0]
		randb = sp.stats.pearsonr(alldat[indices[ntra:],0],alldat[indices[ntra:],1])[0]
		
		corrdiffrand[irand] = randa - randb
		
	if tail == 0:
		p_value = sum(abs(corrdiffrand) >= abs(truecorrdiff)) / float(nrand)
	else:
		p_value = sum(tail*(corrdiffrand) >= tail*(truecorrdiff)) / float(nrand)
	
	return(truecorrdiff, p_value)


def roc_analysis(group1, group2, nrand=1000, tail=1):
	
	import scipy as sp
	import numpy as np
	import random
	from scipy.integrate import cumtrapz
	
	x = group1
	y = group2
	nx = len(x)
	ny = len(y)
	
	z = np.concatenate((x,y)) 
	c = np.sort(z) 
	
	det = np.zeros((c.shape[0],2))
	for ic in range(c.shape[0]):
		det[ic,0] = (x > c[ic]).sum() / float(nx)
		det[ic,1] = (y > c[ic]).sum() / float(ny)
	
	t1 = np.sort(det[:,0])
	t2 = np.argsort(det[:,0])
	
	roc = np.vstack(( [0,0],det[t2,:],[1,1] ))
	t1 = sp.integrate.cumtrapz(roc[:,0],roc[:,1])
	out_i = t1[-1]
	
	# To get the p-value:
	
	trialx = np.zeros(nrand)
	trialy = np.zeros(nrand)
	alldat = np.concatenate((x,y))
	
	fprintf = []
	randi = []
	for irand in range(nrand):
		if not np.remainder(irand,1000):
			fprintf.append('randomization: %d\n' + str(irand))
			
		t1 = np.sort(np.random.rand(nx+ny))
		ind = np.argsort(np.random.rand(nx+ny))
		ranx = z[ind[0:nx]]
		rany = z[ind[nx+1:-1]]
		randc = np.sort( np.concatenate((ranx,rany)) )
		
		randet = np.zeros((randc.shape[0],2))
		for ic in range(randc.shape[0]):
			randet[ic,0] = (ranx > randc[ic]).sum() / float(nx)
			randet[ic,1] = (rany > randc[ic]).sum() / float(ny)
			
		t1 = np.sort(randet[:,0])
		t2 = np.argsort(randet[:,0])
		
		ranroc = np.vstack(( [0,0],randet[t2,:],[1,1] ))
		t1 = sp.integrate.cumtrapz(ranroc[:,0],ranroc[:,1])
		
		randi.append(t1[-1])
		
	randi = np.array(randi)
	
	if tail == 0: # (test for i != 0.5)
		out_p = (abs(randi-0.5) >= abs(out_i-0.5)).sum() / float(nrand)
	if (tail == 1) or (tail == -1): # (test for i > 0.5, and i < 0.5 respectively)
		out_p = (tail*(randi-0.5) >= tail*(out_i-0.5)).sum() / float(nrand)
		
	if (float(1) - out_p) < out_p:
		out_p = float(1) - out_p
		
	return(out_i, out_p)


def sdt_barplot(subject, hit, fa, miss, cr, p1, p2, type_plot = 1, values = False):
	
	def label_diff(i,j,text,X,Y,Z, values = False):
		
		# i = 2
		# j = 3
		# text = '***'
		# X = (ind[0]+width, ind[1], ind[2], ind[3]-width)
		# MEANS = MEANS
		# SEMS = SEMS
	
		middle_x = (X[i]+X[j])/2
		max_value = max(MEANS[i]+SEMS[i], MEANS[j]+SEMS[j])
		min_value = min(MEANS[i]-SEMS[i], MEANS[j]-SEMS[j])
		dx = abs(X[i]-X[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':10,'shrinkB':10,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		ax.annotate('', xy=(X[i],max_value), xytext=(X[j],max_value), arrowprops=props)
	
		if values == False:
			if text == 'n.s.':
				kwargs = {'zorder':10, 'size':16, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.0/10))), **kwargs)
			if text != 'n.s.':
				kwargs = {'zorder':10, 'size':24, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(0.60/10))), **kwargs)
		if values == True:
			kwargs = {'zorder':10, 'size':12, 'ha':'center'}
			ax.annotate('p = ' + str(text), xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.15/10))), **kwargs)
	
	def simpleaxis(ax):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		
	def spine_shift(ax, shift = 10):
		for loc, spine in ax.spines.iteritems():
			if loc in ['left','bottom']:
				spine.set_position(('outward', shift)) # outward by 10 points
			elif loc in ['right','top']:
				spine.set_color('none') # don't draw spine
			else:
				raise ValueError('unknown spine location: %s'%loc)
	
	# Type_plot = 1: SDT categories
	# Type_plot = 2: Yes vs No, Correct vs Incorrect
	
	# hit = -HIT
	# fa = -FA
	# miss = -MISS
	# cr = -CR
	# p1 = 0.05
	# p2 = 0.05
	
	hit_mean = sp.mean(hit)
	fa_mean = sp.mean(fa)
	miss_mean = sp.mean(miss)
	cr_mean = sp.mean(cr)
	
	hit_sem = stats.sem(hit)
	fa_sem = stats.sem(fa)
	miss_sem = stats.sem(miss)
	cr_sem = stats.sem(cr)
	
	yes_mean = sp.mean(np.concatenate((hit, fa), axis=0))
	no_mean = sp.mean(np.concatenate((miss, cr), axis=0))
	correct_mean = sp.mean(np.concatenate((hit, cr), axis=0))
	incorrect_mean = sp.mean(np.concatenate((fa, miss), axis=0))
	
	yes_sem = stats.sem(np.concatenate((hit, fa), axis=0))
	no_sem = stats.sem(np.concatenate((miss, cr), axis=0))
	correct_sem = stats.sem(np.concatenate((hit, cr), axis=0))
	incorrect_sem = stats.sem(np.concatenate((fa, miss), axis=0))
	
	y_axis_swap = False
	if hit_mean < 0:
		if fa_mean < 0:
			if miss_mean < 0:
				if cr_mean < 0:
					y_axis_swap = True
	
	if type_plot == 1:
		MEANS = (hit_mean, miss_mean, fa_mean, cr_mean)
		SEMS = (hit_sem, miss_sem, fa_sem, cr_sem)
	if type_plot == 2:
		MEANS = (yes_mean, no_mean, correct_mean, incorrect_mean)
		SEMS = (yes_sem, no_sem, correct_sem, incorrect_sem)	
		
	if values == False:
		sig1 = 'n.s.'
		if p1 <= 0.05:
			sig1 = '*'
		if p1 <= 0.01:
			sig1 = '**'
		if p1 <= 0.001:
			sig1 = '***'
			
		sig2 = 'n.s.'
		if p2 <= 0.05:
			sig2 = '*'
		if p2 <= 0.01:
			sig2 = '**'
		if p2 <= 0.001:
			sig2 = '***'
	else:
		sig1 = round(p1,5)
		sig2 = round(p2,5)
	
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
	
	N = 4
	ind = np.linspace(0,2,4)  # the x locations for the groups
	bar_width = 0.30       # the width of the bars
	spacing = [0.30, 0, 0, -0.30]
	
	# FIGURE 1
	fig = plt.figure(figsize=(4,3))
	ax = fig.add_subplot(111)
	if type_plot == 1:
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','r','b'][i], alpha = [1,.5,.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('H', 'M','FA', 'CR') )
	if type_plot == 2:
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','k','k'][i], alpha = [1,1,.5,.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('Yes', 'No','Corr.', 'Incorr.') )
	ax.set_xticks( (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width) )
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	if y_axis_swap == True:
		ax.set_ylim(ymin=minvalue-(minvalue/8.0), ymax=0)
	if y_axis_swap == False:
		ax.set_ylim(ymin=0, ymax=maxvalue+(maxvalue/4.0))
	left = 0.2
	top = 0.915
	bottom = 0.2
	plt.subplots_adjust(top=top, bottom=bottom, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)
	
	# STATS:
	
	if y_axis_swap == False:
		X = (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width)
		if values == True:
			label_diff(0,1,sig1,X,MEANS, SEMS, values = True)
			label_diff(2,3,sig2,X,MEANS, SEMS, values = True)
		if values == False:
			label_diff(0,1,sig1,X,MEANS, SEMS)
			label_diff(2,3,sig2,X,MEANS, SEMS)
	
	return(fig)

def sdt_barplot_bpd(subject, hit, fa, miss, cr, p1, p2, type_plot = 1, values = False):
	
	def label_diff(i,j,text,X,Y,Z, values = False):
		
		# i = 2
		# j = 3
		# text = '***'
		# X = (ind[0]+width, ind[1], ind[2], ind[3]-width)
		# MEANS = MEANS
		# SEMS = SEMS
	
		middle_x = (X[i]+X[j])/2
		max_value = max(MEANS[i]+SEMS[i], MEANS[j]+SEMS[j])
		min_value = min(MEANS[i]-SEMS[i], MEANS[j]-SEMS[j])
		dx = abs(X[i]-X[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':10,'shrinkB':10,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		ax.annotate('', xy=(X[i],max_value), xytext=(X[j],max_value), arrowprops=props)
	
		if values == False:
			if text == 'n.s.':
				kwargs = {'zorder':10, 'size':16, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.0/10))), **kwargs)
			if text != 'n.s.':
				kwargs = {'zorder':10, 'size':24, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(0.60/10))), **kwargs)
		if values == True:
			kwargs = {'zorder':10, 'size':12, 'ha':'center'}
			ax.annotate('p = ' + str(text), xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.15/10))), **kwargs)
	
	def simpleaxis(ax):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		
	def spine_shift(ax, shift = 10):
		for loc, spine in ax.spines.iteritems():
			if loc in ['left','bottom']:
				spine.set_position(('outward', shift)) # outward by 10 points
			elif loc in ['right','top']:
				spine.set_color('none') # don't draw spine
			else:
				raise ValueError('unknown spine location: %s'%loc)
	
	# Type_plot = 1: SDT categories
	# Type_plot = 2: Yes vs No, Correct vs Incorrect
	
	# hit = -HIT
	# fa = -FA
	# miss = -MISS
	# cr = -CR
	# p1 = 0.05
	# p2 = 0.05
	
	hit_mean = sp.mean(hit)
	fa_mean = sp.mean(fa)
	miss_mean = sp.mean(miss)
	cr_mean = sp.mean(cr)
	
	hit_sem = stats.sem(hit)
	fa_sem = stats.sem(fa)
	miss_sem = stats.sem(miss)
	cr_sem = stats.sem(cr)
	
	yes_mean = sp.mean(np.concatenate((hit, fa), axis=0))
	no_mean = sp.mean(np.concatenate((miss, cr), axis=0))
	correct_mean = sp.mean(np.concatenate((hit, cr), axis=0))
	incorrect_mean = sp.mean(np.concatenate((fa, miss), axis=0))
	
	yes_sem = stats.sem(np.concatenate((hit, fa), axis=0))
	no_sem = stats.sem(np.concatenate((miss, cr), axis=0))
	correct_sem = stats.sem(np.concatenate((hit, cr), axis=0))
	incorrect_sem = stats.sem(np.concatenate((fa, miss), axis=0))
	
	y_axis_swap = False
	if hit_mean < 0:
		if fa_mean < 0:
			if miss_mean < 0:
				if cr_mean < 0:
					y_axis_swap = True
	
	if type_plot == 1:
		MEANS = (hit_mean, miss_mean, fa_mean, cr_mean)
		SEMS = (hit_sem, miss_sem, fa_sem, cr_sem)
	if type_plot == 2:
		MEANS = (yes_mean, no_mean, correct_mean, incorrect_mean)
		SEMS = (yes_sem, no_sem, correct_sem, incorrect_sem)	
		
	if values == False:
		sig1 = 'n.s.'
		if p1 <= 0.05:
			sig1 = '*'
		if p1 <= 0.01:
			sig1 = '**'
		if p1 <= 0.001:
			sig1 = '***'
			
		sig2 = 'n.s.'
		if p2 <= 0.05:
			sig2 = '*'
		if p2 <= 0.01:
			sig2 = '**'
		if p2 <= 0.001:
			sig2 = '***'
	else:
		sig1 = round(p1,5)
		sig2 = round(p2,5)
	
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
	
	N = 4
	ind = np.linspace(0,2,4)  # the x locations for the groups
	bar_width = 0.30       # the width of the bars
	spacing = [0.30, 0, 0, -0.30]
	
	# FIGURE 1
	fig = plt.figure(figsize=(4,3))
	ax = fig.add_subplot(111)
	if type_plot == 1:
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','r','b'][i], alpha = [1,.5,.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('H', 'M','FA', 'CR') )
	if type_plot == 2:
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','k','k'][i], alpha = [1,1,.5,.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('Yes', 'No','Corr.', 'Incorr.') )
	ax.set_xticks( (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width) )
	# ax.yaxis.set_major_locator(MultipleLocator(0.25))
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	diff = maxvalue - minvalue
	ax.set_ylim(ymin=minvalue-(diff/16.0), ymax=maxvalue+(diff/4.0))
	left = 0.20
	top = 0.915
	bottom = 0.20
	plt.subplots_adjust(top=top, bottom=bottom, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)
	
	# STATS:
	
	if y_axis_swap == False:
		X = (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width)
		if values == True:
			label_diff(0,1,sig1,X,MEANS, SEMS, values = True)
			label_diff(2,3,sig2,X,MEANS, SEMS, values = True)
		if values == False:
			label_diff(0,1,sig1,X,MEANS, SEMS)
			label_diff(2,3,sig2,X,MEANS, SEMS)
	
	return(fig)

def confidence_barplot(subject, hit, fa, miss, cr, p1, p2, p3, values=False):
	
	def label_diff(i,j,text,X,Y,Z, values = False):
	
		middle_x = (X[i]+X[j])/2
		max_value = max(MEANS[i]+SEMS[i], MEANS[j]+SEMS[j])
		min_value = min(MEANS[i]-SEMS[i], MEANS[j]-SEMS[j])
		dx = abs(X[i]-X[j])
	
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':10,'shrinkB':10,'lw':2}
		ax.annotate('', xy=(X[i],max_value), xytext=(X[j],max_value), arrowprops=props)
	
		if values == False:
			if text == 'n.s.':
				kwargs = {'zorder':10, 'size':16, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.0/10))), **kwargs)
			if text != 'n.s.':
				kwargs = {'zorder':10, 'size':24, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(0.60/10))), **kwargs)
		if values == True:
			kwargs = {'zorder':10, 'size':12, 'ha':'center'}
			ax.annotate('p = ' + str(text), xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.15/10))), **kwargs)
	
	def simpleaxis(ax):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		
	def spine_shift(ax, shift = 10):
		for loc, spine in ax.spines.iteritems():
			if loc in ['left','bottom']:
				spine.set_position(('outward', shift)) # outward by 10 points
			elif loc in ['right','top']:
				spine.set_color('none') # don't draw spine
			else:
				raise ValueError('unknown spine location: %s'%loc)
	
	
	hit_mean = sp.mean(hit)
	fa_mean = sp.mean(fa)
	miss_mean = sp.mean(miss)
	cr_mean = sp.mean(cr)
	
	hit_sem = stats.sem(hit)
	fa_sem = stats.sem(fa)
	miss_sem = stats.sem(miss)
	cr_sem = stats.sem(cr)
	
	y_axis_swap = False
	
	MEANS = (hit_mean, miss_mean, fa_mean, cr_mean)
	SEMS = (hit_sem, miss_sem, fa_sem, cr_sem)

	if values == False:
		sig1 = 'n.s.'
		if p1 <= 0.05:
			sig1 = '*'
		if p1 <= 0.01:
			sig1 = '**'
		if p1 <= 0.001:
			sig1 = '***'
			
		sig2 = 'n.s.'
		if p2 <= 0.05:
			sig2 = '*'
		if p2 <= 0.01:
			sig2 = '**'
		if p2 <= 0.001:
			sig2 = '***'
		
		sig3 = 'n.s.'
		if p3 <= 0.05:
			sig3 = '*'
		if p3 <= 0.01:
			sig3 = '**'
		if p3 <= 0.001:
			sig3 = '***'
	else:
		sig1 = round(p1,5)
		sig2 = round(p2,5)
		sig3 = round(p3,5)
	
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
	
	N = 4
	ind = np.linspace(0,2,4) # the x locations for the groups
	bar_width = 0.5 # the width of the bars
	spacing = [0, 0, 0, 0]
	
	# FIGURE 1
	fig = plt.figure(figsize=(4,3))
	ax = fig.add_subplot(111)
	for i in range(N):
		ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['g','g','g','g'][i], alpha = [0.25,.5,.75,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
	simpleaxis(ax)
	spine_shift(ax)
	ax.set_xticklabels( ('--', '-','+', '++'), size=18 )
	ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	ax.set_ylim(ymin=0, ymax=maxvalue+(maxvalue/4.0))
	left = 0.2
	top = 0.915
	bottom = 0.2
	plt.subplots_adjust(top=top, bottom=bottom, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)
	
	# STATS:
	X = (ind[0], ind[1], ind[2], ind[3])
	if values == True:
		label_diff(0,1,sig1,X,MEANS, SEMS, values = True)
		label_diff(1,2,sig2,X,MEANS, SEMS, values = True)
		label_diff(2,3,sig3,X,MEANS, SEMS, values = True)
	if values == False:
		label_diff(0,1,sig1,X,MEANS, SEMS)
		label_diff(1,2,sig2,X,MEANS, SEMS)
		label_diff(2,3,sig3,X,MEANS, SEMS)
	
	return(fig)

def plot_permutations_and_ROC(perm_results, observed_mean_difference, significance, out_i, out_p):
	
	import matplotlib.pyplot as plt
	
	fig = plt.figure(figsize=(20,24))
	
	ax1 = plt.subplot2grid((4,2), (0,0), colspan=2)
	ax1.hist(perm_results[0], 50, color = 'b', alpha = 0.75, linewidth = 1, edgecolor=('w'))
	ax1.set_title('permutation test PPD - Yes vs No')
	ax1.set_xlabel('difference between means')
	ax1.axvline(observed_mean_difference[0],-1,1, color = 'r', linewidth = 3)
	
	if significance[0] < 0.005:
		ax1.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/40), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[0],4)) + ', p < 0.005')
	else:
		ax1.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/40), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[0],4)) + ', p = ' + str(significance[0]))
		
	if out_p[0] < 0.005:
		ax1.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/40), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/4), 'ROC index = ' + str(round(out_i[0],4)) + ', p < 0.005')
	else:
		ax1.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/40), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/4), 'ROC index = ' + str(round(out_i[0],4)) + ', p = ' + str(out_p[0]))
		
	ax2 = plt.subplot2grid((4,2), (1, 0))
	ax2.hist(perm_results[1], 25, color = 'b', alpha = 0.75, linewidth = 1, edgecolor=('w'))
	ax2.set_title('permutation test PPD - Hit vs Miss')
	ax2.set_xlabel('difference between means')
	ax2.axvline(observed_mean_difference[1],-1,1, color = 'r', linewidth = 3)
	
	if significance[1] < 0.005:
		ax2.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/20), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[1],4)) + ', p < 0.005')
	else:
		ax2.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/20), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[1],4)) + ', p = ' + str(significance[1]))
	
	ax3 = plt.subplot2grid((4,2), (1, 1))
	ax3.hist(perm_results[2], 25, color = 'b', alpha = 0.75, linewidth = 1, edgecolor=('w'))
	ax3.set_title('permutation test PPD - FA vs CR')
	ax3.set_xlabel('difference between means')
	ax3.axvline(observed_mean_difference[2],-1,1, color = 'r', linewidth = 3)
	
	if significance[2] < 0.005:
		ax3.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/20), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[2],4)) + ', p < 0.005')
	else:
		ax3.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/20), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[2],4)) + ', p = ' + str(significance[2]))
	
	ax4 = plt.subplot2grid((4,2), (2, 0))
	ax4.hist(perm_results[3], 25, color = 'b', alpha = 0.75, linewidth = 1, edgecolor=('w'))
	ax4.set_title('permutation test PPD - Hit vs FA')
	ax4.set_xlabel('difference between means')
	ax4.axvline(observed_mean_difference[3],-1,1, color = 'r', linewidth = 3)
	
	if significance[3] < 0.005:
		ax4.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/20), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[3],4)) + ', p < 0.005')
	else:
		ax4.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/20), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[3],4)) + ', p = ' + str(significance[3]))
	
	ax5 = plt.subplot2grid((4,2), (2, 1))
	ax5.hist(perm_results[4], 25, color = 'b', alpha = 0.75, linewidth = 1, edgecolor=('w'))
	ax5.set_title('permutation test PPD - Miss vs CR')
	ax5.set_xlabel('difference between means')
	ax5.axvline(observed_mean_difference[4],-1,1, color = 'r', linewidth = 3)
	
	if significance[4] < 0.005:
		ax5.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/20), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[4],4)) + ', p < 0.005')
	else:
		ax5.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/20), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[4],4)) + ', p = ' + str(significance[4]))
	
	ax6 = plt.subplot2grid((4,2), (3,0), colspan=2)
	ax6.hist(perm_results[5], 50, color = 'b', alpha = 0.75, linewidth = 1, edgecolor=('w'))
	ax6.set_title('permutation test PPD - Correct vs Incorrect')
	ax6.set_xlabel('difference between means')
	ax6.axvline(observed_mean_difference[5],-1,1, color = 'r', linewidth = 3)
		
	if significance[5] < 0.005:
		ax6.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/40), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[5],4)) + ', p < 0.005')
	else:
		ax6.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/40), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/6), 'observed difference between\nmeans = ' + str(round(observed_mean_difference[5],4)) + ', p = ' + str(significance[5]))	
	
	if out_p[1] < 0.005:
		ax6.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/40), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/4), 'ROC index = ' + str(round(out_i[1],4)) + ', p < 0.005')
	else:
		ax6.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/40), plt.axis()[3]-((abs(plt.axis()[2])+abs(plt.axis()[3]))/4), 'ROC index = ' + str(round(out_i[1],4)) + ', p = ' + str(out_p[1]))
		
	return(fig)


def plot_feed_conf(subject, feedback_locked_array_joined, omission_indices_joined, hit_indices_joined, fa_indices_joined, cr_indices_joined, miss_indices_joined, confidence_0, confidence_1, confidence_2, confidence_3):
	
	import scipy as sp
	import scipy.stats as stats
	import scipy.signal as signal
	import bottleneck
	import numpy as np
	import matplotlib.pyplot as plt
	
	if subject == 'dh':
		confidence_0 = confidence_1
	else:
		pass
	
	# Compute mean pupil responses HITS
	feed_locked_hit_0_mean = bottleneck.nanmean(feedback_locked_array_joined[((hit_indices_joined[0])*(confidence_0)),:], axis=0)
	feed_locked_hit_1_mean = bottleneck.nanmean(feedback_locked_array_joined[((hit_indices_joined[0])*(confidence_1)),:], axis=0)
	feed_locked_hit_2_mean = bottleneck.nanmean(feedback_locked_array_joined[((hit_indices_joined[0])*(confidence_2)),:], axis=0)
	feed_locked_hit_3_mean = bottleneck.nanmean(feedback_locked_array_joined[((hit_indices_joined[0])*(confidence_3)),:], axis=0)
	# Compute sem pupil responses HITS
	feed_locked_hit_0_sem = bottleneck.nanstd(feedback_locked_array_joined[((hit_indices_joined[0])*(confidence_0)),:], axis=0) / sp.sqrt( (hit_indices_joined[0]*confidence_0).sum() )
	feed_locked_hit_1_sem = bottleneck.nanstd(feedback_locked_array_joined[((hit_indices_joined[0])*(confidence_1)),:], axis=0) / sp.sqrt( (hit_indices_joined[0]*confidence_1).sum() )
	feed_locked_hit_2_sem = bottleneck.nanstd(feedback_locked_array_joined[((hit_indices_joined[0])*(confidence_2)),:], axis=0) / sp.sqrt( (hit_indices_joined[0]*confidence_2).sum() )
	feed_locked_hit_3_sem = bottleneck.nanstd(feedback_locked_array_joined[((hit_indices_joined[0])*(confidence_3)),:], axis=0) / sp.sqrt( (hit_indices_joined[0]*confidence_3).sum() )
	# Compute mean pupil responses FA
	feed_locked_fa_0_mean = bottleneck.nanmean(feedback_locked_array_joined[((fa_indices_joined[0])*(confidence_0)),:], axis=0)
	feed_locked_fa_1_mean = bottleneck.nanmean(feedback_locked_array_joined[((fa_indices_joined[0])*(confidence_1)),:], axis=0)
	feed_locked_fa_2_mean = bottleneck.nanmean(feedback_locked_array_joined[((fa_indices_joined[0])*(confidence_2)),:], axis=0)
	feed_locked_fa_3_mean = bottleneck.nanmean(feedback_locked_array_joined[((fa_indices_joined[0])*(confidence_3)),:], axis=0)
	# Compute sem pupil responses FA
	feed_locked_fa_0_sem = bottleneck.nanstd(feedback_locked_array_joined[((fa_indices_joined[0])*(confidence_0)),:], axis=0) / sp.sqrt( (fa_indices_joined[0]*confidence_0).sum() )
	feed_locked_fa_1_sem = bottleneck.nanstd(feedback_locked_array_joined[((fa_indices_joined[0])*(confidence_1)),:], axis=0) / sp.sqrt( (fa_indices_joined[0]*confidence_1).sum() )
	feed_locked_fa_2_sem = bottleneck.nanstd(feedback_locked_array_joined[((fa_indices_joined[0])*(confidence_2)),:], axis=0) / sp.sqrt( (fa_indices_joined[0]*confidence_2).sum() )
	feed_locked_fa_3_sem = bottleneck.nanstd(feedback_locked_array_joined[((fa_indices_joined[0])*(confidence_3)),:], axis=0) / sp.sqrt( (fa_indices_joined[0]*confidence_3).sum() )
	# Compute mean pupil responses MISS
	feed_locked_miss_0_mean = bottleneck.nanmean(feedback_locked_array_joined[((miss_indices_joined[0])*(confidence_0)),:], axis=0)
	feed_locked_miss_1_mean = bottleneck.nanmean(feedback_locked_array_joined[((miss_indices_joined[0])*(confidence_1)),:], axis=0)
	feed_locked_miss_2_mean = bottleneck.nanmean(feedback_locked_array_joined[((miss_indices_joined[0])*(confidence_2)),:], axis=0)
	feed_locked_miss_3_mean = bottleneck.nanmean(feedback_locked_array_joined[((miss_indices_joined[0])*(confidence_3)),:], axis=0)
	# Compute sem pupil responses MISS
	feed_locked_miss_0_sem = bottleneck.nanstd(feedback_locked_array_joined[((miss_indices_joined[0])*(confidence_0)),:], axis=0) / sp.sqrt( (miss_indices_joined[0]*confidence_0).sum() )
	feed_locked_miss_1_sem = bottleneck.nanstd(feedback_locked_array_joined[((miss_indices_joined[0])*(confidence_1)),:], axis=0) / sp.sqrt( (miss_indices_joined[0]*confidence_1).sum() )
	feed_locked_miss_2_sem = bottleneck.nanstd(feedback_locked_array_joined[((miss_indices_joined[0])*(confidence_2)),:], axis=0) / sp.sqrt( (miss_indices_joined[0]*confidence_2).sum() )
	feed_locked_miss_3_sem = bottleneck.nanstd(feedback_locked_array_joined[((miss_indices_joined[0])*(confidence_3)),:], axis=0) / sp.sqrt( (miss_indices_joined[0]*confidence_3).sum() )
	# Compute mean pupil responses CR
	feed_locked_cr_0_mean = bottleneck.nanmean(feedback_locked_array_joined[((cr_indices_joined[0])*(confidence_0)),:], axis=0)
	feed_locked_cr_1_mean = bottleneck.nanmean(feedback_locked_array_joined[((cr_indices_joined[0])*(confidence_1)),:], axis=0)
	feed_locked_cr_2_mean = bottleneck.nanmean(feedback_locked_array_joined[((cr_indices_joined[0])*(confidence_2)),:], axis=0)
	feed_locked_cr_3_mean = bottleneck.nanmean(feedback_locked_array_joined[((cr_indices_joined[0])*(confidence_3)),:], axis=0)
	# Compute sem pupil responses CR
	feed_locked_cr_0_sem = bottleneck.nanstd(feedback_locked_array_joined[((cr_indices_joined[0])*(confidence_0)),:], axis=0) / sp.sqrt( (cr_indices_joined[0]*confidence_0).sum() )
	feed_locked_cr_1_sem = bottleneck.nanstd(feedback_locked_array_joined[((cr_indices_joined[0])*(confidence_1)),:], axis=0) / sp.sqrt( (cr_indices_joined[0]*confidence_1).sum() )
	feed_locked_cr_2_sem = bottleneck.nanstd(feedback_locked_array_joined[((cr_indices_joined[0])*(confidence_2)),:], axis=0) / sp.sqrt( (cr_indices_joined[0]*confidence_2).sum() )
	feed_locked_cr_3_sem = bottleneck.nanstd(feedback_locked_array_joined[((cr_indices_joined[0])*(confidence_3)),:], axis=0) / sp.sqrt( (cr_indices_joined[0]*confidence_3).sum() )
	
	# # Compute mean pupil responses
	# feed_locked_correct_high_conf_mean = bottleneck.nanmean(feedback_locked_array_joined[((hit_indices_joined[0]+cr_indices_joined[0])*(confidence_2 + confidence_3)),:], axis=0)
	# feed_locked_correct_low_conf_mean = bottleneck.nanmean(feedback_locked_array_joined[((hit_indices_joined[0]+cr_indices_joined[0])*(confidence_0 + confidence_1)),:], axis=0)
	# feed_locked_incorrect_high_conf_mean = bottleneck.nanmean(feedback_locked_array_joined[((fa_indices_joined[0]+miss_indices_joined[0])*(confidence_2 + confidence_3)),:], axis=0)
	# feed_locked_incorrect_low_conf_mean = bottleneck.nanmean(feedback_locked_array_joined[((fa_indices_joined[0]+miss_indices_joined[0])*(confidence_0 + confidence_1)),:], axis=0)
	# # Compute std mean pupil responses
	# feed_locked_correct_high_conf_std = ( bottleneck.nanstd(feedback_locked_array_joined[((hit_indices_joined[0]+cr_indices_joined[0])*(confidence_2 + confidence_3)),:], axis=0) / sp.sqrt(((hit_indices_joined[0]+cr_indices_joined[0])*(confidence_2 + confidence_3)).sum()) )
	# feed_locked_correct_low_conf_std = ( bottleneck.nanstd(feedback_locked_array_joined[((hit_indices_joined[0]+cr_indices_joined[0])*(confidence_0 + confidence_1)),:], axis=0) / sp.sqrt(((hit_indices_joined[0]+cr_indices_joined[0])*(confidence_0 + confidence_1)).sum()) )
	# feed_locked_incorrect_high_conf_std = ( bottleneck.nanstd(feedback_locked_array_joined[((fa_indices_joined[0]+miss_indices_joined[0])*(confidence_2 + confidence_3)),:], axis=0) / sp.sqrt(((fa_indices_joined[0]+miss_indices_joined[0])*(confidence_2 + confidence_3)).sum()) )
	# feed_locked_incorrect_low_conf_std = ( bottleneck.nanstd(feedback_locked_array_joined[((fa_indices_joined[0]+miss_indices_joined[0])*(confidence_0 + confidence_1)),:], axis=0) / sp.sqrt(((fa_indices_joined[0]+miss_indices_joined[0])*(confidence_0 + confidence_1)).sum()) )
	
	# Make the plot
	figure_mean_pupil_locked_to_feedback_conf = plt.figure(figsize=(20, 18))
	# HITS
	plt.subplot(411)
	xb = np.arange(-999,2000)
	p1, = plt.plot(xb, feed_locked_hit_0_mean, color = 'r', alpha = 0.10, linewidth=2)
	p2, = plt.plot(xb, feed_locked_hit_1_mean, color = 'r', alpha = 0.40, linewidth=2)
	p3, = plt.plot(xb, feed_locked_hit_2_mean, color = 'r', alpha = 0.70, linewidth=2)
	p4, = plt.plot(xb, feed_locked_hit_3_mean, color = 'r', alpha = 1.00, linewidth=2)
	plt.fill_between( xb, (feed_locked_hit_0_mean+feed_locked_hit_0_sem), (feed_locked_hit_0_mean-feed_locked_hit_0_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (feed_locked_hit_1_mean+feed_locked_hit_1_sem), (feed_locked_hit_1_mean-feed_locked_hit_1_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (feed_locked_hit_2_mean+feed_locked_hit_2_sem), (feed_locked_hit_2_mean-feed_locked_hit_2_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (feed_locked_hit_3_mean+feed_locked_hit_3_sem), (feed_locked_hit_3_mean-feed_locked_hit_3_sem), alpha=0.1, color = 'r' )
	plt.title(subject + "_pupil_change_feedback_locked_HITS_confidence", size = 'xx-large')
	plt.ylabel("Pupil change (Z)", size = 'xx-large')
	# plt.xlabel("Time in milliseconds", size = 'xx-large')
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.axvline(0, -1, 1)
	plt.legend([p1, p2, p3, p4], ["Hit - Conf0; " + str( ((hit_indices_joined[0])*(confidence_0)).sum() ) + " trials", "Hit - Conf1; " + str( ((hit_indices_joined[0])*(confidence_1)).sum() ) + " trials", "Hit - Conf2; " + str( ((hit_indices_joined[0])*(confidence_2)).sum() ) + " trials", "Hit - Conf3; " + str( ((hit_indices_joined[0])*(confidence_3)).sum() ) + " trials"], loc = 2)
	# FA
	plt.subplot(412)
	xb = np.arange(-999,2000)
	p1, = plt.plot(xb, feed_locked_fa_0_mean, color = 'r', alpha = 0.10, linewidth=2)
	p2, = plt.plot(xb, feed_locked_fa_1_mean, color = 'r', alpha = 0.40, linewidth=2)
	p3, = plt.plot(xb, feed_locked_fa_2_mean, color = 'r', alpha = 0.70, linewidth=2)
	p4, = plt.plot(xb, feed_locked_fa_3_mean, color = 'r', alpha = 1.00, linewidth=2)
	plt.fill_between( xb, (feed_locked_fa_0_mean+feed_locked_fa_0_sem), (feed_locked_fa_0_mean-feed_locked_fa_0_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (feed_locked_fa_1_mean+feed_locked_fa_1_sem), (feed_locked_fa_1_mean-feed_locked_fa_1_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (feed_locked_fa_2_mean+feed_locked_fa_2_sem), (feed_locked_fa_2_mean-feed_locked_fa_2_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (feed_locked_fa_3_mean+feed_locked_fa_3_sem), (feed_locked_fa_3_mean-feed_locked_fa_3_sem), alpha=0.1, color = 'r' )
	plt.title(subject + "_pupil_change_feedback_locked_FA_confidence", size = 'xx-large')
	plt.ylabel("Pupil change (Z)", size = 'xx-large')
	# plt.xlabel("Time in milliseconds", size = 'xx-large')
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.axvline(0, -1, 1)
	plt.legend([p1, p2, p3, p4], ["Fa - Conf0; " + str( ((fa_indices_joined[0])*(confidence_0)).sum() ) + " trials", "Fa - Conf1; " + str( ((fa_indices_joined[0])*(confidence_1)).sum() ) + " trials", "Fa - Conf2; " + str( ((fa_indices_joined[0])*(confidence_2)).sum() ) + " trials", "Fa - Conf3; " + str( ((fa_indices_joined[0])*(confidence_3)).sum() ) + " trials"], loc = 2)
	# MISS
	plt.subplot(413)
	xb = np.arange(-999,2000)
	p1, = plt.plot(xb, feed_locked_miss_0_mean, color = 'b', alpha = 0.10, linewidth=2)
	p2, = plt.plot(xb, feed_locked_miss_1_mean, color = 'b', alpha = 0.40, linewidth=2)
	p3, = plt.plot(xb, feed_locked_miss_2_mean, color = 'b', alpha = 0.70, linewidth=2)
	p4, = plt.plot(xb, feed_locked_miss_3_mean, color = 'b', alpha = 1.00, linewidth=2)
	plt.fill_between( xb, (feed_locked_miss_0_mean+feed_locked_miss_0_sem), (feed_locked_miss_0_mean-feed_locked_miss_0_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xb, (feed_locked_miss_1_mean+feed_locked_miss_1_sem), (feed_locked_miss_1_mean-feed_locked_miss_1_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xb, (feed_locked_miss_2_mean+feed_locked_miss_2_sem), (feed_locked_miss_2_mean-feed_locked_miss_2_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xb, (feed_locked_miss_3_mean+feed_locked_miss_3_sem), (feed_locked_miss_3_mean-feed_locked_miss_3_sem), alpha=0.1, color = 'b' )
	plt.title(subject + "_pupil_change_feedback_locked_MISS_confidence", size = 'xx-large')
	plt.ylabel("Pupil change (Z)", size = 'xx-large')
	# plt.xlabel("Time in milliseconds", size = 'xx-large')
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.axvline(0, -1, 1)
	plt.legend([p1, p2, p3, p4], ["Miss - Conf0; " + str( ((miss_indices_joined[0])*(confidence_0)).sum() ) + " trials", "Miss - Conf1; " + str( ((miss_indices_joined[0])*(confidence_1)).sum() ) + " trials", "Miss - Conf2; " + str( ((miss_indices_joined[0])*(confidence_2)).sum() ) + " trials", "Miss - Conf3; " + str( ((miss_indices_joined[0])*(confidence_3)).sum() ) + " trials"], loc = 2)
	# CR
	plt.subplot(414)
	xb = np.arange(-999,2000)
	p1, = plt.plot(xb, feed_locked_cr_0_mean, color = 'b', alpha = 0.10, linewidth=2)
	p2, = plt.plot(xb, feed_locked_cr_1_mean, color = 'b', alpha = 0.40, linewidth=2)
	p3, = plt.plot(xb, feed_locked_cr_2_mean, color = 'b', alpha = 0.70, linewidth=2)
	p4, = plt.plot(xb, feed_locked_cr_3_mean, color = 'b', alpha = 1.00, linewidth=2)
	plt.fill_between( xb, (feed_locked_cr_0_mean+feed_locked_cr_0_sem), (feed_locked_cr_0_mean-feed_locked_cr_0_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xb, (feed_locked_cr_1_mean+feed_locked_cr_1_sem), (feed_locked_cr_1_mean-feed_locked_cr_1_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xb, (feed_locked_cr_2_mean+feed_locked_cr_2_sem), (feed_locked_cr_2_mean-feed_locked_cr_2_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xb, (feed_locked_cr_3_mean+feed_locked_cr_3_sem), (feed_locked_cr_3_mean-feed_locked_cr_3_sem), alpha=0.1, color = 'b' )
	plt.title(subject + "_pupil_change_feedback_locked_CR_confidence", size = 'xx-large')
	plt.ylabel("Pupil change (Z)", size = 'xx-large')
	plt.xlabel("Time in milliseconds", size = 'xx-large')
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.axvline(0, -1, 1)
	plt.legend([p1, p2, p3, p4], ["CR - Conf0; " + str( ((cr_indices_joined[0])*(confidence_0)).sum() ) + " trials", "CR - Conf1; " + str( ((cr_indices_joined[0])*(confidence_1)).sum() ) + " trials", "CR - Conf2; " + str( ((cr_indices_joined[0])*(confidence_2)).sum() ) + " trials", "CR - Conf3; " + str( ((cr_indices_joined[0])*(confidence_3)).sum() ) + " trials"], loc = 2)
	return(figure_mean_pupil_locked_to_feedback_conf)


def plot_correlation(subject, bpd, ppd):
	" plot correlation bpd - ppd"
	
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	
	# Correlation:
	slope, intercept, r_value, p_value, std_err = stats.linregress(bpd,ppd)
	
	figure_correlation_BPD_PPD = plt.figure(figsize=(16,12))
	(m,b) = sp.polyfit(bpd, ppd, 1)
	phasic_pupil_diameter_p = sp.polyval([m,b], bpd)
	plt.plot(bpd,phasic_pupil_diameter_p, color = 'r', linewidth = 3)
	plt.scatter(bpd, ppd)
	plt.xlabel('BPD (Z)')
	plt.ylabel('PPD - linearly projected')
	plt.title(str(subject) + " - Correlation BPD and PPD")
	# plt.text(plt.axis()[1]-2,plt.axis()[3]-3,'slope = ' + str(round(slope, 3)))
	if round(p_value,5) < 0.005:
		plt.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'regression = ' + str(round(r_value, 3)) + '\np-value < 0.005')
	else:	
		plt.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'regression = ' + str(round(r_value, 3)) + '\np-value = ' + str(round(p_value, 5)))
	
	return(figure_correlation_BPD_PPD)
	


def plot_PPDs1(subject, hit, fa, miss, cr, p1, p2, values = False):
	
	# Height text is not good yet... Now I used +0.35 for Remy and me, and +0.45 for David.
	# For David, use label_diff2 --> for the 0.45, and also for 'n.s.' printed in large, instead of xx-large.
	# Finally, y-axis is still hardcoded yet...
	
	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	
	# hit = ppd_lin_A_joined[hit_indices_joined[0]]
	# fa = ppd_lin_A_joined[fa_indices_joined[0]]
	# miss = ppd_lin_A_joined[miss_indices_joined[0]]
	# cr = ppd_lin_A_joined[cr_indices_joined[0]]
	# values = False	
	# p1 = 0.2
	# p2 = 0.0001
	
	hit_mean = sp.mean(hit)
	fa_mean = sp.mean(fa)
	miss_mean = sp.mean(miss)
	cr_mean = sp.mean(cr)
	
	hit_sem = stats.sem(hit)
	fa_sem = stats.sem(fa)
	miss_sem = stats.sem(miss)
	cr_sem = stats.sem(cr)
	
	yes_mean = sp.mean(np.concatenate((hit, fa), axis=0))
	no_mean = sp.mean(np.concatenate((miss, cr), axis=0))
	correct_mean = sp.mean(np.concatenate((hit, cr), axis=0))
	incorrect_mean = sp.mean(np.concatenate((fa, miss), axis=0))
	
	yes_sem = stats.sem(np.concatenate((hit, fa), axis=0))
	no_sem = stats.sem(np.concatenate((miss, cr), axis=0))
	correct_sem = stats.sem(np.concatenate((hit, cr), axis=0))
	incorrect_sem = stats.sem(np.concatenate((fa, miss), axis=0))
	
	MEANS = (hit_mean, miss_mean, fa_mean, cr_mean)
	SEMS = (hit_sem, miss_sem, fa_sem, cr_sem)
	
	MEANS2 = (yes_mean, no_mean, correct_mean, incorrect_mean)
	SEMS2 = (yes_sem, no_sem, correct_sem, incorrect_sem)	
	
	if values == False:
		sig1 = 'n.s.'
		if p1 <= 0.05:
			sig1 = '*'
		if p1 <= 0.01:
			sig1 = '**'
		if p1 <= 0.001:
			sig1 = '***'
			
		sig2 = 'n.s.'
		if p2 <= 0.05:
			sig2 = '*'
		if p2 <= 0.01:
			sig2 = '**'
		if p2 <= 0.001:
			sig2 = '***'
	else:
		sig1 = p1
		sig2 = p2
	
	def label_diff1(i,j,text,X,Y,Z):
		x = (X[i]+X[j])/2
		y = max(Y[i]+Z[i], Y[j]+Z[j])
		dx = abs(X[i]-X[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':22,'shrinkB':22,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		if text == 'n.s.':
			kwargs = {'zorder':10, 'size':'large', 'ha':'center'}
			if plt.axis()[3] == 1.5:
				ax.annotate(text, xy=(x,y+0.17), **kwargs)
			if plt.axis()[3] == 2:
				ax.annotate(text, xy=(x,y+0.22), **kwargs)
			if plt.axis()[3] == 2.5:
				ax.annotate(text, xy=(x,y+0.28), **kwargs)
			if plt.axis()[3] == 3:
				ax.annotate(text, xy=(x,y+0.35), **kwargs)
			if plt.axis()[3] == 3.5:
				ax.annotate(text, xy=(x,y+0.40), **kwargs)
		if text != 'n.s.':
			kwargs = {'zorder':10, 'size':'xx-large', 'ha':'center'}
			if plt.axis()[3] == 1.5:
				ax.annotate(text, xy=(x,y+0.13), **kwargs)
			if plt.axis()[3] == 2:
				ax.annotate(text, xy=(x,y+0.18), **kwargs)
			if plt.axis()[3] == 2.5:
				ax.annotate(text, xy=(x,y+0.24), **kwargs)
			if plt.axis()[3] == 3:
				ax.annotate(text, xy=(x,y+0.30), **kwargs)
			if plt.axis()[3] == 3.5:
				ax.annotate(text, xy=(x,y+0.35), **kwargs)
	
	
	def label_diff2(i,j,text,X,Y,Z):
		x = (X[i]+X[j])/2
		y = max(Y[i]+Z[i], Y[j]+Z[j])
		dx = abs(X[i]-X[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':22,'shrinkB':22,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		kwargs = {'zorder':10, 'size':'large', 'ha':'center'}
		if plt.axis()[3] == 1.5:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.17), **kwargs)
		if plt.axis()[3] == 2:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.22), **kwargs)
		if plt.axis()[3] == 2.5:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.28), **kwargs)
		if plt.axis()[3] == 3:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.35), **kwargs)
		if plt.axis()[3] == 3:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.40), **kwargs)
	
	
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'align': 'center'}
		
	N = 4
	ind = np.arange(N)  # the x locations for the groups
	width = 0.45       # the width of the bars
	
	# FIGURE 1
	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind[0]+width, MEANS[0], width, yerr=SEMS[0], color='r', alpha = 0.75, **my_dict)
	rects2 = ax.bar(ind[1], MEANS[1], width, yerr=SEMS[1], color='b', alpha = 0.5, **my_dict)
	rects3 = ax.bar(ind[2], MEANS[2], width, yerr=SEMS[2], color='r', alpha = 0.5, **my_dict)
	rects4 = ax.bar(ind[3]-width, MEANS[3], width, yerr=SEMS[3], color='b', alpha = 0.75, **my_dict)
	# ax.set_ylim( (0.5) )
	ax.set_ylabel('PPD - linearly projected', size = 'xx-large')
	ax.set_title(str(subject) + ' - mean PPD around response', size = 'xx-large')
	ax.set_xticks( (ind[0]+width, ind[1], ind[2], ind[3]-width) )
	ax.set_xticklabels( ('HIT', 'MISS','FA', 'CR') )
	ax.tick_params(axis='x', which='major', labelsize=24)
	ax.tick_params(axis='y', which='major', labelsize=16)
	ax.set_ylim(ymax = round((plt.axis()[3]+0.5)*2.0, 0)/2 )
	# STATS:
	X = (ind[0]+width, ind[1], ind[2], ind[3]-width)
	if values == True:
		label_diff2(0,1,sig1,X,MEANS, SEMS)
		label_diff2(2,3,sig2,X,MEANS, SEMS)
	else:
		label_diff1(0,1,sig1,X,MEANS, SEMS)
		label_diff1(2,3,sig2,X,MEANS, SEMS)
	
	return(fig)


def plot_PPDs2(subject, hit, fa, miss, cr, p1, p2, values = False):
	
	# Height text is not good yet... Now I used +0.35 for Remy and me, and +0.45 for David.
	# For David, use label_diff2 --> for the 0.45, and also for 'n.s.' printed in large, instead of xx-large.
	# Finally, y-axis is still hardcoded yet...
	
	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	
	# hit = ppd_lin_A_joined[hit_indices_joined[0]]
	# fa = ppd_lin_A_joined[fa_indices_joined[0]]
	# miss = ppd_lin_A_joined[miss_indices_joined[0]]
	# cr = ppd_lin_A_joined[cr_indices_joined[0]]
	# values = False
	# type_plot = 1
	# p1 = 0.2
	# p2 = 0.0001
	
	hit_mean = sp.mean(hit)
	fa_mean = sp.mean(fa)
	miss_mean = sp.mean(miss)
	cr_mean = sp.mean(cr)
	
	hit_sem = stats.sem(hit)
	fa_sem = stats.sem(fa)
	miss_sem = stats.sem(miss)
	cr_sem = stats.sem(cr)
	
	yes_mean = sp.mean(np.concatenate((hit, fa), axis=0))
	no_mean = sp.mean(np.concatenate((miss, cr), axis=0))
	correct_mean = sp.mean(np.concatenate((hit, cr), axis=0))
	incorrect_mean = sp.mean(np.concatenate((fa, miss), axis=0))
	
	yes_sem = stats.sem(np.concatenate((hit, fa), axis=0))
	no_sem = stats.sem(np.concatenate((miss, cr), axis=0))
	correct_sem = stats.sem(np.concatenate((hit, cr), axis=0))
	incorrect_sem = stats.sem(np.concatenate((fa, miss), axis=0))
	
	MEANS = (hit_mean, miss_mean, fa_mean, cr_mean)
	SEMS = (hit_sem, miss_sem, fa_sem, cr_sem)
	
	MEANS2 = (yes_mean, no_mean, correct_mean, incorrect_mean)
	SEMS2 = (yes_sem, no_sem, correct_sem, incorrect_sem)	
	
	if values == False:
		sig1 = 'n.s.'
		if p1 <= 0.05:
			sig1 = '*'
		if p1 <= 0.01:
			sig1 = '**'
		if p1 <= 0.001:
			sig1 = '***'
			
		sig2 = 'n.s.'
		if p2 <= 0.05:
			sig2 = '*'
		if p2 <= 0.01:
			sig2 = '**'
		if p2 <= 0.001:
			sig2 = '***'
			
	else:
		sig1 = p1
		sig2 = p2
		
	def label_diff1(i,j,text,X,Y,Z):
		x = (X[i]+X[j])/2
		y = max(Y[i]+Z[i], Y[j]+Z[j])
		dx = abs(X[i]-X[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':22,'shrinkB':22,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		if text == 'n.s.':
			kwargs = {'zorder':10, 'size':'large', 'ha':'center'}
			if plt.axis()[3] == 1.5:
				ax.annotate(text, xy=(x,y+0.17), **kwargs)
			if plt.axis()[3] == 2:
				ax.annotate(text, xy=(x,y+0.22), **kwargs)
			if plt.axis()[3] == 2.5:
				ax.annotate(text, xy=(x,y+0.28), **kwargs)
			if plt.axis()[3] == 3:
				ax.annotate(text, xy=(x,y+0.35), **kwargs)
			if plt.axis()[3] == 3.5:
				ax.annotate(text, xy=(x,y+0.40), **kwargs)
		if text != 'n.s.':
			kwargs = {'zorder':10, 'size':'xx-large', 'ha':'center'}
			if plt.axis()[3] == 1.5:
				ax.annotate(text, xy=(x,y+0.13), **kwargs)
			if plt.axis()[3] == 2:
				ax.annotate(text, xy=(x,y+0.18), **kwargs)
			if plt.axis()[3] == 2.5:
				ax.annotate(text, xy=(x,y+0.24), **kwargs)
			if plt.axis()[3] == 3:
				ax.annotate(text, xy=(x,y+0.30), **kwargs)
			if plt.axis()[3] == 3.5:
				ax.annotate(text, xy=(x,y+0.35), **kwargs)
	
	
	def label_diff2(i,j,text,X,Y,Z):
		x = (X[i]+X[j])/2
		y = max(Y[i]+Z[i], Y[j]+Z[j])
		dx = abs(X[i]-X[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':22,'shrinkB':22,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		kwargs = {'zorder':10, 'size':'large', 'ha':'center'}
		if plt.axis()[3] == 1.5:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.17), **kwargs)
		if plt.axis()[3] == 2:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.22), **kwargs)
		if plt.axis()[3] == 2.5:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.28), **kwargs)
		if plt.axis()[3] == 3:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.35), **kwargs)
		if plt.axis()[3] == 3:
			ax.annotate('p = ' + str(round(text,3)), xy=(x,y+0.40), **kwargs)
	
	
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'align': 'center'}
	
	N = 4
	ind = np.arange(N)  # the x locations for the groups
	width = 0.45       # the width of the bars
	
	# FIGURE
	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind[0]+width, MEANS2[0], width, yerr=SEMS2[0], color='r', alpha = 0.75, **my_dict)
	rects2 = ax.bar(ind[1], MEANS2[1], width, yerr=SEMS2[1], color='b', alpha = 0.75, **my_dict)
	rects3 = ax.bar(ind[2], MEANS2[2], width, yerr=SEMS2[2], color='k', alpha = 0.5, **my_dict)
	rects4 = ax.bar(ind[3]-width, MEANS2[3], width, yerr=SEMS2[3], color='k', alpha = 0.5, **my_dict)
	# ax.set_ylim( (0.5) )
	ax.set_ylabel('PPD - linearly projected', size = 'xx-large')
	ax.set_title(str(subject) + ' - mean PPD around response', size = 'xx-large')
	ax.set_xticks( (ind[0]+width, ind[1], ind[2], ind[3]-width) )
	ax.set_xticklabels( ('YES', 'NO','CORR.', 'INCORR.') )
	ax.tick_params(axis='x', which='major', labelsize=24)
	ax.tick_params(axis='y', which='major', labelsize=16)
	ax.set_ylim(ymax = round((plt.axis()[3]+0.5)*2.0, 0)/2 )
	# STATS:
	X = (ind[0]+width, ind[1], ind[2], ind[3]-width)
	if values == True:
		label_diff2(0,1,sig1,X,MEANS2, SEMS2)
		label_diff2(2,3,sig2,X,MEANS2, SEMS2)
	else:
		label_diff1(0,1,sig1,X,MEANS2, SEMS2)
		label_diff1(2,3,sig2,X,MEANS2, SEMS2)
	
	return(fig)


def plot_d_prime(d_prime_means, d_prime_sems):
	
	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	
	MEANS = d_prime_means
	SEMS = d_prime_sems
	
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'align': 'center'}
	
	N = 2
	ind = np.linspace(0,1.5,4)  # the x locations for the groups
	width = 0.45       # the width of the bars
	
	# FIGURE 1
	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind[0], MEANS[0], width, yerr=SEMS[0], color='r', alpha = 0.75, **my_dict)
	rects2 = ax.bar(ind[1], MEANS[1], width, yerr=SEMS[1], color='b', alpha = 0.75, **my_dict)
	rects3 = ax.bar(ind[2], MEANS[2], width, yerr=SEMS[2], color='g', alpha = 0.75, **my_dict)
	rects4 = ax.bar(ind[3], MEANS[3], width, yerr=SEMS[3], color='y', alpha = 0.75, **my_dict)
	ax.set_ylabel("d'", size = 'xx-large')
	ax.set_title('All - d_prime', size = 'xx-large')
	ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
	ax.set_xticklabels( ('JWG', 'RN', 'DH', 'DL') )
	ax.tick_params(axis='x', which='major', labelsize=24)
	ax.tick_params(axis='y', which='major', labelsize=16)
	ax.set_ylim(ymax = 2)
	
	return(fig)


def plot_criterion(criterion_means, criterion_sems):
	
	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	
	MEANS = criterion_means
	SEMS = criterion_sems
	
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'align': 'center'}
	
	N = 2
	ind = np.linspace(0,1.5,4)  # the x locations for the groups
	width = 0.45       # the width of the bars
	
	# FIGURE 1
	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind[0], MEANS[0], width, yerr=SEMS[0], color='r', alpha = 0.75, **my_dict)
	rects2 = ax.bar(ind[1], MEANS[1], width, yerr=SEMS[1], color='b', alpha = 0.75, **my_dict)
	rects3 = ax.bar(ind[2], MEANS[2], width, yerr=SEMS[2], color='g', alpha = 0.75, **my_dict)
	rects4 = ax.bar(ind[3], MEANS[3], width, yerr=SEMS[3], color='y', alpha = 0.75, **my_dict)
	ax.set_ylabel('criterion', size = 'xx-large')
	ax.set_title('All - criterion', size = 'xx-large')
	ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
	ax.set_xticklabels( ('JWG', 'RN', 'DH', 'DL') )
	ax.tick_params(axis='x', which='major', labelsize=24)
	ax.tick_params(axis='y', which='major', labelsize=16)
	# ax.set_ylim(ymax = 2)
	
	return(fig)


def plot_PPDs_feed(subject, ppd_feed, hit, fa, miss, cr, confidence_0, confidence_1, confidence_2, confidence_3):
	
	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	
	if subject == 'dh':
		confidence_0 = confidence_1
	else:
		pass
		
	hit2 = ( sp.mean(ppd_feed[hit * confidence_0]),sp.mean(ppd_feed[hit * confidence_1]),sp.mean(ppd_feed[hit * confidence_2]),sp.mean(ppd_feed[hit * confidence_3]) )
	fa2 = ( sp.mean(ppd_feed[fa * confidence_0]),sp.mean(ppd_feed[fa * confidence_1]),sp.mean(ppd_feed[fa * confidence_2]),sp.mean(ppd_feed[fa * confidence_3]) )
	miss2 = ( sp.mean(ppd_feed[miss * confidence_0]),sp.mean(ppd_feed[miss * confidence_1]),sp.mean(ppd_feed[miss * confidence_2]),sp.mean(ppd_feed[miss * confidence_3]) )
	cr2 = ( sp.mean(ppd_feed[cr * confidence_0]),sp.mean(ppd_feed[cr * confidence_1]),sp.mean(ppd_feed[cr * confidence_2]),sp.mean(ppd_feed[cr * confidence_3] ))
	
	hit_sem2 = ( stats.sem(ppd_feed[hit * confidence_0]),stats.sem(ppd_feed[hit * confidence_1]),stats.sem(ppd_feed[hit * confidence_2]),stats.sem(ppd_feed[hit * confidence_3]) )
	fa_sem2 = ( stats.sem(ppd_feed[fa * confidence_0]),stats.sem(ppd_feed[fa * confidence_1]),stats.sem(ppd_feed[fa * confidence_2]),stats.sem(ppd_feed[fa * confidence_3]) )
	miss_sem2 = ( stats.sem(ppd_feed[miss * confidence_0]),stats.sem(ppd_feed[miss * confidence_1]),stats.sem(ppd_feed[miss * confidence_2]),stats.sem(ppd_feed[miss * confidence_3]) )
	cr_sem2 = ( stats.sem(ppd_feed[cr * confidence_0]),stats.sem(ppd_feed[cr * confidence_1]),stats.sem(ppd_feed[cr * confidence_2]),stats.sem(ppd_feed[cr * confidence_3]) )
	
	N = 16
	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars
	fig = plt.figure(figsize=(20,8))
	ax = fig.add_subplot(111)
	rects5 = ax.bar(ind[0:4]+width, hit2, width, color=('r'), ecolor = 'k', yerr=hit_sem2, linewidth = 0, alpha=0.75)
	rects6 = ax.bar(ind[4:8]+width, fa2, width, color=('r'), ecolor = 'k', yerr=fa_sem2, linewidth = 0, alpha=0.5)
	rects7 = ax.bar(ind[8:12]+width, miss2, width, color=('b'), ecolor = 'k', yerr=miss_sem2, linewidth = 0, alpha=0.5)
	rects8 = ax.bar(ind[12:16]+width, cr2, width, color=('b'), ecolor = 'k', yerr=cr_sem2, linewidth = 0, alpha=0.75)
	ax.set_ylim(ymin=0)
	ax.set_ylabel('Pupil size - linearly projected', size = 16)
	ax.set_xlabel('Confidence rating', size = 16)
	ax.set_title(str(subject) + ' - mean PPD after feedback', size = 20)
	ax.set_xticks(ind+(1.5*width))
	ax.set_xticklabels( ('not', 'little', 'quite', 'very','not', 'little', 'quite', 'very','not', 'little', 'quite', 'very','not', 'little', 'quite', 'very') )
	
	return(fig)
	
	# stats.ttest_rel( (ppd[target_indices_joined][0:797]), (ppd[no_target_indices_joined]) )
	# stats.ttest_rel( (ppd[answer_yes_indices_joined[0]][0:732]), (ppd[answer_no_indices_joined[0]]) )


def plot_PPDs_feed2(subject, ppd_feed, hit, cr, confidence_0, confidence_1, confidence_2, confidence_3, p1, p2, p3, p4):
	
	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	
	# ppd_feed = ppd_feed_lin_A_joined
	# p1 = 0.2
	# p2 = 0.0001
	# p3 = 0.03
	# p4 = 0
		
	hit1 = sp.mean(ppd_feed[hit*confidence_0])
	hit2 = sp.mean(ppd_feed[hit*confidence_1])
	hit3 = sp.mean(ppd_feed[hit*confidence_2])
	hit4 = sp.mean(ppd_feed[hit*confidence_3])
	hit1_sem = stats.sem(ppd_feed[hit*confidence_0])
	hit2_sem = stats.sem(ppd_feed[hit*confidence_1])
	hit3_sem = stats.sem(ppd_feed[hit*confidence_2])
	hit4_sem = stats.sem(ppd_feed[hit*confidence_3])
	
	cr1 = sp.mean(ppd_feed[cr*confidence_0])
	cr2 = sp.mean(ppd_feed[cr*confidence_1])
	cr3 = sp.mean(ppd_feed[cr*confidence_2])
	cr4 = sp.mean(ppd_feed[cr*confidence_3])
	cr1_sem = stats.sem(ppd_feed[cr*confidence_0])
	cr2_sem = stats.sem(ppd_feed[cr*confidence_1])
	cr3_sem = stats.sem(ppd_feed[cr*confidence_2])
	cr4_sem = stats.sem(ppd_feed[cr*confidence_3])
	
	MEANS = (hit1, hit2, hit3, hit4, cr1, cr2, cr3, cr4)
	SEMS = (hit1_sem, hit2_sem, hit3_sem, hit4_sem, cr1_sem, cr2_sem, cr3_sem, cr4_sem)
	
	# hit2 = ( sp.mean(ppd_feed[hit_indices_joined[0] * confidence_0]),sp.mean(ppd_feed[hit_indices_joined[0] * confidence_1]),sp.mean(ppd_feed[hit_indices_joined[0] * confidence_2]),sp.mean(ppd_feed[hit_indices_joined[0] * confidence_3]) )
	# cr2 = ( sp.mean(ppd_feed[cr_indices_joined[0] * confidence_0]),sp.mean(ppd_feed[cr_indices_joined[0] * confidence_1]),sp.mean(ppd_feed[cr_indices_joined[0] * confidence_2]),sp.mean(ppd_feed[cr_indices_joined[0] * confidence_3] ))
	# 
	# hit_sem2 = ( stats.sem(ppd_feed[hit_indices_joined[0] * confidence_0]),stats.sem(ppd_feed[hit_indices_joined[0] * confidence_1]),stats.sem(ppd_feed[hit_indices_joined[0] * confidence_2]),stats.sem(ppd_feed[hit_indices_joined[0] * confidence_3]) )
	# cr_sem2 = ( stats.sem(ppd_feed[cr_indices_joined[0] * confidence_0]),stats.sem(ppd_feed[cr_indices_joined[0] * confidence_1]),stats.sem(ppd_feed[cr_indices_joined[0] * confidence_2]),stats.sem(ppd_feed[cr_indices_joined[0] * confidence_3]) )
	
	def label_diff(i,j,text,X,Y,Z):
		x = (X[i]+X[j])/2
		y = max(Y[i]+Z[i], Y[j]+Z[j])
		dx = abs(X[i]-X[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':12,'shrinkB':12,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		ax.annotate(text, xy=(x,y+0.15), zorder=10, size=24, ha='center')
	
	def simpleaxis(ax):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
	
	def spine_shift(ax, shift = 10):
		for loc, spine in ax.spines.iteritems():
			if loc in ['left','bottom']:
				spine.set_position(('outward', shift)) # outward by 10 points
			elif loc in ['right','top']:
				spine.set_color('none') # don't draw spine
			else:
				raise ValueError('unknown spine location: %s'%loc)
	
	N = 8
	ind = np.arange(N)  # the x locations for the groups
	width = 0.45       # the width of the bars
	fig = plt.figure(figsize=(8,3))
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind[0]+width, hit1, width, yerr=hit1_sem, color='r', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.15, align ='center', capsize=0 )
	rects2 = ax.bar(ind[1], hit2, width, yerr=hit2_sem, color='r', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.40, align ='center', capsize=0 )
	rects3 = ax.bar(ind[2]-width, hit3, width, yerr=hit3_sem, color='r', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.65, align ='center', capsize=0 )
	rects4 = ax.bar(ind[2]+(0.25*width), hit4, width, yerr=hit4_sem, color='r', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.90, align ='center', capsize=0 )
	rects5 = ax.bar(ind[3]+width, cr1, width, yerr=cr1_sem, color='b', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.15, align ='center', capsize=0 )
	rects6 = ax.bar(ind[4], cr2, width, yerr=cr2_sem, color='b', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.40, align ='center', capsize=0 )
	rects7 = ax.bar(ind[5]-width, cr3, width, yerr=cr3_sem, color='b', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.65, align ='center', capsize=0 )
	rects8 = ax.bar(ind[5]+(0.25*width), cr4, width, yerr=cr4_sem, color='b', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.90, align ='center', capsize=0 )
	simpleaxis(ax)
	spine_shift(ax)
	ax.set_ylabel('PPR amplitude (linearly projected)', size = 10)
	ax.set_title('mean PPR amplitude feedback, per confidence', size = 12)
	ax.set_xticks( (ind[0]+width, ind[1], ind[2]-width, ind[2]+(0.25*width), ind[3]+width, ind[4], ind[5]-width, ind[5]+(0.25*width)) )
	ax.set_xticklabels( ('H --', 'H -', 'H +', 'H ++', 'CR --', 'CR -', 'CR +', 'CR ++') )
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	ax.set_ylim(ymin=0, ymax=maxvalue+(maxvalue/3.0))
	ax.set_xlim([0,5.5])
	left = 0.2
	top = 0.915
	bottom = 0.2
	plt.subplots_adjust(top=top, bottom=bottom, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)
	
	X = (ind[0]+width, ind[1], ind[2]-width, ind[2]+(0.25*width), ind[3]+width, ind[4], ind[5]-width, ind[5]+(0.25*width))
	
	sig1 = 'n.s.'
	if p1 < 0.05:
		sig1 = '*'
	if p1 < 0.01:
		sig1 = '**'
	if p1 < 0.001:
		sig1 = '***'
		
	sig2 = 'n.s.'
	if p2 < 0.05:
		sig2 = '*'
	if p2 < 0.01:
		sig2 = '**'
	if p2 < 0.001:
		sig2 = '***'
		
	sig3 = 'n.s.'
	if p3 < 0.05:
		sig3 = '*'
	if p3 < 0.01:
		sig3 = '**'
	if p3 < 0.001:
		sig2 = '***'
		
	sig4 = 'n.s.'
	if p4 < 0.05:
		sig4 = '*'
	if p4 < 0.01:
		sig4 = '**'
	if p4 < 0.001:
		sig4 = '***'
		
	label_diff(0,1,sig1,X,MEANS, SEMS)
	label_diff(1,2,sig2,X,MEANS, SEMS)
	label_diff(2,3,sig3,X,MEANS, SEMS)
	
	props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':12,'shrinkB':12,'lw':2}
	
	ax.annotate('', xy=( X[4],MEANS[4]+SEMS[4]), xytext=( X[5],MEANS[4]+SEMS[4]), arrowprops=props)
	ax.annotate('', xy=( X[6],MEANS[4]+SEMS[4]), xytext=( X[7],MEANS[4]+SEMS[4]), arrowprops=props)
	ax.annotate('', xy=( (X[4]+X[5])/2,MEANS[4]+SEMS[4]+0.035), xytext=( (X[6]+X[7])/2,MEANS[4]+SEMS[4]+0.035), arrowprops=props)
	
	ax.annotate(sig4, xy=((X[4]+X[5]+X[6]+X[7])/4,MEANS[4]+SEMS[4]+0.35), zorder=10, size='24', ha='center')
	
	return(fig)
	
	# stats.ttest_rel( (ppd[target_indices_joined][0:797]), (ppd[no_target_indices_joined]) )
	# stats.ttest_rel( (ppd[answer_yes_indices_joined[0]][0:732]), (ppd[answer_no_indices_joined[0]]) )


def plot_PPDs_feed3(subject, ppd_feed, hit, cr, confidence_0, confidence_1, confidence_2, confidence_3, p1, p2):
	
	# ppd_feed = ppd_feed_lin_A_joined
	# p1 = 0.2
	# p2 = 0.0001
	# p3 = 0.03
	# p4 = 0
	
	hit1 = sp.mean(ppd_feed[hit*confidence_0])
	hit2 = sp.mean(ppd_feed[hit*confidence_1])
	hit3 = sp.mean(ppd_feed[hit*confidence_2])
	hit4 = sp.mean(ppd_feed[hit*confidence_3])
	hit1_sem = stats.sem(ppd_feed[hit*confidence_0])
	hit2_sem = stats.sem(ppd_feed[hit*confidence_1])
	hit3_sem = stats.sem(ppd_feed[hit*confidence_2])
	hit4_sem = stats.sem(ppd_feed[hit*confidence_3])
	
	cr1 = sp.mean(ppd_feed[cr*confidence_0])
	cr2 = sp.mean(ppd_feed[cr*confidence_1])
	cr3 = sp.mean(ppd_feed[cr*confidence_2])
	cr4 = sp.mean(ppd_feed[cr*confidence_3])
	cr1_sem = stats.sem(ppd_feed[cr*confidence_0])
	cr2_sem = stats.sem(ppd_feed[cr*confidence_1])
	cr3_sem = stats.sem(ppd_feed[cr*confidence_2])
	cr4_sem = stats.sem(ppd_feed[cr*confidence_3])
	
	MEANS = (hit1, hit2, hit3, hit4, cr1, cr2, cr3, cr4)
	SEMS = (hit1_sem, hit2_sem, hit3_sem, hit4_sem, cr1_sem, cr2_sem, cr3_sem, cr4_sem)
	
	# hit2 = ( sp.mean(ppd_feed[hit_indices_joined[0] * confidence_0]),sp.mean(ppd_feed[hit_indices_joined[0] * confidence_1]),sp.mean(ppd_feed[hit_indices_joined[0] * confidence_2]),sp.mean(ppd_feed[hit_indices_joined[0] * confidence_3]) )
	# cr2 = ( sp.mean(ppd_feed[cr_indices_joined[0] * confidence_0]),sp.mean(ppd_feed[cr_indices_joined[0] * confidence_1]),sp.mean(ppd_feed[cr_indices_joined[0] * confidence_2]),sp.mean(ppd_feed[cr_indices_joined[0] * confidence_3] ))
	# 
	# hit_sem2 = ( stats.sem(ppd_feed[hit_indices_joined[0] * confidence_0]),stats.sem(ppd_feed[hit_indices_joined[0] * confidence_1]),stats.sem(ppd_feed[hit_indices_joined[0] * confidence_2]),stats.sem(ppd_feed[hit_indices_joined[0] * confidence_3]) )
	# cr_sem2 = ( stats.sem(ppd_feed[cr_indices_joined[0] * confidence_0]),stats.sem(ppd_feed[cr_indices_joined[0] * confidence_1]),stats.sem(ppd_feed[cr_indices_joined[0] * confidence_2]),stats.sem(ppd_feed[cr_indices_joined[0] * confidence_3]) )
	
	def label_diff(i,j,text,X,Y,Z):
		x = (X[i]+X[j])/2
		y = max(Y[i]+Z[i], Y[j]+Z[j])
		dx = abs(X[i]-X[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':12,'shrinkB':12,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		
		ax.annotate(text, xy=(x,y+0.3), zorder=10, size=24, ha='center')
		
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
	
	def simpleaxis(ax):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
	
	def spine_shift(ax, shift = 10):
		for loc, spine in ax.spines.iteritems():
			if loc in ['left','bottom']:
				spine.set_position(('outward', shift)) # outward by 10 points
			elif loc in ['right','top']:
				spine.set_color('none') # don't draw spine
			else:
				raise ValueError('unknown spine location: %s'%loc)
	
	N = 8
	ind = np.arange(N)  # the x locations for the groups
	width = 0.45       # the width of the bars
	fig = plt.figure(figsize=(8,3))
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind[0]+width, hit1, width, yerr=hit1_sem, color='r', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.15, align ='center', capsize=0 )
	rects2 = ax.bar(ind[1], hit2, width, yerr=hit2_sem, color='r', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.40, align ='center', capsize=0 )
	rects3 = ax.bar(ind[2]-width, hit3, width, yerr=hit3_sem, color='r', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.65, align ='center', capsize=0 )
	rects4 = ax.bar(ind[2]+(0.25*width), hit4, width, yerr=hit4_sem, color='r', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.90, align ='center', capsize=0 )
	rects5 = ax.bar(ind[3]+width, cr1, width, yerr=cr1_sem, color='b', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.15, align ='center', capsize=0 )
	rects6 = ax.bar(ind[4], cr2, width, yerr=cr2_sem, color='b', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.40, align ='center', capsize=0 )
	rects7 = ax.bar(ind[5]-width, cr3, width, yerr=cr3_sem, color='b', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.65, align ='center', capsize=0 )
	rects8 = ax.bar(ind[5]+(0.25*width), cr4, width, yerr=cr4_sem, color='b', edgecolor=('k'), ecolor = 'k', linewidth = 0, alpha = 0.90, align ='center', capsize=0 )
	simpleaxis(ax)
	spine_shift(ax)
	ax.set_ylabel('PPR amplitude (linearly projected)', size = 10)
	ax.set_title('mean PPR amplitude feedback, per confidence', size = 12)
	ax.set_xticks( (ind[0]+width, ind[1], ind[2]-width, ind[2]+(0.25*width), ind[3]+width, ind[4], ind[5]-width, ind[5]+(0.25*width)) )
	ax.set_xticklabels( ('H --', 'H -', 'H +', 'H ++', 'CR --', 'CR -', 'CR +', 'CR ++') )
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	ax.set_ylim(ymin=0, ymax=maxvalue+(maxvalue/2.5))
	ax.set_xlim([0,5.5])
	left = 0.2
	top = 0.915
	bottom = 0.2
	plt.subplots_adjust(top=top, bottom=bottom, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)
	
	X = (ind[0]+width, ind[1], ind[2]-width, ind[2]+(0.25*width), ind[3]+width, ind[4], ind[5]-width, ind[5]+(0.25*width))
	
	sig1 = 'n.s.'
	if p1 < 0.05:
		sig1 = '*'
	if p1 < 0.01:
		sig1 = '**'
	if p1 < 0.001:
		sig1 = '***'
		
	sig2 = 'n.s.'
	if p2 < 0.05:
		sig2 = '*'
	if p2 < 0.01:
		sig2 = '**'
	if p2 < 0.001:
		sig2 = '***'
			
	# label_diff(0,1,sig1,X,MEANS, SEMS)
	# label_diff(1,2,sig2,X,MEANS, SEMS)
	# label_diff(2,3,sig3,X,MEANS, SEMS)
	
	
	props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':12,'shrinkB':12,'lw':2}
	
	ax.annotate('', xy=( X[0],MEANS[0]+SEMS[0]), xytext=( X[1],MEANS[0]+SEMS[0]), arrowprops=props)
	ax.annotate('', xy=( X[2],MEANS[0]+SEMS[0]), xytext=( X[3],MEANS[0]+SEMS[0]), arrowprops=props)
	ax.annotate('', xy=( (X[0]+X[1])/2,MEANS[0]+SEMS[0]+0.035), xytext=( (X[2]+X[3])/2,MEANS[0]+SEMS[0]+0.035), arrowprops=props)
	
	ax.annotate('', xy=( X[4],MEANS[4]+SEMS[4]), xytext=( X[5],MEANS[4]+SEMS[4]), arrowprops=props)
	ax.annotate('', xy=( X[6],MEANS[4]+SEMS[4]), xytext=( X[7],MEANS[4]+SEMS[4]), arrowprops=props)
	ax.annotate('', xy=( (X[4]+X[5])/2,MEANS[4]+SEMS[4]+0.035), xytext=( (X[6]+X[7])/2,MEANS[4]+SEMS[4]+0.035), arrowprops=props)
	
	ax.annotate(sig1, xy=((X[0]+X[1]+X[2]+X[3])/4,MEANS[0]+SEMS[0]+0.3), zorder=10, size='24', ha='center')
	ax.annotate(sig2, xy=((X[4]+X[5]+X[6]+X[7])/4,MEANS[4]+SEMS[4]+0.3), zorder=10, size='24', ha='center')
	
	return(fig)
	
	# stats.ttest_rel( (ppd[target_indices_joined][0:797]), (ppd[no_target_indices_joined]) )
	# stats.ttest_rel( (ppd[answer_yes_indices_joined[0]][0:732]), (ppd[answer_no_indices_joined[0]]) )
	
	


def plot_confidence(confidence_ratings_joined, hit_indices_joined, fa_indices_joined, bpd_low_indices_joined, bpd_high_indices_joined, ppd_low_indices_joined, ppd_high_indices_joined):
	
	import scipy as sp
	import scipy.stats as stats
	import scipy.signal as signal
	import bottleneck
	import numpy as np
	import matplotlib.pyplot as plt
	
	## CONFIDENCE:
	# BPD
	mean_confidence_hit_low_bpd = sp.mean(confidence_ratings_joined[bpd_low_indices_joined*hit_indices_joined[0]])
	mean_confidence_hit_high_bpd = sp.mean(confidence_ratings_joined[bpd_high_indices_joined*hit_indices_joined[0]])
	mean_confidence_fa_low_bpd = sp.mean(confidence_ratings_joined[bpd_low_indices_joined*fa_indices_joined[0]])
	mean_confidence_fa_high_bpd = sp.mean(confidence_ratings_joined[bpd_high_indices_joined*fa_indices_joined[0]])
	sem_confidence_hit_low_bpd = stats.sem(confidence_ratings_joined[bpd_low_indices_joined*hit_indices_joined[0]]) 
	sem_confidence_hit_high_bpd = stats.sem(confidence_ratings_joined[bpd_high_indices_joined*hit_indices_joined[0]])
	sem_confidence_fa_low_bpd = stats.sem(confidence_ratings_joined[bpd_low_indices_joined*fa_indices_joined[0]])
	sem_confidence_fa_high_bpd = stats.sem(confidence_ratings_joined[bpd_high_indices_joined*fa_indices_joined[0]])
	# PPD
	mean_confidence_hit_low_ppd = sp.mean(confidence_ratings_joined[ppd_low_indices_joined*hit_indices_joined[0]])
	mean_confidence_hit_high_ppd = sp.mean(confidence_ratings_joined[ppd_high_indices_joined*hit_indices_joined[0]])
	mean_confidence_fa_low_ppd = sp.mean(confidence_ratings_joined[ppd_low_indices_joined*fa_indices_joined[0]])
	mean_confidence_fa_high_ppd = sp.mean(confidence_ratings_joined[ppd_high_indices_joined*fa_indices_joined[0]])
	sem_confidence_hit_low_ppd = stats.sem(confidence_ratings_joined[ppd_low_indices_joined*hit_indices_joined[0]])
	sem_confidence_hit_high_ppd = stats.sem(confidence_ratings_joined[ppd_high_indices_joined*hit_indices_joined[0]])
	sem_confidence_fa_low_ppd = stats.sem(confidence_ratings_joined[ppd_low_indices_joined*fa_indices_joined[0]])
	sem_confidence_fa_high_ppd = stats.sem(confidence_ratings_joined[ppd_high_indices_joined*fa_indices_joined[0]])
	
	## CONFIDENCE:
	# BPD
	bpdMeans = (mean_confidence_hit_low_bpd, mean_confidence_hit_high_bpd, mean_confidence_fa_low_bpd, mean_confidence_fa_high_bpd)
	bpdStd = (sem_confidence_hit_low_bpd, sem_confidence_hit_high_bpd, sem_confidence_fa_low_bpd, sem_confidence_fa_high_bpd)
	MEANS = (mean_confidence_hit_low_ppd, mean_confidence_hit_high_ppd, mean_confidence_fa_low_ppd, mean_confidence_fa_high_ppd)
	ppdStd = (sem_confidence_hit_low_ppd, sem_confidence_hit_high_ppd, sem_confidence_fa_low_ppd, sem_confidence_fa_high_ppd)
	N = 4
	ind = np.arange(N)+1
	width = 0.35       # the width of the bars
	fig = plt.figure(figsize = (10,8))
	ax = fig.add_subplot(211)
	rects1 = ax.bar(ind+0.5, MEANS, width, color=('b','b','g','g'), yerr=ppdStd, ecolor = 'r', align = 'center')
	ax.set_ylabel('Mean confidence')
	ax.set_title('Mean Confidence per PPD')
	ax.set_xticks(ind+0.5)
	ax.set_xticklabels( ('Hit - low PPD', 'Hit - high PPD', 'Fa - low PPD', 'Fa - high PPD') )
	ax = fig.add_subplot(212)
	rects1 = ax.bar(ind+0.5, bpdMeans, width, color=('b','b','g','g'), yerr=bpdStd, ecolor = 'r', align = 'center')
	ax.set_ylabel('Mean confidence')
	ax.set_title('Mean Confidence per BPD')
	ax.set_xticks(ind+0.5)
	ax.set_xticklabels( ('Hit - low BPD', 'Hit - high BPD', 'Fa - low BPD', 'Fa - high BPD') )
	return(fig)


def SDT_measures_per_subject(subject, target_indices_joined, no_target_indices_joined, hit_indices_joined, fa_indices_joined):
	"calculate d_primes"
	
	import scipy as sp
	import scipy.stats as stats
	import scipy.signal as signal
	import bottleneck
	import numpy as np
	import matplotlib.pyplot as plt
	
	# Third, calculate z-scored hit_rate and fa_rate 
	# All trials pooled:
	hit_rate_joined = float(np.sum(hit_indices_joined))/float(np.sum(target_indices_joined))
	fa_rate_joined = float(np.sum(fa_indices_joined))/float(np.sum(no_target_indices_joined))
	hit_rate_joined_zscored = stats.norm.isf(1-hit_rate_joined)
	fa_rate_joined_zscored = stats.norm.isf(1-fa_rate_joined)
	
	# Calculate d_prime:
	d_prime_joined = hit_rate_joined_zscored - fa_rate_joined_zscored
	criterion_joined = -((hit_rate_joined_zscored + fa_rate_joined_zscored) / 2)
	
	return(d_prime_joined, criterion_joined)


def SDT_measures_per_subject_per_run(subject, target_indices, no_target_indices, hit_indices, fa_indices):
	"calculate d_primes"
	
	import scipy as sp
	import scipy.stats as stats
	import scipy.signal as signal
	import bottleneck
	import numpy as np
	import matplotlib.pyplot as plt
	
	d_prime = []	
	criterion = []
	for i in range(len(target_indices)): 
		# target_indices is a list object. Thus, here I loop over all runs.
		# An add-one smoothing (Laplace smoothing with lambda = 1) is applied, in order to avoid 0-counts, which can result in d_prime = inf.
		
		# Third, calculate z-scored hit_rate and fa_rate 
		# For whole trial:
		hit_rate_joined = float(np.sum(hit_indices[i])+1.0)/float(np.sum(target_indices[i])+2.0)
		fa_rate_joined = float(np.sum(fa_indices[i])+1.0)/float(np.sum(no_target_indices[i])+2.0)
		hit_rate_joined_zscored = stats.norm.isf(1-hit_rate_joined)
		fa_rate_joined_zscored = stats.norm.isf(1-fa_rate_joined)
		
		# Calculate d_prime:
		d_prime.append( hit_rate_joined_zscored - fa_rate_joined_zscored )
		# Calculate criterion
		criterion.append( -((hit_rate_joined_zscored + fa_rate_joined_zscored) / 2) )
		
	return(d_prime, criterion)



def GLM_betas_barplot(subject, beta1, beta2, beta3, beta4, p1, p2, p3, p4, p5, p6):
	
	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MultipleLocator
		
	def simpleaxis(ax):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
	
	def spine_shift(ax, shift = 10):
		for loc, spine in ax.spines.iteritems():
			if loc in ['left','bottom']:
				spine.set_position(('outward', shift)) # outward by 10 points
			elif loc in ['right','top']:
				spine.set_color('none') # don't draw spine
			else:
				raise ValueError('unknown spine location: %s'%loc)
	
	def label_diff(i,j,text,X,Y,Z, values = False):
	
		# i = 2
		# j = 3
		# text = '***'
		# X = (ind[0]+width, ind[1], ind[2], ind[3]-width)
		# MEANS = MEANS
		# SEMS = SEMS
	
		middle_x = (X[i]+X[j])/2
		max_value = max(MEANS[i]+SEMS[i], MEANS[j]+SEMS[j])
		min_value = min(MEANS[i]-SEMS[i], MEANS[j]-SEMS[j])
		dx = abs(X[i]-X[j])
	
		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':10,'shrinkB':10,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		ax.annotate('', xy=(X[i],max_value), xytext=(X[j],max_value), arrowprops=props)
	
		if values == False:
			if text == 'n.s.':
				kwargs = {'zorder':10, 'size':16, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.0/10))), **kwargs)
			if text != 'n.s.':
				kwargs = {'zorder':10, 'size':24, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(0.60/10))), **kwargs)
		if values == True:
			kwargs = {'zorder':10, 'size':12, 'ha':'center'}
			ax.annotate('p = ' + str(text), xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.15/10))), **kwargs)
	
	
	beta1_mean = sp.mean(beta1)
	beta2_mean = sp.mean(beta2)
	beta3_mean = sp.mean(beta3)
	beta4_mean = sp.mean(beta4)
	# beta5_mean = sp.mean(beta5)
	
	beta1_sem = stats.sem(beta1)
	beta2_sem = stats.sem(beta2)
	beta3_sem = stats.sem(beta3)
	beta4_sem = stats.sem(beta4)
	# beta5_sem = stats.sem(beta5)
	
	MEANS = (beta1_mean, beta2_mean, beta3_mean, beta4_mean)
	SEMS = (beta1_sem, beta2_sem, beta3_sem, beta4_sem)
	
	sig1 = 'n.s.'
	if p1 <= 0.05:
		sig1 = '*'
	if p1 <= 0.01:
		sig1 = '**'
	if p1 <= 0.001:
		sig1 = '***'
	
	sig2 = 'n.s.'
	if p2 <= 0.05:
		sig2 = '*'
	if p2 <= 0.01:
		sig2 = '**'
	if p2 <= 0.001:
		sig2 = '***'
		
	sig3 = 'n.s.'
	if p3 <= 0.05:
		sig3 = '*'
	if p3 <= 0.01:
		sig3 = '**'
	if p3 <= 0.001:
		sig3 = '***'
		
	sig4 = 'n.s.'
	if p4 <= 0.05:
		sig4 = '*'
	if p4 <= 0.01:
		sig4 = '**'
	if p4 <= 0.001:
		sig4 = '***'
		
	sig5 = 'n.s.'
	if p5 <= 0.05:
		sig5 = '*'
	if p5 <= 0.01:
		sig5 = '**'
	if p5 <= 0.001:
		sig5 = '***'
		
	sig6 = 'n.s.'
	if p6 <= 0.05:
		sig6 = '*'
	if p6 <= 0.01:
		sig6 = '**'
	if p6 <= 0.001:
		sig6 = '***'
	
	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
	
	# N = 5
	# ind = np.linspace(0,2.6667,5)  # the x locations for the groups
	# bar_width = 0.30       # the width of the bars
	# spacing = [0.30, 0, 0, -0.30, -.30]

	N = 4
	ind = np.linspace(0,2,4)  # the x locations for the groups
	bar_width = 0.30       # the width of the bars
	spacing = [0.30, 0, 0, -0.30]
	
	# FIGURE 1
	fig = plt.figure(figsize=(4,3))
	ax = fig.add_subplot(111)
	for i in range(N):
		ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['k','k','k','k'][i], alpha = [0.80, 0.80, 0.80, 0.80][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
	simpleaxis(ax)
	spine_shift(ax)
	ax.set_xticklabels( ('Stim', 'Resp','Down', 'Up') )
	ax.set_xticks( (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width) )
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	diffvalue = maxvalue - minvalue
	ax.set_ylim(ymin=minvalue-(diffvalue/20.0), ymax=maxvalue+(diffvalue/4.0))
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	left = 0.2
	top = 0.915
	bottom = 0.2
	plt.subplots_adjust(bottom=bottom, top=top, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)
	
	if p1 < 0.05:
		ax.text(ind[0]+spacing[0],0,sig1, size=24)
	if p2 < 0.05:
		ax.text(ind[1]+spacing[1],0,sig2, size=24)
	if p3 < 0.05:
		ax.text(ind[2]+spacing[2],0,sig3, size=24)
	if p4 < 0.05:
		ax.text(ind[3]+spacing[3],0,sig4, size=24)
	
	# STATS:
	
	X = (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width)
	
	label_diff(0,1,sig5,X,MEANS,SEMS)
	label_diff(2,3,sig6,X,MEANS,SEMS)
	
	return(fig)










































# EXTRA STUFF:

def plot_resp_confidence(subject, response_locked_array_joined, x, xx, confidence_0, confidence_1, confidence_2, confidence_3, decision_time_joined):
	" plot stimulus locked and response locked mean pupil time series"
	
	import scipy as sp
	import scipy.stats as stats
	import scipy.signal as signal
	import bottleneck
	import numpy as np
	import matplotlib.pyplot as plt
	
	## RESPONSE LOCKED DATA:
	# Compute mean pupil responses
	resp_locked_hits_0_mean = bottleneck.nanmean(response_locked_array_joined[x*confidence_0,:], axis=0)
	resp_locked_hits_1_mean = bottleneck.nanmean(response_locked_array_joined[x*confidence_1,:], axis=0)
	resp_locked_hits_2_mean = bottleneck.nanmean(response_locked_array_joined[x*confidence_2,:], axis=0)
	resp_locked_hits_3_mean = bottleneck.nanmean(response_locked_array_joined[x*confidence_3,:], axis=0)
	# Smooth the data:
	cut_off = 10
	nyq = 1000 / 2.0
	[param2,param1] = signal.butter(3,(cut_off/nyq))
	resp_locked_hits_0_mean = signal.filtfilt(param2, param1, resp_locked_hits_0_mean)
	resp_locked_hits_1_mean = signal.filtfilt(param2, param1, resp_locked_hits_1_mean)
	resp_locked_hits_2_mean = signal.filtfilt(param2, param1, resp_locked_hits_2_mean)
	resp_locked_hits_3_mean = signal.filtfilt(param2, param1, resp_locked_hits_3_mean)
	# Compute std mean pupil responses
	resp_locked_hits_0_std = ( bottleneck.nanstd( response_locked_array_joined[x*confidence_0,:], axis=0) / sp.sqrt((x*confidence_0).sum()) )
	resp_locked_hits_1_std = ( bottleneck.nanstd( response_locked_array_joined[x*confidence_1,:], axis=0) / sp.sqrt((x*confidence_1).sum()) )
	resp_locked_hits_2_std = ( bottleneck.nanstd( response_locked_array_joined[x*confidence_2,:], axis=0) / sp.sqrt((x*confidence_2).sum()) )
	resp_locked_hits_3_std = ( bottleneck.nanstd( response_locked_array_joined[x*confidence_3,:], axis=0) / sp.sqrt((x*confidence_3).sum()) )
	
	# Make the plt.plot
	figure_mean_response_locked_confidence = plt.figure(figsize=(4, 6))
	plt.subplot(111)
	xb = np.arange(-3499,1500)
	p1, = plt.plot(xb, resp_locked_hits_0_mean, color = 'r', alpha = 0.25, linewidth=2)
	p2, = plt.plot(xb, resp_locked_hits_1_mean, color = 'r', alpha = 0.50, linewidth=2)
	p3, = plt.plot(xb, resp_locked_hits_2_mean, color = 'r', alpha = 0.75, linewidth=2)
	p4, = plt.plot(xb, resp_locked_hits_3_mean, color = 'r', alpha = 1.00, linewidth=2)
	plt.fill_between( xb, (resp_locked_hits_0_mean+resp_locked_hits_0_std), (resp_locked_hits_0_mean-resp_locked_hits_0_std), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (resp_locked_hits_1_mean+resp_locked_hits_1_std), (resp_locked_hits_1_mean-resp_locked_hits_1_std), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (resp_locked_hits_2_mean+resp_locked_hits_2_std), (resp_locked_hits_2_mean-resp_locked_hits_2_std), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (resp_locked_hits_3_mean+resp_locked_hits_3_std), (resp_locked_hits_3_mean-resp_locked_hits_3_std), alpha=0.1, color = 'r' )
	plt.axvline(0-sp.mean(decision_time_joined[x]), -1, 1, color = 'r')
	plt.axvline(0, -1, 1)
	plt.hist(0-decision_time_joined[x], bins=10, weights = (np.ones(response_locked_array_joined[x].shape[0]) / len(decision_time_joined[x])), bottom = 0.2, color = 'r', alpha = 0.5)
	plt.text(0-sp.mean(decision_time_joined[x])+10,0.15,"'beep!'")
	plt.xlim( (-3500, 1500) )
	plt.title(subject + "_pupil_diameter_response_locked_" + xx + "_confidence")
	plt.ylabel("Pupil diameter (Z)")
	plt.xlabel("Time in milliseconds")
	# plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit_indices_joined[0].sum()) + " trials", "FA; " + str(fa_indices_joined[0].sum()) + " trials", "CR; " + str(cr_indices_joined[0].sum()) + " trials", "MISS; " + str(miss_indices_joined[0].sum()) + " trials"], loc = 2)
	return(figure_mean_response_locked_confidence)
	
	
	# def d_primes(subject, bpd, ppd, answer_yes_indices_joined, answer_no_indices_joined, target_indices_joined, no_target_indices_joined, hit_indices_joined, fa_indices_joined, omission_indices_joined, bpd_low_indices, bpd_high_indices, ppd_low_indices, ppd_high_indices):
	"calculate d_primes"
	
	import scipy as sp
	import scipy.stats as stats
	import scipy.signal as signal
	import bottleneck
	import numpy as np
	import matplotlib.pyplot as plt
	
	# ########### GET D_PRIME FOR ALL TRIALS POOLED TOGETHER, FOR TRIALS BINNEND BY PPD, FOR TRIALS BINNEN BY BPD ###########
	# Low BPD Bin:
	bpd_lowbin_target_indices = bpd_low_indices * target_indices_joined
	bpd_lowbin_no_target_indices = bpd_low_indices * no_target_indices_joined
	bpd_lowbin_hit_indices = bpd_low_indices * hit_indices_joined[0]
	bpd_lowbin_fa_indices = bpd_low_indices * fa_indices_joined[0]
	# High BPD Bin:
	bpd_highbin_target_indices = bpd_high_indices * target_indices_joined
	bpd_highbin_no_target_indices = bpd_high_indices * no_target_indices_joined
	bpd_highbin_hit_indices = bpd_high_indices * hit_indices_joined[0]
	bpd_highbin_fa_indices = bpd_high_indices * fa_indices_joined[0]
	# Low PPD Bin:
	ppd_lowbin_target_indices = ppd_low_indices * target_indices_joined
	ppd_lowbin_no_target_indices = ppd_low_indices * no_target_indices_joined
	ppd_lowbin_hit_indices = ppd_low_indices * hit_indices_joined[0]
	ppd_lowbin_fa_indices = ppd_low_indices * fa_indices_joined[0]
	# High PPD Bin:
	ppd_highbin_target_indices = ppd_high_indices * target_indices_joined
	ppd_highbin_no_target_indices = ppd_high_indices * no_target_indices_joined
	ppd_highbin_hit_indices = ppd_high_indices * hit_indices_joined[0]
	ppd_highbin_fa_indices = ppd_high_indices * fa_indices_joined[0]
	
	# Third, calculate z-scored hit_rate and fa_rate 
	# All trials pooled:
	hit_rate_joined = float(np.sum(hit_indices_joined))/float(np.sum(target_indices_joined))
	fa_rate_joined = float(np.sum(fa_indices_joined))/float(np.sum(no_target_indices_joined))
	hit_rate_joined_zscored = stats.norm.isf(1-hit_rate_joined)
	fa_rate_joined_zscored = stats.norm.isf(1-fa_rate_joined)
	# Low BPD Bin:
	bpd_lowbin_hit_rate = float(np.sum(bpd_lowbin_hit_indices))/float(np.sum(bpd_lowbin_target_indices))
	bpd_lowbin_fa_rate = float(np.sum(bpd_lowbin_fa_indices))/float(np.sum(bpd_lowbin_no_target_indices))
	bpd_lowbin_hit_rate_zscored = stats.norm.isf(1-bpd_lowbin_hit_rate)
	bpd_lowbin_fa_rate_zscored = stats.norm.isf(1-bpd_lowbin_fa_rate)
	# High BPD Bin:
	bpd_highbin_hit_rate = float(np.sum(bpd_highbin_hit_indices))/float(np.sum(bpd_highbin_target_indices))
	bpd_highbin_fa_rate = float(np.sum(bpd_highbin_fa_indices))/float(np.sum(bpd_highbin_no_target_indices))
	bpd_highbin_hit_rate_zscored = stats.norm.isf(1-bpd_highbin_hit_rate)
	bpd_highbin_fa_rate_zscored = stats.norm.isf(1-bpd_highbin_fa_rate)
	# Low PPD Bin:
	ppd_lowbin_hit_rate = float(np.sum(ppd_lowbin_hit_indices))/float(np.sum(ppd_lowbin_target_indices))
	ppd_lowbin_fa_rate = float(np.sum(ppd_lowbin_fa_indices))/float(np.sum(ppd_lowbin_no_target_indices))
	ppd_lowbin_hit_rate_zscored = stats.norm.isf(1-ppd_lowbin_hit_rate)
	ppd_lowbin_fa_rate_zscored = stats.norm.isf(1-ppd_lowbin_fa_rate)
	# High PPD Bin:
	ppd_highbin_hit_rate = float(np.sum(ppd_highbin_hit_indices))/float(np.sum(ppd_highbin_target_indices))
	ppd_highbin_fa_rate = float(np.sum(ppd_highbin_fa_indices))/float(np.sum(ppd_highbin_no_target_indices))
	ppd_highbin_hit_rate_zscored = stats.norm.isf(1-ppd_highbin_hit_rate)
	ppd_highbin_fa_rate_zscored = stats.norm.isf(1-ppd_highbin_fa_rate)
	
	# Calculate d_prime:
	d_prime_joined = hit_rate_joined_zscored - fa_rate_joined_zscored
	bpd_lowbin_d_prime = bpd_lowbin_hit_rate_zscored - bpd_lowbin_fa_rate_zscored
	bpd_highbin_d_prime = bpd_highbin_hit_rate_zscored - bpd_highbin_fa_rate_zscored
	ppd_lowbin_d_prime = ppd_lowbin_hit_rate_zscored - ppd_lowbin_fa_rate_zscored
	ppd_highbin_d_prime = ppd_highbin_hit_rate_zscored - ppd_highbin_fa_rate_zscored
	
	# Calculate criterion
	criterion_joined = -((hit_rate_joined_zscored + fa_rate_joined_zscored) / 2)
	bpd_lowbin_criterion = -((bpd_lowbin_hit_rate_zscored + bpd_lowbin_fa_rate_zscored) / 2)
	bpd_highbin_criterion = -((bpd_highbin_hit_rate_zscored + bpd_highbin_fa_rate_zscored) / 2)
	ppd_lowbin_criterion = -((ppd_lowbin_hit_rate_zscored + ppd_lowbin_fa_rate_zscored) / 2)
	ppd_highbin_criterion = -((ppd_highbin_hit_rate_zscored + ppd_highbin_fa_rate_zscored) / 2)
	
	return(d_prime_joined, bpd_lowbin_d_prime, bpd_highbin_d_prime, ppd_lowbin_d_prime, ppd_highbin_d_prime, criterion_joined, bpd_lowbin_criterion, bpd_highbin_criterion, ppd_lowbin_criterion, ppd_highbin_criterion)


	# def d_primes2(subject, bpd, ppd, answer_yes_indices, answer_no_indices, target_indices, no_target_indices, hit_indices, fa_indices, omission_indices, bpd_low_indices, bpd_high_indices, ppd_low_indices, ppd_high_indices, confidence_ratings):
	"calculate d_primes"
	
	import scipy as sp
	import scipy.stats as stats
	import scipy.signal as signal
	import bottleneck
	import numpy as np
	import matplotlib.pyplot as plt
	
	d_prime = []
	bpd_lowbin_d_prime = []
	bpd_highbin_d_prime = []
	ppd_lowbin_d_prime = []
	ppd_highbin_d_prime = []
	d_prime_low_confidence = []
	d_prime_high_confidence = []
	
	criterion = []
	bpd_lowbin_criterion = []
	bpd_highbin_criterion = []
	ppd_lowbin_criterion = []
	ppd_highbin_criterion = []
	criterion_low_confidence = []
	criterion_high_confidence = []
	
	for i in range(len(bpd)): 
		# bpd is a list object, with a np.array for every run. Thus, here I loop over all runs.
		# An add-one smoothing (Laplace smoothing with lambda = 1) is applied, in order to avoid 0-counts, which can resulting in d_prime = inf.
		
		# ########### GET D_PRIME FOR ALL TRIALS POOLED TOGETHER, FOR TRIALS BINNEND BY PPD, FOR TRIALS BINNEN BY BPD ###########
		# Low BPD Bin:
		bpd_lowbin_target_indices = bpd_low_indices[i] * target_indices[i]
		bpd_lowbin_no_target_indices = bpd_low_indices[i] * no_target_indices[i]
		bpd_lowbin_hit_indices = bpd_low_indices[i] * hit_indices[i]
		bpd_lowbin_fa_indices = bpd_low_indices[i] * fa_indices[i]
		# High BPD Bin:
		bpd_highbin_target_indices = bpd_high_indices[i] * target_indices[i]
		bpd_highbin_no_target_indices = bpd_high_indices[i] * no_target_indices[i]
		bpd_highbin_hit_indices = bpd_high_indices[i] * hit_indices[i]
		bpd_highbin_fa_indices = bpd_high_indices[i] * fa_indices[i]
		# Low PPD Bin:
		ppd_lowbin_target_indices = ppd_low_indices[i] * target_indices[i]
		ppd_lowbin_no_target_indices = ppd_low_indices[i] * no_target_indices[i]
		ppd_lowbin_hit_indices = ppd_low_indices[i] * hit_indices[i]
		ppd_lowbin_fa_indices = ppd_low_indices[i] * fa_indices[i]
		# High PPD Bin:
		ppd_highbin_target_indices = ppd_high_indices[i] * target_indices[i]
		ppd_highbin_no_target_indices = ppd_high_indices[i] * no_target_indices[i]
		ppd_highbin_hit_indices = ppd_high_indices[i] * hit_indices[i]
		ppd_highbin_fa_indices = ppd_high_indices[i] * fa_indices[i]
		# Low Confidence Bin:
		low_confidence_target_indices = [[confidence_ratings[i] == 0]+[confidence_ratings[i] == 1]] * target_indices[i]
		low_confidence_no_target_indices = [[confidence_ratings[i] == 0]+[confidence_ratings[i] == 1]] * no_target_indices[i]
		low_confidence_hit_indices = [[confidence_ratings[i] == 0]+[confidence_ratings[i] == 1]] * hit_indices[i]
		low_confidence_fa_indices = [[confidence_ratings[i] == 0]+[confidence_ratings[i] == 1]] * fa_indices[i]
		# High Confidence Bin:
		high_confidence_target_indices = [[confidence_ratings[i] == 2]+[confidence_ratings[i] == 3]] * target_indices[i]
		high_confidence_no_target_indices = [[confidence_ratings[i] == 2]+[confidence_ratings[i] == 3]] * no_target_indices[i]
		high_confidence_hit_indices = [[confidence_ratings[i] == 2]+[confidence_ratings[i] == 3]] * hit_indices[i]
		high_confidence_fa_indices = [[confidence_ratings[i] == 2]+[confidence_ratings[i] == 3]] * fa_indices[i]
		
		# Third, calculate z-scored hit_rate and fa_rate 
		# For whole trial:
		hit_rate_joined = float(np.sum(hit_indices[i])+1)/float(np.sum(target_indices[i])+1)
		fa_rate_joined = float(np.sum(fa_indices[i])+1)/float(np.sum(no_target_indices[i])+1)
		hit_rate_joined_zscored = stats.norm.isf(1-hit_rate_joined)
		fa_rate_joined_zscored = stats.norm.isf(1-fa_rate_joined)
		# Low BPD Bin:
		bpd_lowbin_hit_rate = float(np.sum(bpd_lowbin_hit_indices)+1)/float(np.sum(bpd_lowbin_target_indices)+1)
		bpd_lowbin_fa_rate = float(np.sum(bpd_lowbin_fa_indices)+1)/float(np.sum(bpd_lowbin_no_target_indices)+1)
		bpd_lowbin_hit_rate_zscored = stats.norm.isf(1-bpd_lowbin_hit_rate)
		bpd_lowbin_fa_rate_zscored = stats.norm.isf(1-bpd_lowbin_fa_rate)
		# High BPD Bin:
		bpd_highbin_hit_rate = float(np.sum(bpd_highbin_hit_indices)+1)/float(np.sum(bpd_highbin_target_indices)+1)
		bpd_highbin_fa_rate = float(np.sum(bpd_highbin_fa_indices)+1)/float(np.sum(bpd_highbin_no_target_indices)+1)
		bpd_highbin_hit_rate_zscored = stats.norm.isf(1-bpd_highbin_hit_rate)
		bpd_highbin_fa_rate_zscored = stats.norm.isf(1-bpd_highbin_fa_rate)
		# Low PPD Bin:
		ppd_lowbin_hit_rate = float(np.sum(ppd_lowbin_hit_indices)+1)/float(np.sum(ppd_lowbin_target_indices)+1)
		ppd_lowbin_fa_rate = float(np.sum(ppd_lowbin_fa_indices)+1)/float(np.sum(ppd_lowbin_no_target_indices)+1)
		ppd_lowbin_hit_rate_zscored = stats.norm.isf(1-ppd_lowbin_hit_rate)
		ppd_lowbin_fa_rate_zscored = stats.norm.isf(1-ppd_lowbin_fa_rate)
		# High PPD Bin:
		ppd_highbin_hit_rate = float(np.sum(ppd_highbin_hit_indices)+1)/float(np.sum(ppd_highbin_target_indices)+1)
		ppd_highbin_fa_rate = float(np.sum(ppd_highbin_fa_indices)+1)/float(np.sum(ppd_highbin_no_target_indices)+1)
		ppd_highbin_hit_rate_zscored = stats.norm.isf(1-ppd_highbin_hit_rate)
		ppd_highbin_fa_rate_zscored = stats.norm.isf(1-ppd_highbin_fa_rate)
		# Low Confidence Bin:
		low_confidence_hit_rate = float(np.sum(low_confidence_hit_indices)+1)/float(np.sum(low_confidence_target_indices)+1)
		low_confidence_fa_rate = float(np.sum(low_confidence_fa_indices)+1)/float(np.sum(low_confidence_no_target_indices)+1)
		low_confidence_hit_rate_zscored = stats.norm.isf(1-low_confidence_hit_rate)
		low_confidence_fa_rate_zscored = stats.norm.isf(1-low_confidence_fa_rate)
		# High Confidence Bin:
		high_confidence_hit_rate = float(np.sum(high_confidence_hit_indices)+1)/float(np.sum(high_confidence_target_indices)+1)
		high_confidence_fa_rate = float(np.sum(high_confidence_fa_indices)+1)/float(np.sum(high_confidence_no_target_indices)+1)
		high_confidence_hit_rate_zscored = stats.norm.isf(1-high_confidence_hit_rate)
		high_confidence_fa_rate_zscored = stats.norm.isf(1-high_confidence_fa_rate)
		
		# Calculate d_prime:
		d_prime.append( hit_rate_joined_zscored - fa_rate_joined_zscored )
		bpd_lowbin_d_prime.append( bpd_lowbin_hit_rate_zscored - bpd_lowbin_fa_rate_zscored )
		bpd_highbin_d_prime.append( bpd_highbin_hit_rate_zscored - bpd_highbin_fa_rate_zscored )
		ppd_lowbin_d_prime.append( ppd_lowbin_hit_rate_zscored - ppd_lowbin_fa_rate_zscored )
		ppd_highbin_d_prime.append( ppd_highbin_hit_rate_zscored - ppd_highbin_fa_rate_zscored )
		d_prime_low_confidence.append( low_confidence_hit_rate_zscored - low_confidence_fa_rate_zscored )
		d_prime_high_confidence.append( high_confidence_hit_rate_zscored - high_confidence_fa_rate_zscored )
		
		# Calculate criterion
		criterion.append( -((hit_rate_joined_zscored + fa_rate_joined_zscored) / 2) )
		bpd_lowbin_criterion.append( -((bpd_lowbin_hit_rate_zscored + bpd_lowbin_fa_rate_zscored) / 2) )
		bpd_highbin_criterion.append( -((bpd_highbin_hit_rate_zscored + bpd_highbin_fa_rate_zscored) / 2) )
		ppd_lowbin_criterion.append( -((ppd_lowbin_hit_rate_zscored + ppd_lowbin_fa_rate_zscored) / 2) )
		ppd_highbin_criterion.append( -((ppd_highbin_hit_rate_zscored + ppd_highbin_fa_rate_zscored) / 2) )
		criterion_low_confidence.append( -((low_confidence_hit_rate_zscored + low_confidence_fa_rate_zscored) / 2) )
		criterion_high_confidence.append( -((high_confidence_hit_rate_zscored + high_confidence_fa_rate_zscored) / 2) )
	
	return(d_prime, bpd_lowbin_d_prime, bpd_highbin_d_prime, ppd_lowbin_d_prime, ppd_highbin_d_prime, d_prime_low_confidence, d_prime_high_confidence, criterion, bpd_lowbin_criterion, bpd_highbin_criterion, ppd_lowbin_criterion, ppd_highbin_criterion, criterion_low_confidence, criterion_high_confidence)







	# PLOTS:
	
	
	# def plot_stim_resp_feed(subject, stimulus_locked_array_joined, response_locked_array_joined, feedback_locked_array_joined, omission_indices_joined, hit_indices_joined, fa_indices_joined, cr_indices_joined, miss_indices_joined, decision_time_joined,answer_yes_indices_joined, answer_no_indices_joined):
	" plot stimulus locked and response locked mean pupil time series"
	
	import scipy as sp
	import scipy.stats as stats
	import scipy.signal as signal
	import bottleneck
	import numpy as np
	import matplotlib.pyplot as plt
		
	def simpleaxis(ax):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
	
	def spine_shift(ax, shift = 10):
		for loc, spine in ax.spines.iteritems():
			if loc in ['left','bottom']:
				spine.set_position(('outward', shift)) # outward by 10 points
			elif loc in ['right','top']:
				spine.set_color('none') # don't draw spine
			else:
				raise ValueError('unknown spine location: %s'%loc)
	
	
	
	## STIMULUS LOCKED DATA:
	# Compute mean pupil responses
	stim_locked_grand_mean = bottleneck.nanmean(stimulus_locked_array_joined[-omission_indices_joined,500:], axis=0)
	stim_locked_hits_mean = bottleneck.nanmean(stimulus_locked_array_joined[hit_indices_joined[0],500:], axis=0)
	stim_locked_fa_mean = bottleneck.nanmean(stimulus_locked_array_joined[fa_indices_joined[0],500:], axis=0)
	stim_locked_cr_mean = bottleneck.nanmean(stimulus_locked_array_joined[cr_indices_joined[0],500:], axis=0)
	stim_locked_miss_mean = bottleneck.nanmean(stimulus_locked_array_joined[miss_indices_joined[0],500:], axis=0)
	# Compute std mean pupil responses
	stim_locked_grand_sem = ( bottleneck.nanstd(stimulus_locked_array_joined[-omission_indices_joined,500:], axis=0) / sp.sqrt(len(stim_locked_grand_mean)) )
	stim_locked_hits_sem = ( bottleneck.nanstd(stimulus_locked_array_joined[hit_indices_joined[0],500:], axis=0) / sp.sqrt(hit_indices_joined[0].sum()) )
	stim_locked_fa_sem = ( bottleneck.nanstd(stimulus_locked_array_joined[fa_indices_joined[0],500:], axis=0) / sp.sqrt(fa_indices_joined[0].sum()) )
	stim_locked_cr_sem = ( bottleneck.nanstd(stimulus_locked_array_joined[cr_indices_joined[0],500:], axis=0) / sp.sqrt(cr_indices_joined[0].sum()) )
	stim_locked_miss_sem = ( bottleneck.nanstd(stimulus_locked_array_joined[miss_indices_joined[0],500:], axis=0) / sp.sqrt(miss_indices_joined[0].sum()) )
	## RESPONSE LOCKED DATA:
	# Compute mean pupil responses
	resp_locked_grand_mean = bottleneck.nanmean(response_locked_array_joined[-omission_indices_joined,250:], axis=0)
	resp_locked_hits_mean = bottleneck.nanmean(response_locked_array_joined[hit_indices_joined[0],250:], axis=0)
	resp_locked_fa_mean = bottleneck.nanmean(response_locked_array_joined[fa_indices_joined[0],250:], axis=0)
	resp_locked_cr_mean = bottleneck.nanmean(response_locked_array_joined[cr_indices_joined[0],250:], axis=0)
	resp_locked_miss_mean = bottleneck.nanmean(response_locked_array_joined[miss_indices_joined[0],250:], axis=0)
	# Compute std mean pupil responses
	resp_locked_grand_sem = ( bottleneck.nanstd(response_locked_array_joined[:,250:], axis=0) / sp.sqrt(len(resp_locked_grand_mean)) )
	resp_locked_hits_sem = ( bottleneck.nanstd(response_locked_array_joined[hit_indices_joined[0],250:], axis=0) / sp.sqrt(hit_indices_joined[0].sum()) )
	resp_locked_fa_sem = ( bottleneck.nanstd(response_locked_array_joined[fa_indices_joined[0],250:], axis=0) / sp.sqrt(fa_indices_joined[0].sum()) )
	resp_locked_cr_sem = ( bottleneck.nanstd(response_locked_array_joined[cr_indices_joined[0],250:], axis=0) / sp.sqrt(cr_indices_joined[0].sum()) )
	resp_locked_miss_sem = ( bottleneck.nanstd(response_locked_array_joined[miss_indices_joined[0],250:], axis=0) / sp.sqrt(miss_indices_joined[0].sum()) )
	## FEEDBACK LOCKED DATA:
	# Compute mean pupil responses
	feed_locked_grand_mean = bottleneck.nanmean(feedback_locked_array_joined[-omission_indices_joined,500:2499], axis=0)
	feed_locked_hits_mean = bottleneck.nanmean(feedback_locked_array_joined[hit_indices_joined[0],500:2499], axis=0)
	feed_locked_fa_mean = bottleneck.nanmean(feedback_locked_array_joined[fa_indices_joined[0],500:2499], axis=0)
	feed_locked_cr_mean = bottleneck.nanmean(feedback_locked_array_joined[cr_indices_joined[0],500:2499], axis=0)
	feed_locked_miss_mean = bottleneck.nanmean(feedback_locked_array_joined[miss_indices_joined[0],500:2499], axis=0)
	# Compute std mean pupil responses
	feed_locked_grand_sem = ( bottleneck.nanstd(feedback_locked_array_joined[:,500:2499], axis=0) / sp.sqrt(len(feed_locked_grand_mean)) )
	feed_locked_hits_sem = ( bottleneck.nanstd(feedback_locked_array_joined[hit_indices_joined[0],500:2499], axis=0) / sp.sqrt(hit_indices_joined[0].sum()) )
	feed_locked_fa_sem = ( bottleneck.nanstd(feedback_locked_array_joined[fa_indices_joined[0],500:2499], axis=0) / sp.sqrt(fa_indices_joined[0].sum()) )
	feed_locked_cr_sem = ( bottleneck.nanstd(feedback_locked_array_joined[cr_indices_joined[0],500:2499], axis=0) / sp.sqrt(cr_indices_joined[0].sum()) )
	feed_locked_miss_sem = ( bottleneck.nanstd(feedback_locked_array_joined[miss_indices_joined[0],500:2499], axis=0) / sp.sqrt(miss_indices_joined[0].sum()) )
	
	# Make the plt.plot
	figure_mean_pupil_locked_to_stimulus_response_feedback_SDT = plt.figure(figsize=(16, 36))
	
	# Stimulus 
	a = plt.subplot(311)
	xa = np.arange(-499,4000)
	p1, = plt.plot(xa, stim_locked_hits_mean, color = 'r', linewidth=2)
	p2, = plt.plot(xa, stim_locked_fa_mean, color = 'r', alpha = 0.5, linewidth=2)
	p3, = plt.plot(xa, stim_locked_miss_mean, color = 'b', alpha = 0.5, linewidth=2)
	p4, = plt.plot(xa, stim_locked_cr_mean, color = 'b', linewidth=2)
	plt.fill_between( xa, (stim_locked_hits_mean+stim_locked_hits_sem), (stim_locked_hits_mean-stim_locked_hits_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xa, (stim_locked_fa_mean+stim_locked_fa_sem), (stim_locked_fa_mean-stim_locked_fa_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xa, (stim_locked_miss_mean+stim_locked_miss_sem), (stim_locked_miss_mean-stim_locked_miss_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xa, (stim_locked_cr_mean+stim_locked_cr_sem), (stim_locked_cr_mean-stim_locked_cr_sem), alpha=0.1, color = 'b' )
	plt.axvline(sp.mean(decision_time_joined[answer_yes_indices_joined[0]*-omission_indices_joined]), -1, 1, color = 'r', linestyle = '--', alpha = 0.5)
	plt.axvline(sp.mean(decision_time_joined[answer_no_indices_joined[0]*-omission_indices_joined]), -1, 1, color = 'b', linestyle = '--', alpha = 0.5)
	plt.axvline(0, -1, 1, linewidth=1)
	lowest_response = min(stim_locked_hits_mean[sp.mean(decision_time_joined)], stim_locked_fa_mean[sp.mean(decision_time_joined)], stim_locked_miss_mean[sp.mean(decision_time_joined)], stim_locked_cr_mean[sp.mean(decision_time_joined)])
	plt.hist(decision_time_joined[answer_yes_indices_joined[0]*-omission_indices_joined], bins=20, weights = (np.ones(response_locked_array_joined[answer_yes_indices_joined[0]*-omission_indices_joined].shape[0]) / len(decision_time_joined) * 2.5), bottom = round(lowest_response-0.2,1), color = 'r', alpha = 0.5)
	plt.hist(decision_time_joined[answer_no_indices_joined[0]*-omission_indices_joined], bins=20, weights = (np.ones(response_locked_array_joined[answer_no_indices_joined[0]*-omission_indices_joined].shape[0]) / len(decision_time_joined) * 2.5), bottom = round(lowest_response-0.3,1), color = 'b', alpha = 0.5)
	plt.text(sp.mean(decision_time_joined[answer_yes_indices_joined[0]*-omission_indices_joined])+30,plt.axis()[3]-0.05,"'yes!'", size=18)
	plt.text(sp.mean(decision_time_joined[answer_no_indices_joined[0]*-omission_indices_joined])+30,plt.axis()[3]-0.05,"'no!'", size=18)
	plt.title(subject + " - pupil change stimulus locked", size=36)
	plt.ylabel("Pupil change (Z)", size=36)
	plt.xlabel("Time in milliseconds", size=36)
	leg = plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit_indices_joined[0].sum()) + " trials", "FA; " + str(fa_indices_joined[0].sum()) + " trials", "MISS; " + str(miss_indices_joined[0].sum()) + " trials", "CR; " + str(cr_indices_joined[0].sum()) + " trials"], loc = 2, fancybox = True)
	leg.get_frame().set_alpha(0.9)
	if leg:
		for t in leg.get_texts():
			t.set_fontsize(18)    # the legend text fontsize
		for l in leg.get_lines():
			l.set_linewidth(3.5)  # the legend line width
	plt.tick_params(axis='both', which='major', labelsize=24)
	# plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit_indices_joined[0].sum()) + " trials", "FA; " + str(fa_indices_joined[0].sum()) + " trials", "MISS; " + str(miss_indices_joined[0].sum()) + " trials", "CR; " + str(cr_indices_joined[0].sum()) + " trials"], loc = 2)
	simpleaxis(a)
	spine_shift(a)
	plt.xticks([0,1000,2000,3000,4000])
		
	# Response
	b = plt.subplot(312)
	xb = np.arange(-3249,1500)
	p1, = plt.plot(xb, resp_locked_hits_mean, color = 'r', linewidth=2)
	p2, = plt.plot(xb, resp_locked_fa_mean, color = 'r', alpha = 0.5, linewidth=2)
	p3, = plt.plot(xb, resp_locked_cr_mean, color = 'b', linewidth=2)
	p4, = plt.plot(xb, resp_locked_miss_mean, color = 'b', alpha = 0.5, linewidth=2)
	plt.fill_between( xb, (resp_locked_hits_mean+resp_locked_hits_sem), (resp_locked_hits_mean-resp_locked_hits_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (resp_locked_fa_mean+resp_locked_fa_sem), (resp_locked_fa_mean-resp_locked_fa_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xb, (resp_locked_cr_mean+resp_locked_cr_sem), (resp_locked_cr_mean-resp_locked_cr_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xb, (resp_locked_miss_mean+resp_locked_miss_sem), (resp_locked_miss_mean-resp_locked_miss_sem), alpha=0.1, color = 'b' )
	plt.axvline(0-sp.mean(decision_time_joined[answer_yes_indices_joined[0]*-omission_indices_joined]), -1, 1, color = 'r', linestyle = '--', alpha = 0.5)
	plt.axvline(0-sp.mean(decision_time_joined[answer_no_indices_joined[0]*-omission_indices_joined]), -1, 1, color = 'b', linestyle = '--', alpha = 0.5)
	plt.axvline(0, -1, 1, linewidth=1)
	plt.hist(0-decision_time_joined[answer_yes_indices_joined[0]*-omission_indices_joined], bins=20, weights = (np.ones(response_locked_array_joined[answer_yes_indices_joined[0]*-omission_indices_joined].shape[0]) / len(decision_time_joined) * 2.5), bottom = plt.axis()[3]-0.2, color = 'r', alpha = 0.5)
	plt.hist(0-decision_time_joined[answer_no_indices_joined[0]*-omission_indices_joined], bins=20, weights = (np.ones(response_locked_array_joined[answer_no_indices_joined[0]*-omission_indices_joined].shape[0]) / len(decision_time_joined) * 2.5), bottom = plt.axis()[3]-0.3, color = 'b', alpha = 0.5)
	# plt.text(0-sp.mean(decision_time_joined[answer_yes_indices_joined[0]*-omission_indices_joined])+10,0.15,"'yes!'")
	# plt.text(0-sp.mean(decision_time_joined[answer_no_indices_joined[0]*-omission_indices_joined])+10,0.35,"'no!'")
	plt.xlim( (-3225, 1500) )
	plt.title(subject + " - pupil change response locked", size=36)
	plt.ylabel("Pupil change (Z)", size=36)
	plt.xlabel("Time in milliseconds", size=36)
	plt.xticks([-3000,-2000,-1000,0,1000])
	plt.tick_params(axis='both', which='major', labelsize=24)
	# plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit_indices_joined[0].sum()) + " trials", "FA; " + str(fa_indices_joined[0].sum()) + " trials", "CR; " + str(cr_indices_joined[0].sum()) + " trials", "MISS; " + str(miss_indices_joined[0].sum()) + " trials"], loc = 2)
	simpleaxis(b)
	spine_shift(b)
	
	# Feedback
	c = plt.subplot(313)
	xc = np.arange(-499,1500)
	p1, = plt.plot(xc, feed_locked_hits_mean, color = 'r', linewidth=2)
	p2, = plt.plot(xc, feed_locked_fa_mean, color = 'r', alpha = 0.5, linewidth=2)
	p3, = plt.plot(xc, feed_locked_miss_mean, color = 'b', alpha = 0.5, linewidth=2)
	p4, = plt.plot(xc, feed_locked_cr_mean, color = 'b', linewidth=2)
	plt.fill_between( xc, (feed_locked_hits_mean+feed_locked_hits_sem), (feed_locked_hits_mean-feed_locked_hits_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xc, (feed_locked_fa_mean+feed_locked_fa_sem), (feed_locked_fa_mean-feed_locked_fa_sem), alpha=0.1, color = 'r' )
	plt.fill_between( xc, (feed_locked_cr_mean+feed_locked_cr_sem), (feed_locked_cr_mean-feed_locked_cr_sem), alpha=0.1, color = 'b' )
	plt.fill_between( xc, (feed_locked_miss_mean+feed_locked_miss_sem), (feed_locked_miss_mean-feed_locked_miss_sem), alpha=0.1, color = 'b' )
	plt.title(subject + " - pupil change feedback locked", size=36)
	plt.ylabel("Pupil change (Z)", size=36)
	plt.xlabel("Time in milliseconds", size=36)
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.axvline(0, -1, 1, linewidth=1)
	# plt.legend([p1, p2, p3, p4], ["HIT; " + str(hit_indices_joined[0].sum()) + " trials", "FA; " + str(fa_indices_joined[0].sum()) + " trials", "MISS; " + str(miss_indices_joined[0].sum()) + " trials", "CR; " + str(cr_indices_joined[0].sum()) + " trials"], loc = 2)
	simpleaxis(c)
	spine_shift(c)
	
	return(figure_mean_pupil_locked_to_stimulus_response_feedback_SDT)
