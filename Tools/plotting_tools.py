#!/usr/bin/env python
# encoding: utf-8

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