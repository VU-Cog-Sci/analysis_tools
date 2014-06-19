# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:15:24 2014

@author: Daniel schreij (d.schreij@vu.nl)

This module contains functions to analyze datafiles output by the pupil eyetracker
Currently only datafiles for surfaces can be processed.

The function to start with is analyze_file. This reads in the numpy datafile
generated by the pupil recorder and stores it in a pandas Dataframe. This dataframe
object can then be passed on to most of the other functions to create fixation lists,
heatmaps, etc.

This module can also be executed directly (opposed to importing it). The first 
argument should then be the location of the surface_gaze_positions datafile. For
this datafile, the fixation list is then created.
"""

import os, sys		
import numpy as np
import pandas as pd	
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------

class InterestArea(object):
	"""Creates a rectangular interest area of specified dimensions at the indicated position.
		
	Parameters
	----------
	Specify the dimensions of the interest area here. Specify the topleft corner 
	in (x,y) and its width and height (w,h) with ints. 
	You can do this in one of the following formats:
	
	- x,y,w,h (4 ints)		
	- (x,y)(w,h) (2 tuples with each 2 ints)
	- [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] 
	  (1 sequence with 4 points designating te corners of the area
	   starting at the top-left corner and going in clockwise direction.)
	
	You can also optionally specify the label of this interest area by
	passing the keyword argument 'label' and its value (defaul: Unnamed)
				
	Raises
	-------
	ValueError: if parameters not supplied in one of the desired formats.
	"""
	
	def __init__(self, *args, **kwargs):
		"""Constructor"""
		self.set_dimensions(*args)	
		
		# Set default label, to be overwritten by kwarg "label"
		self.label = "Unnamed"				
		
		if kwargs:
			if "label" in kwargs and type(kwargs["label"]) in [str, unicode]:
				self.label = kwargs["label"]								
								
	def __repr__(self):				
		return "Interest area \"{4}\": x:{0} y:{1} w:{2} h:{3}".format(self.x, self.y, self.w, self.h, self.label)
		
	def get_dimensions(self):
		""" Returns positions and dimensions of the interest area
		Returns
		-------
		tuple with (x,y,w,h) (ints)
		"""		
		
		return (self.x, self.y, self.w, self.h)

	def get_x(self):
		"""Returns topleft x-coordinate (int) of interest area."""
		return self.x
		
	def get_y(self):
		"""Returns topleft y-coordinate (int) of interest area."""
		return self.y	
	
	def get_w(self):
		"""Returns width in pixels (int) of interest area."""
		return self.w	
	
	def get_h(self):
		"""Returns height in pixels (int) of interest area."""
		return self.h
		
	def get_label(self):
		"""Returns interest area name."""
		return self.label
		
	def set_dimensions(self, *args):
		"""Sets the dimensions and/or position of the interest area.
		
		Parameters
		----------
		Specify the topleft corner in (x,y) and its width and height (w,h) with ints. 
		You can do this in one of the following formats:
		
		- x,y,w,h (4 ints)		
		- (x,y)(w,h) (2 tuples with each 2 ints)
		- [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] 
		  (1 sequence with 4 points designating te corners of the area
		   starting at the top-left corner and going in clockwise direction.)
					
		Raises
		-------
		ValueError: if parameters not supplied in desired format
		"""
		
		# When passed all dimensions (x,y,w,h) in one iterable, such as
		# (1,2,3,4) or [1,2,4,5]
		if len(args) == 1:								
			if hasattr(args[0], "__iter__") and len(args[0]) == 4:								
				self.x = args[0][0][0]
				self.y = args[0][0][1]												
				self.w = args[0][2][0] - self.x
				self.h = args[0][2][1] - self.y		
			else:
				raise ValueError("Invalid coordinate format supplied for IA")
		# When passed (x,y) and (w,h) as 2 separate iterables	
		if len(args) == 2:
			if hasattr(args[0], "__iter__") and hasattr(args[1], "__iter__") and len(args[0]) == 2 and len(args[1]) == 2:
				self.x = args[0][0]
				self.y = args[0][1]
				self.w = args[1][0]
				self.h = args[1][1]
			else:
				raise ValueError("Invalid coordinate format supplied for IA")
		# When passed x,y,w,h as separate ints
		if len(args) == 4:
			for i in args:
				if not type(i) in [int,float]:
					raise ValueError("Invalid coordinate format supplied for IA")
			else:
				self.x = args[0]
				self.y = args[1]
				self.w = args[2]
				self.h = args[3]
		
		# Cast to integer, just to be sure
		self.x = int(self.x)
		self.y = int(self.y)
		self.w = int(self.w)
		self.h = int(self.h)
		
				
	def set_x(self, value):
		"""
		Parameters
		----------
		value (int/float)
			the top-left x-coordinate of interest area
		"""
		if type(value) in [int,float]:
			self.x = int(value)
		else:
			raise ValueError("Value must be integer or float")
			
	def set_y(self, value):
		"""
		Parameters
		----------
		value (int/float)
			the top-left y-coordinate of interest area
		"""
		if type(value) in [int,float]:	
			self.y = int(value)
		else:
			raise ValueError("Value must be integer or float")
	
	def set_w(self, value):
		"""
		Parameters
		----------
		value (int/float)
			the width of interest area
		"""
		if type(value) in [int,float]:	
			self.w = int(value)
		else:
			raise ValueError("Value must be integer or float")
	
	def set_h(self, value):
		"""
		Parameters
		----------
		value: int/float)
			height of interest area
		"""
		if type(value) in [int,float]:
			self.h = int(value)
		else:
			raise ValueError("Value must be integer or float")
			
	def set_label(self, value):
		"""
		Parameters
		----------
		value: str/unicode)
			The name of the interest area
		"""
		if type(value) in [str,unicode]:
			self.label = value
		else:
			raise ValueError("Value must be string or unicde")
			
	def get_corners(self, scale=None):
		""" Calculates x,y coordinates of each corner of the interest area
		
		Returns
		-------
		tuple with (x,y) coordinate of each corner of the interest area						
		"""
		if scale:
			x,y,w,h = self.scale(scale)		
		else:
			x,y,w,h = self.x, self.y, self.w, self.h
		
		top_left = (x,y)
		top_right = (x+w, y)
		bottom_right = (x+w, y+h)
		bottom_left	 = (x, y+h)
		return (top_left, top_right, bottom_right, bottom_left)
	
	def inside(self, (x,y)):
		""" Checks if specified point falls inside the interest area's boundaries
		
		Parameters
		----------
		(x,y) : tuple with 2 ints
			The point to check for if it falls inside the interest area.		

		Returns
		-------
		True if point is located inside interest area
		
		False if point is located outside of interest area
		"""
		
		area = self.get_corners()				
		
		n = len(area)
		
		inside = False
		p1x,p1y = area[0]
		for i in range(n+1):
			p2x,p2y = area[i % n]
			if y > min(p1y,p2y):
				if y <= max(p1y,p2y):
					if x <= max(p1x,p2x):
						if p1y != p2y:
							xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
						if p1x == p2x or x <= xints:
							inside = not inside
			p1x,p1y = p2x,p2y
		return inside
		
	def scale(self, value):
		"""Increase or decrease interest area value by certain ratio.
		This is useful if you want to plot the IAs and the image size you plot on
		is larger or smaller than the image size you specified the IA coords for
		
		Parameters
		----------
		value: tuple with 2 floats
			[0] The ratio value to enlarge or shrink the IA coordinates with on x-axis and width
			[1] The ratio value to enlarge or shrink the IA coordinates with on y-axis and height
			
		Raises
		------
		ValueError: if value is other thant a float or int
		"""
		
		if type(value) == tuple and len(value) == 2:			
			x = self.x * value[0]
			y = self.y * value[1]
			w = self.w * value[0]
			h = self.h * value[1]		
		else:
			raise ValueError("Scalar value must be tuple with 2 floats")
		
		return (x,y,w,h)
					
			
#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

def analyze_file(datafile, sacc_threshold=0.9):
	"""Analyze the supplied datafile generated by the pupil eye tracker

	This function parses the saccades and the fixations from the list of xy coordinates.
	It creates a CDF of all difference scores for x and y separately. 
	
	Difference scores that fall above the threshold (default 90% or 0.9) will be counted as a saccade
	Needless to say (but doing it anyway) difference scores that fall within the 90% are part of a fixation
	Thus: 
	 	- the lower the threshold, the more liberal the saccade detection
		- the higher the threshold, the more conservative the saccade detection

	Parameters
	----------
	datafile: string
		path to the pupil surface datafile to be parsed
	sacc_threshold: float, default 0.9
		Threshold in CDF above which to count diff value as saccade
		
	Returns
	-------
	pandas.Dataframe with:
		- Filename/participant record
		- Trial no.
		- label of surface
		- index of current frame
		- x on surface (normalized)
		- y on surface (normalized)
		- timestamp of x,y measurement
		- x diff compared to previous trial (hor sacc amplitude) 
		- y diff compared to previous trial (ver sacc amplitude)
		- Saccade (did sample occur after saccade (True/False))
		- fixation indices
		
		Each row in this dataframe corresponds with a sample (which is 24 or
		30 per second depending on the recorder setting)
			
	Raises
	------
	IOError if file not found or file is not a numpy array	
	"""
	
	# Read in datafile. Exit is file not found
	print "Analyzing {0}".format(datafile)
	if not os.path.isfile(datafile):
		raise IOError("File not found. Please specify a valid file location")
	
	# Load data into numpy series and create pandas dataframe
	eye_data = np.load(datafile)
	data = pd.DataFrame(eye_data, columns=["frame_index","surface_label","x","y","timestamp"])
	
	# Parse some extra information from the file path. This sadly isn't included in the datafile itself,
	# so it is absolutely *crucial* to adhere to the pupil folder structure when analyzing data!
	(rest, filename) = os.path.split(datafile)
	(rest, trial_no) = os.path.split(rest)
	participant = os.path.split(rest)[1]
	
	data["subject_file"] = participant
	data["trial_no"] = int(trial_no)
	# It is nice to set this information as the first two columns, so reindex the dataframe by
	# respecifying the order of columns
	data = data.reindex(columns=["subject_file","trial_no","surface_label","frame_index", "x","y","timestamp"])	
	data.x = data.x.astype(float)
	data.y = data.y.astype(float)	
	
	# Timestamps might be read as strings. Convert to floats
	data["timestamp"] = data["timestamp"].astype(float)
	
	# Subract eye data to calculate difference scores from it
	xy = eye_data[:,2:4].astype(float).T	
	vel_data = np.diff(xy)
	
	# Insert 0 for the first diff values of the array (for which no scores can be calculated)
	vel_data = np.insert(vel_data, 0, 0.0, axis=1)
	
	# Add difference scores to array (for now)	
	data["x_diff"] = pd.Series(vel_data[0,:].T, index=data.index)
	data["y_diff"] = pd.Series(vel_data[1,:].T, index=data.index)
	
	# Calculate speed above which to cound x,y shift as saccade
	x_min_sacc_speed = calculate_min_distance(vel_data[0],sacc_threshold)
	y_min_sacc_speed = calculate_min_distance(vel_data[1],sacc_threshold)
	
	# Get rows at which saccades took place. Increment fixation index value after each saccade
	saccade_rows = (np.abs(vel_data[0]) > x_min_sacc_speed) | (np.abs(vel_data[1]) > y_min_sacc_speed)
	# Get indices of saccade rows. Add 0 to beginning to take in start of first fixation too
	saccades_idx = data[saccade_rows].index.insert(0,0)
	
	# Store True at rows at which saccade is assumed to have taken place in column "saccade"
	data["saccade"] = saccade_rows	
	
	# Create a new series containing the fixation index values, to be added to the data array
	# The indices of the series correspond to the indices of the saccades in the data array
	# That is: saccade 1 demarkates the end of fixation 1, sacc 2 for fix 2, etc.		
	fixation_indices = pd.Series(data=np.arange(1,len(saccades_idx)+1), index=saccades_idx)	
	data["fixation_index"] = fixation_indices
	
	# At row at which no fixation index was entered the value by default is NaN. Use the
	# handy function bfill and ffill to fill these values with the next occurring saccade index value
	data["fixation_index"].ffill(inplace=True)

	return data
	
def calculate_min_distance(data, threshold):
	""" Determine the minimum difference score to be counted as a saccade
	by calculating a cummulative distribution of the difference scores. The
	cutoff value is determined by the value of threshold. Any value that falls
	in the cdf range above threshold can be counted as a saccade.
	
	Parameters
	----------
	data: (Nx2) np.array 
		Vector with velocity data
		
	threshold: float 
		The cut-off value above which speeds are counted as saccade
		
	Returns
	-------
	float: minimum diff value to count as saccade
	"""

	# Remove negative values (convert to positive)	
	vel = np.abs(data)	
	# Create empirical CDF of velocities
	ecdf = sm.distributions.ECDF(vel)
	# Create the 'bins' by interpolating between the min and max values (1000 points)
	bins = np.linspace(min(vel),max(vel),1000)
	# Get the probability value of each bin
	prob = ecdf(bins)
	# The min sacc speed is the value of the first bin that exceeds the threshold
	min_sacc_speed = bins[prob > threshold][0]
	
	return min_sacc_speed
	
def generate_fixations_list(data, strict=True, keep_fix_index=False):
	"""
	Compose a list of fixations from the Dataframe created by analyze_file(). 
	Additionally calculate the duration of each fixation
	
	Parameters
	----------
	data: pandas.Dataframe
		output of analyze_file function
		
	strict: bool, default True
		Due to the low sampling frequency of the pupil
		eye tracker, it is possible that gaze positions that are mid-saccade are
		incorrectly registered as fixations, which have an exceptionally short latency
		of 30 ms. This options specifies if these errononeous fixations should be
		filtered out. (default: True)
		
	keep_fix_index: bool, default False
		Depends on the value of 'strict'.
		Indicates if the original fixation indices should be kept, or if the filtered 
		saccades should be reindexed.
		
	Returns
	-------
	
	pandas.Dataframe with:
		
		- Filename/participant record
		- Trial no.
		- label of surface
		- x on surface (normalized)
		- y on surface (normalized)
		- timestamp of x,y measurement
		- x diff compared to previous trial (hor sacc amplitude)
		- y diff compared to previous trial (ver sacc amplitude)
		- fixation indices
		- fixation durations
			
		Each row in this dataframe corresponds to a fixation
	
	"""
	# Get indices of saccade rows. Add 0 to beginning to take in start of first fixation too
	saccade_idx = data[data.saccade == True].index.insert(0,0)
	# Extract relevant rows from total dataframe and remove saccade column
	fixations = data.ix[saccade_idx].drop("saccade", axis=1)

	# Calculate duration of each fixation. diff takes next element and subtracts the previous.
	# Thus diff actually gives duration of *previous* fixation (index is 1 to high by default)
	# shift(-1) corrects for this.
	fixation_durations = fixations["timestamp"].diff().shift(-1)	
	fixations["fixation_duration_ms"] = (fixation_durations*1000).round()
	
	# Instead of just taking the first (x,y) of each fixation at its start as the gaze position,
	# calculate the average (x,y) of the duration of that fixation, as sometimes
	# there is a small shift in gaze that is not large enough to be counted as a saccade but is
	# a slight displacement in the fixation.
	
	# Use a pivot table to calculate the average per fixation index
	pt = data.pivot_table(["x","y"],rows='fixation_index',aggfunc='mean')	
	# Replace the originals with these averaged x,y coords.
	fixations["x"] = pt["x"].values
	fixations["y"] = pt["y"].values
		
	#The last fixation duration is NaN with this method and needs to be calculated separately
	fixations["fixation_duration_ms"][fixations.index[-1]] = ((data["timestamp"][data.index[-1]] - fixations["timestamp"][fixations.index[-1]])*1000).round()
		
	# If strict is true, register only the 'probable' fixations (so no < 2 frames duration)
	if strict:
		fixations = fixations[fixations.fixation_duration_ms > 40]
		# Reindex the remaining saccades		
		if not keep_fix_index:
			fixations.fixation_index = np.arange(1,len(fixations)+1)
			
	return fixations


def generate_heatmap(eye_data, size=(1000,1200), surface_image=None, gauss_outflow=20, title=False):
	"""Generates heatmap of supplied samples
	
	Plots a pyplot.hist2d of the supplied list of samples
	
	Parameters
	----------
	eye_data: pandas.Dataframe
		With at least the x and y columns of the eye data
	
	size: tuple of ints
		Dimensions of the plot
	
	surface_image: string, default None:
		Path to the image to use as the background of the fixation plot
	
	gauss_outflow: int, default 20
		The extent to which the 'heat' of a fixation flows out to the neighboring pixels
		The lower this number, the more confined the hotspots are to the location of the fixation		
	
	title: bool, default False
		Title to display above graph.
	"""

	# Set plot properties	
	plt.figure(figsize=(size[0]/200, size[1]/200))
	plt.xlim(0,size[0])
	plt.ylim(0,size[1])	

	# Somehow the hist2d function crashes if the index of the data array does not start with 0,
	# which is not always the case if the data is queried from a larger dataset.
	# Reset the indices of the passed data array just to be sure they always start at 0
	eye_data.copy().reset_index(inplace=True)
	
	h, x_edge, y_edge = np.histogram2d(eye_data.x*size[0], eye_data.y*size[1], bins=(range(size[0]+1), range(size[1]+1)))
	h = ndi.gaussian_filter(h, (30, 30))  ## gaussian convolution
	
	# First check if image location is specified and exists as a file.
	if not surface_image is None:
		if not type(surface_image) == str:
 			raise TypeError("surface_image must be a path to the image")
		else:
			if not (os.path.isfile(surface_image) and os.access(surface_image, os.R_OK)):
				raise IOError("Image file not found at path or is unreadable (check permissions)")
			else:
				# Flip image upside down
				im = plt.imread(surface_image)
				plt.imshow(im, aspect='auto', extent=[0,size[0], 0, size[1]])				
		
	ax = plt.imshow(h.T)
	ax.set_alpha(0.6)
	
	if title:
		plt.title(title)

	
	
def plot_fixations(fixations, size=(1000,1200), surface_image=None, annotated=True):
	"""Plots the fixations on a surface specified by size. Saccades are shown by lines
	and fixations by circles at the start and end points of these lines. The longer the
	fixation was at a specific point, the bigger the diameter of the circle at that point.
	
	Parameters
	----------
	fixations: pandas.Dataframe
		This supplied dataframe at least needs to contain the columns:
			
			- x
			- y
			- fixation_index
			- fixation_duration
			
	size: tuple, default (1000,1200)
		The space to plot the fixations in. The fixation coordinates inside a surface are 
		normalized by the pupil eye tracker so dimensions need to be specified to be able
		to plot them.
	
	surface_image: string, default None:
		Path to the image to use as the background of the fixation plot	

	Raises
	------
	TypeError: if surface_image is not a string to an image
	IOError: if image is not found at specified path	
	"""
	plt.figure(figsize=(size[0]/200, size[1]/200))
	
	plt.xlim(0,size[0])
	plt.ylim(0,size[1])	
	
	# First check if image location is specified and exists as a file.
	if not surface_image is None:
		if not type(surface_image) == str:
 			raise TypeError("surface_image must be a path to the image")
		else:
			if not (os.path.isfile(surface_image) and os.access(surface_image, os.R_OK)):
				raise IOError("Image file not found at path or is unreadable (check permissions)")
			else:
				# Flip image upside down
				im = plt.imread(surface_image)
				plt.imshow(im, aspect='auto', extent=[0,size[0], 0, size[1]])
				
	# First saccade somehow is almost always logged outside of the surface
	# Discard it if so.
	first_x = fixations["x"].head(1).values[0]
	first_y = fixations["y"].head(1).values[0]
	if first_x < 0 or first_y < 0 or first_x > 1 or first_y > 1:
		fixations = fixations[fixations.fixation_index != 1]
	
	x_coords = list(fixations["x"]*size[0])
	y_coords = list(fixations["y"]*size[1])

	# Plot the lines representing the fixation displacements (i.e. saccades)
	plt.plot(x_coords, y_coords,'b+-')	
	
	# Plot fixations. The longer the fixation the bigger the radius of its circle
	# Use a scalar value to adapt the relative radius to the size of the plot
	scalar = size[0]/1000.0		
	plt.scatter(x=x_coords, y=y_coords, c='r', alpha=0.5, s=scalar*fixations["fixation_duration_ms"], linewidths=1)	
	
	# If required, show the saccade indices next to their data points (slightly offset to the top left)
	if annotated:
		for i in range(len(x_coords)):
			plt.annotate(i+1, xy=(x_coords[i], y_coords[i]), xytext=(-10,10), textcoords = 'offset points')
	plt.show()
	
	
def plot_interest_areas(IA_set, size=(1000,1200), surface_image=None, annotated=True):
	"""Plots the interest areas as designated by IA_set, which should be a sequence of
	pupil.InterestArea objects. It draws them as rectangles in the given size dimensions.
	Optionally, one can provide a surface_image to draw the IAs on.
	
	Parameters
	----------
	IA_set : sequence of InterestArea objects
		The Interest areas to plot.
			
	size: tuple, default (1000,1200)
		The space to plot the fixations in. The fixation coordinates inside a surface are 
		normalized by the pupil eye tracker so dimensions need to be specified to be able
		to plot them.
	
	surface_image: string, default None:
		Path to the image to use as the background of the fixation plot	

	Raises
	------
	TypeError : if surface_image is not a string to an image
	
	IOError : if image is not found at specified path		
	"""
	scaler = (size[0]/1000.0, size[1]/1200.0)
	plt.figure(figsize=(size[0]/200, size[1]/200))
	
	plt.xlim(0,size[0])
	plt.ylim(0,size[1])	
	
	# First check if image location is specified and exists as a file.
	if not surface_image is None:
		if not type(surface_image) == str:
 			raise TypeError("surface_image must be a path to the image")
		else:
			if not (os.path.isfile(surface_image) and os.access(surface_image, os.R_OK)):
				raise IOError("Image file not found at path or is unreadable (check permissions)")
			else:
				# Flip image upside down
				im = plt.imread(surface_image)
				plt.imshow(im, aspect='auto', extent=[0,size[0], 0, size[1]])
	
	for ia in IA_set:			
		x,y = zip(*ia.get_corners(scaler))
		# Add first coordinate again to close shape		
		x = list(x) + [x[0]]
		y = list(y) + [y[0]]		
		plt.plot(x,y)		

		if annotated:
			plt.annotate(ia.get_label(), xy=(x[0],y[0]), xytext=(5,ia.get_h()/4), textcoords = 'offset points')
		
	plt.show()
	
def assign_fixations_to_IAs(fix_list, IAs, scaler=(1000,1200)):
	"""Determines if a fixation falls within a certain interest area, and if so
	assigns this IA to the row of this fixation in the passed fixations dataframe.
	
	Parameters
	----------
	fix_list : pandas.Dataframe
          List of fixations. Preferred is the output format of
	        generate_fixations_list()
	    but can be any arbitrary dataframe in which a single row corresponds
	    to a fixation. This dataframe should at least have a "x" and "y" column
	    representing these coordinates of the fixation.
	IAs : sequence of InterestArea objects 
	
	Returns
	-------
	pandas.Dataframe, which is the passed fix_list with the interest_area column added	
	
	Raises
	------
	TypeError : if surface_image is not a string to an image
	
	IOError : if image is not found at specified path					
	"""
			
	# Initialze
	fix_list["interest_area"] = "None"
	for row in fix_list.iterrows():
		index = row[0]
		data = row[1]
				
		coord = (data["x"]*scaler[0], data["y"]*scaler[1])
		
		for ia_candidate in IAs:
			if ia_candidate.inside(coord):
				fix_list.loc[index,"interest_area"] = ia_candidate.get_label()
				break
	return fix_list


def analyze_files_in_folder(folder, sacc_threshold=0.9, _sort_result=True):
	"""Travese through folders looking for datafilea generated by the pupil eye tracker

	This function recursively traverses through a folder structure looking for pupil data files.
	A good starting point always is the main folder of the recording session.
	For instance if you perform a recording with pupil, it's data structure 
	commonly is
	
	<pupil_folder>/recordings/<session_name>/XXX
	
	where XXX is a number starting from 000, incrementing with each recording.
	The <session_name> variable will be used for the datafile/participant name
	and the XXX for the trial no.	
	
	Parameters
	----------
	folder: string
		path to the starting folder which to traverse from.
		
	sacc_threshold: float, default 0.9
		distance threshold above which to count an (x,y) as a saccade.
		
	Returns
	-------
	pandas.Dataframe with:
		
		- Filename/participant record
		- Trial no.
		- label of surface
		- index of current frame
		- x on surface (normalized)
		- y on surface (normalized)
		- timestamp of x,y measurement
		- x diff compared to previous trial (hor sacc amplitude) 
		- y diff compared to previous trial (ver sacc amplitude)
		- Saccade (did sample occur after saccade (True/False))
		- fixation indices
		
		Each row in this dataframe corresponds with a sample (which is 24 or
		30 per second depending on the recorder setting)
			
	Raises
	------
	IOError if folder not found.
	
	"""	
	if not os.path.isdir(folder):
		raise IOError("Folder not found!")
		
	folder_contents = os.listdir(folder)
	
	data = None
	fixations = None
	for item in folder_contents:
		# Skip directory specifics
		if item in [".",".."]:
			continue
		
		item_path = os.path.join(folder,item)
		
		# If item is a folder, dive into it and restart this process!
		if os.path.isdir(item_path):
			new_data, new_fixations = analyze_files_in_folder(item_path, sacc_threshold, _sort_result=False)
			if not new_data is None:						
				if not data is None:
					data = pd.concat([data,new_data], ignore_index=True)
				else:			
					data = new_data
					
			if not new_fixations is None:
				if not fixations is None:
					fixations = pd.concat([fixations,new_fixations], ignore_index=True)
				else:
					fixations = new_fixations
						
		# If item is a file, check if is the file we are looking for and analyze
		elif os.path.isfile(item_path) and os.path.split(item_path)[1] == "surface_gaze_positions.npy":
			new_data = analyze_file(item_path, sacc_threshold)
			new_fixations = generate_fixations_list(new_data)
			if not new_data is None:						
				if not data is None:
					data = pd.concat([data,new_data], ignore_index=True)
				else:			
					data = new_data
			if not new_fixations is None:						
				if not fixations is None:
					fixations = pd.concat([fixations, new_fixations], ignore_index=True)
				else:			
					fixations = new_fixations		
								
	# Due to the recursive nature of this function, things might not happen in order
	# or in a sorted ways as one might expect. Therefore, explicitly sort the result afterwards
	# Only do this for the top-level function (so not for the recursive calls)
	if _sort_result:
		if not data is None:
			data = data.sort_index(by=['subject_file','trial_no','timestamp']).reset_index(drop=True)
		if not fixations is None:
			fixations = fixations.sort_index(by=['subject_file','trial_no','timestamp']).reset_index(drop=True)
	
	return data, fixations

#------------------------------------------------------------------------------
# Main script
#------------------------------------------------------------------------------

if __name__ == "__main__":
	if len(sys.argv)	< 2:
		print "Please supply data file location"
	else:	
		datafile = sys.argv[1]
			
		try:
			if os.path.isfile(datafile):					
				# Get data from numpy file
				eye_data = analyze_file(datafile,0.9)
				
				# Compose a list of fixations
				fixations = generate_fixations_list(eye_data)
				
				eye_data.to_csv(os.path.join(os.getcwd(),"raw_data.csv"), index=False)
				fixations.to_csv(os.path.join(os.getcwd(),"fixations.csv"), index=False)
			
			# If a folder is given as argument, traverse it while looking for datafiles
			if os.path.isdir(datafile):		
				eye_data = analyze_files_in_folder(datafile, 0.9)
				
				# Write data to csv file
				eye_data.to_csv(os.path.join(os.getcwd(),"raw_data.csv"), index=False)
		
		except IOError as e:
			print >> sys.stderr, e.message
			sys.exit(1)		
			
		

		
			
		