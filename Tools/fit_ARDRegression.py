import numpy as np
import scipy, scipy.signal
from sklearn.linear_model import ARDRegression
from sklearn.cross_validation import KFold

def fit_ARDRegression( timeseries, model, n_iter = 300, compute_score = True ):
	"""fit_ARDRegression fits a large model to a voxel timeseries"""
	
	clf = ARDRegression(n_iter = n_iter, compute_score=compute_score)
	clf.fit(X, y)
	
	return clf

def CV_ARDRegression( timeseries, model, folds = 8 ):
	"""CV fits a ARD model to the timeseries, and does a folds cross-validation"""
	
	kf = KFold(timeseries.shape, n_folds = folds, indices = False)
	
	scores = []
	for train, test in kf:
		clf = fit_ARDRegression(timeseries[train], model[train])
		scores.append(clf.score(timeseries[test], model[test]))
	
	return clf.coef_, scores
	
