from __future__ import division
import scipy.io
import numpy as np
import re
import matplotlib.pyplot as plt
import curvefitting
import scipy.optimize as opt
import matplotlib

#change font to bold
font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

def makeresp(stimuli, x):
	#find indices of cat or dog responses and put in resp dict
	resp = dict()
	for s in stimuli:
		resp[s] = list(np.where(x['resp'] == s))[0]
	#	r = re.compile(s)
	#	vmatch = np.vectorize(lambda x:bool(r.match(x[0])))
	#	vmatch = np.vectorize(lambda x:bool(re.match(x[0],s)))
	#	sel = vmatch(x['resp']).nonzero()
	#	resp[s] = sel[0]
	return resp

def makestim(levels,x):
	#find indices of morph level stimuli and put in stim dict
	stim = dict()
	for level in levels:
		morph = '^decision_' + level
		r = re.compile(morph)
		vmatch = np.vectorize(lambda x:bool(r.match(x[0])))
		sel = vmatch(x['event']).nonzero()
		stim[level] = sel[0]
	return stim

def proportion(stim, resp, category,levels):
	#takes stim and resp and returns the proportion of category A (dog, male) answers in each morph level
	#category = 'd', 'c', 'w','m' (category that want proportion of)
	y = []
	for l in levels:
		n = len(np.intersect1d(stim[l], resp[category]))
		y.append(n/len(stim[l]))
	return y