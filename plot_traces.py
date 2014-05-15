from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import Image
from matplotlib import _png
from matplotlib.offsetbox import OffsetImage
import scipy.io
import pylab
#from matplotlib import rcParams
#rcParams['path.simplify'] = True
#rcParams['pdf.compression'] =1

def resample(ms,srate):
	return int(round(ms/1000*srate))

def formatdata(data,Params):
	"""
	reads in TrialsMTX data structure, pulls out relevant data
	"""
	mndata = dict()
	alltrials = np.array([])
	for k in range(len(Params["conditions"])):
		conditionmean = data[0,k].mean(axis = 0)
		mndata.update({Params["conditions"][k]: {'data' : data[0,k].mean(axis = 0), 'cmax' : conditionmean.max(), 'cmin' : conditionmean.min()}})
	return mndata

def traces(mndata,Params,srate,	colors = ['red','orange','green','blue'], ylim = np.array([-25, 110])):
	"""
	plots traces of high gamma data for the trial duration. separated by condition
	"""
	#plot high gamma traces
	#data should be bandpassed (todo)
	#resample to srate
	st = resample(Params["st"],srate)
	en = resample(Params["en"],srate)
	bl_en = resample(Params["bl_en"],srate)
	bl_st = resample(Params["bl_st"],srate)
	plot_tp = resample(Params["plot"],srate)
	#cue = resample(500,srate) # VISUAL (600 for AUDITORY)
	cue = resample(600,srate) # AUDITORY
	#stimlength = resample(600, srate) #VISUAL
	stimlength = resample(550, srate) #AUDITORY 

	x = np.array(range(st,en+1))
	f, ax = plt.subplots(1,1)
	ax.axhline(y = 0,color = 'k',linewidth=2)
	ax.axvline(x = 0,color='k',linewidth=2)
	ax.axvline(x = cue,color = 'gray',linewidth = 2)
	ax.axvline(x = cue+stimlength,color = 'gray',linewidth = 2)
	ax.axvspan(cue, cue+stimlength, facecolor='0.5', alpha=0.25,label = 'cue')

	for j in range(len(Params["conditions"])):
		condition = Params['conditions'][j]
		#if condition not in ['0', '100']:
		y = mndata[condition]['data']
		ax.plot(x,y, label = condition,linewidth = 2,color = colors[j])
	
	ax.set_ylim((-25,150))
	ax.set_xlim(st,en)
	#ax.set_xlim(st, resample(2500,srate))

	#ax.legend()
	ax.xaxis.set_ticklabels(['', '0', '','500', '', '1000', '', '1500', '', '2000','','2500','', '3000'],minor=False)
	ax.xaxis.set_ticks(range(st,en,plot_tp))
	
	#ax.xaxis.set_ticklabels(['', '0', '','500', '', '1000', '', '1500', '', '2000', '', '2500'],minor=False)
	#ax.xaxis.set_ticks(range(st,resample(2500, srate),plot_tp))
	

	ax.xaxis.set_tick_params(labelsize = 14)
	ax.yaxis.set_tick_params(labelsize=14)
	
	plt_str = list()
	[plt_str.append(str(z)) for z in range(ylim[0],ylim[1], 25)]

	ax.yaxis.set_ticklabels(plt_str,minor=False)
	ax.yaxis.set_ticks(range(ylim[0],ylim[1], 25))
	ax.set_ylim(ylim[0], ylim[1])
	
	#ax.yaxis.set_ticklabels(['','0','','100','','200', '','300','','400','','500','','600','','700'],minor=False)
	#ax.yaxis.set_ticks(range(-50,601,50))
	
	xticklabels = plt.getp(plt.gca(), 'xticklabels')
	yticklabels = plt.getp(plt.gca(), 'yticklabels')
	plt.setp(xticklabels, fontsize=14, weight='bold')
	plt.setp(yticklabels, fontsize=14, weight='bold')

	for pos in ['top','bottom','right','left']:
		ax.spines[pos].set_edgecolor('gray')
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	#ax.set_xlabel("time (ms)")
	#ax.set_ylabel("% change baseline")
	#ax.set_title('Analytic Amplitude - High Gamma (70-150Hz)', fontsize = 18)


## EXAMPLE FOR HOW TO START CALLING FUNCTIONS
"""
dataDir = "/Users/matar/Dropbox/figsforBob/CNS2013/"
matdata = scipy.io.loadmat(dataDir+'TrialsMTX_e44bandpass10_all',struct_as_record = True)
data = matdata["TrialsMTX"]['data'][0,0]

#define parameters
Params={"f1":70, "f2": 150, "st" :-250, "en":3000, "plot":250, "bl_st" : -250, "bl_en":-50, "caxis":150, "conditions":['0','20','40','60','80','100']}
#srate = 3.0518e3
srate = 1000; #resampled in making TrialsMTX

#format data
mndata = formatdata(data, Params)
traces(mndata,Params,srate, colors = ['darkred','red','orange','green','blue','navy'])
plt.savefig('/Users/matar/Dropbox/figsforBob/CNS2013/ST27e44_traces_all.pdf')
"""
