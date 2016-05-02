from PAC_func import _phase_amplitude_coupling_edited

#get/format data
data = gdat  #this is an elec x time matrix
sfreq = srate  #1017.3Hz
f_phase = np.array((5,9))
f_amp = np.array((70,150))
ixs = ixs[0]  #just the first pair
ev = onsets_resp.astype('int')  #onset times of response
t_inc = 0.01
tmin = -.50
tmax = tmin+t_inc

#original method - filter inside
p = []
while tmin < 0:
	pac_out = cnn._phase_amplitude_coupling(data, sfreq, f_phase, f_amp, [ix],
		pac_func='plv', ev=ev,
		tmin=tmin, tmax=tmax,
		baseline=None,
		npad='auto',
		concat_epochs=True,
		verbose=False)
	p.append(pac_out)
	tmin+=t_inc
	tmax+=t_inc


#edited method - filter outside
t_inc = 0.01
tmin = -.50
tmax = tmin+t_inc
p = []

ixs = np.array(ixs, ndmin=2)
f_phase = np.atleast_2d(f_phase)
f_amp = np.atleast_2d(f_amp)

#filter data
data_ph, data_am, ix_map_ph, ix_map_am = _pre_filter_ph_am(
	data, sfreq, ixs, f_phase, f_amp, npad='auto', hi_phase='plv')

data = [(data_ph, ix_map_ph), (data_am, ix_map_am)]
while tmin < 0:
	pac_out = _phase_amplitude_coupling_edited(data, sfreq, f_phase, f_amp, ixs,
		pac_func='plv', ev=ev,
		tmin=tmin, tmax=tmax,
		baseline=None,
		npad='auto',
		concat_epochs=True,
		verbose=None,
		to_filter=False, return_data = False)
	p.append(pac_out)
	tmin+=t_inc
	tmax+=t_inc
