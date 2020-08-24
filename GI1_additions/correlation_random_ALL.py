import matplotlib.pyplot as plt
import numpy as np
import green as g
import source as s
import processing as proc
import parameters
import time

#==============================================================================
#- The original script, correaltion_random.py, required 2 stations to be listed  
#-  as parameters. Here we rearrange a little bit, to go ahead and compute all
#-  stations' raw timeseries and all station-station pairs' correlations.
#-
#- All synthetic data is saved in OUTPUT/
#==============================================================================

def correlation_random_ALL(verbose=False,plot=False,save=True):
#def generate_random_signals(verbose=False,plot=False,save=True):

	"""
	ut,traw,cct,t,ccf,f = generate_random_signals(verbose=False,plot=False,save=True)

	Compute and plot correlation function based on random source summation.
	This function is based directly on "correlation_random", except that all station-station
	pairs are computed and saved. It also outputs the raw time domain traces.

	INPUT:
	------
	plot:			plot when True.
	verbose:		give screen output when True.
	save:			store individual correlations to OUTPUT/correlations_individual

	OUTPUT:
	-------
	ut,traw:        Time-domain raw traces and time axis
	cct, t:		Time-domain correlation function and time axis [N^2 s / m^4],[s].
	ccf, f:		Frequency-domain correlation function and frequency axis [N^2 s^2 / m^4],[1/s].
	
	Last updated: 18 August 2020.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	#- Start time.
	t1=time.time()

	#- Input parameters.
	p=parameters.Parameters()

	#- Spatial grid.
	x_line=np.arange(p.xmin,p.xmax,p.dx)
	y_line=np.arange(p.ymin,p.ymax,p.dy)
	x,y=np.meshgrid(x_line,y_line)

	#- Compute number of samples as power of 2.
	n=int(2**np.ceil(np.log2((p.Twindow)/p.dt)))

	#- Frequency axis.
	df=1.0/(n*p.dt)
	f=np.arange(0.0,1.0/p.dt,df)
	omega=2.0*np.pi*f

	#- Compute time axis
	t=np.arange(p.tmin,p.tmax,p.dt)
	traw=np.arange(0,n*p.dt,p.dt)

	#- Compute instrument response and natural source spectrum.
	S,indeces=s.space_distribution()
	instrument,natural=s.frequency_distribution(f)

	#- Issue some information if wanted.
	if verbose:

		print('number of samples: '+str(n))
		print('maximum time: '+str(np.max(t))+' s')
		print('maximum frequency: '+str(np.max(f))+' Hz')

	#- Warnings.
	if (p.fmax>1.0/p.dt):
		print('WARNING: maximum bandpass frequency cannot be represented with this time step!')

	if (p.fmin<1.0/(n*p.dt)):
		print('WARNING: minimum bandpass frequency cannot be represented with this window length!')

	#==============================================================================
	#- March through source locations and compute raw frequency-domain noise traces.
	#==============================================================================

	#- Set a specific random seed to make simulation repeatable, e.g. for different receiver pair.
	np.random.seed(p.seed)

	#- Initialise frequency-domain wavefields.
	u=np.zeros([n,p.Nwindows,p.Nreceivers],dtype=complex)
	G=np.zeros([n,p.Nwindows],dtype=complex)

	#- Regularise zero-frequency to avoid singularity in Green function.
	omega[0]=0.01*2.0*np.pi*df

	#- March through source indices.
	for k in indeces:

		#- Random phase matrix, frequency steps times time windows.
		phi=2.0*np.pi*(np.random.rand(n,p.Nwindows)-0.5)
		ff=np.exp(1j*phi)

		#- Green function for a specific source point.
		for rec in range(0,p.Nreceivers):
			G[:,0]=g.green_input(p.x[rec],p.y[rec],x[k],y[k],omega,p.dx,p.dy,p.rho,p.v,p.Q)

			#- Apply instrument response and source spectrum
			G[:,0]=G[:,0]*instrument*np.sqrt(natural)

			#- Copy this Green function to all time intervals.
			for i in range(p.Nwindows):
				G[:,i]=G[:,0]

			#- Matrix of random frequency-domain wavefields.
			u[:,:,rec]+=S[k]*ff*G

	#- March through time windows to add earthquakes.
	for win in range(p.Nwindows):

		neq=len(p.eq_t[win])
		for i in range(neq):
			for rec in range(0,p.Nreceivers):
				G[:,0]=g.green_input(p.x[rec],p.y[rec],p.eq_x[win][i],p.eq_y[win][i],omega,p.dx,p.dy,p.rho,p.v,p.Q)
				G[:,0]=G[:,0]*instrument*np.sqrt(natural)
				u[:,win,rec]+=(p.eq_m[win][i]*G[:,0]*np.exp(-1j*omega*p.eq_t[win][i]))

	#==============================================================================
	#- Define the standard bandpass.
	#==============================================================================

	bandpass=np.zeros(np.shape(f))

	Nminmax=int(np.round((p.bp_fmin)/df))
	Nminmin=int(np.round((p.bp_fmin-p.bp_width)/df))
	Nmaxmin=int(np.round((p.bp_fmax)/df))
	Nmaxmax=int(np.round((p.bp_fmax+p.bp_width)/df))

	bandpass[Nminmin:Nminmax]=np.linspace(0.0,1.0,Nminmax-Nminmin)
	bandpass[Nmaxmin:Nmaxmax]=np.linspace(1.0,0.0,Nmaxmax-Nmaxmin)
	bandpass[Nminmax:Nmaxmin]=1.0


	#==============================================================================
	#- Apply the standard bandpass to raw observations, save if desired, plot if desired
	#==============================================================================
	ubp=np.zeros([n,p.Nwindows,p.Nreceivers],dtype=complex)
	for rec in range(p.Nreceivers):
		for i in range(p.Nwindows):
			ubp[:,i,rec]=bandpass*u[:,i,rec]
	
	ut=np.zeros([len(traw),p.Nwindows,p.Nreceivers])
	for rec in range(p.Nreceivers):
		ut[:,:,rec]=np.real(np.fft.ifft(ubp[:,:,rec],axis=0,n=n)/p.dt)

	if plot:
		plt.rcParams["font.family"] = "serif"
		plt.rcParams.update({'font.size': 10})
		for rec in range(p.Nreceivers):
			plt.plot(traw,ut[:,0,rec],linewidth=1.0)
		plt.ylabel('u(t) [N/m^2]')
		plt.title('Recordings for first time window')
		plt.xlabel('Time')
		plt.show()

	if save:

		fn='OUTPUT/raw_synthetics/traw'
		np.save(fn,traw)

		for rec in range(p.Nreceivers):
			fn='OUTPUT/raw_synthetics/syn_'+str(rec)
			np.save(fn,ut[:,:,rec])

	#==============================================================================
	#- Processing. Cross-correlating
	#==============================================================================
	# Ignoring processed version for now...

	#==============================================================================
	#- Time-domain correlation function.
	#==============================================================================

	npairs=p.Nreceivers**2
	ccf_all=np.zeros([n,p.Nwindows,npairs],dtype=complex)
	cct_all=np.zeros([len(t),p.Nwindows,npairs])
	counter=0

	#- Compute correlation function, raw only
	for rec0 in range(p.Nreceivers):
		print(rec0)
		for rec1 in range(p.Nreceivers):
			u1 = u[:,:,rec0]
			u2 = u[:,:,rec1]

			ccf=u1*np.conj(u2)

			# Apply bandpass to each window
			for i in range(p.Nwindows):
				ccf[:,i]=bandpass*ccf[:,i]

			#==============================================================================
			#- Time-domain correlation function.
			#==============================================================================

			#- Some care has to be taken here with the inverse FFT convention of numpy.

			maxlag2=int(np.size(t)) 
			maxlag=int(np.floor(maxlag2/2))

			cct=np.zeros([maxlag2,p.Nwindows],dtype=float)

			dummy=np.real(np.fft.ifft(ccf,axis=0)/p.dt)
			cct[maxlag:maxlag2,:]=dummy[0:maxlag,:]
			cct[0:maxlag,:]=dummy[n-maxlag:n,:]

			ccf_all[:,:,counter]=ccf
			cct_all[:,:,counter]=cct
			counter+=1

			#==============================================================================
			#- Save results if wanted.
			#==============================================================================

			if save:

				#- Store frequency and time axes.

				fn='OUTPUT/correlations_individual/f'
				np.save(fn,f)

				fn='OUTPUT/correlations_individual/t'
				np.save(fn,t)

				#- Store raw and processed correlations in the frequency domain.

				fn='OUTPUT/correlations_individual/ccf_'+str(rec0)+'_'+str(rec1)
				np.save(fn,ccf)

				#- Store raw and processed correlations in the time domain.
				fn='OUTPUT/correlations_individual/cct_'+str(rec0)+'_'+str(rec1)
				np.save(fn,cct)


	#==============================================================================
	#- Plot results if wanted.
	#==============================================================================

	if plot:

		plt.rcParams["font.family"] = "serif"
		plt.rcParams.update({'font.size': 10})

		# Plot correlations as a record section
		counter=0
		xx=p.x/1000.0
		yy=p.y/1000.0
		
		for i in range(p.Nreceivers):
			for j in range(p.Nreceivers):
				if(i<j): # Only plot one direction, don't need both for this plot
					total_distance=np.sqrt((xx[i]-xx[j])**2 + (yy[i]-yy[j])**2)
					this_corr=np.sum(cct_all[:,:,counter],1)
					this_corr=this_corr / np.max(this_corr)
					plt.plot(t,this_corr + total_distance,'k',linewidth=.5)
				counter+=1
		plt.xlabel('Time')
		plt.ylabel('Normalised record section')
		plt.title('Stacked, Observed Noise Correlations')
		plt.show()

	
	#- End time.
	t2=time.time()

	if verbose:
		print('elapsed time: '+str(t2-t1)+' s')

	#==============================================================================
	#- Output.
	#==============================================================================

	return ut,traw,cct_all,t,ccf_all,f

