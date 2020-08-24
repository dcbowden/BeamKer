import matplotlib.pyplot as plt
import numpy as np
import green as g
import source as s
import processing as proc
import parameters
import time

def correlation_random(rec0=0,rec1=1,verbose=False,plot=False,save=True):

	"""
	cct,cct_proc,t,ccf,ccf_proc,f = correlation_random(rec0=0,rec1=1,verbose=False,plot=False,save=True)

	Compute and plot correlation function based on random source summation.

	INPUT:
	------
	rec0, rec1:		indeces of the receivers used in the correlation. 
	plot:			plot when True.
	verbose:		give screen output when True.
	save:			store individual correlations to OUTPUT/correlations_individual

	OUTPUT:
	-------
	cct, t:		Time-domain correlation function and time axis [N^2 s / m^4],[s].
	ccf, f:		Frequency-domain correlation function and frequency axis [N^2 s^2 / m^4],[1/s].
	
	Last updated: 12 August 2019.
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
	u1=np.zeros([n,p.Nwindows],dtype=complex)
	u2=np.zeros([n,p.Nwindows],dtype=complex)
	G1=np.zeros([n,p.Nwindows],dtype=complex)
	G2=np.zeros([n,p.Nwindows],dtype=complex)

	#- Regularise zero-frequency to avoid singularity in Green function.
	omega[0]=0.01*2.0*np.pi*df

	#- March through source indices.
	for k in indeces:

		#- Green function for a specific source point.
		G1[:,0]=g.green_input(p.x[rec0],p.y[rec0],x[k],y[k],omega,p.dx,p.dy,p.rho,p.v,p.Q)
		G2[:,0]=g.green_input(p.x[rec1],p.y[rec1],x[k],y[k],omega,p.dx,p.dy,p.rho,p.v,p.Q)

		#- Apply instrument response and source spectrum
		G1[:,0]=G1[:,0]*instrument*np.sqrt(natural)
		G2[:,0]=G2[:,0]*instrument*np.sqrt(natural)

		#- Copy this Green function to all time intervals.
		for i in range(p.Nwindows):

			G1[:,i]=G1[:,0]
			G2[:,i]=G2[:,0]

		#- Random phase matrix, frequency steps times time windows.
		phi=2.0*np.pi*(np.random.rand(n,p.Nwindows)-0.5)
		ff=np.exp(1j*phi)

		#- Matrix of random frequency-domain wavefields.
		u1+=S[k]*ff*G1
		u2+=S[k]*ff*G2

	#- March through time windows to add earthquakes.
	for win in range(p.Nwindows):

		neq=len(p.eq_t[win])

		for i in range(neq):

			G1=g.green_input(p.x[rec0],p.y[rec0],p.eq_x[win][i],p.eq_y[win][i],omega,p.dx,p.dy,p.rho,p.v,p.Q)
			G2=g.green_input(p.x[rec1],p.y[rec1],p.eq_x[win][i],p.eq_y[win][i],omega,p.dx,p.dy,p.rho,p.v,p.Q)

			G1=G1*instrument*np.sqrt(natural)
			G2=G2*instrument*np.sqrt(natural)

			u1[:,win]+=p.eq_m[win][i]*G1*np.exp(-1j*omega*p.eq_t[win][i])
			u2[:,win]+=p.eq_m[win][i]*G2*np.exp(-1j*omega*p.eq_t[win][i])

	#==============================================================================
	#- Processing.
	#==============================================================================

	#- Apply single-station processing.
	u1_proc,u2_proc=proc.processing_single_station(u1,u2,f,verbose)

	#- Compute correlation function, raw and processed.
	ccf=u1*np.conj(u2)
	ccf_proc=u1_proc*np.conj(u2_proc)

	#- Apply correlation processing.
	ccf_proc=proc.processing_correlation(ccf_proc,f,verbose)

	#==============================================================================
	#- Apply the standard bandpass.
	#==============================================================================

	bandpass=np.zeros(np.shape(f))

	Nminmax=int(np.round((p.bp_fmin)/df))
	Nminmin=int(np.round((p.bp_fmin-p.bp_width)/df))
	Nmaxmin=int(np.round((p.bp_fmax)/df))
	Nmaxmax=int(np.round((p.bp_fmax+p.bp_width)/df))

	bandpass[Nminmin:Nminmax]=np.linspace(0.0,1.0,Nminmax-Nminmin)
	bandpass[Nmaxmin:Nmaxmax]=np.linspace(1.0,0.0,Nmaxmax-Nmaxmin)
	bandpass[Nminmax:Nmaxmin]=1.0

	for i in range(p.Nwindows):

		ccf[:,i]=bandpass*ccf[:,i]
		ccf_proc[:,i]=bandpass*ccf_proc[:,i]

	#==============================================================================
	#- Time-domain correlation function.
	#==============================================================================

	#- Some care has to be taken here with the inverse FFT convention of numpy.

	maxlag2=int(np.size(t)) 
	maxlag=int(np.floor(maxlag2/2))

	cct=np.zeros([maxlag2,p.Nwindows],dtype=float)
	cct_proc=np.zeros([maxlag2,p.Nwindows],dtype=float)

	dummy=np.real(np.fft.ifft(ccf,axis=0)/p.dt)
	cct[maxlag:maxlag2,:]=dummy[0:maxlag,:]
	cct[0:maxlag,:]=dummy[n-maxlag:n,:]

	dummy=np.real(np.fft.ifft(ccf_proc,axis=0)/p.dt)
	cct_proc[maxlag:maxlag2,:]=dummy[0:maxlag,:]
	cct_proc[0:maxlag,:]=dummy[n-maxlag:n,:]

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

		fn='OUTPUT/correlations_individual/ccf_proc_'+str(rec0)+'_'+str(rec1)
		np.save(fn,ccf_proc)

	#==============================================================================
	#- Plot results if wanted.
	#==============================================================================

	if plot:

		plt.rcParams["font.family"] = "serif"
		plt.rcParams.update({'font.size': 10})

		#- Noise traces for first window.
		plt.subplot(2,1,1)
		plt.plot(traw,np.real(np.fft.ifft(u1[:,0]))/p.dt,'k')
		plt.ylabel('u1(t) [N/m^2]')
		plt.title('recordings for first time window')
		plt.subplot(2,1,2)
		plt.plot(traw,np.real(np.fft.ifft(u2[:,0]))/p.dt,'k')
		plt.ylabel('u2(t) [N/m^2]')
		plt.xlabel('t [s]')

		plt.show()

		#- Spectrum of the pressure wavefield for first window.
		plt.subplot(2,1,1)
		plt.plot(f,np.abs(np.sqrt(bandpass)*u1[:,0]),'k',linewidth=2)
		plt.plot(f,np.real(np.sqrt(bandpass)*u1[:,0]),'b',linewidth=1)
		plt.plot(f,np.imag(np.sqrt(bandpass)*u1[:,0]),'r',linewidth=1)
		plt.ylabel('u1(f) [Ns/m^2]')
		plt.title('raw and processed spectra for first window (abs=black, real=blue, imag=red)')
		
		plt.subplot(2,1,2)
		plt.plot(f,np.abs(np.sqrt(bandpass)*u1_proc[:,0]),'k',linewidth=2)
		plt.plot(f,np.real(np.sqrt(bandpass)*u1_proc[:,0]),'b',linewidth=1)
		plt.plot(f,np.imag(np.sqrt(bandpass)*u1_proc[:,0]),'r',linewidth=1)
		plt.ylabel('u1_proc(f) [?]')
		plt.xlabel('f [Hz]')

		plt.show()

		#- Raw time- and frequency-domain correlation for first window.
		plt.subplot(2,1,1)
		plt.semilogy(f,np.abs(ccf[:,0]),'k',linewidth=2)
		plt.title('raw frequency-domain correlation for first window')
		plt.ylabel('correlation [N^2 s^2 / m^4]')
		plt.xlabel('f [Hz]')
		
		plt.subplot(2,1,2)
		plt.plot(t,np.real(cct[:,0]),'k')
		plt.title('raw time-domain correlation for first window')
		plt.ylabel('correlation [N^2 s / m^4]')
		plt.xlabel('t [s]')

		plt.show()

		#- Processed time- and frequency-domain correlation for first window.
		plt.subplot(2,1,1)
		plt.semilogy(f,np.abs(ccf_proc[:,0]),'k',linewidth=2)
		plt.title('processed frequency-domain correlation for first window')
		plt.ylabel('correlation [N^2 s^2 / m^4]*unit(T)')
		plt.xlabel('f [Hz]')
		
		plt.subplot(2,1,2)
		plt.plot(t,np.real(cct_proc[:,0]))
		plt.title('processed time-domain correlation for first window')
		plt.ylabel('correlation [N^2 s / m^4]*unit(T)')
		plt.xlabel('t [s]')

		plt.show()

		#- Raw and processed ensemble correlations. 
		plt.plot(t,np.sum(cct,1)/np.max(cct),'k')
		plt.plot(t,np.sum(cct_proc,1)/np.max(cct_proc),'r')
		plt.title('ensemble time-domain correlation (black=raw, red=processed)')
		plt.ylabel('correlation [N^2 s / m^4]*unit(T)')
		plt.xlabel('t [s]')

		plt.show()

	
	#- End time.
	t2=time.time()

	if verbose:
		print('elapsed time: '+str(t2-t1)+' s')

	#==============================================================================
	#- Output.
	#==============================================================================

	return cct,cct_proc,t,ccf,ccf_proc,f

