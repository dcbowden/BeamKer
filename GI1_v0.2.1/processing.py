import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import parameters
import time

def processing_single_station(u1,u2,f,verbose=False):

	"""
	u1, u2 = processing_single_station(u1,u2,f,verbose=False)

	Perform single-trace processing in the time and frequency domains.

	INPUT:
	------
	u1,u2:			Frequency-domain wavefields at two receivers.
	f:				Frequency axis.
	verbose:		Give screen output when True.

	OUTPUT:
	-------
	u1, u2:			Processed frequency-domain wavefields at two receivers.
	
	Last updated: 12 August 2019.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	#- Input parameters.
	p=parameters.Parameters()

	#- Time and frequency axes.
	n=np.shape(u1)[0]
	df=f[1]
	t=np.arange(-0.5*n*p.dt,0.5*n*p.dt,p.dt)

	#- Initialise processed recordings.
	u1t=np.zeros([n,p.Nwindows],dtype=float)
	u2t=np.zeros([n,p.Nwindows],dtype=float)
	
	#==============================================================================
	#- Frequency-domain processing.
	#==============================================================================

	#- Spectral whitening. --------------------------------------------------------

	if p.process_whiten==1:

		if verbose==1: print('spectral whitening')

		#- Regularisation term.
		reg1=np.zeros([n,p.Nwindows],dtype=complex)
		reg2=np.zeros([n,p.Nwindows],dtype=complex)

		for k in range(p.Nwindows):
			reg1[:,k]=0.01*np.max(np.abs(u1[:,k]))
			reg2[:,k]=0.01*np.max(np.abs(u2[:,k]))

		u1=u1/(np.abs(u1)+reg1)
		u2=u2/(np.abs(u2)+reg2)

	#==============================================================================
	#- Time-domain processing.
	#==============================================================================

	#- Compute time-domain recordings. --------------------------------------------


	if p.process_onebit==1 or p.process_rms_clip==1:

		u1t=np.real(np.fft.ifft(u1,axis=0))
		u2t=np.real(np.fft.ifft(u2,axis=0)) 

	#- One-bit normalisation. -----------------------------------------------------

	if p.process_onebit==1:

		if verbose==1: print('one-bit normalisation')

		u1t=np.sign(u1t)
		u2t=np.sign(u2t)

	#- Rms clipping. --------------------------------------------------------------

	if p.process_rms_clip==1:

		if verbose==1: print('rms clipping')

		for k in range(p.Nwindows):

			rms1=np.sqrt(np.sum(u1t[:,k]**2)/n)
			rms2=np.sqrt(np.sum(u2t[:,k]**2)/n) 

			u1t[:,k]=np.clip(u1t[:,k],-rms1,rms1)
			u2t[:,k]=np.clip(u2t[:,k],-rms2,rms2) 

	#- Return to frequency domain. ------------------------------------------------

	if p.process_onebit==1 or p.process_rms_clip==1:

		u1=np.fft.fft(u1t,axis=0)
		u2=np.fft.fft(u2t,axis=0)

	#==============================================================================
	#- Clean up and return.
	#==============================================================================

	#- Return.
	return u1, u2

#==================================================================================
#==================================================================================
#==================================================================================

def processing_correlation(ccf,f,verbose=False):

	"""
	ccf = processing_correlation(ccf,f,verbose=False)

	Perform correlation function processing in the time and frequency domains.

	INPUT:
	------
	ccf:			Frequency-domain correlation function.
	f:				Frequency axis.
	verbose:		Give screen output when 1.

	OUTPUT:
	-------
	ccf:			Processed frequency-domain correlation function.
	
	Last updated: 12 July 2019.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	#- Start time.
	t1=time.time()

	#- Input parameters.
	p=parameters.Parameters()

	#- Time and frequency axes.
	n=np.shape(ccf)[0]
	df=f[1]
	t=np.arange(-0.5*n*p.dt,0.5*n*p.dt,p.dt)

	#- Initialise processed recordings.
	cct=np.zeros([n,p.Nwindows],dtype=float)

	#==============================================================================
	#- Time-domain processing.
	#==============================================================================

	#- Compute time-domain correlation. -------------------------------------------

	if p.process_causal_acausal_average==1 or p.process_correlation_normalisation==1 or p.process_phase_weighted_stack==1:

		cct=np.real(np.fft.ifft(ccf,axis=0))

	#- Average causal and acausal branches. ---------------------------------------

	if p.process_causal_acausal_average==1:

		if verbose==1: print('average causal and acausal branch')

		cct[:,:]=0.5*(cct[:,:]+cct[::-1,:])

	#- Time-domain normalisation. -------------------------------------------------

	if p.process_correlation_normalisation==1:

		if verbose==1: print('normalise correlations')

		for k in range(p.Nwindows):
			cct[:,k]=cct[:,k]/np.max(np.abs(cct[:,k]))

	#- Phase-weighted stack. ------------------------------------------------------

	if p.process_phase_weighted_stack==1:

		if verbose==1: print('phase-weighted stack')

		#- Compute phase stack.
		s=signal.hilbert(cct,axis=0)
		phi=np.angle(s)
		ps=np.abs(np.sum(np.exp(1j*phi),1)/n)

		#- Apply phase stack
		for k in range(p.Nwindows):
			cct[:,k]=cct[:,k]*ps

	#- Return to frequency domain. ------------------------------------------------

	if p.process_causal_acausal_average==1 or p.process_correlation_normalisation==1 or p.process_phase_weighted_stack==1:

		ccf=np.fft.fft(cct,axis=0)

	#==============================================================================
	#- Clean up and return.
	#==============================================================================

	#- Return.
	return ccf