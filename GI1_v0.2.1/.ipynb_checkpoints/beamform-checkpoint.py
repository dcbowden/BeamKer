import matplotlib.pyplot as plt
import numpy as np
import green as g
import source as s
import processing as proc
import parameters
import time

# This package is newly added, DCB August 2019.
# Currently only function is specshift(), but we could hardcode other beamform functions here.

def specshift(signal,shift):
	"""
	shifted_signal = specshift(signal,shift)

	shift a time-domain by some number of samples, 
	can handle sub-sample shifts.
	
	Modified from matlab scripts from Jean-Paul Ampuero, ampuero@erdw.ethz.ch

	INPUT:
	------
	signal:			data [nsamples, nstations]
	shift:			shifts [nstations], must be number of samples, not seconds        

	"""
	# Assumes signal is size (nsamp, nstations)
	# Assumes shift is size (nstations)
	nshift = np.size(shift)
	#[nsamp,nsta] = np.shape(np.atleast_2d(signal))
	if( len(signal.shape) > 1):
	    nsamp = signal.shape[0]
	    nsta  = signal.shape[1]
	else:
	    nsamp = signal.shape[0]
	    nsta = 1
	    signal=signal[:,np.newaxis]


	if(nshift!=nsta):
	    print("error! incorrect sizes")
	    return -1

	# NFFT, as next power of two, with padding for wraparound
	n=int(2.0**np.ceil(np.log2(nsamp+np.max(nshift))))
	n2=int(n/2)
	spect=np.fft.fft(signal,n,axis=0)
	ik = np.zeros([n,1],dtype='complex')
	ik[:n2+1,0] = 2*1j*np.pi*np.arange(0,n2+1)/n
	ik[n2+1:,0]= -ik[ np.arange(n2-1,0,-1),0]
	fshift=np.exp(ik*shift)
	fshift[n2+1] = np.real(fshift[n2+1]) #real nyquist freq.
	spect = spect*fshift
	signalout = np.fft.irfft(spect,n,axis=0)[0:nsamp,:]
	return signalout


#def beamform_simple(u,f):
	## UNFINISHED!
	# Make more beamforming here?

