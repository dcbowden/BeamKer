import matplotlib.pyplot as plt
import numpy as np
import parameters

def correlations(rec0=0,rec1=1,effective=True):
	"""
	correlations(idx1=0,idx2=1,effective=True)

	Plot unprocessed, processed and effective correlation functions.

	INPUT:
	------
	rec0, rec1:		indeces of the receivers used in the correlation. 
	effective:		plot effective correlation. Must already be stored in OUTPUT/correlations	

	OUTPUT:
	-------
	none
	
	Last updated: 24 March 2016.
	"""

	#==============================================================================
	#- Input.
	#==============================================================================

	#- Load frequency and time axes. ----------------------------------------------

	fn='OUTPUT/correlations/f.npy'
	f=np.load(fn)

	f[0]=0.1*f[1]

	df=f[1]-f[0]
	n=len(f)

	fn='OUTPUT/correlations/t.npy'
	t=np.load(fn)

	dt=t[1]-t[0]

	#- Load frequency-domain correlation functions. -------------------------------

	fn='OUTPUT/correlations/ccf_'+str(rec0)+'_'+str(rec1)+'.npy'
	ccf=np.load(fn)

	fn='OUTPUT/correlations/ccf_proc_'+str(rec0)+'_'+str(rec1)+'.npy'
	ccf_proc=np.load(fn)

	if effective==True:
		fn='OUTPUT/correlations/ccf_eff_'+str(rec0)+'_'+str(rec1)+'.npy'
		ccf_eff=np.load(fn)

	#- Read receiver positions and compute frequency-domain Green function. -------

	p=parameters.Parameters()

	d=np.sqrt((p.x[rec0]-p.x[rec1])**2 + (p.y[rec0]-p.y[rec1])**2)

	#==============================================================================
	#- Time-domain correlations.
	#==============================================================================

	maxlag2=int(np.size(t)) 
	maxlag=int(np.floor(maxlag2/2))

	dummy=np.real(np.fft.ifft(ccf)/p.dt)
	cct=np.zeros(maxlag2,dtype=float)
	cct[maxlag:maxlag2]=dummy[0:maxlag]
	cct[0:maxlag]=dummy[n-maxlag:n]

	dummy=np.real(np.fft.ifft(ccf_proc)/p.dt)
	cct_proc=np.zeros(maxlag2,dtype=float)
	cct_proc[maxlag:maxlag2]=dummy[0:maxlag]
	cct_proc[0:maxlag]=dummy[n-maxlag:n]

	if effective==True:
		dummy=np.real(np.fft.ifft(ccf_eff)/dt)
		cct_eff=np.zeros(maxlag2,dtype=float)
		cct_eff[maxlag:maxlag2]=dummy[0:maxlag]
		cct_eff[0:maxlag]=dummy[n-maxlag:n]

	#==============================================================================
	#- Plot results.
	#==============================================================================

	scale_proc=np.max((np.max(np.abs(cct_proc)), np.max(np.abs(cct_eff))))
	scale=np.max(np.abs(cct))

	if (1==1):

		plt.plot(t,cct_proc/scale_proc,'r',linewidth=1.5)
		if effective==True: plt.plot(t,cct_eff/scale_proc,'b',linewidth=1.5)
		plt.plot(t,cct/scale,'--k',linewidth=1.5)
	
		plt.text(0.5*np.max(t),0.7,'x %0.2g' % scale,color=(0.1,0.1,0.1),fontsize=14)
		plt.text(0.5*np.max(t),0.6,'x %0.2g' % scale_proc,color=(0.9,0.1,0.1),fontsize=14)
		plt.text(0.5*np.max(t),0.5,'x %0.2g' % scale_proc,color=(0.1,0.1,0.9),fontsize=14)

		plt.xlim((0.7*np.min(t),0.7*np.max(t)))
		plt.ylim((-1.1,1.1))

		plt.title('Correlation functions (black=original, red=processed, blue=effective)')
		plt.xlabel('time [s]')
		plt.ylabel('Green functions [s/kg], [s/k]*unit(T)')
		plt.show()

	if (1==0):

		if effective==True: 
			plt.plot(t,(cct_proc-cct_eff)/scale_proc,'--',color=(0.8,0.8,0.8),linewidth=1.5)
			plt.plot(t,cct_eff/scale_proc,'r',linewidth=1.5)
			
		plt.plot(t,cct_proc/scale_proc,'k',linewidth=2.0)
	
		plt.text(0.25*np.max(t),-0.9,'x %0.2g' % scale_proc,color=(0.1,0.1,0.1),fontsize=14)

		plt.xlim((0.35*np.min(t),0.35*np.max(t)))
		plt.ylim((-1.1,1.1))

		plt.title('Correlation functions (black=processed, red=effective, dashed=error)')
		plt.xlabel('time [s]')
		plt.ylabel('Green functions [s/kg], [s/k]*unit(T)')
		plt.show()
