import matplotlib.pyplot as plt
import numpy as np
import green as g
import parameters
import source as s

def propagation(rec0=0,rec1=1,average=True):

	"""
	propagation(idx1=0,idx2=1,average=True)

	Plot time- and frequency-domain propagation corrector, and time- and frequency-domain effective
	Green function. Requires propagation corrector files located in OUTPUT/correctors.

	INPUT:
	------
	rec0, rec1:		indeces of the receivers used in the correlation. 
	average: 		plot average over all receiver pairs (True or False)
	

	OUTPUT:
	-------
	none
	
	Last updated: 12 August 2019.
	"""

	#==============================================================================
	#- Input.
	#==============================================================================

	#- Load frequency and time axes. ----------------------------------------------

	fn='OUTPUT/correctors/f.npy'
	f=np.load(fn)

	df=f[1]-f[0]
	f[0]=0.01*f[1]

	n=len(f)

	fn='OUTPUT/correctors/t.npy'
	t=np.load(fn)

	dt=t[1]-t[0]

	#- Read receiver positions and compute frequency-domain Green function. -------

	p=parameters.Parameters()
	G=g.green(p.x[rec0],p.y[rec0],p.x[rec1],p.y[rec1],2.0*np.pi*f)

	d=np.sqrt((p.x[rec0]-p.x[rec1])**2 + (p.y[rec0]-p.y[rec1])**2)

	#- Load frequency-domain propagation corrector. -------------------------------

	fn='OUTPUT/correctors/g_'+str(rec0)+'_'+str(rec1)+'.npy'
	gf=np.load(fn)

	#- Load all frequency-domain propagation correctors to compute average. -------

	if average:

		gf_all=np.zeros(np.shape(gf),dtype=complex)

		for i in range(p.Nreceivers):
			for k in range(i,p.Nreceivers):

				fn='OUTPUT/correctors/g_'+str(i)+'_'+str(k)+'.npy'
				dummy=np.load(fn)

				gf_all+=dummy

		gf_all=gf_all/float((p.Nreceivers*(p.Nreceivers-1)))


	#- Bandpass, instrument and natural spectra. ----------------------------------

	bandpass=np.zeros(np.shape(f))

	Nminmax=int(np.round((p.bp_fmin)/df))
	Nminmin=int(np.round((p.bp_fmin-p.bp_width)/df))
	Nmaxmin=int(np.round((p.bp_fmax)/df))
	Nmaxmax=int(np.round((p.bp_fmax+p.bp_width)/df))

	bandpass[Nminmin:Nminmax]=np.linspace(0.0,1.0,Nminmax-Nminmin)
	bandpass[Nmaxmin:Nmaxmax]=np.linspace(1.0,0.0,Nmaxmax-Nmaxmin)
	bandpass[Nminmax:Nmaxmin]=1.0

	instrument,natural=s.frequency_distribution(f)

	#==============================================================================
	#- Time-domain corrector.
	#==============================================================================

	n2=int(n/2)
	maxlag2=int(np.size(t)) 
	maxlag=int(np.floor(maxlag2/2))

	dummy=np.real(np.fft.ifft(gf)/dt)
	gt=np.zeros(maxlag2)
	gt[maxlag:maxlag2]=dummy[0:maxlag]
	gt[0:maxlag]=dummy[n-maxlag:n]

	if average:

		dummy=np.real(np.fft.ifft(gf_all)/dt)
		gt_all=np.zeros(maxlag2)
		gt_all[maxlag:maxlag2]=dummy[0:maxlag]
		gt_all[0:maxlag]=dummy[n-maxlag:n]

	#==============================================================================
	#- Time-domain Green functions.
	#==============================================================================

	dummy=np.real(np.fft.ifft(instrument*np.sqrt(natural)*G)/dt)
	Gt=np.zeros(maxlag2)
	Gt[maxlag:maxlag2]=dummy[0:maxlag]
	Gt[0:maxlag]=dummy[n-maxlag:n]

	dummy=np.real(np.fft.ifft(instrument*np.sqrt(natural)*G*gf)/dt)
	Gt_corr=np.zeros(maxlag2)
	Gt_corr[maxlag:maxlag2]=dummy[0:maxlag]
	Gt_corr[0:maxlag]=dummy[n-maxlag:n]

	#==============================================================================
	#- Plot results.
	#==============================================================================

	plt.rcParams["font.family"] = "serif"
	plt.rcParams.update({'font.size': 10})

	#- Plot time-domain corrector. ------------------------------------------------

	plt.subplot(2,1,1)
	if average==True: plt.plot(t,gt_all,'--',color=(0.7,0.7,0.7),linewidth=2)
	plt.plot(t,gt,'k',linewidth=2)
	plt.xlim((0.25*np.min(t),0.25*np.max(t)))

	plt.title('time-domain propagation corrector in [1/s]*unit(T)')
	plt.xlabel('time [s]')

	#- Plot frequency-domain corrector. ------------------------------------------

	plt.subplot(2,1,2)
	if average==True: plt.plot(f,gf_all,'--',color=(0.7,0.7,0.7),linewidth=2)
	plt.plot(f,np.abs(gf),'k',linewidth=2)
	plt.plot(f,np.imag(gf),'r',linewidth=1)
	plt.xlim((0.0,0.2*np.max(f)))

	plt.title('frequency-domain propagation corrector in unit(T)')
	plt.xlabel('frequency [Hz]')

	plt.show()

	#- Plot time-domain Green functions. ------------------------------------------

	scale_eff=np.max(np.abs(Gt_corr))
	scale=np.max(np.abs(Gt))

	plt.subplot(2,1,1)
	plt.plot(t,Gt_corr/scale_eff,'r',linewidth=2)
	plt.plot(t,Gt/scale,'k',linewidth=2)
	plt.plot([d/p.v,d/p.v],[-1.1,1.1],'k--')
	
	plt.text(d/p.v+200.0,0.7,'x %0.2g' % scale,color=(0.7,0.7,0.7),fontsize=14)
	plt.text(d/p.v+200.0,0.5,'x %0.2g' % scale_eff,color=(0.9,0.2,0.2),fontsize=14)

	plt.xlim((d/p.v-700.0,d/p.v+700.0))
	plt.ylim((-1.1,1.1))

	plt.title('scaled time-domain Green functions (black=original, red=effective)')
	plt.xlabel('time [s]')
	plt.ylabel('Green function [s/kg]*unit(T)')

	#- Plot frequency-domain Green functions. -------------------------------------

	scale=np.max(np.abs(instrument*np.sqrt(natural)*G))
	scale_eff=np.max(np.abs(instrument*np.sqrt(natural)*G*gf))

	plt.subplot(2,1,2)
	plt.plot(f,np.real(instrument*np.sqrt(natural)*G*gf)/scale_eff,'r',linewidth=2)
	plt.plot(f,np.abs(instrument*np.sqrt(natural)*G*gf)/scale_eff,'r--',color=(0.9,0.2,0.2),linewidth=2)
	plt.plot(f,np.real(instrument*np.sqrt(natural)*G)/scale,'k',linewidth=2)
	plt.plot(f,np.abs(instrument*np.sqrt(natural)*G)/scale,'--',color=(0.3,0.3,0.3),linewidth=2)
	
	plt.text(0.15*np.max(f),0.7,'x %0.2g' % scale,color=(0.7,0.7,0.7),fontsize=14)
	plt.text(0.15*np.max(f),0.5,'x %0.2g' % scale_eff,color=(0.9,0.2,0.2),fontsize=14)
	
	plt.xlim((0.0,0.2*np.max(f)))
	plt.ylim((-1.1,1.1))

	plt.title('scaled frequency-domain Green functions, real part and absolute value (black=original, red=effective)')
	plt.ylabel('Green function [s^2/kg]*unit(T)')
	plt.xlabel('frequency [Hz]')

	plt.show()

	return f


def source(n_win=0,average=True):

	"""
	source(n=0,average=True)

	Plot time- and frequency-domain source corrector. 
	Requires propagation corrector files located in OUTPUT/correctors.

	INPUT:
	------
	n_win:				index of the time window. 
	average: 		plot average over all time windows (True or False).
	

	OUTPUT:
	-------
	none
	
	Last updated: 12 August 2019.
	"""

	#==============================================================================
	#- Input.
	#==============================================================================

	p=parameters.Parameters()

	plt.rcParams["font.family"] = "serif"
	plt.rcParams.update({'font.size': 10})

	#- Load frequency and time axes. ----------------------------------------------

	fn='OUTPUT/correctors/f.npy'
	f=np.load(fn)

	f[0]=0.1*f[1]

	df=f[1]-f[0]
	n=len(f)

	fn='OUTPUT/correctors/t.npy'
	t=np.load(fn)

	dt=t[1]-t[0]

	#- Load frequency-domain source corrector. ------------------------------------

	fn='OUTPUT/correctors/'+str(n_win)+'_f.npy'
	nff=np.load(fn)

	if average==True:

		nff_average=np.zeros(np.shape(nff))

		for k in range(p.Nwindows):
			fn='OUTPUT/correctors/'+str(k)+'_f.npy'
			dummy=np.load(fn)
			nff_average+=dummy

			#- Plot frequency-domain correctors. ----------------------------------

			plt.semilogy(f,np.abs(dummy),'k',linewidth=2)
			plt.xlim((0.0,0.3*np.max(f)))

		plt.title('frequency-domain source correctors [1]')
		plt.ylabel('frequency-domain source correctors [1]')
		plt.xlabel('frequency [Hz]')
		plt.ylim((1e-7,1e2))

		plt.show()

		nff_average=nff_average/p.Nwindows

	#==============================================================================
	#- Time-domain corrector.
	#==============================================================================

	n2=int(n/2)
	maxlag2=int(np.size(t)) 
	maxlag=int(np.floor(maxlag2/2))

	dummy=np.real(np.fft.ifft(nff)/dt)
	nft=np.zeros(maxlag2)
	nft[maxlag:maxlag2]=dummy[0:maxlag]
	nft[0:maxlag]=dummy[n-maxlag:n]

	if average:

		dummy=np.real(np.fft.ifft(nff_average)/dt)
		nft_average=np.zeros(maxlag2)
		nft_average[maxlag:maxlag2]=dummy[0:maxlag]
		nft_average[0:maxlag]=dummy[n-maxlag:n]

	#==============================================================================
	#- Plot results.
	#==============================================================================

	#- Plot time-domain corrector. ------------------------------------------------

	plt.subplot(2,1,1)
	if average==True: plt.plot(t,nft_average,'--',color=(0.7,0.7,0.7),linewidth=2)
	plt.plot(t,nft,'k',linewidth=2)
	plt.xlim((0.25*np.min(t),0.25*np.max(t)))

	plt.title('time-domain source corrector [1/s]')
	plt.ylabel('time-domain source corrector [1/s]')
	plt.xlabel('time [s]')

	#- Plot frequency-domain corrector. ------------------------------------------

	plt.subplot(2,1,2)
	plt.plot(f,np.real(nff),'k',linewidth=2)
	if average==True: plt.plot(f,np.real(nff_average),'--',color=(0.7,0.7,0.7),linewidth=2)
	plt.xlim((0.0,0.3*np.max(f)))

	plt.title('frequency-domain source corrector [1]')
	plt.ylabel('frequency-domain source corrector [1]')
	plt.xlabel('frequency [Hz]')

	plt.show()


def geometric_spreading(freq=0.015):
	"""
	geometric_spreading(freq=0.015)

	Plot effective geometric spreading for a specific frequency.

	INPUT:
	------
	freq:				frequency [Hz].
	
	OUTPUT:
	-------
	none
	
	Last updated: 12 August 2019.
	"""


	#==============================================================================
	#- Input and initialisation.
	#==============================================================================

	#- Load frequency axis. -------------------------------------------------------

	fn='OUTPUT/correctors/f.npy'
	f=np.load(fn)

	df=f[1]-f[0]
	f[0]=0.01*f[1]

	n=len(f)

	p=parameters.Parameters()

	idx=int(np.round((freq-np.min(f))/df))

	#- Bandpass, instrument and natural spectra. ----------------------------------

	bandpass=np.zeros(np.shape(f))

	Nminmax=int(np.round((p.bp_fmin)/df))
	Nminmin=int(np.round((p.bp_fmin-p.bp_width)/df))
	Nmaxmin=int(np.round((p.bp_fmax)/df))
	Nmaxmax=int(np.round((p.bp_fmax+p.bp_width)/df))

	bandpass[Nminmin:Nminmax]=np.linspace(0.0,1.0,Nminmax-Nminmin)
	bandpass[Nmaxmin:Nmaxmax]=np.linspace(1.0,0.0,Nmaxmax-Nmaxmin)
	bandpass[Nminmax:Nmaxmin]=1.0

	instrument,natural=s.frequency_distribution(f)

	#==============================================================================
	#- March through all receiver pairs.
	#==============================================================================

	npairs=int(p.Nreceivers*(p.Nreceivers+1)/2)
	d=np.zeros(npairs)
	a=np.zeros(npairs)
	a_eff=np.zeros(npairs)

	count=0
	average=0.0
	count_average=0.0

	for i in range(p.Nreceivers):
		for k in range(i,p.Nreceivers):

			#- Read propagation corrector. ----------------------------------------

			fn='OUTPUT/correctors/g_'+str(i)+'_'+str(k)+'.npy'
			g_ik=np.load(fn)

			#- Compute effective Green function. ----------------------------------

			G=g.green(p.x[i],p.y[i],p.x[k],p.y[k],2.0*np.pi*f)
			G_eff=G*g_ik

			d[count]=np.sqrt((p.x[i]-p.x[k])**2 + (p.y[i]-p.y[k])**2)
			a_eff[count]=np.abs(G_eff[idx])
			a[count]=np.abs(G[idx])

			#- Compute scaling factor excluding auto-correlations. ----------------

			if d[count]>0.0:
				average+=a_eff[count]/a[count]
				count_average+=1.0

			count+=1

	scale=average/count_average

	#==============================================================================
	#- Plot.
	#==============================================================================

	plt.rcParams["font.family"] = "serif"
	plt.rcParams.update({'font.size': 10})

	plt.semilogy(d,a_eff,'ro')
	plt.semilogy(d,a*scale,'ko')
	plt.xlim((-0.1*np.max(d),1.1*np.max(d)))
	plt.ylabel('Green function unit(T) [s/kg]')
	plt.xlabel('inter-station distance')
	plt.title('amplitude vs. distance (effective=red, original scaled=black')
	plt.show()






