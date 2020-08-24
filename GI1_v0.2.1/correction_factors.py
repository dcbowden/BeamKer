import matplotlib.pyplot as plt
import numpy as np
import parameters
import time
import correlation_random as cr

def correction_factors(reg_level=100000.0,save=False,verbose=False,plot=False):

	"""
	f, gf_ik, n_ff, t, gt_ik, n_ft = correction_factors(reg_level=100000.0,save=False,verbose=False,plot=False)

	Compute and store source and propagation correctors.

	INPUT:
	------
	reg_level:		water level regularisation parameter for the computation of the transfer function
	plot:			plot when True.
	verbose:		give screen output when True.
	save:			store individual correctors to OUTPUT/correctors/ and ensemble correlations (raw and processed) for all receiver pairs to /OUTPUT/correlations.

	OUTPUT:
	-------
	gf_ik, n_ff:	frequency-domain propagation correctors and source correctors.
	gt_ik, n_ft:	time-domain propagation correctors and source correctors.
	f:				frequency axis.
	t:				time axis.
	
	Last updated: 12 August 2019.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	#- Start time.
	t1=time.time()

	#- Input parameters.
	p=parameters.Parameters()
	
	#- Compute number of samples as power of 2.
	n=int(2**np.ceil(np.log2((p.Twindow)/p.dt)))

	df=1.0/(n*p.dt)
	t=np.arange(p.tmin,p.tmax,p.dt)
	
	#- List of receiver pairs.
	pairs=[]

	for i in range(p.Nreceivers):
		for k in range(i,p.Nreceivers):

			pairs.append((i,k))

	#- Initialise propagation and source correction factors.
	gf_ik=np.zeros([n,len(pairs)],dtype=complex)
	gt_ik=np.zeros([len(t),len(pairs)],dtype=float)
	n_ff=np.zeros([n,p.Nwindows],dtype=float)
	n_ft=np.zeros([len(t),p.Nwindows],dtype=float)
	n_f_norm=np.zeros(n,dtype=float)

	#==============================================================================
	#- Propagation and source correction factors.
	#==============================================================================

	#- March through all receiver pair combinations. ------------------------------
	#- Compute propagation correctors and accumulate source correctors. -----------

	if verbose: print('compute propagation and source correction factors -------------')

	for k in range(len(pairs)):
		
		#- Print receiver pair. ---------------------------------------------------
		if verbose: print('receiver pair: '+str(pairs[k][0]),str(pairs[k][1]))

		#- Compute correlation functions for this receiver pair. ------------------
		cct,cct_proc,t,ccf,ccf_proc,f=cr.correlation_random(rec0=pairs[k][0],rec1=pairs[k][1],verbose=False,plot=False)
		t_intermediate=time.time()

		#- Store. -----------------------------------------------------------------

		if save:

			#- Store frequency and time axes. -------------------------------------

			fn='OUTPUT/correlations/f'
			np.save(fn,f)

			fn='OUTPUT/correlations/t'
			np.save(fn,t)

			#- Store raw and processed ensemble correlations in the frequency domain.

			fn='OUTPUT/correlations/ccf_'+str(pairs[k][0])+'_'+str(pairs[k][1])
			np.save(fn,np.sum(ccf,axis=1)/p.Nwindows)

			fn='OUTPUT/correlations/ccf_proc_'+str(pairs[k][0])+'_'+str(pairs[k][1])
			np.save(fn,np.sum(ccf_proc,axis=1)/p.Nwindows)

		#- Print time. ------------------------------------------------------------
		if verbose: print('time: '+str(t_intermediate-t1)+' s')

		#- Compute regularised transfer function for this receiver pair. ----------
		n_T_ik=np.zeros(np.shape(ccf),dtype=complex)

		for win in range(p.Nwindows):
			reg=np.max(ccf[:,win]*np.conj(ccf[:,win]))/reg_level
			n_T_ik[:,win]=ccf_proc[:,win]*np.conj(ccf[:,win])/(ccf[:,win]*np.conj(ccf[:,win])+reg)

		#- Compute propagation correction factor from weighted sum. ---------------
		gf_ik[:,k]=np.sum(n_T_ik,1)/p.Nwindows

		#- Accumulate source correction factors. ----------------------------------
		for win in range(p.Nwindows):
			n_ff[:,win]+=np.real(n_T_ik[:,win]*np.conj(gf_ik[:,k]))

		n_f_norm+=np.abs(gf_ik[:,k])**2

	#- Compute final source correctors by normalisation. --------------------------

	for win in range(p.Nwindows):
		n_ff[:,win]=n_ff[:,win]/(n_f_norm+np.max(n_f_norm)/reg_level)

	#==============================================================================
	#- Compute time-domain propagation and source correctors. 
	#==============================================================================

	maxlag2=int(np.size(t)) 
	maxlag=int(np.floor(maxlag2/2))

	dummy=np.real(np.fft.ifft(gf_ik,axis=0)/p.dt)
	gt_ik[maxlag:maxlag2,:]=dummy[0:maxlag,:]
	gt_ik[0:maxlag,:]=dummy[n-maxlag:n,:]

	dummy=np.real(np.fft.ifft(n_ff,axis=0)/p.dt)
	n_ft[maxlag:maxlag2,:]=dummy[0:maxlag,:]
	n_ft[0:maxlag,:]=dummy[n-maxlag:n,:]

	#==============================================================================
	#- Store.
	#==============================================================================

	if save:

		#- Store frequency and time axes. -----------------------------------------

		fn='OUTPUT/correctors/f'
		np.save(fn,f)

		fn='OUTPUT/correctors/t'
		np.save(fn,t)

		#- Store frequency-domain propagation correctors. -------------------------

		for k in range(len(pairs)):

			fn='OUTPUT/correctors/g_'+str(pairs[k][0])+'_'+str(pairs[k][1])
			np.save(fn,gf_ik[:,k])

		#- Store frequency-domain source correctors. ------------------------------

		for k in range(p.Nwindows):

			fn='OUTPUT/correctors/'+str(k)+'_f'
			np.save(fn,n_ff[:,k])

	#==============================================================================
	#- Plot.
	#==============================================================================

	if plot:

		plt.rcParams["font.family"] = "serif"
		plt.rcParams.update({'font.size': 10})

		for k in range(len(pairs)): plt.plot(t,gt_ik[:,k],color=np.random.rand(3))
		plt.show()

		for k in range(p.Nwindows): plt.plot(t,n_ft[:,k],color=np.random.rand(3))
		plt.show()

	#==============================================================================
	#- Clean up and output.
	#==============================================================================

	#- End time.
	t2=time.time()

	if verbose==1:
		print('elapsed time: '+str(t2-t1)+' s')

	return f, gf_ik, n_ff, t, gt_ik, n_ft
