# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import green as g
import source as s
import parameters
import get_propagation_corrector as gpc

def correlation_function(rec0=0,rec1=1,effective=False,plot=False,save=False):

	"""
	cct, t, ccf, f = correlation_function(rec0=0,rec1=1,effective=False,plot=False,save=False)

	Compute time- and frequency-domain correlation functions. 


	INPUT:
	------
	rec0, rec1:		indeces of the receivers used in the correlation. 
	plot:			When plot=True, the source distribution, and the time- and frequency domain correlation functions are plotted.
	save:			Save pdf figures to OUTPUT when True.
	effective:		When effective==True, effective correlations are computed using the propagation correctors stored in OUTPUT/correctors.
					The source power-spectral density is then interpreted as the effective one.

	OUTPUT:
	-------
	cct, t:		Time-domain correlation function and time axis [N^2 s / m^4],[s].
	ccf, f:		Frequency-domain correlation function and frequency axis [N^2 s^2 / m^4],[1/s].

	Last updated: 16 July 2019.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	p=parameters.Parameters()

	x_line=np.arange(p.xmin,p.xmax,p.dx)
	y_line=np.arange(p.ymin,p.ymax,p.dy)

	x,y=np.meshgrid(x_line,y_line)

	f=np.arange(p.fmin-p.fwidth,p.fmax+p.fwidth,p.df)
	omega=2.0*np.pi*f

	t=np.arange(p.tmin,p.tmax,p.dt)

	#- Frequency- and space distribution of the source. ---------------------------

	S,indices=s.space_distribution(plot=plot,save=save)
	instrument,natural=s.frequency_distribution(f)
	filt=natural*instrument*instrument

	#- Read propagation corrector if needed. --------------------------------------

	if effective:

		gf=gpc.get_propagation_corrector(rec0,rec1,plot=False)

	else:

		gf=np.ones(len(f),dtype=complex)

	#==============================================================================
	#- Compute inter-station correlation function.
	#==============================================================================

	cct=np.zeros(np.shape(t),dtype=float)
	ccf=np.zeros(np.shape(f),dtype=complex)

	for idf in range(len(omega)):

		P=g.conjG1_times_G2(p.x[rec0],p.y[rec0],p.x[rec1],p.y[rec1],x,y,omega[idf],p.dx,p.dy,p.rho,p.v,p.Q)
		ccf[idf]=gf[idf]*np.conj(np.sum(P*S))

		cct=cct+np.real(filt[idf]*ccf[idf]*np.exp(1j*omega[idf]*t))

	cct=cct*p.dx*p.dy*p.df

	#==============================================================================
	#- Plot result.
	#==============================================================================

	if (plot or save):

		plt.rcParams["font.family"] = "serif"
		plt.rcParams.update({'font.size': 10})

		#- Frequency domain.
		plt.semilogy(f,np.abs(ccf),'k')
		plt.semilogy(f,np.real(ccf),'b')
		plt.title('frequency-domain correlation function (black=abs, blue=real)')
		plt.xlabel('frequency [Hz]')
		plt.ylabel(r'correlation [N$^2$ s$^2$/m$^4$]')
		
		if plot: 
			plt.show()
		else:
			fn='OUTPUT/correlations_computed/c_frequency_domain_'+str(rec0)+'-'+str(rec1)+'.pdf'
			plt.savefig(fn,format='pdf')
			plt.clf()

		#- Time domain.

		tt=np.sqrt((p.x[rec0]-p.x[rec1])**2+(p.y[rec0]-p.y[rec1])**2)/p.v
		cct_max=np.max(np.abs(cct))

		plt.plot(t,cct,'k',linewidth=2.0)
		plt.plot([tt,tt],[-1.1*cct_max,1.1*cct_max],'--',color=(0.5,0.5,0.5),linewidth=1.5)
		plt.plot([-tt,-tt],[-1.1*cct_max,1.1*cct_max],'--',color=(0.5,0.5,0.5),linewidth=1.5)

		plt.ylim((-1.1*cct_max,1.1*cct_max))
		plt.title('correlation function')
		plt.xlabel('time [s]')
		plt.ylabel(r'correlation [N$^2$ s/m$^4$]')
		
		if plot: 
			plt.show()
		else:
			fn='OUTPUT/correlations_computed/c_time_domain_'+str(rec0)+'-'+str(rec1)+'.pdf'
			plt.savefig(fn,format='pdf')
			plt.clf()

	#==============================================================================
	#- Save results if wanted.
	#==============================================================================

	if save==1:

		#- Store frequency and time axes.
		fn='OUTPUT/correlations_computed/t'
		np.save(fn,t)

		fn='OUTPUT/correlations_computed/t'
		np.save(fn,t)

		#- Store computed correlations in the time and frequency domain.
		fn='OUTPUT/correlations_computed/cct_'+str(rec0)+'-'+str(rec1)
		np.save(fn,cct)

		fn='OUTPUT/correlations_computed/ccf_'+str(rec0)+'-'+str(rec1)
		np.save(fn,ccf)

	#==============================================================================
	#- Return.
	#==============================================================================

	return cct, t, ccf, f
