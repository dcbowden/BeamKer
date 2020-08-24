# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import adsrc as adsrc
import green as g
import source as s
import parameters
import time
import get_propagation_corrector as gpc

def source_kernel(cct, t, rec0=0, rec1=1, measurement='cctime', effective=0, plot=0):
	"""
	x,y,K = source_kernel(cct, t, rec0=0, rec1=1, measurement='cctime', effective=0, plot=0):

	Compute source kernel for a frequency-independent source power-spectral density.

	INPUT:
	------

	cct, t:			Time-domain correlation function and time axis as obtained from correlation_function.py.
	rec0, rec1:		Indeces of the receivers used in the correlation. 
	measurement:	Type of measurement used to compute the adjoint source. See adsrc.py for options.
	plot:			When plot=1, plot source kernel.
	effective:		When effective==1, effective correlations are computed using the propagation correctors stored in OUTPUT/correctors.
					The source power-spectral density is then interpreted as the effective one.


	OUTPUT:
	-------
	x,y:			Space coordinates.
	K:				Source kernel [unit of measurement * m^2 / Pa^4 s^2].

	Last updated: 27 May 2016.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	p=parameters.Parameters()

	## DCB-note: Changing this to apply 1.5 stretching before arange.
	## Previously the dx, dy parameters also got stretched 1.5, which may affect
	## other lines later...
	##
	## Original:
	#x_line=np.arange(p.xmin,p.xmax,p.dx)
	#y_line=np.arange(p.ymin,p.ymax,p.dy)
	#
	#nx=len(x_line)
	#ny=len(y_line)
	#
	#x,y=np.meshgrid(1.5*x_line,1.5*y_line)
	#

	x_line=np.arange(p.xmin*1.5,p.xmax*1.5,p.dx)
	y_line=np.arange(p.ymin*1.5,p.ymax*1.5,p.dy)
	#x_line=np.arange(p.xmin*0.25,p.xmax*0.25,p.dx)
	#y_line=np.arange(p.ymin*0.25,p.ymax*0.25,p.dy)
	nx=len(x_line)
	ny=len(y_line)
	x,y=np.meshgrid(x_line,y_line)

	f=np.arange(p.fmin-p.fwidth,p.fmax+p.fwidth,p.df)
	omega=2.0*np.pi*f

	K_source=np.zeros(np.shape(x))

	#- Read propagation corrector if needed. --------------------------------------

	if (effective==1):

		gf=gpc.get_propagation_corrector(rec0,rec1,plot=0)

	else:

		gf=np.ones(len(f),dtype=complex)

	#- Compute number of grid points corresponding to the minimum wavelength. -----

	L=int(np.ceil(p.v/(p.fmax*p.dx)))

	#==============================================================================
	#- Compute the adjoint source.
	#==============================================================================

	a=adsrc.adsrc(cct, t, measurement, plot)

	#==============================================================================
	#- Compute kernel.
	#==============================================================================

	for k in range(len(omega)):

		#- Green functions.
		G1=g.green_input(x,y,p.x[rec0],p.y[rec0],omega[k],p.dx,p.dy,p.rho,p.v,p.Q)
		G2=g.green_input(x,y,p.x[rec1],p.y[rec1],omega[k],p.dx,p.dy,p.rho,p.v,p.Q)

		#- Compute kernel.
		K_source+=2.0*np.real(gf[k]*G1*np.conj(G2)*a[k])

	#==============================================================================
	#- Smooth over minimum wavelength.
	#==============================================================================

	for k in range(L):
		K_source[1:ny-2,:]=(K_source[1:ny-2,:]+K_source[0:ny-3,:]+K_source[2:ny-1,:])/3.0

	for k in range(L):
		K_source[:,1:nx-2]=(K_source[:,1:nx-2]+K_source[:,0:nx-3]+K_source[:,2:nx-1])/3.0


	#==============================================================================
	#- Visualise if wanted.
	#==============================================================================

	if plot==1:
		cmap = plt.get_cmap('RdBu')
		plt.pcolormesh(x-p.dx/2,y-p.dy/2,K_source,cmap=cmap,shading='interp')
		plt.clim(-np.max(np.abs(K_source))*0.15,np.max(np.abs(K_source))*0.15)
		plt.axis('image')
		plt.colorbar()
		plt.title('Source kernel [unit of measurement s^2 / m^2]')
		plt.xlabel('x [km]')
		plt.ylabel('y [km]')
		
		plt.plot(p.x[rec0],p.y[rec0],'ro')
		plt.plot(p.x[rec1],p.y[rec1],'ro')
		
		plt.show()

	return x,y,K_source


def structure_kernel(cct, t, rec0=0, rec1=1, measurement='cctime', dir_forward='OUTPUT/', effective=0, plot=0):
	"""
	x,y,K_kappa = structure_kernel(cct, t, rec0=0, rec1=1, measurement='cctime', dir_forward='OUTPUT/', effective=0, plot=0):

	Compute structure kernel K_kappa for a frequency-independent source power-spectral density.

	INPUT:
	------

	cct, t:			Time-domain correlation function and time axis as obtained from correlation_function.py.
	rec0, rec1:		Indeces of the receivers used in the correlation. 
	measurement:	Type of measurement used to compute the adjoint source. See adsrc.py for options.
	dir_forward:	Location of the forward interferometric fields from rec0 and rec1. Must exist.
	plot:			When plot=1, plot structure kernel.
	effective:		When effective==1, effective correlations are computed using the propagation correctors stored in OUTPUT/correctors.
					The source power-spectral density is then interpreted as the effective one.


	OUTPUT:
	-------
	x,y:			Space coordinates.
	K:				Structure kernel [unit of measurement * 1/N].

	Last updated: 11 July 2016.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	p=parameters.Parameters()

	x_line=np.arange(p.xmin,p.xmax,p.dx)
	y_line=np.arange(p.ymin,p.ymax,p.dy)

	x,y=np.meshgrid(x_line,y_line)

	f=np.arange(p.fmin-p.fwidth,p.fmax+p.fwidth,p.df)
	df=f[1]-f[0]
	omega=2.0*np.pi*f

	K_kappa=np.zeros(np.shape(x))

	C1=np.zeros((len(y_line),len(x_line),len(omega)),dtype=complex)
	C2=np.zeros((len(y_line),len(x_line),len(omega)),dtype=complex)

	nx=len(x_line)
	ny=len(y_line)

	kappa=p.rho*(p.v**2)

    #- Frequency- and space distribution of the source. ---------------------------

	S,indeces=s.space_distribution(plot=0)
	instrument,natural=s.frequency_distribution(f)
	filt=natural*instrument*instrument

	#- Compute the adjoint source. ------------------------------------------------

	a=adsrc.adsrc(cct, t, measurement, plot)

	#- Compute number of grid points corresponding to the minimum wavelength. -----

	L=int(np.ceil(p.v/(p.fmax*p.dx)))

	#- Read propagation corrector if needed. --------------------------------------

	if (effective==1):

		gf=gpc.get_propagation_corrector(rec0,rec1,plot=0)

	else:

		gf=np.ones(len(f),dtype=complex)

	#==============================================================================
	#- Load forward interferometric wavefields.
	#==============================================================================

	fn=dir_forward+'/cf_'+str(rec0)
	fid=open(fn,'r')
	C1=np.load(fid)
	fid.close()

	fn=dir_forward+'/cf_'+str(rec1)
	fid=open(fn,'r')
	C2=np.load(fid)
	fid.close()

    #==============================================================================
	#- Loop over frequencies.
	#==============================================================================

	for k in range(len(omega)):

		w=omega[k]

		#- Adjoint fields. --------------------------------------------------------
		G1=-w**2*g.green_input(x,y,p.x[rec0],p.y[rec0],w,p.dx,p.dy,p.rho,p.v,p.Q)*gf[k]
		G2=-w**2*g.green_input(x,y,p.x[rec1],p.y[rec1],w,p.dx,p.dy,p.rho,p.v,p.Q)*gf[k]

		#- Multiplication with adjoint fields. ------------------------------------
		K_kappa=K_kappa-2.0*np.real(G2*C1[:,:,k]*np.conj(a[k])+G1*C2[:,:,k]*a[k])

	K_kappa=K_kappa/kappa

	#==============================================================================
	#- Smooth over minimum wavelength.
	#==============================================================================

	for k in range(L):
		K_kappa[1:ny-2,:]=(K_kappa[1:ny-2,:]+K_kappa[0:ny-3,:]+K_kappa[2:ny-1,:])/3.0

	for k in range(L):
		K_kappa[:,1:nx-2]=(K_kappa[:,1:nx-2]+K_kappa[:,0:nx-3]+K_kappa[:,2:nx-1])/3.0

	#==============================================================================
	#- Visualise if wanted.
	#==============================================================================

	if plot==1:

		cmap = plt.get_cmap('RdBu')
		plt.pcolormesh(x-p.dx/2,y-p.dy/2,K_kappa,cmap=cmap,shading='interp')
		plt.clim(-np.max(np.abs(K_kappa))*0.25,np.max(np.abs(K_kappa))*0.25)
		plt.axis('image')
		plt.colorbar()
		plt.title('Structure (kappa) kernel [unit of measurement / m^2]')
		plt.xlabel('x [km]')
		plt.ylabel('y [km]')
		
		plt.plot(p.x[rec0],p.y[rec0],'ro')
		plt.plot(p.x[rec1],p.y[rec1],'ro')
		
		plt.show()


	return x,y,K_kappa
