# -*- coding: utf-8 -*-

import numpy as np
import parameters

#==============================================================================
#- Green function where input is read inside the function.
#==============================================================================

def green(x0,y0,x,y,omega):

	"""
	G = green(x0,y0,x,y,omega)

	Frequency-domain Green function for the 2D acoustic wave equation -w^2 rho u - kappa rho (d_i 1/rho d_i u) = f .
	Frequency-domain quantities are:

	w:		circular frequency [1/s]
	rho:	mass density [kg/m^3]
	kappa:	bulk modulus [N/m^2]
	u:		Fourier transform of acoustic pressure [Ns/m^2]
	f:		Fourier transform of external source [kg^2/(s^3 m^4)]

	INPUT:
	------
	x0, y0:	source coordinates [m]
	x,y:	space variables [m]
	omega:	circular frequency [1/s]

	OUTPUT:
	-------
	G:		Frequency-domain Green function.

	Last updated: 4 May 2016.
	"""
    
	p=parameters.Parameters()

	r=np.sqrt((x-x0)**2+(y-y0)**2)
	r_reg=np.sqrt((x-x0-0.5*p.dx)**2+(y-y0-0.5*p.dy)**2)

	A=-1j/(4.0*p.rho*p.v**2) * np.sqrt(2.0*p.v/(np.pi*omega*r_reg))
	G=A*np.exp(-1j*omega*r/p.v)*np.exp(-omega*r/(2.0*p.v*p.Q))*np.exp(1j*np.pi/4.0)

	return G

#==============================================================================
#- Green function where input is provided to the function.
#==============================================================================

def green_input(x0,y0,x,y,omega,dx,dy,rho,v,Q):

	"""
	G = green_input(x0,y0,x,y,omega,dx,dy,rho,v,Q)

	Frequency-domain Green function for the 2D acoustic wave equation -w^2 rho u - kappa rho (d_i 1/rho d_i u) = f .
	
	The difference to the function defined above is that input is given to the function directly, instead of being read by the function. This speeds up computations quite a bit.

	Frequency-domain quantities are:

	w:		circular frequency [1/s]
	rho:	mass density [kg/m^3]
	kappa:	bulk modulus [N/m^2]
	u:		Fourier transform of acoustic pressure [Ns/m^2]
	f:		Fourier transform of external source [kg^2/(s^3 m^4)]

	INPUT:
	------
	x0, y0:	source coordinates [m]
	x,y:	space variables [m]
	omega:	circular frequency [1/s]
	dx, dy:	grid spacing in x- and y-direction [km]
	rho:	density [kg/m^3]
	v:		acoustic velocity [m/s]
	Q:		quality factor

	OUTPUT:
	-------
	G:		Frequency-domain Green function.

	Last updated: 16 May 2016.
	"""

	wrv=omega*np.sqrt((x-x0-0.05*dx)**2+(y-y0-0.05*dy)**2)/v

	A=-1j/(4.0*rho*v**2) * np.sqrt(2.0/(np.pi*wrv))
	G=A*np.exp(-1j*wrv)*np.exp(-wrv/(2.0*Q))*np.exp(1j*np.pi/4.0)

	return G


def conjG1_times_G2(x0,y0,x1,y1,xs,ys,omega,dx,dy,rho,v,Q):

	"""
	P = conjG1_times_G2(x0,y0,x1,y1,xs,ys,omega,dx,dy,rho,v,Q)

	Computes the following:

	G1=g.green_input(x0,y0,xs,ys,omega,dx,dy,rho,v,Q)
	G2=g.green_input(x1,y1,xs,ys,omega,dx,dy,rho,v,Q)
	P=np.conj(G1)*G2

	INPUT:
	------

	See above for green_input.

	OUTPUT:
	-------
	P:		Frequency-domain product conj(G1)*G2, needed to compute correlation functions.

	Last updated: 16 May 2016.
	"""

	#- (regularised) distances

	r0=np.sqrt((xs-x0-0.1*dx)**2+(ys-y0-0.1*dy)**2)
	r1=np.sqrt((xs-x1-0.1*dx)**2+(ys-y1-0.1*dy)**2)

	#- amplitude factor
	A=1.0/(8.0*np.pi*omega*(rho**2)*np.sqrt(r1)*np.sqrt(r0)*(v**3))

	#- exponentials
	P=A*np.exp(-omega*( 1j*(r1-r0)/v + (r1+r0)/(2.0*v*Q)  ))

	return P