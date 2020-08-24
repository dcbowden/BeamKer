import matplotlib.pyplot as plt
import numpy as np
import parameters


def get_propagation_corrector(rec0=0,rec1=1,plot=0):

	"""
	gf_interp=get_propagation_corrector(rec0=0,rec1=1,plot=0)

	Read propagation corrector and interpolate to the frequency axis used to compute
	ensemble correlations and kernels (correlation_function.py, correlation_field.py, kernels.py).

	INPUT:
	------
	rec0, rec1:		indeces of the receivers used in the correlation. 
	plot: 			plot original and interpolated corrector when plot==1.
	

	OUTPUT:
	-------
	gf_interp:		interpolated, frequency-domain propagation corrector.
	
	Last updated: 27 May 2016.
	"""

	#==============================================================================
	#- Input.
	#==============================================================================

	#- Read parameters. -----------------------------------------------------------

	p=parameters.Parameters()

	#- Load frequency and time axes. ----------------------------------------------

	fn='OUTPUT/correctors/f'
	fid=open(fn,'r')
	f=np.load(fid)
	fid.close()

	df=f[1]-f[0]
	f[0]=0.01*f[1]

	n=len(f)

	fn='OUTPUT/correctors/t'
	fid=open(fn,'r')
	t=np.load(fid)
	fid.close()

	dt=t[1]-t[0]

	#- Load frequency-domain propagation corrector. -------------------------------

	fn='OUTPUT/correctors/g_'+str(rec0)+'_'+str(rec1)
	fid=open(fn,'r')
	gf=np.load(fid)
	fid.close()

	#==============================================================================
	#- Interpolation.
	#==============================================================================

	f_interp=np.arange(p.fmin-p.fwidth,p.fmax+p.fwidth,p.df)

	gf_interp_real=np.interp(f_interp,f,np.real(gf))

	gf_interp_imag=np.interp(f_interp,f,np.imag(gf))


	#==============================================================================
	#- Plot.
	#==============================================================================

	if (plot==1):

		plt.plot(f,np.real(gf),'k')
		plt.plot(f_interp,gf_interp_real,'--k')
		plt.plot(f,np.imag(gf),'r')
		plt.plot(f_interp,gf_interp_imag,'--r')

		plt.title('frequency-domain propagation corrector in unit(T) [red=imaginary, black=real, dashed=interpolated, solid=original]')
		plt.xlabel('frequency [Hz]')

		plt.show()

	#==============================================================================
	#- Return.
	#==============================================================================

	gf_interp=gf_interp_real+1.0j*gf_interp_imag

	return gf_interp