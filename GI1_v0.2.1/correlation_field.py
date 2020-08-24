import numpy as np
import matplotlib.pyplot as plt
import source as s
import green as g
import parameters
import time

def snapshot(rec=0,t=1.0, minvalplot=0.0, maxvalplot=0.0, plot=False, save=False, verbose=False, dir_precomputed='OUTPUT/'):

	"""
	
	snapshot(rec=0,t=1.0, minvalplot=0.0, maxvalplot=0.0, plot=False, save=False, verbose=False, dir_precomputed='OUTPUT/')

	Compute and plot correlation wavefield for a specific time t. This requires that the frequency-domain wavefield has been precomputed using the precompute function below.

	INPUT:
	------

	rec:				index of receiver.
	t: 					time in s.
	minvalplot:			minimum of colour scale, ignored when 0.
	maxvalplot: 		maximum of colour scale, ignored when 0.
	plot:				plot when True.
	save:				save as pdf when True.
	verbose:			give screen output when True.
	dir_precomputed:	directory where precomputed correlation field is located.

	OUTPUT:
	-------

	C:		2D time-domain correlation wavefield [N^2 s / m^4].
	x,y:	2D axes [m].

	Last updated: 18 July 2019.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	p=parameters.Parameters()

	#- Spatial grid.
	x_line=np.arange(p.xmin,p.xmax,p.dx)
	y_line=np.arange(p.ymin,p.ymax,p.dy)
	x,y=np.meshgrid(x_line,y_line)

	#- Frequency line.
	f=np.arange(p.fmin-p.fwidth,p.fmax+p.fwidth,p.df)
	omega=2.0*np.pi*f

	#- Power-spectral density.
	S,indeces=s.space_distribution()
	instrument,natural=s.frequency_distribution(f)
	filt=natural*instrument*instrument

	C=np.zeros(np.shape(x))

	#==============================================================================
	#- Load forward interferometric wavefields in the frequency domain.
	#==============================================================================

	fn=dir_precomputed+'cf_'+str(rec)+'.npy'
	Cfull=np.load(fn)

	#==============================================================================
	#- Compute correlation field for a specific time.
	#==============================================================================

	for idx in range(len(x_line)):
		for idy in range(len(y_line)):
			C[idy,idx]=np.real(np.sum(Cfull[idy,idx,:]*np.exp(1j*omega*t)))*p.df

	#==============================================================================
	#- Plot.
	#==============================================================================

	if (plot or save):

		plt.rcParams["font.family"] = "serif"
		plt.rcParams.update({'font.size': 10})

		if (minvalplot==0.0 and maxvalplot==0.0):
			maxvalplot=0.8*np.max(np.abs(C))
			minvalplot=-maxvalplot

		#- Plot interferometric wavefield. ----------------------------------------

		#plt.pcolor(x/1000.0,y/1000.0,C,cmap='RdBu',vmin=minvalplot,vmax=maxvalplot)
		plt.pcolor(x/1000.0,y/1000.0,np.abs(C),cmap='Greys',vmin=0.0,vmax=maxvalplot)

		#- Plot receiver positions. -----------------------------------------------

		for k in range(p.Nreceivers):

			plt.plot(p.x[k]/1000.0,p.y[k]/1000.0,'ko',markersize='5')
			plt.text(p.x[k]/1000.0+4.0*p.dx/1000.0,p.y[k]/1000.0+4.0*p.dx/1000.0,str(k))

		plt.plot(p.x[rec]/1000.0,p.y[rec]/1000.0,'kx')
		plt.text(p.x[rec]/1000.0+4.0*p.dx/1000.0,p.y[rec]/1000.0+4.0*p.dx/1000.0,str(rec))

		#- Embellish the plot. ----------------------------------------------------

		plt.colorbar()
		plt.axis('image')
		plt.title('correlation field, t='+str(t)+' s')
		plt.xlim((p.xmin/1000.0,p.xmax/1000.0))
		plt.ylim((p.ymin/1000.0,p.ymax/1000.0))
		plt.xlabel('x [km]')
		plt.ylabel('y [km]')

		if plot:
			plt.show()
		if save:
			fn='OUTPUT/'+str(t)+'.pdf'
			plt.savefig(fn,format='pdf')
			plt.clf()

	#==============================================================================
	#- Return.
	#==============================================================================
	
	return C, x, y



def precompute(rec=0,verbose=False,mode='individual'):

	"""
	precompute(rec=0,verbose=False,mode='individual')

	Compute correlation wavefield in the frequency domain and store for in /OUTPUT for re-use in snapshot kernel computation.

	INPUT:
	------

	rec:		index of reference receiver.
	verbose:	give screen output when True.
	mode:		'individual' sums over individual sources. This is very efficient when there are only a few sources. This mode requires that the indeces array returned by source.space_distribution is not empty.
				'random' performs a randomised, down-sampled integration over a quasi-continuous distribution of sources. This is more efficient for widely distributed and rather smooth sources.
				'combined' is the sum of 'individual' and 'random'. This is efficient when a few point sources are super-imposed on a quasi-continuous distribution.

	OUTPUT:
	-------

	Frequency-domain interferometric wavefield stored in /OUTPUT.

	Last updated: 18 July 2019.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	p=parameters.Parameters()

	#- Spatial grid.
	x_line=np.arange(p.xmin,p.xmax,p.dx)
	y_line=np.arange(p.ymin,p.ymax,p.dy)
	x,y=np.meshgrid(x_line,y_line)

	nx=len(x_line)
	ny=len(y_line)

	#- Frequency line.
	f=np.arange(p.fmin-p.fwidth,p.fmax+p.fwidth,p.df)
	omega=2.0*np.pi*f

	#- Power-spectral density.
	S,indeces=s.space_distribution()
	instrument,natural=s.frequency_distribution(f)
	filt=natural*instrument*instrument

	C=np.zeros((len(y_line),len(x_line),len(omega)),dtype=complex)

	#==============================================================================
	#- Compute correlation field by summing over individual sources.
	#==============================================================================

	if (mode=='individual'):

		#- March through the spatial grid. ----------------------------------------
		for idx in range(nx):

			if verbose: print(str(100*float(idx)/float(len(x_line)))+' %')

			for idy in range(ny):

				#- March through all sources.
				for k in indeces:

					C[idy,idx,:]+=S[k]*filt*g.conjG1_times_G2(x[idy,idx],y[idy,idx],p.x[rec],p.y[rec],x[k],y[k],omega,p.dx,p.dy,p.rho,p.v,p.Q)
					
		#- Normalisation.
		C=np.conj(C)*p.dx*p.dy

	#==============================================================================
	#- Compute correlation field by random integration over all sources
	#==============================================================================

	downsampling_factor=5.0
	n_samples=np.floor(float(nx*ny)/downsampling_factor)

	if (mode=='random'):

		#- March through frequencies. ---------------------------------------------

		for idf in range(0,len(f),3):

			if verbose: print('f=', f[idf], ' Hz')

			if (filt[idf]>0.05*np.max(filt)):

				#- March through downsampled spatial grid. ------------------------

				t0=time.time()

				for idx in range(0,nx,3):
					for idy in range(0,ny,3):

						samples_x=np.random.randint(0,nx,n_samples)
						samples_y=np.random.randint(0,ny,n_samples)
						
						G1=g.green_input(x[samples_y,samples_x],y[samples_y,samples_x],x_line[idx],y_line[idy],omega[idf],p.dx,p.dy,p.rho,p.v,p.Q)
						G2=g.green_input(x[samples_y,samples_x],y[samples_y,samples_x],p.x[rec],   p.y[rec],   omega[idf],p.dx,p.dy,p.rho,p.v,p.Q)
				
						C[idy,idx,idf]=downsampling_factor*filt[idf]*np.sum(S[samples_y,samples_x]*G1*np.conj(G2))
					
				t1=time.time()
				if verbose: print('time per frequency: ', t1-t0, 's')

		#- Normalisation. ---------------------------------------------------------

		C=C*p.dx*p.dy

		#- Spatial interpolation. -------------------------------------------------

		for idx in range(0,nx-3,3):
			C[:,idx+1,:]=0.67*C[:,idx,:]+0.33*C[:,idx+3,:]
			C[:,idx+2,:]=0.33*C[:,idx,:]+0.67*C[:,idx+3,:]

		for idy in range(0,ny-3,3):
			C[idy+1,:,:]=0.67*C[idy,:,:]+0.33*C[idy+3,:,:]
			C[idy+2,:,:]=0.33*C[idy,:,:]+0.67*C[idy+3,:,:]

		#- Frequency interpolation. -----------------------------------------------

		for idf in range(0,len(f)-3,3):
			C[:,:,idf+1]=0.67*C[:,:,idf]+0.33*C[:,:,idf+3]
			C[:,:,idf+2]=0.33*C[:,:,idf]+0.67*C[:,:,idf+3]

	#==============================================================================
	#- Compute correlation field by random integration over all sources + individual sources
	#==============================================================================

	downsampling_factor=5.0
	n_samples=np.floor(float(nx*ny)/downsampling_factor)

	if (mode=='combined'):

		#--------------------------------------------------------------------------
		#- March through frequencies for random sampling. -------------------------

		for idf in range(0,len(f),3):

			if verbose: print('f=', f[idf], ' Hz')

			if (filt[idf]>0.05*np.max(filt)):

				#- March through downsampled spatial grid. ------------------------

				t0=time.time()

				for idx in range(0,nx,3):
					for idy in range(0,ny,3):

						samples_x=np.random.randint(0,nx,n_samples)
						samples_y=np.random.randint(0,ny,n_samples)
						
						G1=g.green_input(x[samples_y,samples_x],y[samples_y,samples_x],x_line[idx],y_line[idy],omega[idf],p.dx,p.dy,p.rho,p.v,p.Q)
						G2=g.green_input(x[samples_y,samples_x],y[samples_y,samples_x],p.x[rec],   p.y[rec],   omega[idf],p.dx,p.dy,p.rho,p.v,p.Q)
				
						C[idy,idx,idf]=downsampling_factor*filt[idf]*np.sum(S[samples_y,samples_x]*G1*np.conj(G2))
					
				t1=time.time()
				if verbose: print('time per frequency: ', t1-t0, 's')


		#- Spatial interpolation. -------------------------------------------------

		for idx in range(0,nx-3,3):
			C[:,idx+1,:]=0.67*C[:,idx,:]+0.33*C[:,idx+3,:]
			C[:,idx+2,:]=0.33*C[:,idx,:]+0.67*C[:,idx+3,:]

		for idy in range(0,ny-3,3):
			C[idy+1,:,:]=0.67*C[idy,:,:]+0.33*C[idy+3,:,:]
			C[idy+2,:,:]=0.33*C[idy,:,:]+0.67*C[idy+3,:,:]

		#- Frequency interpolation. -----------------------------------------------

		for idf in range(0,len(f)-3,3):
			C[:,:,idf+1]=0.67*C[:,:,idf]+0.33*C[:,:,idf+3]
			C[:,:,idf+2]=0.33*C[:,:,idf]+0.67*C[:,:,idf+3]


		#--------------------------------------------------------------------------
		#- March through the spatial grid for individual sources. -----------------
		
		for idx in range(nx):

			if verbose: print(str(100*float(idx)/float(len(x_line)))+' %')

			for idy in range(ny):

				#- March through all sources.
				for k in indeces:

					C[idy,idx,:]+=S[k]*filt*np.conj(g.conjG1_times_G2(x[idy,idx],y[idy,idx],p.x[rec],p.y[rec],x[k],y[k],omega,p.dx,p.dy,p.rho,p.v,p.Q))
					
		
		#- Normalisation. ---------------------------------------------------------

		C=C*p.dx*p.dy

	#==============================================================================
	#- Save interferometric wavefield.
	#==============================================================================

	fn='OUTPUT/cf_'+str(rec)
	np.save(fn,C)