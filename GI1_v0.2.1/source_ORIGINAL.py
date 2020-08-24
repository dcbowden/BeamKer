import numpy as np
import matplotlib.pyplot as plt
import parameters

def space_distribution(plot=False,save=False):
	"""
	S, indices = space_distribution(plot=False,save=False)

	Spatial frequency-domain power-spectral density distribution mask of the noise sources. 

	INPUT:
	------
	plot:		Plot source distribution when plot==True.
	save:		Save to OUTPUT/ when True.

	OUTPUT:
	-------
	S:		Frequency-domain power-spectral density distribution of the noise sources [1]. The actual units come from the frequency dependence of the spectrum below.
	indices:	Indices where S is non-zero.

	Last updated: 18 July 2019.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	p=parameters.Parameters()

	x_line=np.arange(p.xmin,p.xmax,p.dx)
	y_line=np.arange(p.ymin,p.ymax,p.dy)

	x,y=np.meshgrid(x_line,y_line)

	nx=len(x_line)
	ny=len(y_line)

	#==============================================================================
	#- Define power-spectral density distribution.
	#==============================================================================

	#- A Gaussian blob. -----------------------------------------------------------

	if (p.type=='gauss'):

		x0=0.5*p.xmax
		y0=0.5*(p.ymax+p.ymin)
		sigma=0.1*p.xmax

		exponent=((x-x0)**2+(y-y0)**2)/(2.0*sigma**2)
		S=np.exp(-exponent)

		maxS=np.max(S)

		indices=[]

		for i in range(len(x_line)):
			for k in range(len(y_line)):

				if S[k,i]>0.5*maxS:
					indices.append((k,i))

	#- Gaussian blobs. ------------------------------------------------------------

	elif (p.type=='gauss2'):

		#- First blob.

		x0=0.75*p.xmax
		y0=-0.3*p.ymax
		sigma_x=0.03*p.xmax
		sigma_y=0.15*p.xmax

		exponent=((x-x0)**2/(2.0*sigma_x**2)+(y-y0)**2/(2.0*sigma_y**2))
		S=np.exp(-exponent)

		#- Second blob.

		x0=-0.35*p.xmax
		y0=-0.5*p.ymax
		sigma_x=0.03*p.xmax
		sigma_y=0.10*p.xmax

		exponent=((x-x0)**2/(2.0*sigma_x**2)+(y-y0)**2/(2.0*sigma_y**2))
		S+=np.exp(-exponent)

		#- Compute indices.

		maxS=np.max(S)

		indices=[]

		for i in range(len(x_line)):
			for k in range(len(y_line)):

				if S[k,i]>0.3*maxS:
					indices.append((k,i))

	#- Gaussian blobs. ------------------------------------------------------------

	elif (p.type=='gauss3'):

		#- First blob.

		x0=0.75*p.xmax
		y0=-0.3*p.ymax
		sigma_x=0.03*p.xmax
		sigma_y=0.15*p.xmax

		exponent=((x-x0)**2/(2.0*sigma_x**2)+(y-y0)**2/(2.0*sigma_y**2))
		S=np.exp(-exponent)

		#- Second blob.

		x0=-0.60*p.xmax
		y0=0.60*p.ymax
		sigma_x=0.30*p.xmax
		sigma_y=0.10*p.xmax

		exponent=((x-x0)**2/(2.0*sigma_x**2)+(y-y0)**2/(2.0*sigma_y**2))
		S+=np.exp(-exponent)

		#- Compute indices.

		maxS=np.max(S)

		indices=[]

		for i in range(len(x_line)):
			for k in range(len(y_line)):

				if S[k,i]>0.3*maxS:
					indices.append((k,i))

	#- Gaussian blobs. ------------------------------------------------------------

	elif (p.type=='gauss3_sparse'):

		#- First blob.

		x0=0.75*p.xmax
		y0=-0.3*p.ymax
		sigma_x=0.03*p.xmax
		sigma_y=0.15*p.xmax

		exponent=((x-x0)**2/(2.0*sigma_x**2)+(y-y0)**2/(2.0*sigma_y**2))
		S=np.exp(-exponent)

		#- Second blob.

		x0=-0.60*p.xmax
		y0=0.60*p.ymax
		sigma_x=0.30*p.xmax
		sigma_y=0.10*p.xmax

		exponent=((x-x0)**2/(2.0*sigma_x**2)+(y-y0)**2/(2.0*sigma_y**2))
		S+=np.exp(-exponent)

		#- Compute indices.

		maxS=np.max(S)

		indices=[]

		for i in range(0,len(x_line),5):
			for k in range(0,len(y_line),5):

				if S[k,i]>0.3*maxS:
					indices.append((k,i))

	#- Single point source. -------------------------------------------------------

	elif (p.type=='point'):

		S=np.zeros(np.shape(x))

		S[80,300]=1.0e6
		S[130,280]=1.0e6

		indices=[]
		indices.append((80,300))
		indices.append((130,280))

	#- Single point source on homogeneous background. -----------------------------

	elif (p.type=='point_homogeneous'):

		#- Point sources on homogeneous background.
		S=10000.0*np.ones(np.shape(x))

		S[80,300]=1.0e6
		S[130,280]=1.0e6

		#- Assign only the indices of point sources for precompute in 'combined' mode.
		indices=[]
		indices.append((80,300))
		indices.append((130,280))


	#- Two smoothed blocks. -------------------------------------------------------

	elif (p.type=='blocks1'):

		#- Make smoothed blocks.

		S=np.zeros(np.shape(x))
		indices=[]

		S[20:80,295:305]=1.0e4
		S[10:40,95:105]=2.0e4

		for k in range(10): S[1:ny-2,:]=(S[1:ny-2,:]+S[0:ny-3,:]+S[2:ny-1,:])/3.0
		for k in range(10): S[:,1:nx-2]=(S[:,1:nx-2]+S[:,0:nx-3]+S[:,2:nx-1])/3.0

		#- Compute indices.

		maxS=np.max(S)

		indices=[]

		for i in range(nx):
			for k in range(ny):

				if S[k,i]>0.3*maxS:
					indices.append((k,i))

	#- Homogeneous distribution. --------------------------------------------------

	elif (p.type=='homogeneous'):

		S=np.ones(np.shape(x))
		indices=[]

	#- Two lines of receivers (used for figure 1). --------------------------------

	elif (p.type=='line'):

		S=np.zeros(np.shape(x))
		indices=[]

		for i in range(20,80):
			S[i,300]=1.0e6
			indices.append((i,300))

		for i in range(10,40):
			S[i,100]=1.0e6
			indices.append((i,100))

	#- Two lines of receivers. ----------------------------------------------------

	elif (p.type=='line2'):

		S=np.zeros(np.shape(x))
		indices=[]

		for i in range(50,100):
			S[i,300]=1.0e6
			indices.append((i,300))

		for i in range(50,100):
			S[i,50]=1.0e6
			indices.append((i,50))

	#- Nearly equally distributed quasi-random sources distribution. --------------

	elif (p.type=='random1'):

		S=np.zeros(np.shape(x))
		indices=[]
		np.random.seed(1)

		Nx=float(len(x_line))
		Ny=float(len(y_line))

		for k in range(100):
			idx=int(np.floor(Nx*float(np.random.rand(1))))
			idy=int(np.floor(Ny*float(np.random.rand(1))))

			S[idy,idx]=1.0e6
			indices.append((idy,idx))

	#- Biased random distribution. ------------------------------------------------

	elif (p.type=='random2'):

		S=np.zeros(np.shape(x))
		indices=[]
		np.random.seed(1)

		Nx=float(len(x_line))
		Ny=float(len(y_line))

		for k in range(100):
			idx=int(np.floor(Nx*float(np.random.rand(1))))
			idy=int(np.floor(Ny*float(np.random.rand(1))))

			S[idy,idx]=1.0e6
			indices.append((idy,idx))

		for k in range(200):
			idx=int(np.floor(Nx*float((0.20+0.15*np.random.rand(1)))))
			idy=int(np.floor(Ny*float((0.10+0.30*np.random.rand(1)))))

			S[idy,idx]=1.0e6
			indices.append((idy,idx))

	#- The same as random2, but scaled. -------------------------------------------

	elif (p.type=='random3'):

		S=np.zeros(np.shape(x))
		indices=[]
		np.random.seed(1)

		Nx=float(len(x_line))
		Ny=float(len(y_line))

		for k in range(100):
			idx=int(np.floor(Nx*float(np.random.rand(1))))
			idy=int(np.floor(Ny*float(np.random.rand(1))))

			S[idy,idx]=1.0e3
			indices.append((idy,idx))

		for k in range(200):
			idx=int(np.floor(Nx*float((0.20+0.15*np.random.rand(1)))))
			idy=int(np.floor(Ny*float((0.10+0.30*np.random.rand(1)))))

			S[idy,idx]=1.0e3
			indices.append((idy,idx))

	#==============================================================================
	#- Plot.
	#==============================================================================

	if (plot or save):

		plt.rcParams["font.family"] = "serif"
		plt.rcParams.update({'font.size': 10})

		#- Plot source power-spectral density distribution. -----------------------

		plt.pcolor(x/1000.0,y/1000.0,S,cmap='Greys')

		#- Plot receiver positions. -----------------------------------------------

		for k in range(p.Nreceivers):

			plt.plot(p.x[k]/1000.0,p.y[k]/1000.0,'ko',markersize='5')
			plt.text(p.x[k]/1000.0+3.0*p.dx/1000.0,p.y[k]/1000.0+3.0*p.dx/1000.0,str(k))

		plt.axis('image')
		plt.xlim((p.xmin/1000.0,p.xmax/1000.0))
		plt.ylim((p.ymin/1000.0,p.ymax/1000.0))
		plt.xlabel('x [km]')
		plt.ylabel('y [km]')
		plt.title('source-receiver configuration')
		plt.colorbar()

		if plot:
			plt.show()
		else:
			plt.savefig('OUTPUT/source_receiver_geometry.pdf',format='pdf')
			plt.clf()

	#==============================================================================
	#- Return.
	#==============================================================================

	return S, indices

#==================================================================================
#==================================================================================
#==================================================================================

def frequency_distribution(f,plot=False):
	"""
	instrument,natural = frequency_distribution(f,plot=0)

	Power-spectral density distribution of the noise sources [1 /s^2] and instrument response.

	INPUT:
	------
	f:			Frequency axis [Hz].
	plot:		Plot source distribution when plot==True.

	OUTPUT:
	-------
	instrument:	Frequency-domain instrument response [1].
	natural:	Power-spectral density of the noise sources as function of frequency [1 /s^2].

	Last updated: 16 July 2019.
	"""
	
	#==============================================================================
	#- Initialisation.
	#==============================================================================

	#- Input. ---------------------------------------------------------------------

	p=parameters.Parameters()

	natural=np.zeros(len(f))
	instrument=np.zeros(len(f))

	T=1.0/f

	#==============================================================================
	#- Instrument response.
	#==============================================================================

	#- Flat response (white). -----------------------------------------------------

	if p.instrument=='flat':

		instrument[:]=1.0

	#- Bandpass. ------------------------------------------------------------------

	if p.instrument=='bandpass':

		df=f[1]-f[0]

		Nminmax=int(np.round((p.bp_fmin-np.min(f))/df))
		Nminmin=int(np.round((p.bp_fmin-p.bp_width-np.min(f))/df))
		Nmaxmin=int(np.round((p.bp_fmax-np.min(f))/df))
		Nmaxmax=int(np.round((p.bp_fmax+p.bp_width-np.min(f))/df))

		instrument[Nminmin:Nminmax]=np.linspace(0.0,1.0,Nminmax-Nminmin)
		instrument[Nmaxmin:Nmaxmax]=np.linspace(1.0,0.0,Nmaxmax-Nmaxmin)
		instrument[Nminmax:Nmaxmin]=1.0

	#==============================================================================
	#- Natural ambient noise power-spectral density.
	#==============================================================================

	#- Flat spectrum (white). -----------------------------------------------------

	if p.natural=='flat':

		natural[:]=1.0

	#- Bandpass. ------------------------------------------------------------------

	if p.natural=='bandpass':

		df=f[1]-f[0]

		Nminmax=int(np.round((p.bp_fmin-np.min(f))/df))
		Nminmin=int(np.round((p.bp_fmin-np.min(f)-p.bp_width)/df))
		Nmaxmin=int(np.round((p.bp_fmax-np.min(f))/df))
		Nmaxmax=int(np.round((p.bp_fmax-np.min(f)+p.bp_width)/df))

		natural[Nminmin:Nminmax]=np.linspace(0.0,1.0,Nminmax-Nminmin)
		natural[Nmaxmin:Nmaxmax]=np.linspace(1.0,0.0,Nmaxmax-Nmaxmin)
		natural[Nminmax:Nmaxmin]=1.0

	#- Peterson's NLNM ------------------------------------------------------------

	#- In using Peterson's NLNM, we make the simplifying assumption that the
	#- recorded noise spectrum is equal to the actual noise source spectrum.
	#- In reality, however, the latter is modulated by wave propagation through
	#- the Earth. The spectrum for elastic waves is simply re-interpreted as a
	#- spectrum for acoustic waves.

	if p.natural=='nlnm':

		#- Power-spectral density for acceleration.
		for k in range(len(f)):

			if f[k]==0.0: 
				T[k]=100000.0
			else:
				T[k]=1.0/f[k]

			if (T[k]>=0.1) & (T[k]<0.17): natural[k]=-162.36+5.64*np.log10(T[k])
			elif (T[k]>=0.17) & (T[k]<0.40): natural[k]=-166.70
			elif (T[k]>=0.40) & (T[k]<0.80): natural[k]=-170.00-8.30*np.log10(T[k])
			elif (T[k]>=0.80) & (T[k]<1.24): natural[k]=-166.40+28.90*np.log10(T[k])
			elif (T[k]>=1.24) & (T[k]<2.40): natural[k]=-168.60+52.48*np.log10(T[k])
			elif (T[k]>=2.40) & (T[k]<4.30): natural[k]=-159.98+29.81*np.log10(T[k])
			elif (T[k]>=4.30) & (T[k]<5.00): natural[k]=-141.10
			elif (T[k]>=5.00) & (T[k]<6.00): natural[k]=-71.36-99.77*np.log10(T[k])
			elif (T[k]>=6.00) & (T[k]<10.00): natural[k]=-97.26-66.49*np.log10(T[k])
			elif (T[k]>=10.00) & (T[k]<12.0): natural[k]=-132.18-31.57*np.log10(T[k])
			elif (T[k]>=12.00) & (T[k]<15.60): natural[k]=-205.27+36.16*np.log10(T[k])
			elif (T[k]>=15.60) & (T[k]<21.90): natural[k]=-37.65-104.33*np.log10(T[k])
			elif (T[k]>=21.90) & (T[k]<31.60): natural[k]=-114.37-47.10*np.log10(T[k])
			elif (T[k]>=31.60) & (T[k]<45.00): natural[k]=-160.58-16.28*np.log10(T[k])
			elif (T[k]>=45.00) & (T[k]<70.00): natural[k]=-187.50
			elif (T[k]>=70.00) & (T[k]<101.00): natural[k]=-216.47+15.70*np.log10(T[k])
			elif (T[k]>=101.00) & (T[k]<154.00): natural[k]=-185.00
			elif (T[k]>=154.00) & (T[k]<328.00): natural[k]=-168.34-7.61*np.log10(T[k])
			elif (T[k]>=328.00) & (T[k]<600.00): natural[k]=-217.43+11.90*np.log10(T[k])
			elif (T[k]>=600.00) & (T[k]<10000.00): natural[k]=-258.28+26.60*np.log10(T[k])
			elif (T[k]>=10000.00) & (T[k]<100000.00): natural[k]=-346.88+48.75*np.log10(T[k])

		#- Convert to displacement.
		natural=natural+40.0*np.log10(T/(2.0*np.pi))

		#- Convert from dB to absolute power-spectral density.
		natural=10.0**(0.1*natural)

	#- Peterson's NHNM ------------------------------------------------------------

	#- In using Peterson's NHNM, we make the simplifying assumption that the
	#- recorded noise spectrum is equal to the actual noise source spectrum.
	#- In reality, however, the latter is modulated by wave propagation through
	#- the Earth. The spectrum for elastic waves is simply re-interpreted as a
	#- spectrum for acoustic waves.

	if p.natural=='nhnm':

		#- Power-spectral density for acceleration.
		for k in range(len(T)):

			if f[k]==0.0: 
				T[k]=100000.0
			else:
				T[k]=1.0/f[k]

			if (T[k]>=0.1) & (T[k]<0.22): natural[k]=-108.73-17.23*np.log10(T[k])
			elif (T[k]>=0.22) & (T[k]<0.32): natural[k]=-150.34-80.50*np.log10(T[k])
			elif (T[k]>=0.32) & (T[k]<0.80): natural[k]=-122.31-23.87*np.log10(T[k])
			elif (T[k]>=0.80) & (T[k]<3.80): natural[k]=-116.85+32.51*np.log10(T[k])
			elif (T[k]>=3.80) & (T[k]<4.60): natural[k]=-108.48+18.08*np.log10(T[k])
			elif (T[k]>=4.60) & (T[k]<6.30): natural[k]=-74.66-32.95*np.log10(T[k])
			elif (T[k]>=6.30) & (T[k]<7.90): natural[k]=0.66-127.18*np.log10(T[k])
			elif (T[k]>=7.90) & (T[k]<15.40): natural[k]=-93.37-22.42*np.log10(T[k])
			elif (T[k]>=15.40) & (T[k]<20.00): natural[k]=73.54-162.98*np.log10(T[k])
			elif (T[k]>=20.00) & (T[k]<354.80): natural[k]=-151.25+10.01*np.log10(T[k])
			elif (T[k]>=354.80) & (T[k]<100000.00): natural[k]=-206.66+31.63*np.log10(T[k])
			
		#- Convert to displacement.
		natural=natural+40.0*np.log10(T/(2.0*np.pi))

		#- Convert from dB to absolute power-spectral density.
		natural=10.0**(0.1*natural)


	#==============================================================================
	#- Plot.
	#==============================================================================

	if plot:

		plt.rcParams["font.family"] = "serif"
		plt.rcParams.update({'font.size': 10})

		plt.plot(f,natural,'k',linewidth=2)
		plt.title('source spectrum')
		plt.xlabel('period [s]')
		plt.ylabel(r'source power-spectral density [1 /s$^2$]')
		plt.show()

		plt.plot(f,instrument,'k',linewidth=2)
		plt.title('instrument response')
		plt.xlabel('period [s]')
		plt.ylabel('instrument response [1]')
		plt.show()

		plt.plot(f,np.sqrt(natural),'r')
		plt.title('source and instrument spectrum combined')
		plt.xlabel('period [s]')
		plt.ylabel('combined instrument and source spectrum [1 / s]')
		plt.show()


	return instrument, natural

