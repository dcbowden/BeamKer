import numpy as np

class Parameters():

	def __init__(self):

		#- Read setup.txt. ----------------------------------------------------

		f=open('INPUT/setup.txt','r')

		#- Spatial domain.
		f.readline()
		self.xmin=float(f.readline().strip().split('=')[1])
		self.xmax=float(f.readline().strip().split('=')[1])
		self.dx=float(f.readline().strip().split('=')[1])
		f.readline()
		self.ymin=float(f.readline().strip().split('=')[1])
		self.ymax=float(f.readline().strip().split('=')[1])
		self.dy=float(f.readline().strip().split('=')[1])

		#- Material parameters.
		f.readline()
		f.readline()
		self.rho=float(f.readline().strip().split('=')[1])
		self.v=float(f.readline().strip().split('=')[1])
		self.Q=float(f.readline().strip().split('=')[1])

		#- Spatial source distribution.
		f.readline()
		f.readline()
		self.type=str(f.readline().strip().split('=')[1])

		#- Natural power-spectral density.
		f.readline()
		f.readline()
		self.natural=str(f.readline().strip().split('=')[1])

		#- Instrument response.
		f.readline()
		f.readline()
		self.instrument=str(f.readline().strip().split('=')[1])

		f.close()

		#- Read receivers.txt. ------------------------------------------------

		f=open('INPUT/receivers.txt','r')

		#- Number of receivers.
		f.readline()
		self.Nreceivers=int(f.readline().strip().split('=')[1])

		#- Initialise receiver positions.
		self.x=np.zeros(self.Nreceivers)
		self.y=np.zeros(self.Nreceivers)

		#- Receiver positions.
		f.readline()
		f.readline()

		for k in range(self.Nreceivers):
			self.x[k]=float(f.readline().strip().split('=')[1])
			self.y[k]=float(f.readline().strip().split('=')[1])
			f.readline()
		
		f.close()

		#- Read correlation_field_and_kernels.txt. ----------------------------

		f=open('INPUT/correlation_field_and_kernels.txt','r')

		#- Frequency.
		f.readline()
		self.fmin=float(f.readline().strip().split('=')[1])
		self.fmax=float(f.readline().strip().split('=')[1])
		self.df=float(f.readline().strip().split('=')[1])
		self.fwidth=float(f.readline().strip().split('=')[1])

		#- Time.
		f.readline()
		f.readline()
		self.tmin=float(f.readline().strip().split('=')[1])
		self.tmax=float(f.readline().strip().split('=')[1])
		self.dt=float(f.readline().strip().split('=')[1])

		f.close()

		#- Read ensemble_correlation.txt. ----------------------------

		f=open('INPUT/ensemble_correlation.txt','r')

		#- Length and number of windows.
		f.readline()
		self.Twindow=float(f.readline().strip().split('=')[1])
		self.Nwindows=int(f.readline().strip().split('=')[1])

		#- Random seed.
		f.readline()
		f.readline()
		self.seed=int(f.readline().strip().split('=')[1])

		f.close()

		#- Read processing.txt. ---------------------------------------

		f=open('INPUT/processing.txt','r')

		#- Standard bandpass.
		f.readline()
		self.bp_fmin=float(f.readline().strip().split('=')[1])
		self.bp_fmax=float(f.readline().strip().split('=')[1])
		self.bp_width=float(f.readline().strip().split('=')[1])

		#- Single-trace processing.
		f.readline()
		f.readline()
		self.process_onebit=int(f.readline().strip().split('=')[1])
		self.process_rms_clip=int(f.readline().strip().split('=')[1])
		self.process_whiten=int(f.readline().strip().split('=')[1])

		#- Correlation processing.
		f.readline()
		f.readline()
		self.process_causal_acausal_average=int(f.readline().strip().split('=')[1])
		self.process_correlation_normalisation=int(f.readline().strip().split('=')[1])
		self.process_phase_weighted_stack=int(f.readline().strip().split('=')[1])

		f.close()

		#- Read earthquake_catalogue.txt. -------------------------------------

		f=open('INPUT/earthquake_catalogue.txt','r')

		#- Number of earthquakes.
		f.readline()
		self.Neq=int(f.readline().strip().split('=')[1])

		#- Initialise earthquake positions, amplitudes and timings.
		self.eq_x=[[] for i in range(self.Nwindows)]
		self.eq_y=[[] for i in range(self.Nwindows)]
		self.eq_t=[[] for i in range(self.Nwindows)]
		self.eq_m=[[] for i in range(self.Nwindows)]

		#- Read earthquake positions, amplitudes and timings.
		f.readline()
		f.readline()

		for k in range(self.Neq):

			t_dummy=float(f.readline().strip().split('=')[1])
			window=np.floor(t_dummy/float(self.Twindow))

			self.eq_t[int(window)].append(t_dummy-window*self.Twindow)
			self.eq_x[int(window)].append(float(f.readline().strip().split('=')[1]))
			self.eq_y[int(window)].append(float(f.readline().strip().split('=')[1]))
			self.eq_m[int(window)].append(float(f.readline().strip().split('=')[1]))
			f.readline()

		#- Issue warnings if necessary. ---------------------------------------

		lambda_min=self.v/self.fmax
		if (max(self.dx,self.dy)>0.5*lambda_min):
			print('Maximum grid spacing, '+str(max(self.dx,self.dy))+' km, is larger than half the minimum wavelength, '+str(0.5*lambda_min)+'km.')