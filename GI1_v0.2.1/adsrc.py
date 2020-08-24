import numpy as np
import matplotlib.pyplot as plt
import parameters


def adsrc(cct, t, measurement='cctime', plot=0):
  """
  a = adsrc(cct, t, measurement='cctime', plot=0):

  Compute frequency-domain adjoint source for a given measurement.

  INPUT:
  ------

  cct, t:       Time-domain correlation function and time axis as obtained from correlation_function.py.
  measurement:  Type of measurement used to compute the adjoint source. See adsrc.py for options.
  plot:         When plot=1, plot original and windowed correlation, as well as frequency-domain adjoint source.

  OUTPUT:
  -------
  a:            Frequency-domain adjoint source

  Last updated: 12 May 2016.
  """

  #==============================================================================
  #- Initialisation.
  #==============================================================================

  #- Basic input parameters. ----------------------------------------------------

  p=parameters.Parameters()

  f=np.arange(p.fmin-p.fwidth,p.fmax+p.fwidth,p.df)
  omega=2.0*np.pi*f

  df=f[1]-f[0]

  #- Read time windows. ---------------------------------------------------------

  windows=[]

  fid=open('INPUT/windows.txt','r')

  fid.readline()
  N=int(fid.readline())
  fid.readline()
  fid.readline()

  for n in range(N):
    w=fid.readline().strip().split(' ')
    windows.append([float(w[0]), float(w[1]), float(w[2])])

  fid.close()

  #- Apply windows to time-domain correlation. ----------------------------------

  tap=np.zeros(len(cct))

  for n in range(N):
    tap+=taper(t,windows[n][0],windows[n][1],windows[n][2])

  cct_tap=cct*tap
  cct_tap2=cct_tap*tap

  if plot==1:
    plt.plot(t,cct,'k')
    plt.plot(t,cct_tap,'r')
    plt.xlabel('time [s]')
    plt.ylabel('correlation [N^2 s/ m^4]')
    plt.title('time-domain correlation function (windowed in red)')
    plt.show()

  #- Compute Fourier transform of the time-domain correlation function. ---------
    
  ccf=np.arange(len(f),dtype=complex)
  ccf_tap=np.arange(len(f),dtype=complex)
  ccf_tap2=np.arange(len(f),dtype=complex)

  for k in range(len(f)):
      ccf[k]=np.sum(cct*np.exp(-1.0j*omega[k]*t))*p.dt
      ccf_tap[k]=np.sum(cct_tap*np.exp(-1.0j*omega[k]*t))*p.dt
      ccf_tap2[k]=np.sum(cct_tap2*np.exp(-1.0j*omega[k]*t))*p.dt

  #- Amplitude measurement. -----------------------------------------------------
  #- Simple L2 energy: chi = 0.5 * integral ( |C|^2 ) 
  
  if measurement=='amp':

    a=np.conj(ccf_tap2)

  #- Cross-correlation time shift measurement. ----------------------------------

  if measurement=='cctime':

    normalisation=np.sum(omega*omega*ccf_tap*np.conj(ccf))*2.0*np.pi*df
    a=-1.0j*omega*np.conj(ccf_tap)/normalisation

  #- Return. ---------------------------------------------------------------------- 

  return a

#==================================================================================================
#- Cosine taper.
#==================================================================================================
def taper(t,tmin,tmax,width):
  """ Cosine taper. 
      t=time axis [s]
      tmin=minimum time [s]
      tmax=maximum time [s]
      width=width of the taper [s]
  """

  #- Find indices that define the taper edges.
  N=float(len(t))
  Nminmin=int(np.argwhere(np.min(np.abs(t-tmin+width))==np.abs(t-tmin+width))[0])
  Nminmax=int(np.argwhere(np.min(np.abs(t-tmin))==np.abs(t-tmin))[0])
  Nmaxmin=int(np.argwhere(np.min(np.abs(t-tmax))==np.abs(t-tmax))[0])
  Nmaxmax=int(np.argwhere(np.min(np.abs(t-tmax-width))==np.abs(t-tmax-width))[0])

  #- Plateau.
  f=np.zeros(len(t))
  f[Nminmax:Nmaxmin]=1.0

  #- Left slope.
  x=np.arange(0.0,Nminmax+1-Nminmin,dtype=float)
  x=np.pi*x/x[-1]
  f[Nminmin:Nminmax+1]=0.5-0.5*np.cos(x)

  #- Right slope.
  x=np.arange(0.0,Nmaxmax+1-Nmaxmin,dtype=float)
  x=np.pi*x/x[-1]
  f[Nmaxmin:Nmaxmax+1]=0.5+0.5*np.cos(x)

  return f


