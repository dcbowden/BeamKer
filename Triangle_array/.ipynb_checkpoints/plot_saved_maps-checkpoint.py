# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Python 3.7

# Assumes we're running in a directory like ~/Dropbox/...../GI_code/Array_name/
# The source files are in ~/Dropbox/...../GI_code/GI/

import sys
sys.path.append('../GI1_v2/')
sys.path.append('../GI1_v2/PLOT/')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import green as g
import source as s
import processing as proc
import parameters
import time
import random
from correlation_random import *
from correlation_function import *
from correlation_field import *
from kernels import *
from adsrc import *
from earthquakes import *

# The following are newly added, as compared to Andreas' original code.
from correlation_random_ALL import *
from beamform import *

# %matplotlib inline
# -

mpl.rcParams['figure.figsize'] = [20, 10]

# Check the station and noise-source geometry
S,indeces=s.space_distribution(plot=1)

# +
#- Input parameters.
p=parameters.Parameters()
xx = p.x/1000
yy = p.y/1000
x_center = np.mean(p.x)
y_center = np.mean(p.y)

# Have to assume a velocity! 
rvel=3.0       


#- Spatial grid.
downsample=1
x_line=np.arange(p.xmin*1.5,p.xmax*1.5,p.dx*downsample)/1000.0
y_line=np.arange(p.ymin*1.5,p.ymax*1.5,p.dy*downsample)/1000.0
x,y=np.meshgrid(x_line,y_line)



# We know where the source actually is:
x0=0.75*p.xmax
y0=0.75*p.ymax
ix0=(np.abs(x_line*1000.0-x0)).argmin()
iy0=(np.abs(y_line*1000.0-y0)).argmin()

# Let's also examine a "Bad" point
xB=1.1*p.xmax
yB=0.7*p.ymax
ixB=(np.abs(x_line*1000.0-xB)).argmin()
iyB=(np.abs(y_line*1000.0-yB)).argmin()

np.save('3_PD', PD)
np.save('3_PG', PG)
np.save('3_PS', PS)
np.save('3_PDna', PDna)
np.save('3_PGna', PGna)
np.save('3_PSna', PSna)

# +
# MFP!
# Modified to include Gosia's MFP and SSA, all in one!

plot=True 

# Time domain shifts. This is rather slow...
# Probably because I'm using the fancy fft-domain specshift
#  which is over-engineered for this purpose

#- Input parameters.
p=parameters.Parameters()
xx = p.x/1000
yy = p.y/1000
x_center = np.mean(p.x)
y_center = np.mean(p.y)

# Have to assume a velocity! 
rvel=3.0       

# ONLY work on the first time window. 
# TODO - work on more windows? average results?
iwin=0
data=ut[:,iwin,:]
datacc=cct[:,iwin,:]
nsta = np.shape(ut)[2]


#- Spatial grid.
downsample=1
x_line=np.arange(p.xmin*1.5,p.xmax*1.5,p.dx*downsample)/1000.0
y_line=np.arange(p.ymin*1.5,p.ymax*1.5,p.dy*downsample)/1000.0
x,y=np.meshgrid(x_line,y_line)


# We know where the source actually is:
x0=0.75*p.xmax
y0=0.75*p.ymax
ix0=(np.abs(x_line*1000.0-x0)).argmin()
iy0=(np.abs(y_line*1000.0-y0)).argmin()

# Let's also examine a "Bad" point
xB=1.1*p.xmax
yB=0.7*p.ymax
ixB=(np.abs(x_line*1000.0-xB)).argmin()
iyB=(np.abs(y_line*1000.0-yB)).argmin()



PD=np.zeros([np.size(x_line),np.size(y_line)],dtype=complex) # Daniel's incorrect MFP
PG=np.zeros([np.size(x_line),np.size(y_line)],dtype=complex) # Gosia's MFP (the real one)
PS=np.zeros([np.size(x_line),np.size(y_line)],dtype=complex) # SSA

# The same thing, but excluding autocorrlations
# "na"  = non-auto
PDna=np.zeros([np.size(x_line),np.size(y_line)],dtype=complex) # Daniel's incorrect MFP
PGna=np.zeros([np.size(x_line),np.size(y_line)],dtype=complex) # Gosia's MFP (the real one)
PSna=np.zeros([np.size(x_line),np.size(y_line)],dtype=complex) # SSA


print("starting gridpoints: {0}".format(np.size(x_line)*np.size(y_line)))
counter_grid=0
for idx in range(len(x_line)):
        for idy in range(len(y_line)):
            counter_grid+=1
            if(counter_grid % 1000 == 0):
                print(counter_grid)
                 
            # Distances from the test-point to the various stations
            distances = np.sqrt( (xx-x_line[idx])**2 + (yy-y_line[idy])**2 )
            timeshifts = distances / rvel
            
            # Relative timeshift, the difference that is actually needed to shift
            timeshifts_relative_to_mean = timeshifts - np.mean(timeshifts)
            
            # Work on direct time traces.
            dshiftD = specshift(data,timeshifts_relative_to_mean/p.dt)
             
            # Work on cross-correlations
            # . .... this is currently a bit messy
            # Trying to loop through unique station-station pairs and apply the correct shift
            counter = 0
            non_autocorr = []
            dshiftG = np.zeros([300,6])
            for sta0 in range(nsta):
                for sta1 in range(nsta):
                    this_id = sta0*nsta+sta1
                    if(sta1>=sta0):
                        # Design a list that excludes autocorrelations
                        if(sta1>sta0):
                            non_autocorr.append(counter)                    
                        relative_timeshift = timeshifts[sta1] - timeshifts[sta0]
                        dshiftG[:,counter] = specshift(datacc[:,this_id],-relative_timeshift/p.dt)[:,0]
                        counter += 1  


            
            A = np.cov(dshiftD.T) 
            PD[idx,idy] = np.sum( A )
            
            np.fill_diagonal(A,0) # Exclude autocorrelations
            PDna[idx,idy] = np.sum( A )
            
            ###################################################
            # Gosia's cross-correlation sum
            stack = np.mean(dshiftG.T,axis=0)
            A = np.max(stack) #MFP
            PG[idx,idy] = A
            
            A = stack[150] #SSA, the stack amplitude at zero-lag time 
            PS[idx,idy] = A
            
            ###################################################
            # Gosia's cross-correlation sum, excluding auto-correlations
            stack = np.mean(dshiftG[:,non_autocorr].T,axis=0)
            A = np.max(stack) #MFP
            PGna[idx,idy] = A
            
            A = stack[150] #SSA, the stack amplitude at zero-lag time 
            PSna[idx,idy] = A
            
            
            if(plot):
                if(idx==ix0 and idy==iy0): # Only the slowness pixel that we care about

                    plt.figure(figsize=(10,5))
                    plt.plot(traw,data,'--')
                    ax=plt.gca()
                    ax.set_xlim([50,80])
                    plt.legend(['0','1','2'])
                    plt.title('Unshifted')
                    plt.show()                

                    plt.figure(figsize=(10,5))
                    plt.plot(traw,dshiftD)
                    ax=plt.gca()
                    ax.set_xlim([50,80])
                    plt.title('Properly Shifted Relative to Mean')
                    plt.show()
                    A = np.cov(dshiftD.T)
                    print("Cov matrix:")
                    print(A)
                    
                    plt.figure(figsize=(10,5))
                    plt.plot(dshiftG)
                    plt.plot(stack,linewidth=2,color='black')
                    ax=plt.gca()
                    ax.set_xlim([0,300])
                    plt.title('Properly Shifted Correlations')
                    plt.show()
                    print("value of stack at center:")
                    print(stack[150])
                    print("value of max of stack:")
                    print(np.max(stack))

                if(idx==ixB and idy==iyB): # Only the slowness pixel that we care about

                    plt.figure(figsize=(10,5))
                    plt.plot(traw,data,'--')
                    ax=plt.gca()
                    ax.set_xlim([50,80])
                    plt.legend(['0','1','2'])
                    plt.title('Unshifted')
                    plt.show()                

                    plt.figure(figsize=(10,5))
                    plt.plot(traw,dshiftD)
                    ax=plt.gca()
                    ax.set_xlim([50,80])
                    plt.title('Wrongly Shifted Relative to Mean')
                    plt.show()
                    A = np.cov(dshiftD.T)
                    print("Cov matrix:")
                    print(A)
                    
                    plt.figure(figsize=(10,5))
                    plt.plot(dshiftG)
                    plt.plot(stack,linewidth=2,color='black')
                    ax=plt.gca()
                    ax.set_xlim([0,300])
                    plt.title('Wrongly Shifted Correlations')
                    plt.show()
                    print("value of stack at center:")
                    print(stack[150])
                    print("value of max of stack:")
                    print(np.max(stack))


# +
mpl.rc('font', **{'size':20})

def plot_P(P,title="MFP",save=0,savename='none',cmax=0,scaling='about0'):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_axes([0.1,0.1,0.6,0.6])
    cmap = plt.get_cmap('RdBu')
    i = plt.pcolor(x_line-p.dx/1000/2,y_line-p.dy/1000/2,np.real(P.T),cmap='RdBu',rasterized=True)#,vmin=-4,vmax=4)
    if(cmax==0):
        cmax=np.max(np.real(P))
    if(scaling=='about0'):
        plt.clim(-cmax,cmax)
    elif(scaling=='positive'):
        plt.clim(0,cmax)
    #elif(scaling=='none'):
        
    plt.xlabel('Distance [km]')
    plt.ylabel('Distance [km]')
    plt.plot(x0/1000.0,y0/1000.0,'r*')
    plt.plot(xB/1000.0,yB/1000.0,'k*')
    plt.plot(xx,yy,'k^')
    plt.title(title)
    plt.axis('equal')
    ax.tick_params(top=True,right=True)

    colorbar_ax = fig.add_axes([0.75, 0.1, 0.03, 0.6])
    fig.colorbar(i, cax=colorbar_ax)
    if(save==1):
        plt.savefig(savename, bbox_inches='tight')
    plt.show()

plot_P(PD, title="MFP (with autocorr) - Daniel's",                           scaling='positive',cmax=0,save=1,savename="3_MFP_DanielB.pdf")
# plot_P(PD, title="MFP (with autocorr) - Daniel's",                           scaling='positive',cmax=0.8e-27,save=1,savename="3_MFP_DanielB.pdf")
plot_P(PG, title="MFP (with autocorr) - Gosia, shifting CC, max of stack",   scaling='positive',save=1,savename="3_MFP_Gosia.pdf")
plot_P(PS, title="SSA (with autocorr) - Gosia, shifting CC, zero-lag stack", scaling='positive',save=1,savename="3_SSA.pdf")

plot_P(PDna, title="MFP (no autocorr) - Daniel's'",                        scaling='about0',save=1,savename="3_MFP_Daniel_na.pdf")
plot_P(PGna, title="MFP (no autocorr) - Gosia, shifting CC, max of stack", scaling='positive',save=1,savename="3_MFP_Gosia_na.pdf")
plot_P(PSna, title="SSA (no autocorr) - Gosia, shifting CC, max of stack", scaling='about0',save=1,savename="3_SSA_na.pdf")

# TODO, pixels are represented in llcorner, shift?



# -
np.save('3_PD', PD)
np.save('3_PG', PG)
np.save('3_PS', PS)
np.save('3_PDna', PDna)
np.save('3_PGna', PGna)
np.save('3_PSna', PSna)




# +
# More plotting, to see how individual waveforms line up, or don't line up.


# We know where the source actually is:
x0=0.75*p.xmax
y0=0.75*p.ymax
ix0=(np.abs(x_line*1000.0-x0)).argmin()
iy0=(np.abs(y_line*1000.0-y0)).argmin()

# Let's also examine a "Bad" point
xB=1.1*p.xmax
yB=0.7*p.ymax
ixB=(np.abs(x_line*1000.0-xB)).argmin()
iyB=(np.abs(y_line*1000.0-yB)).argmin()



# here use index ix0,iy0 for the correct source, or ixB,iyB for the Bad source.
# distances = np.sqrt( (xx-x_line[ixB])**2 + (yy-y_line[iyB])**2 )
distances = np.sqrt( (xx-x_line[ix0])**2 + (yy-y_line[iy0])**2 )
timeshifts = distances / rvel

# Relative timeshift, the difference that is actually needed to shift
timeshifts_relative_to_mean = timeshifts - np.mean(timeshifts)

# Work on direct time traces.
dshiftD = specshift(data,timeshifts_relative_to_mean/p.dt)

# Work on cross-correlations
# . .... this is currently a bit messy
# Trying to loop through unique station-station pairs and apply the correct shift
counter = 0
dshiftG = np.zeros([300,6])

counter = 0
non_autocorr = []
autocorr = []
dshiftG = np.zeros([300,6])
for sta0 in range(nsta):
    for sta1 in range(nsta):
        this_id = sta0*nsta+sta1
        if(sta1>=sta0):
            # Design a list that excludes autocorrelations
            if(sta1>sta0):
                non_autocorr.append(counter)
            relative_timeshift = timeshifts[sta1] - timeshifts[sta0]
            dshiftG[:,counter] = specshift(datacc[:,this_id],-relative_timeshift/p.dt)[:,0]
            counter += 1  

plt.figure(figsize=(10,5))
plt.plot(traw,data,'--')
ax=plt.gca()
ax.set_xlim([50,80])
plt.legend(['0','1','2'])
plt.title('Unshifted')
plt.show()                

plt.figure(figsize=(10,5))
plt.plot(traw,dshiftD)
ax=plt.gca()
ax.set_xlim([50,80])
plt.title('Wrongly Shifted Relative to Mean')
plt.show()
A = np.cov(dshiftD.T)
print("Cov matrix:")
print(A)

stack = np.mean(dshiftG.T,axis=0)
plt.figure(figsize=(10,5))
plt.plot(dshiftG)
plt.plot(stack,linewidth=2,color='black')
ax=plt.gca()
ax.set_xlim([0,300])
plt.title('Wrongly Shifted Correlations - all')
plt.show()
print("value of stack at center:")
print(stack[150])
print("value of max of stack:")
print(np.max(stack))


#non_autocorr=[1,2,4]

stack = np.mean(dshiftG[:,non_autocorr].T,axis=0)
plt.figure(figsize=(10,5))
plt.plot(dshiftG[:,non_autocorr])
plt.plot(stack,linewidth=2,color='black')
ax=plt.gca()
ax.set_xlim([0,300])
plt.title('Wrongly Shifted Correlations - excluding autocorr.')
plt.show()
print("value of stack at center (SSA):")
print(stack[150])
print("value of max of stack (MFP):")
print(np.max(stack))

# -

print(non_autocorr)


