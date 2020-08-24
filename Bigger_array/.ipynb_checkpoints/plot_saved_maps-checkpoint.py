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
sys.path.append('../../GI1_v2/')
sys.path.append('../../GI1_v2/PLOT/')

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

#- Slowness grid.
# (green's functions programmed at 3000m/s -> 0.33s/km)
# number of pixes in x and y
sl=.75 # second/km

nux = 101
nuy = 101
ux = np.linspace(-sl,sl,nux)
uy = np.linspace(-sl,sl,nuy)
dux=ux[1]-ux[0]
duy=uy[1]-uy[0]

# Define the index of actual noise source, for verification
# (as defined in source.py)
expected_slowness = 1/3.0
sx = expected_slowness*np.sqrt(2)/2
sy = expected_slowness*np.sqrt(2)/2
ix0=(np.abs(ux-sx)).argmin()
iy0=(np.abs(uy-sy)).argmin()


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



# +
mpl.rc('font', **{'size':20})

def plot_beam(P, title="Beamform",save=0,savename='none',cmin=0,cmax=0):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_axes([0.1,0.1,0.6,0.6])  #x0,y0,dx,dy
    cmap = plt.get_cmap('inferno')
    i = plt.pcolor(ux-dux/2,uy-duy/2,np.real(P.T),cmap=cmap,rasterized=True)#,vmin=-4,vmax=4)
    if(cmax==0):
        cmax=np.max(P)
    plt.clim(cmin,cmax)
    plt.axis('equal')
    plt.axis('tight')
    plt.xlim(min(ux)+dux,max(ux)-dux)
    plt.ylim(min(uy)+duy,max(uy)-duy)
    plt.xlabel('Slowness East-West [s/km]')
    plt.ylabel('Slowness North-South [s/km]')
    ax.tick_params(top=True,right=True)
    plt.plot(sx,sy,'k*')
    plt.plot([np.min(ux), np.max(ux)],[0,0],'k')
    plt.plot([0,0],[np.min(uy), np.max(uy)],'k')
    plt.title(title)
    colorbar_ax = fig.add_axes([0.75, 0.1, 0.03, 0.6])  #x0,y0,dx,dy
    fig.colorbar(i, cax=colorbar_ax)
    if(save==1):
        plt.savefig(savename, bbox_inches='tight')
    plt.show()

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
    #plt.axis('tight')

    ax.tick_params(top=True,right=True)

    colorbar_ax = fig.add_axes([0.75, 0.1, 0.03, 0.6])
    fig.colorbar(i, cax=colorbar_ax)
    if(save==1):
        plt.savefig(savename, bbox_inches='tight')
    plt.show()

def plot_K(K,title="MFP",save=0,savename='none',cmax=0):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_axes([0.1,0.1,0.6,0.6])
    cmap = plt.get_cmap('RdBu')
#     i = plt.pcolor((x-p.dx/2)/1000,(y-p.dy/2)/1000,K,cmap='RdBu',rasterized=True)#,vmin=-4,vmax=4)
    i = plt.pcolor(x-(p.dx/2/1000),y-(p.dy/2/1000),K,cmap='RdBu',rasterized=True)#,vmin=-4,vmax=4)
    if(cmax==0):
        cmax=np.max(np.abs(K))*0.15
    plt.clim(-cmax,cmax)
    plt.xlabel('Distance [km]')
    plt.ylabel('Distance [km]')
    plt.plot(x0/1000.0,y0/1000.0,'r*')
    plt.plot(xx,yy,'k^')
    plt.title(title)
    plt.axis('equal')
    ax.tick_params(top=True,right=True)
    colorbar_ax = fig.add_axes([0.75, 0.1, 0.03, 0.6])
    fig.colorbar(i, cax=colorbar_ax)
    if(save==1):
        plt.savefig(savename, bbox_inches='tight')
    plt.show()


# +
P1 = np.load('1_P.npy')
P2 = np.load('2_P.npy')
P2_surfonly = np.load('2_P_surfonly.npy')

# PD = np.load('3_PD.npy')
# PG = np.load('3_PG.npy')
# PS = np.load('3_PS.npy')
# PDna = np.load('3_PDna.npy')
# PGna = np.load('3_PGna.npy')
# PSna = np.load('3_PSna.npy')

K4_sum = np.load('4_Ksum.npy')


# +
mpl.rc('font', **{'size':20})


# ## From 1_Beamform
plot_beam(P1,title="Standard Beamform (timeshift and correlate)",save=1,savename="1_standard_beamform.pdf")

# ## From 2_first_correlate_then_beamform
plot_beam(P2,title="Correlate, then Beamform (all information)",save=1,savename="2_correlation_beamform.pdf")
plot_beam(P2_surfonly,title="Restrict to Surface Wave Windows",save=1,savename="2a_surface_waves_only.pdf")

## From 3_MFP
# plot_P(PD, title="MFP (with autocorr) - Daniel's",                           scaling='positive',cmax=0,save=1,savename="3_MFP_DanielB.pdf")
# plot_P(PG, title="MFP (with autocorr) - Gosia, shifting CC, max of stack",   scaling='positive',save=1,savename="3_MFP_Gosia.pdf")
# plot_P(PS, title="SSA (with autocorr) - Gosia, shifting CC, zero-lag stack", scaling='positive',save=1,savename="3_SSA.pdf")
# plot_P(PDna, title="MFP (no autocorr) - Daniel's'",                        scaling='about0',save=1,savename="3_MFP_Daniel_na.pdf")
# plot_P(PGna, title="MFP (no autocorr) - Gosia, shifting CC, max of stack", scaling='positive',save=1,savename="3_MFP_Gosia_na.pdf")
# plot_P(PSna, title="SSA (no autocorr) - Gosia, shifting CC, max of stack", scaling='about0',save=1,savename="3_SSA_na.pdf")

## From 4_MFP_by_kerneling
plot_K(K4_sum,title="Source kernel [unit of measurment / m^2]",save=0,savename="4_MFP_by_kernels.pdf")







# -


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


