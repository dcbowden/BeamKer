import matplotlib.pyplot as plt
import numpy as np
import parameters

def earthquakes(scale=1.0e-9):

	"""
	To be written
	"""

	#==============================================================================
	#- Input.
	#==============================================================================

	p=parameters.Parameters()
	
	#==============================================================================
	#- Plot.
	#==============================================================================

	#- Spatial distribution. ------------------------------------------------------

	font = {'family' : 'sansserif', 'color'  : 'darkred', 'weight' : 'normal', 'size'   : 14,}

	#- Map with earthquake position and magnitude.

	for win in range(p.Nwindows):

		neq=len(p.eq_t[win])

		for i in range(neq):

			plt.plot(p.eq_x[win][i],p.eq_y[win][i],'bo',markersize=3+scale*p.eq_m[win][i])

	#- Superimpose receiver locations.
	for k in range(p.Nreceivers):

		plt.plot(p.x[k],p.y[k],'ro')
		plt.text(p.x[k]+3.0*p.dx,p.y[k]+3.0*p.dx,str(k),fontdict=font)

	plt.axis('image')
	plt.xlim((p.xmin,p.xmax))
	plt.ylim((p.ymin,p.ymax))
	plt.xlabel('x [km]')
	plt.ylabel('y [km]')
	plt.title('spatial distribution of earthquakes')
	plt.show()

	#- Temporal distribution. -----------------------------------------------------

	t_max=0.0
	m_max=0.0

	for win in range(p.Nwindows):

		neq=len(p.eq_t[win])

		for i in range(neq):

			x=p.eq_t[win][i]+float(win)*p.Twindow
			y=p.eq_m[win][i]
			plt.plot(x,y,'ko',markersize=5)
			plt.plot((x,x),(0,y),'k')
			t_max=np.max((t_max,p.eq_t[win][i]+float(win)*p.Twindow))
			m_max=np.max((m_max,p.eq_m[win][i]))

	plt.xlim((-0.1*t_max,1.1*t_max))
	plt.ylim((0.0,1.1*m_max))
	plt.xlabel('t [s]')
	plt.ylabel('s [1 / s]')
	plt.show()