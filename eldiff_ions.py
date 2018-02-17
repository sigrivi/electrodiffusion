import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import time
import random
import sys
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import signal


class Ion:
	def __init__(self, c_init, D, z, name ):
		self.c_init = np.asarray(c_init, dtype=np.float)                             # concentration
		self.c      = c_init.copy()
		self.cNew   = c_init.copy()
		self.D      = D                              # diffusion constant
		self.z      = z 								# valence
		self.name   = name                        # name of the ion

# The use of this class imlplies the assumtion of c_Na + c_K = c_Cl
class ConcentrationProfile:
	def __init__(self, x_values, c_values, delta_x, Na_base, K_base, name): # c_values are in mM
		self.N_x     = int(x_values[-1]) + 1
		self.delta_x = delta_x # delta_x is measured in meters
		self.delta_c = 0.001*np.interp(np.linspace(0, self.N_x-1, num = self.N_x), x_values, c_values)
		self.c_Na    = 0.001*Na_base*np.ones(self.N_x) - self.delta_c
		self.c_K     = 0.001*K_base*np.ones(self.N_x) + self.delta_c
		self.c_Cl    = self.c_Na + self.c_K
		self.name    = name

def integrate(v,xmin,xmax):
	Nx = len(v)
	V = np.zeros(Nx)
	Dx = (xmax-xmin)/Nx
	for i in range(0,Nx-1):
		V[i+1] = Dx*(v[i+1]+v[i])/2 + V[i]
	return(V)


def solveEquation(Ions,lambda_n, N_t, delta_t,N, delta_x):
	
	Phi_of_t = np.zeros((N-1, N_t))
	x_min = 0
	x_max = N*delta_x

	for t in range(N_t):
		sum_1 = np.zeros(N-1)
		sum_2 = np.zeros(N-1)

		# We only calculate grad phi_(i+1/2), i.e. grad phi at the half-points.
		for I in Ions:
			sum_1 += I.z*I.D*(I.c[1:]-I.c[:-1])
			sum_2 += I.z**2*I.D*(I.c[1:]+I.c[:-1])/2.
		
		gradient_of_phi_at_halfpoints = (-1./delta_x)*sum_1/sum_2
		Phi = integrate(gradient_of_phi_at_halfpoints,x_min,x_max)

		Phi_of_t[:,t] = Phi[:]

		# Then we use grad phi at the half-points
		for I in Ions:
			alpha = delta_t*I.D/(delta_x**2*lambda_n**2)
			I.cNew[1:N-1] = I.c[1:N-1] + alpha*(I.c[2:N]-2*I.c[1:N-1]+I.c[:N-2]) \
						   + delta_x*alpha*I.z/2.*\
						   ((I.c[2:N] + I.c[1:N-1])*gradient_of_phi_at_halfpoints[1:N-1] - \
						   		(I.c[1:N-1] + I.c[:N-2])*gradient_of_phi_at_halfpoints[:N-2])
		
			I.c = I.cNew.copy()
	
		if t%(N_t/5) == 0:
			if t>0:
				plt.plot(Ions[0].c,label=' t=%.1f' %(t*delta_t))
	plt.title('sodium concentration')
	plt.legend()
	plt.show()
	return(Ions, Phi_of_t)

def electroneutrality(Ions, N, plot = 'false' ):
	valence_times_concentration = np.zeros(N)
	norm_factor = np.zeros(N)
	for I in Ions:
		valence_times_concentration += I.c*I.z
		norm_factor += I.c*np.abs(I.z)

	if plot == 'true':		
		plt.plot(valence_times_concentration/np.sum(norm_factor),label='el_sum')
		plt.legend()
		plt.show() 

	return(valence_times_concentration/np.sum(norm_factor))

def plotIons(Ions,x):
	for I in Ions:
		plt.plot(x,I.c-I.c[0], label = I.name)
	plt.title('deviation from base line concentrations')
	plt.ylabel('c-c_0 (M)')
	plt.xlabel('cortical depth (mm)')
	plt.legend()
	plt.savefig('delta_c', dpi=225)
	plt.show()

def makeAkses(parameters):
	t = np.linspace(0,parameters[0]*parameters[1], num = parameters[0])
	x = np.linspace(0,(parameters[2]-2)*parameters[3]*1000, num=parameters[2]) # NB: Nx-2 because the ends are not included *1000 to get in mm

	return(t,x)

if __name__=="__main__":

	N_t = 10000          # t_final = N_t * delta_t
	delta_t = 1/1000      # delta_t in seconds
	delta_x = 1/10000     # delta_x i meters

# valence: 
	zNa = 1                                         
	zCl = -1
	zK = 1

# diffusion constants: 
	DNa = 1.33e-9  
	DCl = 2.03e-9
	DK = 1.96e-9

# tortuosity
	lambda_n = 1.6    
# Phi is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V 
	Psi = 8.3144598*310/(96485.333)

                           

#	halnes2016 = ConcentrationProfile(np.asarray([1,2,5,12,14]), np.asarray([0,5,1,2,0]), delta_x, 150, 3, 'halnes2016')
#	N_x = halnes2016.N_x

	c_values = np.asarray(\
		      [0,0.1350844277673544, 0.3827392120075046, 0.48780487804878053, 0.74296435272045, 1.1782363977485926,\
	           1.4859287054409005, 1.5759849906191372, 1.4934333958724202, 1.891181988742964, 1.590994371482176, \
	           1.0281425891181983, 0.6979362101313321, 0.6378986866791744, 0.6378986866791747, 0.6829268292682926,\
	           0.8405253283302062, 0.893058161350844, 0.8930581613508446, 0.8780487804878045, 0.8930581613508448, \
	           0.8255159474671664, 0.908067542213884, 0.8180112570356467, 0.8405253283302068, 0.7804878048780483,0])
	x_values = np.linspace(0,26,num = 27)
	Gratiy2017 = ConcentrationProfile(x_values, c_values, delta_x, 150, 3, 'Gratiy2017')


	N_x = Gratiy2017.N_x

	parameters = [N_t, delta_t, N_x, delta_x]
#	np.save("parameters.npy", parameters)

# vectors for the axes
	t,x = makeAkses(parameters)
# -----------------------------------------------------------------------------

# initialize ions
#	Ions = [Ion(halnes2016.c_Na,DNa,zNa,'Na+'),Ion(halnes2016.c_Cl, DCl, zCl,'Cl-' ),Ion(halnes2016.c_K, DK, zK,'K+' )]
	Ions = [Ion(Gratiy2017.c_Na,DNa,zNa,'Na+'),Ion(Gratiy2017.c_Cl, DCl, zCl,'Cl-' ),Ion(Gratiy2017.c_K, DK, zK,'K+' )]


# check electroneutrality
	el_sum = electroneutrality(Ions,N_x, plot = 'true') # plot = 'true' if you want to plot 

# plot initial ion concentration
	plotIons(Ions,x)

# solve the equation
	[sodium, chloride, potassium], Phi_of_t = solveEquation(Ions, lambda_n, N_t, delta_t, N_x, delta_x)

# Check electroneutrality
	el_sum = electroneutrality(Ions,N_x, plot = 'true') # plot = 'true' if you want to plot

# plot final ion concentration
#	plotIons(Ions,x)

# Phi_of_t is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V
# to get Phi_of_t in mV: *1000
	Phi_of_t = Phi_of_t*Psi*1000

#-----------------------------------------------------------------------------

	sys.exit()

	delta_t = 1/1000
	delta_x = 1/10000
	#t_final = 100
	#x_max = 0.01
	N_t = 100000          # t_final = N_t * delta_t
	N_x = 27               # x_max = N_x*delta_x
	fs = N_t*delta_t*delta_t # sampling rate
	parameters = [N_t, delta_t, N_x, delta_x]
	np.save("parameters.npy", parameters)

# vectors for the axes
	t,x = makeAkses(parameters)
# initial concentrations: 
	c1 = np.ones(N_x)*.150                           
	#c1[N_x//2] = .145
	c2 = np.ones(N_x)*.153
	#c2[N_x//2] = .153
	c3 = np.ones(N_x)*.003                           
	#c3[N_x//2] = .008
# the original delta_c-vector has 25 elements, I added 2 more, to keep it zero at the edges
	delta_c = 0.001*np.asarray(\
		      [0,0.1350844277673544, 0.3827392120075046, 0.48780487804878053, 0.74296435272045, 1.1782363977485926,\
	           1.4859287054409005, 1.5759849906191372, 1.4934333958724202, 1.891181988742964, 1.590994371482176, \
	           1.0281425891181983, 0.6979362101313321, 0.6378986866791744, 0.6378986866791747, 0.6829268292682926,\
	           0.8405253283302062, 0.893058161350844, 0.8930581613508446, 0.8780487804878045, 0.8930581613508448, \
	           0.8255159474671664, 0.908067542213884, 0.8180112570356467, 0.8405253283302068, 0.7804878048780483,0])
	c1 -= delta_c
	c3 += delta_c
# valence: 
	z1 = 1                                         
	z2 = -1
	z3 = 1

# diffusion constants: 
	D1 = 1.33e-9  
	D2 = 2.03e-9
	D3 = 1.96e-9

# tortuosity
	lambda_n = 1.6    
# Phi is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V 
	Psi = 8.3144598*310/(96485.333)

                           
# -----------------------------------------------------------------------------

# initialize ions
	Ions = [Ion(c1,D1,z1,'sodium'),Ion(c2, D2, z2,'chloride' ),Ion(c3, D3, z3,'potassium' )]

# check electroneutrality
	el_sum = electroneutrality(Ions,N_x, plot = 'true') # plot = 'true' if you want to plot 

# plot initial ion concentration
	plotIons(Ions,x)

# solve the equation
	[sodium, chloride, potassium], Phi_of_t = solveEquation(Ions, lambda_n, N_t, delta_t, N_x, delta_x)

# Check electroneutrality
	el_sum = electroneutrality(Ions,N_x, plot = 'true') # plot = 'true' if you want to plot

# plot final ion concentration
#	plotIons(Ions,x)

# Phi_of_t is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V
# to get Phi_of_t in mV: *1000
	Phi_of_t = Phi_of_t*Psi*1000

#------------------------------------------------------------------------------
	
# save Phi(x,t) 
	np.save("Phi_of_t.npy", Phi_of_t)
	


	sys.exit()
# save to textfile
	#np.ndarray.tofile(Phi_of_t,'phi.txt')

	plt.plot(t,Phi_of_t[N_x//2,:])
	plt.ylabel('Phi (mV)')
	plt.xlabel('time (s) ')
	plt.title('Phi(x=%.1f)'%(N_x/2))
	plt.show()




#	sys.exit()

	plt.plot(x[1:], Phi_of_t[:,N_t-1])  # Phi is calculated at half-points => Phi is shorter than x
	plt.ylabel('Phi (mV)')
	plt.xlabel('cortical depth (mm)')
	plt.title('Phi(t=%.1f)'%(delta_t*N_t))
	plt.savefig('phi', dpi=225)
	plt.show()



	X,Y = np.meshgrid(t,x[1:])
	plt.figure()
	cp = plt.contourf(X,Y,Phi_of_t)
	plt.colorbar(cp)
	plt.xlabel('time (s)')
	plt.ylabel('cortical depth (mm)')
	plt.title('Phi (mV)')
	plt.savefig('Phi_X_T', dpi =225)
	plt.show()



	sys.exit()
	plt.plot(t,Phi_of_t[N_x//2,:])
	plt.ylabel('Phi (mV)')
	plt.xlabel('time (s) ')
	plt.title('Phi(x=%.1f)'%(N_x/2))
	
	plt.show()

#phi_average is the average of phi(t) over the whole cortical depth
#	phi_average = np.zeros(N_t)
#	for i in range(N_t-1):
#		phi_average[i] = np.sum(Phi_of_t[:,i])
#	phi_average = phi_average/(N_x-1)

#	plt.plot(t[:-1],phi_average[:-1]) # *1000 to plot in mV
#	plt.title('spatial average of Phi')
#	plt.xlabel('time (s)')
#	plt.ylabel('Phi (mV)')
#	plt.savefig('phi_average', dpi = 225)
#	plt.show()