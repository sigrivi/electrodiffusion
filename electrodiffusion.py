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
from scipy import io




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
		
class Model:
	def __init__(self, phi, parameters, name, color): 
		self.phi     = phi
		self.parameters = parameters # parameters = [N_t, delta_t, N_x, delta_x]
		self.name = name
		self.N_t = parameters[0]
		self.delta_t = parameters[1]
		self.f = np.zeros(int(self.N_t/2 +1))
		self.psd = np.zeros(int(self.N_t/2+1))
		self.location = 0
		self.color = color

def integrate(v,xmin,xmax):
	Nx = len(v)
	V = np.zeros(Nx)
	Dx = (xmax-xmin)/Nx
	for i in range(0,Nx-1):
		V[i+1] = Dx*(v[i+1]+v[i])/2 + V[i]
	return(V)


def solveEquation(Ions,lambda_n, N_t, delta_t, N, delta_x):
	
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
	
#		if t%(N_t/5) == 0:
#			if t>0:
#				plt.plot(Ions[0].c,label=' t=%.1f' %(t*delta_t))
#	plt.title('sodium concentration')
#	plt.legend()
#	plt.show()
	return(Ions, Phi_of_t)

def electroneutrality(Ions, N, plot = 'false' ):
	valence_times_concentration = np.zeros(N)
	norm_factor = np.zeros(N)
	for I in Ions:
		valence_times_concentration += I.c*I.z
		norm_factor += I.c*np.abs(I.z)

	if plot == 'true':		
		plt.plot(valence_times_concentration/(norm_factor),label='el_sum')
		plt.legend()
		plt.show() 

	return(valence_times_concentration/(norm_factor))

def plotIons(Ions, x, filename):
	for I in Ions:
		plt.plot(x,I.c-I.c[0]*np.ones(len(I.c)), label = I.name)
	plt.title('deviation from base line concentrations')
	plt.ylabel('$c-c_0$ (M)')
	plt.xlabel('cortical depth (mm)')
	plt.legend()
	plt.savefig(filename +'_delta_c', dpi=225)
	plt.show()

def makeAkses(parameters):
	t = np.linspace(0,parameters[0]*parameters[1], num = parameters[0])
	x = np.linspace(0,(parameters[2]-2)*parameters[3]*1000, num=parameters[2]) # NB: Nx-2 because the ends are not included *1000 to get in mm

	return(t,x)

def makePSD(Phi_of_t, N_t, delta_t):
	fs = 1/delta_t # sampling frequency
	psd_max = np.zeros(int(N_t/2 +1))

	for i in range(int(Phi_of_t.shape[0])-1):
	    f, psd_new = signal.periodogram(Phi_of_t[i,:], fs)
	    if np.amax(psd_new[1:-1]) > np.amax(psd_max[1:-1]):
	   		psd_max = psd_new
	   		location = i

	return(f, psd_max, location)
#-----------------------------------------------
if __name__=="__main__":

	N_t = 100000          # t_final = N_t * delta_t
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

# -----------------------------------------------------------------------------
# concentration profiles                          



# 0 Gratiy2017
	c_values = np.asarray(\
		      [0,0.1350844277673544, 0.3827392120075046, 0.48780487804878053, 0.74296435272045, 1.1782363977485926,\
	           1.4859287054409005, 1.5759849906191372, 1.4934333958724202, 1.891181988742964, 1.590994371482176, \
	           1.0281425891181983, 0.6979362101313321, 0.6378986866791744, 0.6378986866791747, 0.6829268292682926,\
	           0.8405253283302062, 0.893058161350844, 0.8930581613508446, 0.8780487804878045, 0.8930581613508448, \
	           0.8255159474671664, 0.908067542213884, 0.8180112570356467, 0.8405253283302068, 0.7804878048780483,0])
	x_values = np.linspace(0,26,num = 27)

	Gratiy2017 = ConcentrationProfile(x_values, c_values, delta_x, 150, 3, 'Gratiy2017')

# 1 Halnes2016
	halnes_delta_c = np.load('halnes_delta_c.npy')
	halnes_x_values = np.linspace(0,14,num = 15)
	Halnes2016 = ConcentrationProfile(halnes_x_values, halnes_delta_c, delta_x, 150, 3, 'Halnes2016')

# 2 Dietzel1982
	Dietzel1982_1 = ConcentrationProfile([0,2,4,6,8,10,16,17], [0,7.5,4.5,3.5,5,3,0.5,0], delta_x, 148,3, 'Dietzel1982_1')

# 3 Nicholson1987
	Nicholson1987 = ConcentrationProfile([0,1,2,3,4,5,6,7], [0, 4.4, 2.7, 1.6, 1., 0.8, 0.7, 0], delta_x, 150, 3, 'Nicholson1987')

# 4 EkstremeGradient

# List of all profiles
	Profiles = [Gratiy2017, Halnes2016, Dietzel1982_1, Nicholson1987]

# choose a profile from the list of profiles
	choose_profile = 1

# save the parameters used
	N_x = Profiles[choose_profile].N_x
	parameters = [N_t, delta_t, N_x, delta_x]
	np.save(Profiles[choose_profile].name +"_parameters.npy", parameters)

# vectors for the axes
	t,x = makeAkses(parameters)
# -----------------------------------------------------------------------------

# initialize ions
	Ions = [Ion(Profiles[choose_profile].c_Na,DNa,zNa,'$Na^+$'),Ion(Profiles[choose_profile].c_K, DK, zK,'$K^+$' ),Ion(Profiles[choose_profile].c_Cl, DCl, zCl,'$Cl^-$' )]


# check electroneutrality
	el_sum = electroneutrality(Ions, N_x, plot = 'true') # plot = 'true' if you want to plot 
	assert np.amax(el_sum) < 1.e-14       # unit test

# plot initial ion concentration
	plotIons(Ions, x, Profiles[choose_profile].name)

# solve the equation
	[sodium, chloride, potassium], Phi_of_t = solveEquation(Ions, lambda_n, N_t, delta_t, N_x, delta_x)

# Phi_of_t is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V
# to get Phi_of_t in mV: *1000
	Phi_of_t = Phi_of_t*Psi*1000

	f, psd, location = makePSD(Phi_of_t, N_t, delta_t)
	plt.plot(np.log10(f[1:-1]),np.log10(psd[1:-1]), label=Profiles[choose_profile].name)
	plt.legend()
	plt.savefig('PSD',dpi=225)
	plt.show()

	

# Check electroneutrality
	el_sum = electroneutrality(Ions, N_x, plot = 'true') # true = 'true' if you want to plot
#	assert np.amax(el_sum) < 1.e-14       # unit test

# plot final ion concentration
	plotIons(Ions, x, Profiles[choose_profile].name + '_final')



# save Phi(x,t) 
	np.save( Profiles[choose_profile].name + "_Phi_of_t.npy" , Phi_of_t)

# contour plot of Phi
	X,Y = np.meshgrid(t,x[1:])
	plt.figure()
	cp = plt.contourf(X,Y,Phi_of_t)
	plt.colorbar(cp)
	plt.xlabel('time (s)')
	plt.ylabel('cortical depth (mm)')
	plt.title('$\Phi$ (mV)')
	plt.savefig(Profiles[choose_profile].name + '_Phi_X_T', dpi =225)
	plt.show()

#-----------------------------------------------------------------------------
	sys.exit() 


	plt.plot(x[1:], Phi_of_t[:,1])  # Phi is calculated at half-points => Phi is shorter than x
	plt.ylabel('Phi (mV)')
	plt.xlabel('cortical depth (mm)')
	plt.title('Phi(t=%.1f)'%(delta_t*N_t))
	plt.savefig('phi', dpi=225)
	plt.show()



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
