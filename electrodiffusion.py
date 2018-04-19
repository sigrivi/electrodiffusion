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
	V = np.zeros(Nx+1)
	Dx = (xmax-xmin)/Nx
	for i in range(0,Nx-1):
		V[i+1] = Dx*(v[i+1]+v[i])/2 + V[i]
	V[-1] = Dx*v[-1]/2 + V[-2]
	return(V[1:])

def forwardStep(Ions,lambda_n, N_t, delta_t, N_x, delta_x, grad_phi):
	for I in Ions:
		alpha = delta_t*I.D/(delta_x**2*lambda_n**2)
		I.cNew[1:N_x-1] = I.c[1:N_x-1] + alpha*(I.c[2:N_x]-2*I.c[1:N_x-1]+I.c[:N_x-2]) \
					   + delta_x*alpha*I.z/2.*\
					   ((I.c[2:N_x] + I.c[1:N_x-1])*grad_phi[1:N_x-1] - \
					   		(I.c[1:N_x-1] + I.c[:N_x-2])*grad_phi[:N_x-2])
		I.c = I.cNew.copy()

def exponentialDecay(Ions, delta_t, tau): 
	delta_c = Ions[1].c-Ions[1].c[0]
	Ions[1].cNew = delta_c*np.exp(-delta_t/tau) + Ions[1].c[0]
	Ions[0].cNew = -delta_c*np.exp(-delta_t/tau) + Ions[0].c[0]
	Ions[1].c = Ions[1].cNew.copy()
	Ions[0].c = Ions[0].cNew.copy()


def solveEquation(Ions,lambda_n, N_t, delta_t, N_x, delta_x, tau = 0):
	
	Phi_of_t = np.zeros((N_x-1, N_t))
	c_of_t =np.zeros((N_x, N_t)) # for storing the concentrations of one ion species
	x_min = 0
	x_max = N_x*delta_x

	for t in range(N_t):
		sum_1 = np.zeros(N_x-1)
		sum_2 = np.zeros(N_x-1)

		# We only calculate grad phi_(i+1/2), i.e. grad phi at the half-points.
		for I in Ions:
			sum_1 += I.z*I.D*(I.c[1:]-I.c[:-1])
			sum_2 += I.z**2*I.D*(I.c[1:]+I.c[:-1])/2.
		
		grad_phi = (-1./delta_x)*sum_1/sum_2
		Phi = integrate(grad_phi,x_min,x_max)

		Phi_of_t[:,t] = Phi[:]

		if tau == 0:
		# Then we use grad phi at the half-points
			forwardStep(Ions,lambda_n, N_t, delta_t, N_x, delta_x, grad_phi)
		else:
			exponentialDecay(Ions, delta_t, tau)


			
		c_of_t[:,t] = Ions[0].c	
#		if t%(N_t/5) == 0:
#			if t>0:
#				plt.plot(Ions[0].c,label=' t=%.1f' %(t*delta_t))
#	plt.title('sodium concentration')
#	plt.legend()
#	plt.show()
	return(Phi_of_t, c_of_t)

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
		plt.plot(1000*(I.c-I.c[0]*np.ones(len(I.c))),-x, label = I.name)
	plt.title('Deviation from base line concentrations (' + filename +')')
	plt.xlabel('$\Delta c$ (mM)')
	plt.ylabel('cortical depth (mm)')
	plt.legend()
	plt.savefig(filename +'_delta_c',dpi=225)
	plt.show()

def makeAkses(parameters): # parameters = [N_t, delta_t, N_x, delta_x]
	t = np.linspace(0,parameters[0]*parameters[1], num = parameters[0])
	x = np.linspace(0,(parameters[2]-1)*parameters[3]*1000, num=parameters[2]) # NB:  *1000 to get in mm

	return(t,x)

def makePSD(Phi_of_t, N_t, delta_t):
	fs = 1/delta_t # sampling frequency
	psd_max = np.zeros(int(N_t/2 +1))

	for i in range(int(Phi_of_t.shape[0])-1):
	    f, psd_new = signal.periodogram(Phi_of_t[i,:], fs)
	    if np.mean(psd_new[1:-1]) > np.mean(psd_max[1:-1]):
	   		psd_max = psd_new
	   		location = i

	return(f, psd_max, location)

def plotPhi(Phi_of_t, parameters, name):
#	parameters = [N_t, delta_t, N_x, delta_x]
	x_max = (parameters[2]-1)*parameters[3]
	t,x = makeAkses(parameters)
	x = x[:-1] - x_max*1000
#	t = np.linspace(0,parameters[0]*parameters[1], num = parameters[0])
#	x = (np.linspace(0., x_max, num=parameters[2]-1) - x_max)*1000

	Phi_of_t = np.flip(Phi_of_t,0)
	X,Y = np.meshgrid(t,x)
	plt.figure()
	cp = plt.contourf(X,Y,Phi_of_t) #, vmin = -0.14, vmax = 0.04 is not a good idea, because the potiantials differs too much in magnitude
	plt.colorbar(cp)
	plt.xlabel('time (s)')
	plt.ylabel('cortical depth (mm)')
	plt.title('$ \Phi(x,t)$ in mV (' + name + ')')
	plt.savefig(name +'Phi_of_t', dpi =225)
#	print(x)
	plt.show()
#-----------------------------------------------
if __name__=="__main__":
	sys.exit()

