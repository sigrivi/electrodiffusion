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


from electrodiffusion import solveEquation #(Ions,lambda_n, N_t, delta_t, N, delta_x): return(Phi_of_t) NB: Ions are changed in place
from electrodiffusion import Ion # (self, c_init, D, z, name ):
from electrodiffusion import plotIons #(Ions, x, filename):
from electrodiffusion import plotPhi #(Phi_of_t, parameters, title):

#------------------------------------------------------------------------------
if __name__=="__main__":

	N_t = 10000          # t_final = N_t * delta_t
	delta_t = 1/100      # delta_t in seconds
	delta_x = 1/10000     # delta_x i meters
	N_x = 10

# valence: 
	zNa = 1                                         
	zCl = -1
	zK = 1

# diffusion coefficients: 
	DNa = 1.33e-9
#	DCl = DNa
	DCl = 2.03e-9
	DK = 1.96e-9

	D = 2*DNa*DCl/(DNa+DCl) # joint diffusion coefficient
#	D = (DNa + DCl)/2

# tortuosity
	lambda_n = 1.6 

# Phi is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V 
	Psi = 8.3144598*310/(96485.333)

# -----------------------------------------------------------------------------

#	delta_c = .003*np.sin(np.pi *x)*np.exp(-100*(x-.5)**2)
	c_max = .003                # maximum deviation from baseline concentration
	L = (N_x-1)*delta_x                                       #length of the system
	x = np.linspace(0, L, num = N_x)
	t = np.linspace(0, delta_t*N_t, num = N_t)                    # time vector
	delta_c = c_max*np.sin(np.pi*x/L)          # initial concentration deviation
	c_0 = .150*np.ones(N_x) 

	tau = (lambda_n*L)**2/(D*np.pi**2)

	print('tau', tau)

	c_Na = c_0 + delta_c
	c_Cl = c_0 + delta_c

	Ions1 = [Ion(c_Na, DNa, zNa, '$Na^+$'), Ion(c_Cl, DCl, zCl, '$Cl^-$')] # correct diffusion coefficients
	Ions2 = [Ion(c_Na, D, zNa, '$Na^+$'), Ion(c_Cl, D, zCl, '$Cl^-$')] # same diffusion coefficient

# make exact solution of diffusion equation
	c_exact_of_t = np.zeros((N_x, N_t))
	for i in range(1,N_t+1):
		c_exact_of_t[:,i-1] = c_0 + delta_c*np.exp(-i*delta_t/tau)


#	plt.plot(x*1000, c_exact_of_t[:,N_t-1], label = 'c_exact1')
#	plt.plot(Ions[1].c, label = 'Cl1')
	
#	plotIons(Ions, x, 'two_ions1')
	Phi_of_t1, c_of_t1 = solveEquation(Ions1,lambda_n, N_t, delta_t, N_x, delta_x)
	Phi_of_t2, c_of_t2 = solveEquation(Ions2,lambda_n, N_t, delta_t, N_x, delta_x)
	Phi_of_t1 = Phi_of_t1*Psi*1000
	Phi_of_t2 = Phi_of_t2*Psi*1000
#	plotIons(Ions, x, 'two_ions2')

#	plt.plot(x*1000, c_of_t1[:,N_t-1], label = 'Na')
#	plt.plot(x*1000, Ions[1].c, label = 'Cl')
#	plt.legend()
#	plt.show()


	print(c_of_t2[:,N_t-1] == Ions2[1].c)
	print(c_of_t1[:,N_t-1] -Ions1[1].c)
	print(delta_t, np.amax(np.abs(c_of_t1[N_x//2,:] - c_of_t2[N_x//2,:])))
	print(delta_t, np.amax(np.abs(c_of_t1[N_x//2,:] - c_exact_of_t[N_x//2,:]))/c_max)
#	plt.plot(t, c_of_t1[N_x//2,:] - c_of_t2[N_x//2,:])
#	plt.plot(t,c_of_t1[N_x//2,:] - c_exact_of_t[N_x//2,:])
#	plt.show()
	print('max phi', np.amax(Phi_of_t1[:,0]))
	X=x[:-1]+0.5*delta_x
	for i in range(5):

		time = int(i*delta_t*N_t/5)
		plt.plot(X*1000,Phi_of_t1[:, int(i*N_t/5)], label = 't = %d s' %time )
		print('max phi',int(i*N_t/5), np.amax(Phi_of_t1[:,int(i*N_t/5)]))
		print(.10952503128985423*np.exp(-int(i*N_t/500)/tau))

	plt.plot(X*1000, .10952503128985423*np.sin(np.pi*X/(L-delta_x)))
	plt.xlabel('x in mm')
	plt.ylabel('$\Phi(x)$ in mV')
	plt.title('The diffusion potential of the two-ion system at time t')
	plt.legend()
	plt.savefig('two_ions', dpi=500)
	plt.show()
	#plotPhi(Phi_of_t1, [N_t, delta_t, N_x, delta_x], 'Na_and_Cl')
	