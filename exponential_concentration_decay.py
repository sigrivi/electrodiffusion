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


from electrodiffusion import solveEquation #(Ions,lambda_n, N_t, delta_t, N, delta_x, tau=0): return(Phi_of_t, c_of_t) NB: Ions are changed in place
from electrodiffusion import Ion # (self, c_init, D, z, name ):
from electrodiffusion import plotIons #(Ions, x, filename):
from electrodiffusion import plotPhi #(Phi_of_t, parameters, title):
from electrodiffusion import makePSD # (Phi_of_t, N_t, delta_t): return(f, psd_max, location)
from electrodiffusion import ConcentrationProfile # (values, c_values, delta_x, Na_base, K_base, name)
from electrodiffusion import electroneutrality

# This is a program that use the [K⁺] profile from Gratiy2017
# Calculates the diffusion potential and plots the PSDs of the electrodiffusive
# model, and of models where the concentrations decay with a set of time constants,
# not due to diffusion

# use this program to make figure in section "Time constant of the [K⁺] decay"

#------------------------------------------------------------------------------
if __name__=="__main__":

	N_t = 10000            # t_final = N_t * delta_t
	delta_t = 1/100        # delta_t in seconds
	delta_x = 1/100000     # delta_x i meters

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
	x_values = np.linspace(0,26,num = 27)*10

	Gratiy2017 = ConcentrationProfile(x_values, c_values, delta_x, 150, 3, 'Gratiy2017')


# 3 Nicholson1987
#	Gratiy2017 = ConcentrationProfile([0,5,10,15,20,25,30,35,40,45,50,60,70,75], [0,4.4,4.0,3.0,2.3, 1.7,1.3,1.,1.2,1.0, 0.8, 0.7,0.5, 0], delta_x, 150, 3, 'Nicholson1987')


	for i in [0, 1 ,10, 20, 50, 150,250]: # set time constants here
		Ions = [Ion(Gratiy2017.c_Na,DNa,zNa,'$Na^+$'),Ion(Gratiy2017.c_K, DK, zK,'$K^+$' ),Ion(Gratiy2017.c_Cl, DCl, zCl,'$Cl^-$' )]

		Phi_of_t, c_of_t = solveEquation(Ions, lambda_n, N_t, delta_t, Gratiy2017.N_x, delta_x, tau =i)
		Phi_of_t = Phi_of_t*Psi*1000

#		plotPhi(Phi_of_t, [N_t, delta_t, Gratiy2017.N_x, delta_x], 'tau')
#	parameters = [N_t, delta_t, N_x, delta_x]

		el_sum = electroneutrality(Ions, Gratiy2017.N_x) # true = 'true' if you want to plot
		assert np.amax(el_sum) < 1.e-13       # unit test

		f, psd, location = makePSD(Phi_of_t, N_t, delta_t)
		if i == 0: # tau = 0 corresponds to the electrodiffusive model
			plt.plot(np.log10(f[1:-1]), np.log10(psd[1:-1]), 'xkcd:indigo', label = 'KNP')
		else:
			plt.plot(np.log10(f[1:-1]), np.log10(psd[1:-1]), label = '\u03C4 = %d'%i)
		print(location)
		plt.legend()
	plt.title('PSD of the diffusion potential (Cordingley1978)' )
	plt.xlabel('$log_{10}(frequency)$ in Hz')
	plt.ylabel('$log_{10}(PSD)$ in (mV)²/Hz')
	plt.savefig('exponential_decay_Cordingley', dpi=500)
	plt.show()