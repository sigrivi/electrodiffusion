import numpy as np
import matplotlib.pyplot as plt
import sys


from electrodiffusion import solveEquation #(Ions,lambda_n, N_t, delta_t, N, delta_x): return(Phi_of_t, c_of_t) NB: Ions are changed in place
from electrodiffusion import Ion # (self, c_init, D, z, name ):
from electrodiffusion import integrate
from electrodiffusion import ConcentrationProfile # (values, c_values, delta_x, Na_base, K_base, name)
from electrodiffusion import plotIons #(Ions, x, filename):
from electrodiffusion import makePSD #(Phi_of_t, N_t, delta_t): return(f, psd_max, location)
from electrodiffusion import electroneutrality #(Ions, N, plot = 'false' ):
from electrodiffusion import makeAkses # (parameters):return(t,x)
from electrodiffusion import plotPhi #(Phi_of_t, parameters, title):
from electrodiffusion import PSD_of_LFP #(): return() This fuction plots the PSD of the LFP

# this program is used for plotting the PSD reuslting from SD parameters
# I used this program to plot the plots in figure 4.10
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

# 4 Herreras1993 Spreading Depression
	Herreras1993 = ConcentrationProfile([0,10,20,30,40,50,60,70,80,100], [0,3,7,50,45,45,45,45,30,0], delta_x, 150, 3, 'Herreras1993')
# I have used the ConcentrationProfile class because it makes delta_c and N_x for me.

	c_Na = -2*Herreras1993.delta_c + .150
	c_K = Herreras1993.delta_c + .003
	c_Cl = -Herreras1993.delta_c + .153
	N_x = Herreras1993.N_x

	

	for i in [0,10,30,60]: # i=0 corresponds to KNP formalism, i>0 corresponds to ED with tau=i
		Ions =[Ion(c_Na, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(c_Cl, DCl, zCl, 'Cl-' )]
		Phi_of_t, c_of_t = solveEquation(Ions, lambda_n, N_t, delta_t, N_x, delta_x, tau = i)
		Phi_of_t = Phi_of_t*Psi*1000

# check electroneutrality
		el_sum = electroneutrality(Ions, N_x) # plot = 'true' if you want to plot 
		assert np.amax(el_sum) < 1.e-13       # unit test
#		plotPhi(Phi_of_t, [N_t, delta_t, N_x, delta_x], 'Herreras1993')

# find PSD_max		
		f, psd, location = makePSD(Phi_of_t, N_t, delta_t)

# plot the PSD of KNP diff. pot and of the ED diff.pot
		if i == 0: # tau = 0 corresponds to the electrodiffusive model
			plt.plot(np.log10(f[31:-1]), np.log10(psd[31:-1]), 'xkcd:plum',label = 'KNP')
		else:
			plt.plot(np.log10(f[31:-1]), np.log10(psd[31:-1]), label = '\u03C4 = %d s'%i)
		print(location)

	PSD_of_LFP()

	plt.legend()
	plt.title('PSD of the SD diffusion potential' )
	plt.xlabel('$log_{10}(frequency)$ in Hz')
	plt.ylabel('$log_{10}(PSD)$ in (mV)²/Hz')
	plt.savefig('sd', dpi=500)
	plt.show()