import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import sys
from scipy import signal

from electrodiffusion import solveEquation #(Ions,lambda_n, N_t, delta_t, N, delta_x): return(Phi_of_t) NB: Ions are changed in place
from electrodiffusion import Ion # (self, c_init, D, z, name ):
from electrodiffusion import integrate
from electrodiffusion import ConcentrationProfile # (values, c_values, delta_x, Na_base, K_base, name)
from electrodiffusion import plotIons #(Ions, x, filename):
from electrodiffusion import makePSD #(Phi_of_t, N_t, delta_t): return(f, psd_max, location)
from electrodiffusion import electroneutrality #(Ions, N, plot = 'false' ):	return(valence_times_concentration/(norm_factor))
from electrodiffusion import makeAkses # (parameters):return(t,x)

#------------------------------------------------------------------------------
# This is a script using the data from halnes2016 to model the electrodiffusion
# to simulate the diffusion potential for the full model and for the scenarios 1--5

if __name__=="__main__":

# load the data from Halnes 2016
	K  = np.load("data_cK.npy")
	Na = np.load("data_cNa.npy")
	Ca = np.load("data_cCa.npy")
	X  = np.load("data_cX.npy")

	N_t = 10000            # t_final = N_t * delta_t
	delta_t = 1/100        # delta_t in seconds
	delta_x = 1/100000     # delta_x i meters
	N_x =  len(K)*10
	x_values = np.linspace(0,14,num = 15)*10

	c_K = 0.001*np.interp(np.linspace(0, N_x-1, num = N_x), x_values, K)
	c_Na = 0.001*np.interp(np.linspace(0, N_x-1, num = N_x), x_values, Na)
	c_Ca = 0.001*np.interp(np.linspace(0, N_x-1, num = N_x), x_values, Ca)
	c_X = 0.001*np.interp(np.linspace(0, N_x-1, num = N_x), x_values, X)
# valence: 
	zNa = 1                                         
	zCl = -1
	zK  = 1
	zCa = 2

# diffusion constants: 
	DNa = 1.33e-9  
	DCl = 2.03e-9
	DK  = 1.96e-9
	DCa = 0.71e-9
# tortuosity
	lambda_n = 1.6 

# Phi is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V 
	Psi = 8.3144598*310/(96485.333)
	
# -----------------------------------------------------------------------------
# initialize ions

#  The full model with Ca²+
	Ions0 = [Ion(c_Na, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(c_X, DCl, zCl, 'X-' ), Ion(c_Ca, DCa, zCa, 'Ca2+')]

# Delta K⁺ + Delta Na⁺ = 0 | Delta Cl⁻ = 0
	Ions1 =[Ion(-c_K + .153, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(np.ones(N_x)*.153, DCl, zCl, 'Cl-' )]

# Delta K⁺ + .5* Delta Na⁺ = .5* Delta Cl⁻
	Ions2 = [Ion(-.5*c_K + .1515, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(.5*c_K + .1515, DCl, zCl, 'Cl-' )]

# Delta K⁺ - Delta Cl⁻ = 0 | Delta Na⁺ = 0
	Ions3 = [Ion(np.ones(N_x)*.150, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(c_K + .150, DCl, zCl, 'Cl-' )]

# Delta K⁺ =  -2* Delta Na⁺ | Delta K⁺ = -Delta Cl⁻
	Ions4 = [Ion(-2*c_K + .156, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(-c_K + .156, DCl, zCl, 'Cl-' )]

# Delta K⁺ + Delta Na⁺ = Delta Cl⁻
	Ions5 = [Ion(c_Na, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(c_Na + c_K, DCl, zCl, 'Cl-' )]


	Scenarios = [Ions0, Ions1, Ions2, Ions3, Ions4, Ions5]

	Phi = [] # to store the solutions Phi_of_t
	PSD = np.zeros((len(Scenarios), int(N_t/2 +1))) # to store the PSD_max
	i = 0
	
# check that the system is electroneutral initially:
# NB: the full model is not electroneutral, because it has current sources
	j = 0
	for S in Scenarios:
#		plotIons(M, x_values, 'ions_before%d' %j)
		el_sum = electroneutrality(S, N_x)
		print(j,np.amax(el_sum))
		j+= 1

#	location = 20 #NB: this is only used for comparing PSDs at different compartments!
	for S in Scenarios:
		Phi_of_t, c_of_t = solveEquation(S, lambda_n, N_t, delta_t, N_x, delta_x)
		Phi_of_t = Phi_of_t*Psi* 1000
		Phi.append(Phi_of_t)
		print(i,'phimax', np.amax(np.abs(Phi_of_t[:,0])) )
		f, psd, location = makePSD(Phi_of_t, N_t, delta_t)
#		f, psd = signal.periodogram(Phi_of_t[location,:], 1/delta_t) # for comparing PSDs at different compartments
		print(i,location) # to check that the same compartment is used
		PSD[i,:] = psd

# plot PSD:
		plt.plot(np.log10(f[1:-1]), np.log10(psd[1:-1]), label = i)

# plot the potential at t = 0:
#		plt.plot(np.linspace(0, (N_x-1)/100, num = N_x-1), Phi_of_t[:,0], label = i)
		i += 1
	location = location/100

# NB: the full model is not electroneutral, because it has current sources
	j = 0
	for S in Scenarios:
		el_sum = electroneutrality(S, N_x)
		print(j,np.amax(el_sum))
		j+= 1


# to plot the PSDs, the plots in figure 4.3
	plt.title('PSD of the diffusion potential at depth x = %.2f mm' %location)
	plt.xlabel('$log_{10}(frequency) $ in Hz') # frequency is measured in Hz
	plt.ylabel('$log_{10}(PSD) $ in (mV)²/Hz') # PSD is measured in (mV)²/Hz
	plt.legend()
	plt.savefig('psd_scenarios', dpi = 500)
	plt.show()
	sys.exit()

# to plot the potential at t = 0. the polt in figure 4.2
	plt.title('$\Phi (x,0)$')
	plt.xlabel('cortical depth in mm')
	plt.ylabel('$\Phi$ in mV')
	plt.legend()
	plt.savefig('init_c_scenarioes', dpi=500)
	plt.show()
	sys.exit()



# -----------------------------------------------------------------------------
