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
# with and without Ca2+. I have found that the PSD is slgthly higher when 
# ignoring Ca2+, and that the higest amplitude is in compartent 6 when Ca2+ is 
# included, and in compartment 2 when it is not included.

if __name__=="__main__":


	K  = np.load("data_cK.npy")
	Na = np.load("data_cNa.npy")
	Ca = np.load("data_cCa.npy")
	X  = np.load("data_cX.npy")

	N_t = 1000          # t_final = N_t * delta_t
	delta_t = 1/100      # delta_t in seconds
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

# Delta K⁺ + Delta Na⁺ = Delta Cl⁻
	Ions2 = [Ion(c_Na, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(c_Na + c_K, DCl, zCl, 'Cl-' )]

# Delta K⁺ + .5* Delta Na⁺ = .5* Delta Cl⁻
	Ions3 = [Ion(-.5*c_K + .1515, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(.5*c_K + .1515, DCl, zCl, 'Cl-' )]

# Delta K⁺ - Delta Cl⁻ = 0 | Delta Na⁺ = 0
	Ions4 = [Ion(np.ones(N_x)*.150, DNa, zNa, 'Na+'), Ion(c_K, DK, zK, 'K+'), Ion(c_K + .150, DCl, zCl, 'Cl-' )]

	Ions5 = [Ion(c_Na, DNa, zNa, 'Na+'), Ion(np.ones(N_x)*.003, DK, zK, 'K+'), Ion(c_Na, DCl, zCl, 'Cl-')]

	Ions6 = [Ion(c_Na, DNa, zNa, 'Na+'), Ion(-c_Na+.153, DK, zK, 'K+'), Ion(np.ones(N_x)*.153, DCl, zCl, 'Cl-')]

# 
	Models = [Ions0, Ions1, Ions2, Ions3, Ions4, Ions5, Ions6]
	Names = ['full model', '$1\!:\!1\ K^+\!/Na^+$', '$1\!:\!1\ K^+\!/(Na^+-Cl^-)$', '$1\!:\!(1/2)\ K^+\!/Na^+$', '$1\!:\!1\ K^+\!/Cl^-$', '$1\!:\!1\ Na^+\!/Cl^-$', '$1\!:\!1\ Na^+\!/K^+$']


#	sys.exit()
	Phi = []
	PSD = np.zeros((len(Models), int(N_t/2 +1)))
	i = 0
	j = 0
	for M in Models:
#		plotIons(M, x_values, 'ions_before%d' %j)
		el_sum = electroneutrality(M, N_x)
		print(j,np.amax(el_sum))
		j+= 1
	location = 0 #NB: this is only used for comparing PSDs at different compartments!
	for M in Models:
		Phi_of_t, c_of_t = solveEquation(M, lambda_n, N_t, delta_t, N_x, delta_x)
		Phi_of_t = Phi_of_t*Psi* 1000
		Phi.append(Phi_of_t)
#		f, psd, location = makePSD(Phi_of_t, N_t, delta_t)
#		f, psd = signal.periodogram(Phi_of_t[location,:], 1/delta_t)
#		print(i,location)
#		PSD[i,:] = psd
#		plt.plot(np.log10(f[1:-1]), np.log10(psd[1:-1]), label = Names[i])
#		plt.legend()
		plt.plot(np.linspace(0, (N_x-1)/100, num = N_x-1), Phi_of_t[:,0], label = Names[i])
		i += 1
	plt.title('$\Phi (x,0)$')
	plt.xlabel('cortical depth in mm')
	plt.ylabel('$\Phi$ in mV')
	plt.legend()

	plt.savefig('initial_conditions', dpi=500)
	plt.show()
	sys.exit()
	plt.title('PSD of the diffusion potential')
	plt.xlabel('$log_{10}(frequency)$') # frequency is measured in Hz
	plt.ylabel('$log_{10}(PSD)$ ') # PSD is measured in (mV)²/Hz

	plt.savefig('Ca_contribution', dpi = 500)
	plt.show()


	j = 0
	for M in Models:
#		plotIons(M, x_values, 'ions_after%d' %j)
		el_sum = electroneutrality(M, N_x)
		print(j,np.amax(el_sum))
		print('mean deviation',np.mean(np.log10(PSD[j,1:-1]) - np.log10(PSD[0,1:-1])))
		j+= 1

#	print(np.mean(np.log10(PSD[1,1:-1])))
#	print('mean deviation',np.mean(np.log10(PSD[1,1:-1]) - np.log10(PSD[0,1:-1])))
# contour plot of

# -----------------------------------------------------------------------------

	sys.exit()

	Phi_diff = Phi_of_t_without_Ca - Phi_of_t_with_Ca

# contour plot of Phi
	parameters = [N_t, delta_t, N_x, delta_x]
	t, x = makeAkses(parameters)
	X,Y = np.meshgrid(t,x[1:])
	plt.figure()
	cp = plt.contourf(X,Y,Phi_diff)
	plt.colorbar(cp)
	plt.xlabel('time (s)')
	plt.ylabel('cortical depth (mm)')
	plt.title('difference in $ \Phi$ (mV)')
	plt.savefig('diff_Phi_X_T', dpi =225)
	plt.show()