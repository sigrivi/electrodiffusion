import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import sys
from scipy import signal

from eldiff_ions import solveEquation
from eldiff_ions import Ion
from eldiff_ions import integrate
from eldiff_ions import ConcentrationProfile # (values, c_values, delta_x, Na_base, K_base, name)
from eldiff_ions import plotIons #(Ions, x, filename):
from eldiff_ions import makePSD #(Phi_of_t, N_t, delta_t):

# This is a script using the data from halnes2016 to model the electrodiffusion
# with and without Ca2+. I have found that the PSD is slgthly higher when 
# ignoring Ca2+, and that the higest amplitude is in compartent 6 when Ca2+ is 
# included, and in compartment 2 when it is not included.

if __name__=="__main__":


	c_K  = .001*np.load("data_cK.npy")
	c_Na = .001*np.load("data_cNa.npy")
	c_Ca = .001*np.load("data_cCa.npy")
	c_X  = .001*np.load("data_cX.npy")

	delta_c = np.load('halnes_delta_c.npy') # NB: delta_c is in mM !!!
	x_values = np.linspace(0,14,num = 15)


	N_t = 100000          # t_final = N_t * delta_t
	delta_t = 1/1000      # delta_t in seconds
	delta_x = 1/10000     # delta_x i meters
	N_x =  len(c_K)

# tortuosity
	lambda_n = 1.6 

# Phi is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V 
	Psi = 8.3144598*310/(96485.333)
	
# -----------------------------------------------------------------------------

# initialize ions
	Ions_with_Ca = [Ion(c_Na, 1.33e-9, 1, 'Na+'), Ion(c_K, 1.96e-9, 1, 'K+'), \
     		        Ion(c_Ca, .71e-9, 2, 'Ca2+'), Ion(c_X, 2.03e-9, -1, 'X-' )]

	Halnes2016 = ConcentrationProfile(x_values, delta_c, delta_x, 150, 3, 'Halnes2016') # NB: input is in mM !!!

	Ions_without_Ca = [Ion(Halnes2016.c_Na, 1.33e-9, 1, 'Na+'), Ion(Halnes2016.c_K, 1.96e-9, 1, 'K+'), \
     		           Ion(Halnes2016.c_Cl, 2.03e-9, -1, 'Cl-' )]

# solve the equation
	[K_with, Na_with, Ca_with, X_with], Phi_of_t_with_Ca = solveEquation(Ions_with_Ca, lambda_n, N_t, delta_t, N_x, delta_x)
	[K_without, Na_without, Cl_without], Phi_of_t_without_Ca = solveEquation(Ions_without_Ca, lambda_n, N_t, delta_t, N_x, delta_x)

	plotIons([K_with, Na_with, Ca_with, X_with], x_values, 'with_Ca')
	plotIons([K_without, Na_without, Cl_without], x_values, 'without_Ca')

# Phi_of_t is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V
# to get Phi_of_t in mV: *1000
	Phi_of_t_with_Ca = Phi_of_t_with_Ca*Psi*1000
	Phi_of_t_without_Ca = Phi_of_t_without_Ca*Psi*1000


# FFT of the potential at location
	f, PSD_with_Ca, location_with_Ca = makePSD(Phi_of_t_with_Ca, N_t, delta_t)
	f, PSD_without_Ca, location_without_Ca = makePSD(Phi_of_t_without_Ca, N_t, delta_t)
	print(location_with_Ca)
	print(location_without_Ca)

	plt.plot(np.log10(f[1:-1]),np.log10(PSD_with_Ca[1:-1]), label='PSD_with_Ca')
	plt.plot(np.log10(f[1:-1]),np.log10(PSD_without_Ca[1:-1]), label='PSD_without_Ca')
	plt.legend()
	plt.show()