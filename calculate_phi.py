import numpy as np
import matplotlib.pyplot as plt
import sys

from electrodiffusion import solveEquation #(Ions,lambda_n, N_t, delta_t, N, delta_x): return(Phi_of_t) NB: Ions are changed in place
from electrodiffusion import Ion # (self, c_init, D, z, name ):
from electrodiffusion import integrate
from electrodiffusion import ConcentrationProfile # (values, c_values, delta_x, Na_base, K_base, name)
from electrodiffusion import plotIons #(Ions, x, filename):
from electrodiffusion import makePSD #(Phi_of_t, N_t, delta_t): return(f, psd_max, location)
from electrodiffusion import electroneutrality #(Ions, N, plot = 'false' ):
from electrodiffusion import makeAkses # (parameters):return(t,x)
from electrodiffusion import plotPhi #(Phi_of_t, parameters, title):


# all 1:1 Na/K profiles are stored here. Use this file to calculate Phi(x,t)
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

	Cordingley1978 = ConcentrationProfile(x_values, c_values, delta_x, 150, 3, 'Cordingley1978')

# 1 Halnes2016
	halnes_delta_c = np.load('halnes_delta_c.npy')
	halnes_x_values = np.linspace(0,14,num = 15)*10
	Halnes2016 = ConcentrationProfile(halnes_x_values, halnes_delta_c, delta_x, 150, 3, 'Halnes2016')

# 2 Dietzel1982
	Dietzel1982_1 = ConcentrationProfile([0,6,15,30,52,72,90,99, 114,135,165, 170], [0,5.5,3.5,4.0,5.8,5.9,5.0,5.2,2.6, 4.0, 3.0,0], delta_x, 148,3, 'Dietzel1982')

# 3 Nicholson1987
	Nicholson1987 = ConcentrationProfile([0,5,10,15,20,25,30,35,40,45,50,60,70,75], [0,4.4,4.0,3.0,2.3, 1.7,1.3,1.,1.2,1.0, 0.8, 0.7,0.5, 0], delta_x, 150, 3, 'Nicholson1987')

# 4 Herreras1993 Spreading Depression
	Herreras1993 = ConcentrationProfile([0,10,20,30,40,50,60,70,80,100], [0,3,7,50,45,45,45,45,30,0], delta_x, 150, 3, 'Herreras1993')

# List of all profiles
	Profiles = [Cordingley1978, Halnes2016, Dietzel1982_1, Nicholson1987, Herreras1993]

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
	el_sum = electroneutrality(Ions, N_x) # plot = 'true' if you want to plot 
	assert np.amax(el_sum) < 1.e-14       # unit test

# plot initial ion concentration
	print(Profiles[choose_profile].c_values)
	plotIons(Ions, x, Profiles[choose_profile].name,  Profiles[choose_profile].c_values, Profiles[choose_profile].x_values/100)
	
# solve the equation
	Phi_of_t, c_of_t = solveEquation(Ions, lambda_n, N_t, delta_t, N_x, delta_x)

# Phi_of_t is dimensionless, needs to be multiplied with Psi = RT/F = 0.0267 V
# to get Phi_of_t in mV: *1000

	Phi_of_t = Phi_of_t*1000*Psi

# Check electroneutrality
	el_sum = electroneutrality(Ions, N_x, plot = 'true') # true = 'true' if you want to plot
	assert np.amax(el_sum) < 1.e-13       # unit test

# plot final ion concentration
#	plotIons(Ions, x, Profiles[choose_profile].name + '_final')

# save Phi(x,t) 
	np.save( Profiles[choose_profile].name + "_Phi_of_t.npy" , Phi_of_t)

	plotPhi(Phi_of_t, parameters, Profiles[choose_profile].name )

# -----------------------------------------------------------------------------