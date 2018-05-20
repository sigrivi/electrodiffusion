import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from electrodiffusion import makePSD # (Phi_of_t, N_t, delta_t): return(f, psd_max, location)
from electrodiffusion import PSD_of_LFP # () : # make the PSD of the LFP from Gratiy
from electrodiffusion import Model # (self, phi, parameters, name, color): 

# this program is used to calculate and plot the PSDs of the diff.pot.
# and do linear regression on log(PSD)
# use PSD_of_LFP to plot the PSD of the LFP from Gratiy together with the other PSDs

if __name__=="__main__":

	Dietzel1982_1 = Model(np.load('Dietzel1982_Phi_of_t.npy'), np.load('Dietzel1982_parameters.npy'), 'Dietzel1982', 'xkcd:teal') 
	Cordingley1978    = Model(np.load('Cordingley1978_Phi_of_t.npy'), np.load('Cordingley1978_parameters.npy'), 'Cordingley1978', 'xkcd:indigo')
	Halnes2016    = Model(np.load('Halnes2016_Phi_of_t.npy'), np.load('Halnes2016_parameters.npy'), 'Halnes2016', 'xkcd:olive')
	Nicholson1987 = Model(np.load('Nicholson1987_Phi_of_t.npy'), np.load('Nicholson1987_parameters.npy'), 'Nicholson1987', 'xkcd:grey')

	Models = [ Halnes2016,  Dietzel1982_1,Nicholson1987, Cordingley1978]


# make sure that all signals have same sampling frequency
#	assert Dietzel1982_1_parameters[1] == Gratiy2017_parameters[1] == Halnes2016_parameters[1]
	for M in Models:
		M.f, M.psd, M.location = makePSD(M.phi, M.N_t, M.delta_t)
		print(M.name, M.location) # 

# to make the plot in fig 4.6:	
	plt.figure()
	plt.title('PSD of the diffusion potential' )
	plt.xlabel('$log_{10}(frequency)$ in Hz')
	plt.ylabel('$log_{10}(PSD)$ in (mV)²/Hz')
	for M in Models:
		plt.plot(np.log10(M.f[1:-1]), np.log10(M.psd[1:-1]), M.color,label = M.name) 
		print(M.name, np.log10(M.f[100]), np.log10(M.psd[100])) # for table 4.3
		


# linear regression for the slope of the PSDs:
		log_freq = np.log10(M.f[10:1000])
		log_freq = log_freq.reshape((log_freq.shape[0],1))
		# y-values
		log_psd = np.log10(M.psd[10:1000])

		# Create linear regression object
		regr = linear_model.LinearRegression()

		# Train the model using the training sets
		regr.fit(log_freq, log_psd)

		# Make predictions using the testing set
		lin_fit = regr.predict(log_freq)

		# The coefficients
		print('Coefficients: \n', regr.coef_)
		# The mean squared error
		print("Mean squared error: %f"
		      % mean_squared_error(log_psd, lin_fit))

# to plot the PSDs of el.diff. together with PSD of LFP. Fig 4.7
#	PSD_of_LFP()

	plt.legend()
	plt.savefig('PSD', dpi = 500)
	plt.show()