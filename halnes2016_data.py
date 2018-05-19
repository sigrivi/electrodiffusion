import numpy as np
import matplotlib.pyplot as plt
from scipy import io


# This is a script which extract data from the file halnes2016.mat. It makes:
# the files halnes_delta_c.npy, data_cK.npy, data_cNa.npy, data_cCa.npy and data_cX.npy 
# A plot of the laminar concentration profiles based on these data

if __name__=="__main__":

	data = io.loadmat('halnes2016.mat')

	data_cK = np.array(data["A"]['cK'][0][0])
	data_cNa = np.array(data["A"]['cNa'][0][0])
	data_cCa = np.array(data["A"]['cCa'][0][0])
	data_cX = np.array(data["A"]["cX"][0][0])
	data_t = np.array(data["A"]['t'][0][0])

	time42s = 168000
	delta_c = data_cK[time42s] - np.ones(len(data_cK[time42s]))*3
	x = np.linspace(0,1.4, len(data_cX[time42s]))

	concentrations = [data_cK[time42s], data_cNa[time42s], data_cCa[time42s], data_cX[time42s]]

# make the plot in figure 2.4
	names = ['$K^+$', '$Na^+$', '$Ca^{2+}$', '$X^-$']
	dot_styles = ['ko','kD','k^', 'kv']
	i = 0
	for c in concentrations:
		c = np.flip(c,0)
		plt.plot(c-c[0],-x , 'k', linestyle = '--')
		plt.plot(c-c[0],-x , dot_styles[i], label = names[i])
		i += 1
	plt.title('Laminar ionic profiles')
	plt.xlabel('$\Delta c$ (mM)')
	plt.ylabel('cortical depth (mm)')
	plt.legend()
	plt.savefig('laminar_profile_Halnes',dpi=225)
	plt.show()

# save the concentration profiles to npy files:
	np.save("halnes_delta_c.npy", np.flip(delta_c,0))
	np.save("data_cK.npy", concentrations[0])
	np.save("data_cNa.npy", concentrations[1])
	np.save("data_cCa.npy", concentrations[2])
	np.save("data_cX.npy", concentrations[3])

