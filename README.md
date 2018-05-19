# electrodiffusion

This is a repository containing the code I used for my Master's thesis.

electrodiffusion.py contains the definitions of all classes and functions

halnes2016_data.py extract data from the file halnes2016.mat. It makes:
the files halnes_delta_c.npy, data_cK.npy, data_cNa.npy, data_cCa.npy and data_cX.npy which are used in calculate_phi and init_c_cenarios
A plot of the laminar concentration profiles based on these data. This is upper right panel of figure 2.4

calculate_phi.py calculates phi with initial concentrations from Cordingley, Halnes, Dietzel, Nicholson and the 1:1 Na+/K+ assumtion. Stores the PSDs in
Cordingley1978_Phi_of_t.npy, Halnes2016_Phi_of_t.npy, Dietzel1982_Phi_of_t.npy, Nicholson1987_Phi_of_t.npy.

psd.py calculates the PSDs of the diffusion potentials stored in Cordingley1978_Phi_of_t.npy, Halnes2016_Phi_of_t.npy, Dietzel1982_Phi_of_t.npy, Nicholson1987_Phi_of_t.npy
There is and option to include the PSD of the LFP FRom Gratiy in the plot.
Linear regression to find the slopes of the PSDs
I used this program for figure 4.6 and 4.7

init_c_scenarios is used for comparing scenarios for the inital ionic composition to the full model with the Halnes data.
I used this program for the plots in figure 4.2 and 4.3

spreading_depression.py simulates the diffusion potential with parameters from a spreading depression scenario. 
Both the KNP formalism and the exponental decay (ED) are used.
I used this program for the plot in figure 4.10
