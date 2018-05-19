# electrodiffusion

This is a repository containing the code I used for my Master's thesis.


halnes2016_data.py extract data from the file halnes2016.mat. It makes:
the files halnes_delta_c.npy, data_cK.npy, data_cNa.npy, data_cCa.npy and data_cX.npy which are used in calculate_phi and init_c_cenarios
A plot of the laminar concentration profiles based on these data. This is upper right panel of figure 2.4

spreading_depression.py simulates the diffusion potential with parameters from a spreading depression scenario. 
Both the KNP formalism and the exponental decay (ED) are used.
I used this program for the plot in figure 4.10
