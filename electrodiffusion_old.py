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

class Ion:
	def __init__(self, c, cNew, D, z, name ):
		self.c = c                              # concentration
		self.cNew = cNew
		self.D = D                              # diffusion constant
		self.z = z 								# valence
		self.name = name                        # name of the ion
		self.dcdx = differentiate(c)

def differentiate(v):                           # x-values are between 0 and 1
	Nx = len(v)

	dv = np.zeros(Nx)
	dv[1:Nx-1] = (v[2:Nx]-v[:Nx-2])*Nx/2
	return(dv)

def integrate(v,xmin,xmax):
	Nx = len(v)
	V = np.zeros(Nx)
	Dx = (xmax-xmin)/Nx
	#V[:Nx-1] = (v[1:Nx]+v[:Nx-1])/(20)
	for i in range(0,Nx-1):
		V[i+1] = Dx*(v[i+1]+v[i])/2 + V[i]	
	return(V)

N = 10                                         # number of interior points
c1 = np.ones(N)*.145                           # initial concentration vector
c1[5] = 0.15
c2 = c1.copy()
z1 = 1                                         # valence
z2 = -1

D1 = 1.33e-1 #1.33e-9                                   # diffusion constant
D2 = 2.03e-1 #2.03e-9
lambda_n = 1.6                                 # tortuosity
#phi = np.zeros(N)                            # potential gradient



Ions = [Ion(c1,c1,D1,1,'sodium'),Ion(c1,c1, D2,-1,'cloride' )]




deltat = 1/20000
deltax = 1/(N)


def solveEquation(c1,c2,deltat,deltax): 
	c1New = c1.copy()
	c2New = c2.copy()
	alpha1 = deltat*D1/(deltax**2*lambda_n**2)
	alpha2 = deltat*D2/(deltax**2*lambda_n**2)
	
	for t in range(1000):
		dc1dx = differentiate(c1)
		#dc2dx = dc1dx.copy()                                                   # according to theory: dc1dx == dc2dx
		dc2dx = differentiate(c2)

		phi = -((z1*D1*dc1dx + z2*D2*dc2dx)/(z1**2*D1*c1 + z2**2*D2*c2 ))       # phi is the derivative of the potential
		Phi = integrate(phi,0,1)	                                           # Phi is the potential

		#c1New[1:N-1] = c1[1:N-1] + alpha1*(c1[2:N]-2*c1[1:N-1]+c1[0:N-2]) \
		#+ deltax*alpha1*z1/4*((c1[2:N] + c1[1:N-1])*(phi[2:N] + phi[1:N-1]) - (c1[1:N-1] + c1[0:N-2])*(phi[1:N-1] + phi[0:N-2]))

		#c2New[1:N-1] = c2[1:N-1] + alpha2*(c2[2:N]-2*c2[1:N-1]+c2[0:N-2])\
		# + deltax*alpha2*z2/4*((c2[2:N] + c2[1:N-1])*(phi[2:N] + phi[1:N-1]) - (c2[1:N-1] + c2[0:N-2])*(phi[1:N-1] + phi[0:N-2]))





		c1New[2:N-2] = c1[2:N-2] + alpha1*(c1[4:N]-2*c1[2:N-2]+c1[:N-4])/4 \
		+ deltax*alpha1*z1/4*((c1[3:N-1] + c1[2:N-2])*(phi[3:N-1] + phi[2:N-2]) - (c1[2:N-2] + c1[1:N-3])*(phi[2:N-2] + phi[1:N-3]))

		c2New[2:N-2] = c2[2:N-2] + alpha2*(c2[4:N]-2*c2[2:N-2]+c2[:N-4])/4 \
		+ deltax*alpha2*z2/4*((c2[3:N-1] + c2[2:N-2])*(phi[3:N-1] + phi[2:N-2]) - (c2[2:N-2] + c2[1:N-3])*(phi[2:N-2] + phi[1:N-3]))

		#for i in range(1,N-1):
		#	if i%2 != 0: #c1New[i]-0.145 < 1.e-7
		#		c1New[i] = (c1New[i-1]+c1New[i+1])/2
		#		c2New[i] = (c2New[i-1]+c2New[i+1])/2



		#for i in range(1,N-2):
			#c1New[i] = c1[i]*(1-2*alpha1 + alpha1*z1*deltax**2*dphi[i]) + c1[i+1]*(alpha1 + alpha1*z1*deltax*phi[i]/2) + c1[i-1]*(alpha1 - alpha1*z1*deltax*phi[i]/2)
			#c2New[i] = c2[i]*(1-2*alpha2 + alpha2*z2*deltax**2*dphi[i]) + c2[i+1]*(alpha2 + alpha2*z2*deltax*phi[i]/2) + c2[i-1]*(alpha2 - alpha2*z2*deltax*phi[i]/2)

			#c1New[i] = c1[i] + alpha1*(c1[i+1]-2*c1[i]+c1[i-1]) + deltax*alpha1*z1/4*( ( c1[i+1]+c1[i] )*( phi[i+1]+phi[i] ) - ( c1[i]+c1[i-1] )*( phi[i]+phi[i-1] ))
			#c2New[i] = c2[i] + alpha2*(c2[i+1]-2*c2[i]+c2[i-1]) + deltax*alpha2*z2/4*( ( c2[i+1]+c2[i] )*( phi[i+1]+phi[i] ) - ( c2[i]+c2[i-1] )*( phi[i]+phi[i-1] ))

			#c1New[i] = c1[i] + alpha1*(c1[i+1]-2*c1[i]+c1[i-1]) + alpha1*z1/2*( ( c1[i+1]+c1[i] )*( Phi[i+1]-Phi[i] ) - ( c1[i]+c1[i-1] )*( Phi[i]-Phi[i-1] ))
			#c2New[i] = c2[i] + alpha2*(c2[i+1]-2*c2[i]+c2[i-1]) + alpha2*z2/2*( ( c2[i+1]+c2[i] )*( Phi[i+1]-Phi[i] ) - ( c2[i]+c2[i-1] )*( Phi[i]-Phi[i-1] ))


		c1 = c1New.copy()
		c2 = c2New.copy()
	

		if t%200 == 0:
			if t>0:
				plt.plot(c1,label='c1 t=%d' %t)	
	plt.legend()
	plt.show()

	print(c1)
	
	return(c1, c2, Phi)

def solveEquation2(Ions,deltat,deltax):

	                                           # Phi is the potential
	for t in range(1000):

		phi = -((Ions[0].z*Ions[0].D*Ions[0].dcdx + Ions[1].z*Ions[1].D*Ions[1].dcdx)/(Ions[0].z**2*Ions[0].D*Ions[0].c + Ions[1].z**2*Ions[1].D*Ions[1].c))       # phi is the derivative of the potential
		Phi = integrate(phi,0,1)

		for I in Ions:
			alpha = deltat*I.D/(deltax**2*lambda_n**2)
			I.cNew[1:N-1] = I.c[1:N-1] + alpha*(I.c[2:N]-2*I.c[1:N-1]+I.c[0:N-2]) + deltax*alpha*I.z/4*( ( I.c[2:N]+I.c[1:N-1] )*( phi[2:N]+phi[1:N-1] ) - ( I.c[1:N-1]+I.c[0:N-2] )*( phi[1:N-1]+phi[0:N-2] ))
			I.c = I.cNew

		if t%200 == 0:
			if t>0:
				plt.plot(I.c,label='c1 t=%d' %t)
	print(t)
	return(Ions, Phi)

#[Ion1,Ion2], Phi = solveEquation2(Ions,deltat,deltax)

C1,C2,Phi = solveEquation(c1,c2, deltat, deltax)
print(np.sum(C1-C2))

plt.plot(C1,label='c1')	
plt.plot(C2, label = 'c2')
plt.legend()
plt.show()
plt.plot(Phi, label = 'Phi')
plt.legend()
plt.show()



sys.exit()



x = np.linspace(0,10,100)
y = integrate(x,0,10)

plt.plot(x,x**2/2)
plt.plot(x,y)
plt.show()
sys.exit()
# 
