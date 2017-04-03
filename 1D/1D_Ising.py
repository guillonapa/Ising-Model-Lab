
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# TODO: so far the only param quantity that matters is the product BETA * J, combine params into ratio's
# TODO: find criterion for equilibrium
# TODO: implement functions to measure quantities (below) as function of T (KB * T / J) and external magnetic field (B mu / J)
#		TODO: dim-less E per spin (E / JN)
#		TODO: dim-less Magnetization per spin (m / mu N)
# 		TODO: dim-less specific heat (c / K_B)
#		TODO: dim-less susceptibility per spin (chi / mu^2 N)


### PARAMETERS ###

N = 100
T = 300 # Kelvin
BETA = (1.0 / (1.38064852 * T)) * math.pow(10, 23) # find how to set this
J = 1 # e-20 # everything is normalized w.r.t. this, set to 1
H = 1 # = B * mu

MCS = 50000 # number of Monte Carlo steps


### FUNCTIONS ###

# TODO: averages need to taken by averaging over monte carlo steps (need to wait for equilibrium)
def get_spec_heat(ising_array):
	E = get_energy(ising_array)
	
### MAIN ###

ising_array = np.zeros(N, dtype='int32')
for i in range(N):
	ising_array[i] = 2 * random.randint(0, 1) - 1 # initialize to +/- 1
	

energy_list = []
file_object = open('file_name', 'w')
current_energy = get_energy(ising_array)
for i in range(MCS):
	flip_index = random.randint(0, N - 1)
	
	# get delta_E
	delta_E = 0
	candidate = -ising_array[flip_index]
	if flip_index > 0:
		left_element = ising_array[flip_index - 1]
		delta_E += -J * left_element * candidate
	if flip_index < N - 1:
		right_element = ising_array[flip_index + 1]
		delta_E += -J * candidate * right_element
	
	# update ising_array
	if delta_E < 0 or random.random() < math.exp(-BETA * delta_E):
		ising_array[flip_index] = candidate
		current_energy += delta_E
	# else no change
	
	energy_list.append(current_energy)
	file_object.write('{}, {} \n'.format(i, current_energy))
	
	
plt.plot(energy_list)
plt.show()
