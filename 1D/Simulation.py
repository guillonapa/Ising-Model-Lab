
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time


# TODO: so far the only param quantity that matters is the product BETA * J, combine params into ratio's
# TODO: find criterion for equilibrium
# TODO: implement functions to measure quantities (below) as function of T (KB * T / J) and external magnetic field (B mu / J)
#		TODO: dim-less E per spin (E / JN)
#		TODO: dim-less Magnetization per spin (m / mu N)
# 		TODO: dim-less specific heat (c / K_B)
#		TODO: dim-less susceptibility per spin (chi / mu^2 N)


### PARAMETERS ###

N = 50 # 100 #0
M = N
NUM_SPINS = N * M
#T = 300 # Kelvin
#BETA = 1 # (1.0 / (1.38064852 * T)) * math.pow(10, 23) # find how to set this
k_B = 1.38064852e-23 # m^2 kg s^-2 K-1
J = 1 # e-20 # everything is normalized w.r.t. this, set to 1
H = 0 # = B * mu

MCS = int(1.0e6) # 7 # number of Monte Carlo steps

dim1 = False

np.random.seed(7)


### FUNCTIONS ###

def update(ising_array, BETA):
	if dim1: # 1D
		flip_index = random.randint(0, N - 1)

		# get delta_E
		candidate = -ising_array[flip_index]
		delta_E = -H * candidate
		if flip_index > 0:
			left_element = ising_array[flip_index - 1]
			delta_E += -J * left_element * candidate
		if flip_index < N - 1:
			right_element = ising_array[flip_index + 1]
			delta_E += -J * candidate * right_element


		if delta_E < 0 or random.random() < math.exp(-BETA * delta_E): # update ising_array
			ising_array[flip_index] = candidate
			return delta_E
			# return delta_E, 2*candidate
		else: # else no change
			return 0
			# return 0, 0

	else: # 2D
		flip_index_x = random.randint(0, N - 1)
		flip_index_y = random.randint(0, N - 1)

		candidate = -ising_array[flip_index_x, flip_index_y]
		delta_E = -H * candidate

		neighbors = get_neighbors(flip_index_x, flip_index_y)
		for index in neighbors:
			index_x, index_y = index
			delta_E += -J * ising_array[index_x, index_y] * candidate


		if delta_E < 0 or random.random() < math.exp(-BETA * delta_E): # update ising_array
			ising_array[flip_index_x, flip_index_y] = candidate
			return delta_E
			# return delta_E, 2*candidate
		else: # else no change
			return 0
			# return 0, 0

def get_energy(ising_array):
	if dim1: # 1D
		energy_accumulator = 0
		last_element = 0
		for current_element in ising_array:
			energy_accumulator += -J * last_element * current_element - H * current_element
			last_element = current_element
		return energy_accumulator

	else: # 2D
		energy_accumulator = 0
		for i in range(0, N):
			for j in range(0, M):
				current_element = ising_array[i, j]
				energy_accumulator += -H * current_element

				neighbors = get_neighbors(i, j)
				for index in neighbors:
					index_x, index_y = index
					energy_accumulator += -J * ising_array[index_x, index_y] * current_element
		return energy_accumulator / 2

def get_magnetization(ising_array):
	if dim1: # 1D
		magnetization_accumulator = 0
		for current_element in ising_array:
			magnetization_accumulator += current_element
		return magnetization_accumulator / (len(ising_array) * 1.0)
	else: # 2D
		magnetization_accumulator = 0
		for i in range(0, N):
			for j in range(0, M):
				magnetization_accumulator += ising_array[i, j]
		return magnetization_accumulator / ((len(ising_array) * 1.0) * (len(ising_array[0]) * 1.0))

def get_neighbors(i, j):
	neighbor_list = []
	if i > 0:
		neighbor_list.append((i-1, j))
		if j > 0:
			neighbor_list.append((i, j-1))
			neighbor_list.append((i-1, j-1))
		if j < M - 1:
			neighbor_list.append((i, j+1))
			neighbor_list.append((i-1, j+1))
	if i < N - 1:
		neighbor_list.append((i+1, j))
		if j > 0:
			neighbor_list.append((i, j-1))
			neighbor_list.append((i+1, j-1))
		if j < M - 1:
			neighbor_list.append((i, j+1))
			neighbor_list.append((i+1, j+1))

	return neighbor_list


def get_spec_heat(energy_list):
	E_acc = 0
	E2_acc = 0
	for energy in energy_list:
		E_acc += energy
		E2_acc += energy * energy

	num_entries = len(energy_list)
	E_avg = float(E_acc) / num_entries
	E2_avg = float(E2_acc) / num_entries

	return E2_avg - (E_avg * E_avg)

############
### MAIN ###
############

spec_heat_list = []
beta_vals = np.arange(.2, 1, 0.05)
T_vals = np.arange(1.9, 2.6, .1)
#for BETA in beta_vals:
for T in T_vals:
#	print 'Working on BETA = ' + str(BETA)
	print 'Working on T = ' + str(T)

#	T = 1 / (k_B * BETA)
	BETA = 1 / (k_B * T)


	if dim1:
		ising_array = np.zeros(N, dtype='int32')
		for i in range(N):
			ising_array[i] = 2 * random.randint(0, 1) - 1 # initialize to +/- 1
	else:
		ising_array = np.zeros((N, M), dtype='int32')
		for i in range(N):
			for j in range(M):
				ising_array[i, j] = 2 * random.randint(0, 1) - 1 # initialize to +/- 1


	energy_list = []
	# magnetization_list = []
	current_energy = get_energy(ising_array)
	# current_magnetization = get_magnetization(ising_array)
	is_in_equilibrium = False
	for i in range(MCS):

		current_energy += update(ising_array, BETA)
		# current_energy, current_magnetization += update(ising_array) # if we return the delta(magnetization)

		if is_in_equilibrium:
			energy_list.append(current_energy)
			# magnetization_list.append(current_magnetization)
		elif i > MCS / 2:
			is_in_equilibrium = True


#	plt.plot(energy_list)
#	plt.show()

	# energy is in units of J

	temp = get_spec_heat(energy_list)
	print temp
	print np.var(energy_list)
	print '#######'
	spec_heat = temp / (T * T) # get_spec_heat(energy_list) / (T*T)
	spec_heat_list.append(spec_heat)

# TODO add beta_vals and spec_heat_list to a CSV file
#plt.plot(beta_vals, spec_heat_list)
plt.plot(T_vals, spec_heat_list)
plt.show()
