
import numpy as np
import random
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import time
import pickle


# TODO: so far the only param quantity that matters is the product BETA * J, combine params into ratio's
# TODO: find criterion for equilibrium
# TODO: implement functions to measure quantities (below) as function of T (KB * T / J) and external magnetic field (B mu / J)
#		TODO: dim-less E per spin (E / JN)
#		TODO: dim-less Magnetization per spin (m / mu N)
# 		TODO: dim-less specific heat (c / K_B)
#		TODO: dim-less susceptibility per spin (chi / mu^2 N)


### PARAMETERS ###

N = 10 # 10 # 100
M = N
NUM_SPINS = N * M
#T = 300 # Kelvin
#BETA = 1 # (1.0 / (1.38064852 * T)) * math.pow(10, 23) # find how to set this
k_B = 1.38064852e-23 # m^2 kg s^-2 K-1
J = 1 # e-20 # everything is normalized w.r.t. this, set to 1
H = 0 # = B * mu

# N -- Steps
# 10 -- int(1.0e7)
# 25 -- int(5.0e7)
# 100 -- int(1.8e8)
MCS = 1000 # int(5.0e5) #  #8) # 7 # number of Monte Carlo steps

dim1 = False

random.seed(7)


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
			return delta_E, 2*candidate
		else: # else no change
			return 0, 0

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
			return delta_E, 2*candidate
		else: # else no change
			return 0, 0

def get_energy_and_magnetization(ising_array):
	if dim1: # 1D
		energy_accumulator = 0
		last_element = 0

		magnetization_accumulator = 0

		for current_element in ising_array:
			energy_accumulator += -J * last_element * current_element - H * current_element
			last_element = current_element

			magnetization_accumulator += current_element

		tot_magnetization = magnetization_accumulator / float(len(ising_array))
		return energy_accumulator, tot_magnetization

	else: # 2D
		energy_accumulator = 0
		magnetization_accumulator = 0
		for i in range(0, N):
			for j in range(0, M):
				magnetization_accumulator += ising_array[i, j]

				current_element = ising_array[i, j]
				energy_accumulator += -H * current_element

				neighbors = get_neighbors(i, j)
				for index in neighbors:
					index_x, index_y = index
					energy_accumulator += -J * ising_array[index_x, index_y] * current_element
		tot_magnetization = magnetization_accumulator / float( len(ising_array) * len(ising_array[0]) )
		return energy_accumulator / 2, tot_magnetization


def get_neighbors(i, j):
	neighbor_set = set([])
	if i > 0:
		neighbor_set.add((i-1, j))
		if j > 0:
			neighbor_set.add((i, j-1)) # m1
			neighbor_set.add((i-1, j-1))
		if j < M - 1:
			neighbor_set.add((i, j+1)) # m2
			neighbor_set.add((i-1, j+1))
	if i < N - 1:
		neighbor_set.add((i+1, j))
		if j > 0:
			neighbor_set.add((i, j-1)) # m1
			neighbor_set.add((i+1, j-1))
		if j < M - 1:
			neighbor_set.add((i, j+1)) # m2
			neighbor_set.add((i+1, j+1))

	return neighbor_set


############
### MAIN ###
############

energy_list = []
spec_heat_list = []
mag_list = []
susceptibility_list = []

num_snaps = 10 # 50
low_T_snaps = []
hi_T_snaps = []

spacings = np.logspace(-1.5, 0, num=10)
ups = 2.3 + spacings
downs = (2.3 - spacings)[::-1]
KbT_vals = np.concatenate([downs, ups]) # np.arange(1.9, 3.2, .1)

for KbT in KbT_vals:
	print 'Working on KbT = ' + str(KbT)

	BETA = 1 / (KbT)


	if dim1:
		ising_array = np.zeros(N, dtype='int32')
		for i in range(N):
			ising_array[i] = 2 * random.randint(0, 1) - 1 # initialize to +/- 1
	else:
		ising_array = np.zeros((N, M), dtype='int32')
		for i in range(N):
			for j in range(M):
				ising_array[i, j] = 2 * random.randint(0, 1) - 1 # initialize to +/- 1


	current_energy, current_magnetization = get_energy_and_magnetization(ising_array)
	current_abs_mag = abs(current_magnetization)
	energy_avg = 0
	energy_var = 0
	mag_avg = 0
	mag_var = 0 # TODO: get variance of |M| instead of M
	is_in_equilibrium = False
	i = 0
	k = 0 # how many summands we have in our average
	while i < MCS:
		i += 1

		e_update, mag_update = update(ising_array, BETA) # if we return the delta(magnetization)
		current_energy += e_update
		current_magnetization += mag_update
		current_abs_mag = abs(current_magnetization)

		if i % (MCS / num_snaps) == 0:
			if KbT < 1.35:
				low_T_snaps.append(np.copy(ising_array))

			elif KbT > 3.25:
				hi_T_snaps.append(np.copy(ising_array))

		if is_in_equilibrium:
			k += 1
			new_energy_avg = energy_avg + float(current_energy - energy_avg) / k
			energy_var += (current_energy - energy_avg) * (current_energy - new_energy_avg)
			energy_avg = new_energy_avg

			new_mag_avg = mag_avg + float(current_abs_mag - mag_avg) / k
			mag_var += (current_abs_mag - mag_avg) * (current_abs_mag - new_mag_avg)
			mag_avg = new_mag_avg

		elif i > MCS / 2:
			k = 1
			energy_avg = current_energy
			mag_avg = current_abs_mag
			is_in_equilibrium = True

	energy_var /= k - 1
	mag_var /= k - 1

#	plt.plot(energy_list)
#	plt.show()

	# energy is in units of J

#	temp = energy_var
#	print temp
##	print np.var(energy_list)
#	print '#######'
	energy_list.append(energy_avg)
	spec_heat = energy_var / (KbT * KbT) # get_spec_heat(energy_list) / (T*T) TODO: scale by Kb?
	spec_heat_list.append(spec_heat)

	mag_list.append(mag_avg)
	susceptibility = mag_var / KbT # TODO: scale by Kb?
	susceptibility_list.append(susceptibility)

# TODO add beta_vals and spec_heat_list to a CSV file
#plt.plot(beta_vals, spec_heat_list)
energy_fig = plt.figure()
plt.plot(KbT_vals, energy_list)
energy_fig.suptitle('E vs. T', fontsize=20)
plt.xlabel('T (J / Kb)', fontsize=18)
plt.ylabel('E (J)', fontsize=16)
energy_fig.savefig('plots/energy.jpg')
plt.close()

spec_heat_fig = plt.figure()
plt.plot(KbT_vals, spec_heat_list)
spec_heat_fig.suptitle('C vs. T', fontsize=20)
plt.xlabel('T (J / Kb)', fontsize=18)
plt.ylabel('C (Kb)', fontsize=16)
spec_heat_fig.savefig('plots/spec_heat.jpg')
plt.close()

mag_fig = plt.figure()
plt.plot(KbT_vals, mag_list)
mag_fig.suptitle('M vs. T', fontsize=20)
plt.xlabel('T (J / Kb)', fontsize=18)
plt.ylabel('M ()', fontsize=16)
mag_fig.savefig('plots/mag.jpg')
plt.close()

chi_fig = plt.figure()
plt.plot(KbT_vals, susceptibility_list)
chi_fig.suptitle('Chi vs. T', fontsize=20)
plt.xlabel('T (J / Kb)', fontsize=18)
plt.ylabel('Chi ()', fontsize=16)
chi_fig.savefig('plots/chi.jpg')
plt.close()

#plt.plot(KbT_vals, energy_list)
#plt.show()
#plt.plot(KbT_vals, spec_heat_list)
#plt.show()

#plt.plot(KbT_vals, mag_list)
#plt.show()
#plt.plot(KbT_vals, susceptibility_list)
#plt.show()

pickle_file = open('KbT_E_C_M_Chi.pkl', 'w')
data = (KbT_vals, energy_list, spec_heat_list, mag_list, susceptibility_list)
pickle.dump(data, pickle_file)

if dim1:

	print "No images for one dimensional case"

else:

	counter = 0
	for snap in low_T_snaps:
		counter += 1
		cold_fig = plt.figure()
		plt.imshow(snap, vmin=-1, vmax=1, cmap=cm.Greys_r) #, vmin=-1, vmax=1, cmap=cm.Greys_r)
		cold_fig.savefig('cold_snaps/snap_{}.png'.format(counter))
		plt.close()


	print len(low_T_snaps)

	counter = 0
	for snap in hi_T_snaps:
		counter += 1
		cold_fig = plt.figure()
		plt.imshow(snap, vmin=-1, vmax=1, cmap=cm.Greys_r) #, vmin=-1, vmax=1, cmap=cm.Greys_r)
		cold_fig.savefig('hot_snaps/snap_{}.png'.format(counter))
		plt.close()

print len(hi_T_snaps)
