def get_magnetization(ising_array):
	magnetization_accumulator = 0
	for current_element in ising_array:
		magnetization_accumulator += current_element
	return magnetization_accumulator / (len(ising_array) * 1.0)
