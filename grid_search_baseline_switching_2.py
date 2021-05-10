import os
# vector_sizes = [20, 30, 50, 100, 150, 200, 250, 400, 500]
# random_states = [42, 1973, 53022]

if __name__ == "__main__":
	# Best parameters chosen after grid search
	feature_length = [150]
	random_seed = [42]

	with open("results_switching.csv", 'w') as f:
		f.write("Type\trandom_state\tvector_length\taccuracy\tF1\tmicro_f1\tmacro_f1\n")

	for feature in feature_length:
		for random_state in random_seed:
			print(feature, random_state)
			os.system("python recreate_results_additional_2.py " + "All9Signals " + str(feature) + " " + str(random_state))