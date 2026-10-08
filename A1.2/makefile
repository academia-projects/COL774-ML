# Define the Python interpreter
PYTHON = python

# Define the script to run
SCRIPT = logistic.py

# Define the directory containing the argument files
DATA_DIR = data

# Define the arguments for the script with the correct paths
ARGS1 = a $(DATA_DIR)/train.csv $(DATA_DIR)/test.csv $(DATA_DIR)/sample_weights1.txt $(DATA_DIR)/modelpredictions.txt $(DATA_DIR)/modelweights.txt

ARGS2 = b $(DATA_DIR)/train.csv $(DATA_DIR)/test.csv $(DATA_DIR)/regularization.txt $(DATA_DIR)/modelpredictions.txt $(DATA_DIR)/modelweights.txt $(DATA_DIR)/bestlambda.txt

# The default target that runs the script with the specified arguments
run_a:
	py aryan_temp.py a data/train1.csv data/tests/test1/params.txt done_weights.txt

run_b:
	py aryan_temp.py b data/train1.csv test.csv modelweights.txt modelpredictions.csv

run_c :
	py logistic_competitive.py data/train2.csv test.csv output.txt

# Clean target to be used only when you want to delete the output files
clean:
	rm -f $(DATA_DIR)/modelpredictions.txt $(DATA_DIR)/modelweights.txt

# .PHONY: run clean
