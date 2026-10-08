# Define the Python interpreter
PYTHON = python

# Define the script to run
SCRIPT = linear.py

# Define the directory containing the argument files
DATA_DIR = data

# Define the arguments for the script with the correct paths
ARGS1 = a $(DATA_DIR)/train.csv $(DATA_DIR)/test.csv $(DATA_DIR)/sample_weights1.txt $(DATA_DIR)/modelpredictions.txt $(DATA_DIR)/modelweights.txt

ARGS2 = b $(DATA_DIR)/train.csv $(DATA_DIR)/test.csv $(DATA_DIR)/regularization.txt $(DATA_DIR)/modelpredictions.txt $(DATA_DIR)/modelweights.txt $(DATA_DIR)/bestlambda.txt

# The default target that runs the script with the specified arguments
run_a:
	$(PYTHON) $(SCRIPT) $(ARGS1)

run_b:
	$(PYTHON) $(SCRIPT) $(ARGS2)

run_c :
	py linear_comp.py data/train.csv data/test.csv data/output.txt

# Clean target to be used only when you want to delete the output files
clean:
	rm -f $(DATA_DIR)/modelpredictions.txt $(DATA_DIR)/modelweights.txt

.PHONY: run clean
