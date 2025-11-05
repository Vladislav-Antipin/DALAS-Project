DATA_DIR = data
SRC_DIR = src
PYTHON = .venv/bin/python

.PHONY: all mesh indications drugs clean

all: drugs

# Step 1: scrape mesh ids
mesh: $(DATA_DIR)/mesh_ids.pkl

$(DATA_DIR)/mesh_ids.pkl:
	$(PYTHON) $(SRC_DIR)/retrieve_data.py --step mesh

# Step 2: get autoimmune indications
indications: $(DATA_DIR)/mesh_ids.pkl $(DATA_DIR)/indications.pkl

$(DATA_DIR)/indications.pkl: $(DATA_DIR)/mesh_ids.pkl
	$(PYTHON) $(SRC_DIR)/retrieve_data.py --step indications

# Step 3: get drugs
drugs: $(DATA_DIR)/indications.pkl $(DATA_DIR)/drugs.pkl

$(DATA_DIR)/drugs.pkl: $(DATA_DIR)/indications.pkl
	$(PYTHON) $(SRC_DIR)/retrieve_data.py --step drugs

clean:
	rm -rf $(DATA_DIR)/*.pkl