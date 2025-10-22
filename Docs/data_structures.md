
following data frames can be 
- either **merged in a single one** on drug IDs and disease IDs 
	- each row is a combination drug-disease
	- features for drugs and diseases will be duplicated
	- problem with targets
		- one drug has multiple targets
		- one 
		- either not be included or somehow
	- for basic ML models (logistic regression, XGBoost)
- or represented as a **graph**
	- nodes: diseases, drugs and targets
		-  node payloads: for drugs and diseases (see *singleton features*)
	- edges: disease-drug, drug-target, target-disease
		- edge payloads: see *connection to targets*, *connection drug-disease*
	- we want to predict whether a disease drug edge is a promising match 
## Singleton Features

Drugs 
- ChEMBL IDs - we have all we want
- chemical features from ChEMBL 
	- molecular weight, molecular fingerprints, pKa ...

Diseases
- 

## Connection to Targets
Drugs-Targets matches
- 
Disease-Targets matches
- 
## Connection Drug-Disease
for each Drug-Disease pair
- direct 

