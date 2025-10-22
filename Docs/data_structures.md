
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
- ChEMBL IDs - we already have complete list 
- chemical features from ChEMBL 
	- molecular weight, molecular fingerprints, pKa ...
- embeddings 
	- from SapBERT, PubMedBERT, BioBERT
	- of: 
		- textual description 
			- from ChEMBL? maybe DrugBank?
		- name 

Diseases
- MeSH IDs  - we already have complete list 
- embeddings 
	- from SapBERT, PubMedBERT, BioBERT
	- of: 
		- textual description 
			- scrape / get from MeSH
		- name 
- #todo other parameters?
	- PubMed? MeSH API?

## Connection to Targets

Note: I'll call "targets": molecules that are targeted by disease, and molecules that are involved in the pathogenesis of the disease

#todo maybe consider mechanisms as well

Drugs-Targets matches
- from ChEMBL
- consider only protein targets?
- find their UniProt ID
- activation or inhibition (if more options, reduce it to binary - positive or negative action)
Disease-Targets matches
- TODO: find out where to take it from
- get UniProt ID
- up- or down- regulation 

We'll further analyse shared targets btw drugs and diseases
## Connection Drug-Disease

for each Drug-Disease pair
- **prediction goal** = is this drug proven to be effective against this disease?
	- if in `indications_df` from ChEMBL
		- **positive**
	- else 
		- send get request to Clinical Trials API 
		- filter either by drug or disease 
		- if no drug-disease study found 
			- **unknown**
		- else if only TERMINATED studies
			- why terminated?
				- not effective: **negative**
				- other: **unknown**
		- else (if has COMPLETED studies)
			- if p-values non-significant for latest phase studies:
				- **negative**
			- else:
				- **double check** because normally it should have appeared in `indications_df`
		- else if some studies TERMINATED and all p-values non-significant:
	- keep track of max, median and mean p-values, status and reasons for termination (if it's the case)
- timestamp of earliest successful study termination 
	- from Clinical Trials
- with *connection to targets* data, engineer features:
	- e.g. nb of shared targets
- with *singleton features*, engineer features:
	- cosine proximity of embeddings (name-name, description-description)
