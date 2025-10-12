# Drug repurposing

>  **Goal**: predict whether a drug can be repurposed for a new disease 

  “If we do a new clinical trial, which molecule should we prioritise and test first ?”
  
- Scope:
	- disease domain - e.g. autoimmune diseases (psoriasis, IBD, rheumatoid arthritis...)
	- subset of ~100-500 approved drugs (restrain by disease domain and/or mechanism of action)
	- collect drug features, disease features and pairwise features
- Sources:
	- Drug data
		- chemical features
			- [ChEMBL API](https://www.ebi.ac.uk/training/online/courses/embl-ebi-programmatically/wp-content/uploads/sites/128/2020/08/ChEMBL-programmatically.pdf) → molecular weight, logP, SMILES, targets, mechanism 
			- [DrugBank](https://go.drugbank.com/releases/latest) attention: need to verify account for academic purposes
			- [RDKit API](https://www.rdkit.org/docs/api-docs.html) → Morgan fingerprints, similarity matrices, descriptors
	- Clinical drug-disease indications
		- [Open FDA API](https://open.fda.gov/apis/) ground truth for approved drug indications
		- EMA - european analog
		- [DrugBank](https://go.drugbank.com/releases/latest) attention: need to verify account for academic purposes
	- Co-citations
		- scrape PubMed abstracts → count co-occurrences drug:disease
		- [Europe PMC API](https://europepmc.org/RestfulWebService)
	- Disease features 
		- MeSH hierarchy 
		- textual description → embedding?

- Clean data:
	- get rid of synonyms
	- handle NAs
	- ....
- EDA:
	- PCA in chemical feature space → see if they cluster by disease / sth else
- Modelling:
	- baseline ML:
		- consider drug:disease independently
		- binary classification for a pair drug:disease (logistic regression, random forest, gradient boosting) - whether it's a successful repurposing (predicted positive for a pair not used for training and having no known indication)
		- multinomial approach 
		- use co-citation to create a "repurposability score" and regress on it
	- Graph models:
		- consider connections drug - traget, target - disease, drug - disease
		- + assign features to nodes
		- `node2vec` after if we have time, predict potential links drug-disease
- Evaluation:
	- CV
	- if possible, split train-test by time stamp of indication approval (to approach real-life repurposing)
- Interactive dashboard
	 - input a drug → list of candidate diseases + visualizations 
	 - 

    
  

  

