Step 1: 

Create a python3 Jupyter Notebook using Anaconda

Step 2:

Download the dataset from https://portal.gdc.cancer.gov/repository and select
Data Category: Transcriptome Profiliing
Data type: miRNA Expression Quantification
Experimental Strategy: miRNA-Seq

Download the case IDs of the files

Step 3: Run the source codes in Jupyter Notebook in the order mentioned below

To generate the cases related to the files:
Run parse_file_id.py
		
To request meta data for the files and labels:
Run request_meta.py

To generate the matrix of the miRNA data for all the files:
Run gen_matrix.py

KNN with PCA Feature Extraction and generate evaluation metrics:
Run predict_pca.py

Generating ROC curve with PCA and KNN:
Run roc_pca.py

Generating scatter plot for PCA feature visualisation:
Run feature_pca.py

KNN with t-SNE Feature Extraction and generate evaluation metrics:
Run predict_tsne.py

Generating ROC curve with t-SNE and KNN:
Run roc_tsne.py

Generating scatter plot for t-SNE feature visualisation:
Run feature_tsne.py
