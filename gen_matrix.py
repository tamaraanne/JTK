import pandas as pd 
import hashlib
import os 
from logging import Logger
def file_as_bytes(file):
    with file:
        return file.read()

def extractMatrix(dirname):
	'''
	return a dataframe of the miRNA matrix, each row is the miRNA counts for a file_id
	'''
	count = 0

	miRNA_data = []
	for idname in os.listdir(dirname):
		# list all the ids 
		if idname.find("-") != -1:
			idpath = dirname +"/" + idname

			# all the files in each id directory
			for filename in os.listdir(idpath):
				# check the miRNA file
				if filename.find("-") != -1:

					filepath = idpath + "/" + filename
					df = pd.read_csv(filepath,sep="\t")
					# columns = ["miRNA_ID", "read_count"]
					if count ==0:
						# get the miRNA_IDs 
						miRNA_IDs = df.miRNA_ID.values.tolist()

					id_miRNA_read_counts = [idname] + df.read_count.values.tolist()
					miRNA_data.append(id_miRNA_read_counts)


					count +=1
					# print (df)
	columns = ["file_id"] + miRNA_IDs
	df = pd.DataFrame(miRNA_data, columns=columns)
	return df



def extractSiteLabel(inputfile):
	df = pd.read_csv(inputfile, sep="\t")
	#
	# print (df[columns])
	df['cancer_type'] = df['cases.0.project.disease_type'] 
	df['cancer_type'] = df['cases.0.samples.0.sample_type']
	print("Types of Cancer") 
	df.loc[df['cases.0.project.disease_type'].str.contains("Breast"), 'cancer_type'] = 0
	df.loc[df['cases.0.project.disease_type'].str.contains("Lung"), 'cancer_type'] = 1
	df.loc[df['cases.0.project.disease_type'].str.contains("Kidney"), 'cancer_type'] = 2
	df.loc[df['cases.0.project.disease_type'].str.contains("Ovarian"), 'cancer_type'] = 3
	df.loc[df['cases.0.project.disease_type'].str.contains("Brain"), 'cancer_type'] = 4
	df.loc[df['cases.0.project.disease_type'].str.contains("Stomach"), 'cancer_type'] = 5
	df.loc[df['cases.0.project.disease_type'].str.contains("Head and Neck"), 'cancer_type'] = 6
	df.loc[df['cases.0.project.disease_type'].str.contains("Bladder"), 'cancer_type'] = 7
	df.loc[df['cases.0.project.disease_type'].str.contains("Skin"), 'cancer_type'] = 8
	df.loc[df['cases.0.project.disease_type'].str.contains("Leukemia"), 'cancer_type'] = 9
	df.loc[df['cases.0.project.disease_type'].str.contains("Sarcoma"), 'cancer_type'] = 10
	df.loc[df['cases.0.project.disease_type'].str.contains("Cervical"), 'cancer_type'] = 11
	df.loc[df['cases.0.project.disease_type'].str.contains("Liver"), 'cancer_type'] = 12
	df.loc[df['cases.0.project.disease_type'].str.contains("Testicular"), 'cancer_type'] = 13
	df.loc[df['cases.0.project.disease_type'].str.contains("Uterine"), 'cancer_type'] = 14
	df.loc[df['cases.0.project.disease_type'].str.contains("Colon"), 'cancer_type'] = 15
	df.loc[df['cases.0.project.disease_type'].str.contains("Rectum"), 'cancer_type'] = 16
	df.loc[df['cases.0.project.disease_type'].str.contains("Pancreatic"), 'cancer_type'] = 17
	df.loc[df['cases.0.project.disease_type'].str.contains("Esophageal"), 'cancer_type'] = 18
	df.loc[df['cases.0.project.disease_type'].str.contains("Pheochromocytoma"), 'cancer_type'] = 19
	df.loc[df['cases.0.project.disease_type'].str.contains("Adrenocortical"), 'cancer_type'] = 20
	df.loc[df['cases.0.project.disease_type'].str.contains("Mesothelioma"), 'cancer_type'] = 21
	df.loc[df['cases.0.project.disease_type'].str.contains("Prostate"), 'cancer_type'] = 22
	df.loc[df['cases.0.project.disease_type'].str.contains("Wilms"), 'cancer_type'] = 23
	df.loc[df['cases.0.project.disease_type'].str.contains("Thyroid"), 'cancer_type'] = 24
	df.loc[df['cases.0.project.disease_type'].str.contains("Rhabdoid"), 'cancer_type'] = 25
	df.loc[df['cases.0.project.disease_type'].str.contains("Thymoma"), 'cancer_type'] = 26
	df.loc[df['cases.0.project.disease_type'].str.contains("Cholangiocarcinoma"), 'cancer_type'] = 27
	df.loc[df['cases.0.project.disease_type'].str.contains("Uveal"), 'cancer_type'] = 28
	df.loc[df['cases.0.project.disease_type'].str.contains("Neoplasm"), 'cancer_type'] = 29
	df.loc[df['cases.0.samples.0.sample_type'].str.contains("Solid"), 'cancer_type'] = 30









	breast_count = df.loc[df.cancer_type == 0].shape[0]
	lung_count = df.loc[df.cancer_type == 1].shape[0]
	kidney_count = df.loc[df.cancer_type == 2].shape[0]
	ovary_count = df.loc[df.cancer_type == 3].shape[0]
	brain_count = df.loc[df.cancer_type == 4].shape[0]
	stom_count = df.loc[df.cancer_type == 5].shape[0]
	hn_count = df.loc[df.cancer_type == 6].shape[0]
	blad_count = df.loc[df.cancer_type == 7].shape[0]
	skin_count = df.loc[df.cancer_type == 8].shape[0]
	leuk_count = df.loc[df.cancer_type == 9].shape[0]
	sar_count = df.loc[df.cancer_type == 10].shape[0]
	cervix_count = df.loc[df.cancer_type == 11].shape[0]
	liver_count = df.loc[df.cancer_type == 12].shape[0]
	testis_count = df.loc[df.cancer_type == 13].shape[0]
	uterus_count = df.loc[df.cancer_type == 14].shape[0]
	colon_count = df.loc[df.cancer_type == 15].shape[0]
	rectum_count = df.loc[df.cancer_type == 16].shape[0]
	panc_count = df.loc[df.cancer_type == 17].shape[0]
	esop_count = df.loc[df.cancer_type == 18].shape[0]
	adg_count = df.loc[df.cancer_type == 19].shape[0]
	adgc_count = df.loc[df.cancer_type == 20].shape[0]
	pluera_count = df.loc[df.cancer_type == 21].shape[0]
	prostate_count = df.loc[df.cancer_type == 22].shape[0]
	wilms_count = df.loc[df.cancer_type == 23].shape[0]
	thyroid_count = df.loc[df.cancer_type == 24].shape[0]
	rhab_count = df.loc[df.cancer_type == 25].shape[0]
	thy_count = df.loc[df.cancer_type == 26].shape[0]
	bile_count = df.loc[df.cancer_type == 27].shape[0]
	uveal_count = df.loc[df.cancer_type == 28].shape[0]
	lymph_count = df.loc[df.cancer_type == 29].shape[0]
	normal_count = df.loc[df.cancer_type == 30].shape[0]


    
	print("{} Breast, {} Lung, {} Kidney, {} Ovarian, {} Brain, {} Stomach, {} Head and Neck, {} Bladder, {} Skin, {} Leukemia, {} Sarcoma, {} Cervical, {} Liver, {} Testicular, {} Uterine, {} Colon, {} Rectum, {} Pancreatic, {} Esophageal, {} Adrenal, {} Adrenal-cortical, {} Pluera-lung, {} Prostate, {} Wilms tumor, {} Thyroid, {} Rhabdoid, {} Thymus, {} Bile duct, {} Uveal melanoma, {} Lymphoma, {} Normal ".format(breast_count,lung_count,kidney_count,ovary_count,brain_count,stom_count,hn_count,blad_count,skin_count,leuk_count,sar_count,cervix_count,liver_count,testis_count,uterus_count,colon_count,rectum_count,panc_count,esop_count,adg_count,adgc_count,pluera_count,prostate_count,wilms_count,thyroid_count,rhab_count,thy_count,bile_count,uveal_count,lymph_count,normal_count))
	columns = ['file_id','cancer_type']
	return df[columns]
    
if __name__ == '__main__':


	data_dir ="/Users/tamarafernandes/Downloads/data/"
	# Input directory and label file. The directory that holds the data. Modify this when use.
	dirname = data_dir + "live_miRNA"
	label_file = data_dir + "files_meta.tsv"
	site_file = data_dir + "files_meta.tsv"
	#output file
	outputfile = data_dir + "miRNA_matrix.csv"

	# extract data
	matrix_df = extractMatrix(dirname)
	site_df = extractSiteLabel(site_file)

	#merge the two based on the file_id
	result = pd.merge(matrix_df, site_df, on='file_id', how="left")
	
	#print(result)

	#save data
	result.to_csv(outputfile, index=False)
	#print (labeldf)