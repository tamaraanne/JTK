# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import logger

from sklearn.manifold import TSNE

def specificity_score(y_true, y_predict):
    '''
    true_negative rate
    '''
    true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
    real_negative = len(y_true) - sum(y_true)
    return true_negative / real_negative 

def model_fit_predict(X_train,X_test,y_train,y_test):

    np.random.seed(2018)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    models = {
        'LR': LogisticRegression(),
        'EXT': ExtraTreesClassifier(),
        'RF': RandomForestClassifier(),
        'SVC': SVC(),
        'KNN': KNeighborsClassifier()
    }
    tuned_parameters = {
        'LR':{'C': [1, 10]},
        'EXT': { 'n_estimators': [16, 32] },
        'RF': { 'n_estimators': [16, 32] },
        'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
        'KNN' : {'n_neighbors' : [6]}
    }
    scores= {}
    for key in models:
        clf = GridSearchCV(models[key], tuned_parameters[key], scoring=None,  refit=True, cv=10)
        clf.fit(X_train,y_train)
        y_test_predict = clf.predict(X_test)
        precision = precision_score(y_test, y_test_predict,average='micro')
        accuracy = accuracy_score(y_test, y_test_predict)
        f1 = f1_score(y_test, y_test_predict,average='micro')
        recall = recall_score(y_test, y_test_predict,average='micro')
        specificity = specificity_score(y_test, y_test_predict)
        scores[key] = [precision,accuracy,f1,recall,specificity]
    print(scores)
    return scores



def draw(scores):
    '''
    draw scores.
    '''
    import matplotlib.pyplot as plt
    logger.info("scores are {}".format(scores))
    ax = plt.subplot(111)
    ax.set_title('Evaluation Metrics (t-SNE Feature Extraction)')
    precisions = []
    accuracies =[]
    f1_scores = []
    recalls = []
    categories = []
    specificities = []
    N = len(scores)
    ind = np.arange(N)  # set the x locations for the groups
    width = 0.1        # the width of the bars
    for key in scores:
        categories.append(key)
        precisions.append(scores[key][0])
        accuracies.append(scores[key][1])
        f1_scores.append(scores[key][2])
        recalls.append(scores[key][3])
        specificities.append(scores[key][4])

    precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
    accuracy_bar = ax.bar(ind+1*width, accuracies,width=0.1,color='g',align='center')
    f1_bar = ax.bar(ind+2*width, f1_scores,width=0.1,color='r',align='center')
    recall_bar = ax.bar(ind+3*width, recalls,width=0.1,color='y',align='center')
    specificity_bar = ax.bar(ind+4*width,specificities,width=0.1,color='purple',align='center')

    print(categories)
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(categories)
    ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0],specificity_bar[0]), ('precision', 'accuracy','f1','sensitivity','specificity'))
    ax.grid()
    plt.savefig('/Users/tamarafernandes/Downloads/data/tamjan.png')
    plt.show()
    

if __name__ == '__main__':


    data_dir ="/Users/tamarafernandes/Downloads/data/"

    data_file = data_dir + "miRNA_matrix.csv"

    df = pd.read_csv(data_file)
    #print(df)
    y_data = df.pop('cancer_type').values

    df.pop('file_id')

    columns =df.columns
    #print (columns)
    X_data = df.values
    tsne = TSNE()
    X_data = tsne.fit_transform(X_data)
    # split the data to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)

    #standardize the data.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # check the distribution of tumor and normal sampels in traing and test data set.
    logger.info("Percentage of tumor cases in training set is {}".format(sum(y_train)/len(y_train)))
    logger.info("Percentage of tumor cases in test set is {}".format(sum(y_test)/len(y_test))) 

    
    scores = model_fit_predict(X_train,X_test,y_train,y_test)

    draw(scores)