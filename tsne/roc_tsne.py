#Generating the ROC curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

if __name__ == '__main__':

    data_dir ="/Users/tamarafernandes/Downloads/data/"

    data_file = data_dir + "miRNA_matrix.csv"

    df = pd.read_csv(data_file)
    #print(df)
    y_data = df.pop('cancer_type').values
    y_data = label_binarize(y_data, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    n_classes = 31

    df.pop('file_id')

    columns =df.columns
    X_data = df.values
    tsne = TSNE()
    X_data = tsne.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=0)
    
    clf = OneVsRestClassifier(svm.LinearSVC(random_state=0))
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    #print(y_score)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['darksalmon','seagreen','darkgreen','purple','lightgreen','maroon','lightcoral','coral','peachpuff','tan','deepskyblue','aqua', 'darkorange', 'cornflowerblue','red','yellow','blue','indigo','orange','lavender','magenta','green','brown','navy','plum','yellowgreen','lawngreen','indianred','hotpink','deeppink','crimson'])
    
    #plot
    plt.figure(figsize=(17,14))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 10], [0, 10], 'k--',color='red', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('ROC-NCI database(t-SNE)',fontsize=40)
    plt.legend(loc="lower right")
    plt.savefig('/Users/tamarafernandes/Downloads/data/ROCcurvetsne.png')
    plt.show()