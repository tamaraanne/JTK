import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
if __name__ == '__main__':
    df=pd.read_csv("/Users/tamarafernandes/Downloads/data/miRNA_matrix.csv")
    y_label=df.pop('cancer_type').values
    datafr=pd.DataFrame(X_data)
    datafr['2']=pd.Series(y_label, index=datafr.index)
    x=datafr.values
    plt.figure(figsize=(17,14))
    arr=[0 for i in range(31)]
    colors = ['darksalmon','seagreen','darkgreen','purple','lightgreen','maroon','lightcoral','coral','peachpuff','tan','deepskyblue','aqua', 'darkorange', 'cornflowerblue','red','yellow','blue','indigo','orange','lavender','magenta','green','brown','navy','plum','yellowgreen','lawngreen','indianred','hotpink','deeppink','crimson']
    for i in range(11486):
        if arr[int(x[i][2])]==0:
            plt.plot(x[i][0],x[i][1],color=colors[int(x[i][2])],marker='o',markersize=5,label='Class {0}'.format(int(x[i][2])))
            arr[int(x[i][2])]=1
        else:
            plt.plot(x[i][0],x[i][1],color=colors[int(x[i][2])],marker='o',markersize=5)
    plt.title('Feature Visualzation(t-SNE)',fontsize=40)
    plt.xlabel('Feature1',fontsize=20)
    plt.ylabel('Feature2',fontsize=20)
    plt.legend(loc="lower right")
    plt.savefig('/Users/tamarafernandes/Downloads/data/FeatureVisualisationTSNE.png')
    plt.show()