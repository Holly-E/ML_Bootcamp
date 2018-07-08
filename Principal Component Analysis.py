import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#print(cancer.keys())
#print(cancer['DESCR'])

# Standardize data
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(df)
scaled_features = scalar.transform(df)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_features)
x_pca=pca.transform(scaled_features)

# Now we can visualize our clusters. Could pass this into kmeans and use c=kmeans.label_
#plt.scatter(x_pca[:,0],x_pca[:,1], c=cancer['target'])
#plt.xlabel('First Principal Component')
#plt.ylabel('Second Principal Component')

# Visualizing components, which are combinations of the original features
# Heatmap represents the correlation between the feature and the component itself
# Higher the color, the more correlation with the component
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
sns.heatmap(df_comp,cmap='plasma', xticklabels=True)
plt.tight_layout()
plt.show()