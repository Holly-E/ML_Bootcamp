import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#make blobs allows you to make artificial data you can play with
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)

#plot all rows in first column against all rows in second column
#np/pd nodation: [ first_row:last_row , column_X ] - you have 2-dimensional list/matrix/array
# and you get all values in column X (from all rows).
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')

# determine k
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data[0])
    kmeanModel.fit(data[0])
    distortions.append(sum(np.min(cdist(data[0],kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data[0].shape[0])

# Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')

#K Means Model
kmeans = KMeans(n_clusters=4)
#Fit to the features
kmeans.fit(data[0])

kmeans.cluster_centers_
kmeans.labels_

# Can't visualize clusters with lots of features. Use PCA to reduce # features
plt.title('K Means')
plt.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')

plt.show()
