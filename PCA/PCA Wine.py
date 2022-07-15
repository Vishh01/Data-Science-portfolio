#************** Hierarchical Clustering On Dataset**************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"F:\ 8- Dimension Reduction(PCA) unsup learning\Assignments\wine.csv")
df.describe()
df.info()

df1 = df.drop(["Type"],axis=1) #droppin column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data  frame (considering the numerical part of data)
df1_norm = norm_func(df.iloc[:, :])
df1_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df1_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels 
)

plt.show() 

# Cutting the dendrogram at max_d
max_d = 1.80
plt.axhline(y=max_d, c='k')


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df1_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df1['clust'] = cluster_labels # creating a new column and assigning it to new column  

df1 = df1.iloc[:, [13,0,1,2,3,4,5,6,7,8,9,10,11,12]] 
df1.head()

# Aggregate mean of each cluster
df1.iloc[:, 1:].groupby(df1.clust).mean() 

df1.iloc[:, 1:].groupby(df1.clust).count() 


#**********************K-means clustering ***************************

from sklearn.cluster import	KMeans
df2= df.drop(["Type"],axis=1) #droppin column

# Normalized data frame (considering the numerical part of data)
df2_norm = norm_func(df2.iloc[:, :])
df2_norm.describe() 

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df2_norm) 
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3) 
model.fit(df2_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df2['clust'] = mb

df2 = df2.iloc[:, [13,0,1,2,3,4,5,6,7,8,9,10,11,12]] 
df2.head()

# Aggregate mean of each cluster
df2.iloc[:, 1:].groupby(df2.clust).mean() 

df2.iloc[:, 1:].groupby(df2.clust).count() 

# ******************************************************************

# ************PCA***************
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# df3= df.drop(["Type"],axis=1)

# Considering only numerical data 
df3_data = df.iloc[:, 1:] 

# Normalizing the numerical data 
df3_normal = scale(df3_data) 
df3_normal

pca = PCA(n_components = 3)
pca_values = pca.fit_transform(df3_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# PCA weights
pca.components_
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red") 

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2"
final = pd.concat([df.Type, pca_data.iloc[:, 0:3]], axis = 1)

# Scatter diagram
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
final[['comp0', 'comp1', 'Type']].apply(lambda x: ax.text(*x), axis=1) 

# *****************************************************************************
# Hierarchical clustering on PCA dataset
df4 = final.drop(["Type"],axis=1) #droppin column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df4_norm = norm_func(df4.iloc[:, :])
df4_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df4_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels 
)

plt.show()

# Cutting the dendrogram at max_d
max_d = 0.9
plt.axhline(y=max_d, c='k')


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df4_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df4['clust'] = cluster_labels # creating a new column and assigning it to new column  

df4 = df4.iloc[:, [3,0,1,2,]] 
df4.head()

# Aggregate mean of each cluster
df4.iloc[:, 1:].groupby(df4.clust).mean() 

df4.iloc[:, 1:].groupby(df4.clust).count() 

# *********************Kmeans Clustering on PCA Dataset*******************************

from sklearn.cluster import	KMeans
df5= final.drop(["Type"],axis=1) #droppin column

# Normalized data frame (considering the numerical part of data)
df5_norm = norm_func(df5.iloc[:, :])
df5_norm.describe() 

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df5_norm) 
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3) 
model.fit(df5_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df5['clust'] = mb

df5 = df5.iloc[:, [3,0,1,2,]] 
df5.head()

# Aggregate mean of each cluster
df5.iloc[:, 1:].groupby(df5.clust).mean() 

df5.iloc[:, 1:].groupby(df5.clust).count() 


 
# Hierarchical cluster before PCA
ax = df1.plot(x='Alcohol', y='Proline', kind='scatter',figsize=(12,8))
df1[['Alcohol', 'Proline', 'clust']].apply(lambda x: ax.text(*x), axis=1) 


#Scatter- Hierarchical cluster After PCA
ax1 = df4.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
df4[['comp0', 'comp1', 'clust']].apply(lambda x: ax1.text(*x), axis=1) 

# Scatter- KMeans cluster before PCA
ax2 = df2.plot(x='Alcohol', y='Proline', kind='scatter',figsize=(12,8))
df2[['Alcohol', 'Proline', 'clust']].apply(lambda x: ax2.text(*x), axis=1) 


# Scatter- KMeans cluster After PCA
ax3 = df5.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
df5[['comp0', 'comp1', 'clust']].apply(lambda x: ax3.text(*x), axis=1) 


# Inference:-
# Results Are different and improved
# In Hierarchical Clustering- 
# Clustering is improved after doing PCA on dataset.

# In KMeans Clustering- 
# Clustering is improved after doing PCA on dataset.



