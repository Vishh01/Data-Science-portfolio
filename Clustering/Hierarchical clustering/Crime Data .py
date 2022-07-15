# Importing packages
import pandas as pd 
import matplotlib.pyplot as plt 

# Importing Data 
df1 = pd.read_csv(r"F:\6- Data mining Hierarical Clustering unsup learning\Assignments\crime_data.csv") 

df1.describe() 
df1.info()

df1 = df1.rename(columns ={'Unnamed: 0' : 'Country'})


# Normalization function
def norm_func (i):
    x= (i-i.min())/(i.max()-i.min())
    return(x) 

# Normalized data frame 
df_norm = norm_func(df1.iloc[:,1:]) 
df_norm.describe() 

# For creating dendogram 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 

z= linkage(df_norm, method='complete',metric="euclidean")

plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
     leaf_rotation=0,   # rotates the x axis labels
     leaf_font_size=10) # font size for the x axis labels  
 plt.show()               

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering             

h_complete = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='euclidean').fit(df_norm)  
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df1["clust"] = cluster_labels    # creating a new column and assigning it to new column 

df1 = df1.iloc[:, [5,0,1,2,3,4]] 
df1.head()

# Aggregate mean of each cluster
df1.iloc[:, 2: ].groupby(df1.clust).mean() 
df1.iloc[:, 2: ].groupby(df1.clust).count()

# Inferences:-
# 1)"Clust - 0" has low crime rate.
# 2) "Clust - 1" has high crime rate.  
# There are 20 countries among total 50 Countries have higher crime rate.  
