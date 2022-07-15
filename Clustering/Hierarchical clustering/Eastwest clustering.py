# Import Libraries 
import pandas as pd
import matplotlib.pyplot as plt 

# Import dataset
df1 = pd.read_excel(r"F:\6- Data mining Hierarical Clustering unsup learning\Assignments\EastWestAirlines.xlsx", sheet_name="data")

df1.describe() 
df1.info() 

df = df1.drop(["ID#","Award?"], axis=1)    #Droping the ID & Award? columns

# Normalization Function
def norm_fun(i):
    x= (i-i.min()) / (i.max()- i.min())
    return(x) 

# Normalized data frame
df_norm = norm_fun(df) 
df_norm.describe() 

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch


z = linkage(df_norm,method='complete',metric='euclidean')

# Dendogram
plt.figure(figsize=(20,12));plt.title('Hierarical Clustering');plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(z,
    leaf_rotation= 0,
    leaf_font_size = 10) 
plt.show()     

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete= AgglomerativeClustering(n_clusters = 3, linkage='complete', affinity='euclidean').fit(df_norm)
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df1['clust'] = cluster_labels  # creating a new column and assigning it to new column

df1= df1.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]]    


# Aggregate mean & Count of each cluster
a = df1.iloc[:,2:12].groupby(df1.clust).mean()  

df1.iloc[:,2:12].groupby(df1.clust).size() 

# creating a csv file 
df1.to_csv("EastWest Airlines Cluster.csv", encoding = "utf-8") 


# Inferences
# 0 Loyal Customer-  
#                    oldest customer with high balance and bonus miles.
#                    They use the frequent flyer credit card very often. 

# 1 Frequent travellers- 
#                     old customer having high balance and bonus miles & have high number of miles Travelled  

# 2 Not so Frequent customer - 
#                     This cluster have highest no of customers who don’t travel as frequently compared to the other two cluster 
#                     and have the least flight miles eligible for award travel.
