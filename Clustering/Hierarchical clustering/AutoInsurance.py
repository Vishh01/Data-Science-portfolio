#Importing libraries 
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# Import dataset 
df1 = pd.read_csv(r"F:\6- Data mining Hierarical Clustering unsup learning\Assignments\AutoInsurance.csv")

df = df1.drop(['Customer','State'], axis=1 )   #Droping column 
df.describe()
df.info() 

df.isnull().sum()       # No null Values 

# Getting the object type columns
df.select_dtypes(['object']).columns 
df.nunique()

# There is more Categorial data in data so we have to convert it to numerical data 
#Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() 

le_count = 0
for col in df.columns[::]:
    if df[col].dtype == 'object':
        if len(list(df[col].unique())) <= 4:
            le.fit(df[col]) 
            df[col] = le.transform(df[col])
            le_count +=1

#Onehot Encoding    

df = df.iloc[:,[0,1,2,6,7,8,9,10,11,12,13,14,15,17,18,19,21,3,4,5,16,20]] 
df_new= pd.get_dummies(df.iloc[:,16:]) 

df= df.drop(['Education','Effective To Date','EmploymentStatus','Policy','Vehicle Class'], axis=1)

df = pd.concat([df,df_new], axis = 1) #Concating the onehotencoded columns to dataset

df.info()        #No categorical data

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, :])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch  
import matplotlib.pyplot as plt

z = linkage(df_norm, method = "complete", metric = "euclidean") 

# Dendrogram
plt.figure(figsize=(60,45));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels 
)

# Cutting the dendrogram at max_d
max_d = 4.36
plt.axhline(y=max_d, c='k')

# Now applying AgglomerativeClustering choosing 2 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_) 

df1['clust'] = cluster_labels  # creating a new column and assigning it to new column 


df1 = df1.iloc[:, [24,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]] 
df1.head() 

# Aggregate mean of each cluster
a = df1.iloc[:, 2:].groupby(df1.clust).mean() 

b= df1.iloc[:, 2:].groupby(df1.clust).count() 

# Inferences:-
# Cluster 0 -  There are 1303 Customer  
#               Average income 
#               New Customer 
#               monthly premium is average
 
# Cluster 1 -  There are 2200 Customer
#              Average income  
#              monthly premium is highest

# Cluster 2 -  There are 2945 Customer
#              Lowest income of customer  
#              old customer
#              monthly premium is average
#              Claimed Amount is highest 

# Cluster 3 -  There are 2686 Customer 
#             Highest income of customer
#             monthly premium is lowest 
#             have more policies 
#             lowest claim amount
#             Claim amount is lowest

             


