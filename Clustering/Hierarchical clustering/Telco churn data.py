# importing Libraries

import numpy as np 
import pandas as pd

# loading the dataset
df1= pd.read_excel(r"F:\ 6- Data mining Hierarical Clustering unsup learning\Assignments\telco_customer_churn.xlsx")

df1.describe()
df1.info() 
df = df1.drop(["Customer ID", "Count","Quarter"] , axis = 1 )  #Droping columns 
df1.isna().sum()             # No missing values 

  
# Getting the object type columns
df1.select_dtypes(['object']).columns 


# There is more Categorial data in data so we have to convert it to numerical data 
#Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() 

le_count = 0
for col in df.columns[::]:
    if df[col].dtype == 'object':
        if len(list(df[col].unique())) <= 2:
            le.fit(df[col]) 
            df[col] = le.transform(df[col])
            le_count +=1
 
#Onehot Encoding    

df = df.iloc[:,[0,1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,21,22,23,24,25,26,3,8,18,20]]
df_new= pd.get_dummies(df.iloc[:,23:]) 

df= df.drop(["Offer","Internet Type","Contract","Payment Method"], axis=1) #droping categorical column 

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
plt.figure(figsize=(45,30));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels 
)


# Now applying AgglomerativeClustering choosing 2 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_) 

df1['clust'] = cluster_labels # creating a new column and assigning it to new column 

df1 = df1.iloc[:, [30,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,]] 
df1.head()

# Aggregate mean of each cluster
a = df1.iloc[:, 2:].groupby(df1.clust).mean() 

df1.iloc[:, 2:].groupby(df1.clust).count() 

# creating a csv file 
df1.to_csv("telco_churn.csv", encoding = "utf-8") 


# # Inferences:-

# Cluster 0-
#           It has highest number of customer that is 3888
#           Having highest monthly charges and also total charges.
#           Churn rate is low
          
# Cluster 1- 
#            3155 customers have low charges 
#            Having low monrhly charges And total charge is also low.
#            Churn rate is High 
