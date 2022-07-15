import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

df1= pd.read_csv(r"F:\7- K-Means clustering unsup learning\Assignments\Insurance Dataset.csv")
df1.describe()
df1.info()

df1.isnull().sum()          # No null Values 

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df1.iloc[:, :]) 

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb

df1.head()
  
df1 = df1.iloc[:, [5,0,1,2,3,4]] 
df1.head() 

# Aggregate mean of each cluster
df1.iloc[:, 1:].groupby(df1.clust).mean() 

df1.iloc[:, 1:].groupby(df1.clust).count() 

df1.to_csv("Kmeans_Insurance.csv", encoding = "utf-8")

# Inferences:-
# cluster0 :-   There are 30 customers. Old age customers with High income & highest premium paid.
# cluster1 :-   There are 49 Customers. Young customers with lowest premium & claims count. Average income.
# cluster2 :-   There are 21 new Customers. Average Age with average premium & claim count. Lowest income compare to others.
