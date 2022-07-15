import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#impirt dataset
df1 = pd.read_excel(r"F:\7- K-Means clustering unsup learning\Assignments\EastWestAirlines (1).xlsx", sheet_name ="data")

df1.describe()      #getting descriptive stats
df1.info()
df1.isnull().sum()   #no null values

df= df1.drop(["ID#"], axis = 1)   #droping column ID#

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x) 

# Normalized data frame 
df_norm = norm_func(df.iloc[:, :10]) 

###### scree plot or elbow curve ######
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3) 
model.fit(df_norm) 

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column 

df1.head()


df1 = df1.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
df1.head() 
a = df1.iloc[:, 2:12].groupby(df1.clust).mean()
df1.iloc[:, 2:12].groupby(df1.clust).count()

df1.to_csv("Kmeans_EastWestAirlines.csv", encoding = "utf-8")

# Inferences
# Cluster 0 = Frequent travellers
# Cluster 1 = Loyal Customer
# Cluster 2 = Not so Frequent Customer 
