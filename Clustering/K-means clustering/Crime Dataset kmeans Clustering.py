import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

# import dataset
df1 = pd.read_csv(r'F:\7- K-Means clustering unsup learning\Assignments\crime_data (1).csv')

df1 = df1.rename(columns = {'Unnamed: 0' : 'Country' })
df =df1.drop(['Country'], axis=1)

df1.isnull().sum()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, :])

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
df1['clust'] = mb # creating a  new column and assigning it to new column 

df1.head()

df1 = df1.iloc[:,[5,0,1,2,3,4]]
df1.head()

df1.iloc[:, 2:].groupby(df1.clust).mean()
df1.iloc[:, 2:].groupby(df1.clust).count()

# Inferences:-
# Cluster 0:-  Average Crime rate in the 18 cities. 
# Cluster 1:-  Highest crime rate as comapre to other cities.
# Cluster 2:-  Lowest Crime rate as Compare to other citites.
