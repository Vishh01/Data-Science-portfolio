# importing Libraries

import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# loading the dataset
df1= pd.read_excel(r"F:\7- K-Means clustering unsup learning\Assignments\Telco_customer_churn (1).xlsx")

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

# Selecting 2 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 2) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb

df1 = df1.iloc[:, [30,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,]] 
df1.head()

# Aggregate mean of each cluster
a = df1.iloc[:, 2:].groupby(df1.clust).mean()  

df1.iloc[:, 2:].groupby(df1.clust).count() 

# creating a csv file 
df1.to_csv("telco_churn.csv", encoding = "utf-8") 



# Inferences: -

# Cluster 0-
# •	It has number of customers that is 1557.
# •	Having lowest monthly charges and also total charges.
          
# Cluster 1- 
# •	5486 customers have low charges
# •	Having high monthly charges and total charge is also High.
