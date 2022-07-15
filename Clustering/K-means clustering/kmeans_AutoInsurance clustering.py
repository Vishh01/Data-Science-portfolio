# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans

# Importing Dataset
df1= pd.read_csv(r"F:e\7- K-Means clustering unsup learning\Assignments\AutoInsurance (1).csv")

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
model = KMeans(n_clusters = 4) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column 

df.head()
  
df1 = df1.iloc[:, [24,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]] 
df1.head() 

# Aggregate mean of each cluster
a = df1.iloc[:, 2:].groupby(df1.clust).mean() 

b= df1.iloc[:, 2:].groupby(df1.clust).count() 

df1.to_csv("Kmeans_AutoInsurance.csv", encoding = "utf-8")

# Inferences:-
# cluster 0:-  There are 2317 customers. Unemployed with hight monthly premium.Oldest customer.more number of policies. Highest Claim Amount.
# cluster1:-   There are 2009 customers. Low income & low monthly premium. New customer.less number of policies. low claim amount.
# cluster2:-   There are 2771 customers. Highest income. High monthly premium. Average Claim amount.
# cluster3:-  There are 2037 customers. Average income. Average monthly premium Average Claim Amount More number of policies.
