import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# Loading dataset
a = pd.read_csv(r"F:\12-Network Analytics\Assignments\flight_hault.csv")

# Preprocessing
a.columns

a.columns = ["ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time"]
row1= pd.Series(data=[1, 'Goroka', 'Goroka.1', 'Papua New Guinea', 'GKA', 'AYGA','-6.081689', '145.391881', '5282', '10', 'U', 'Pacific/Port_Moresby'],index=["ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time"]) 
a = a.append(row1, ignore_index=True)

a = a.sort_values(by='ID')

#checking null values
a.isna().sum()

a= a.dropna() #dropping null values
a= a.iloc[0:500,:]

# Creating Graph
g = nx.DiGraph()
g = nx.from_pandas_edgelist(a,source='IATA_FAA',target = 'ICAO')

print(nx.info(g)) #getting info

# Degree Centrality 
degree = nx.degree_centrality(g)   
print(degree) 

# pip install decorator==5.0.9
pos =nx.spring_layout(g) 
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')

# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
bewns = nx.betweenness_centrality(g)
print(bewns)

# ********************************************************************************
# loading dataset
b = pd.read_csv(r"F:\12-Network Analytics\Assignments\connecting_routes.csv")
 
# Preprocessing
b = b.drop(["0"],axis=1)
b.columns

b.columns = ["flights", "ID", "main Airport", "main Airport ID", "Destination","Destination ID","haults","machinary"]
b = b.sort_values(by='ID')

#checking null values
b.isna().sum()

b= b.dropna() #dropping null values
b= b.iloc[0:500,:]

# Creating Graph
dg = nx.DiGraph()
dg = nx.from_pandas_edgelist(b,source='main Airport',target = 'Destination')

print(nx.info(dg)) #getting info

# Degree Centrality 
degree1 = nx.degree_centrality(dg)   
print(degree1) 

# pip install decorator==5.0.9
pos =nx.spring_layout(dg) 
nx.draw_networkx(dg, pos, node_size = 25, node_color = 'blue')

# closeness centrality
closeness1 = nx.closeness_centrality(dg)
print(closeness1)

## Betweeness Centrality 
betwns1 = nx.betweenness_centrality(dg)
print(betwns1)
