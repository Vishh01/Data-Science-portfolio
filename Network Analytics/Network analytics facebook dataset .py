import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# Loading facebook dataset
G = pd.read_csv(r"F:\12-Network Analytics\Assignments\facebook.csv")

G.columns = [0,1,2,3,4,5,6,7,8]
g = nx.Graph()
g = nx.from_pandas_adjacency(G)

print(nx.info(g))

b = nx.degree_centrality(g)  # Degree Centrality  
print(b) 

# circular Graph
nx.draw_circular(g)

# ****************************

# Loading instagram dataset
F = pd.read_csv(r"F:\12-Network Analytics\Assignments\instagram.csv")

F.columns = [0,1,2,3,4,5,6,7]
f = nx.Graph()
f = nx.from_pandas_adjacency(F)

print(nx.info(f))

c = nx.degree_centrality(f)  # Degree Centrality  
print(c) 

# star Graph
star= nx.star_graph(f)
nx.draw(star)

# ***********************************************

# Loading linkedIn dataset
A = pd.read_csv(r"F:\12-Network Analytics\Assignments\linkedin.csv")

A.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12]
a = nx.Graph()
a = nx.from_pandas_adjacency(A)

print(nx.info(a)) #Getting info 

d = nx.degree_centrality(a)  # Degree Centrality  
print(d) 

# star Graph
star= nx.star_graph(a)
nx.draw(star)


