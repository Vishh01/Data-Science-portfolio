import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

groceries = []
with open("F:\\10-Association Rules Unsup learning\\Assignments\\transactions_retail1.csv") as f:
    groceries = f.read()
    
# splitting the data into separate transactions using separator as "\n"
groceries= groceries.replace(","," ") 
groceries= groceries.replace("NA"," ") 
groceries = groceries.split("\n") 

from collections import Counter # ,OrderedDict

item_frequencies = Counter(groceries)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies])) 
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries)) 
groceries_series = groceries_series.iloc[:557041, :] # removing the last empty transaction

X = groceries_series['transactions'].str.join(sep = '/n').str.get_dummies(sep = '/n')
