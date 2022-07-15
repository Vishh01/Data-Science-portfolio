# Importing libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Loading Data Frame for the movies  data
movies = pd.read_csv(r"F:\10-Association Rules Unsup learning\Assignments\my_movies.csv")

# Considering the numeric data
movies= movies.iloc[:, 5:15]

# Applying Apriori algorithm on movies datasets
frequent_movies= apriori(movies, min_support = 0.0075, max_len = 5, use_colnames = True)

# Most Frequent movie sets based on support 
frequent_movies.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_movies.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_movies.itemsets[0:11], rotation=20)
plt.xlabel('movies')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_movies, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part ###################################
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
a = rules_no_redudancy.sort_values('lift', ascending = False).head(10)
