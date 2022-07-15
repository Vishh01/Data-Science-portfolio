# Importing libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Loading Data Frame for the movies  data
phones = pd.read_csv(r"F:\10-Association Rules Unsup learning\Assignments\myphonedata.csv")

# Considering the numeric data
phones= phones.iloc[:, 3:9]

# Applying Apriori algorithm on movies datasets
frequent_phones= apriori(phones, min_support = 0.0075, max_len = 5, use_colnames = True)

# Most Frequent movie sets based on support 
frequent_phones.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_phones.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_phones.itemsets[0:11], rotation=20)
plt.xlabel('phones')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_phones, metric = "lift", min_threshold = 1)
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
a = rules_no_redudancy.sort_values('lift', ascending = False)


