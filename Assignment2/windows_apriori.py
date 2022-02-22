from apriori_templete import *

F, support = run_apriori('Assignment2\market_data_transaction.txt', 0.5)

print(F)
print(support)
