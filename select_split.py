import os
import sys
import pandas as pd


table = pd.read_csv(sys.argv[1])
table = table.sort_values(by='hist score', ascending=False)
top_row = table.head(n=1).to_numpy()
print(top_row)
print(os.path.abspath(sys.argv[1].replace('info.csv', top_row[0][1])))
