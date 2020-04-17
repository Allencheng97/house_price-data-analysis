import csv
import json
import os
import pandas as pd
import re

path = r"data.json"
f = open(path)
records = [json.loads(line) for line in f.readlines()]
#read file
df = pd.DataFrame(records)
df.drop_duplicates()

df.to_csv(r'C:\\Users\\Allen\\Desktop\\test\\data.csv',encoding='gb18030')
df.to_csv(r'C:\\Users\\Allen\\Desktop\\test\\data-utf8.csv', encoding='utf-8')