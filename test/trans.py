import json
import pandas as pd

path = r"data.json"
f = open(path)
records = [json.loads(line) for line in f.readlines()]
df = pd.DataFrame(records)

df.to_csv(r"data.csv",encoding='gb18030')