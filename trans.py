import json
import pandas as pd
f1 = open('C:\\Users\\Allen\\Desktop\\test\\data.json', 'r')
records = [json.loads(line) for line in f1.readlines()]
# read file
df = pd.DataFrame(records)
# df.drop_duplicates()

# delete_list = df[(df.building_type.isnull())].index.tolist()
# delete_list += df[(df.building_structure.isnull())].index.tolist()
# delete_set = set(delete_list)
# delete_list = list(delete_set)
# df = df.drop(delete_list)
# # delete all rows which building_structure or building_type is null
# head, tail = cap(df['unit_price'])
# delete_list = df[df['unit_price'] < head].index.tolist(
# ) + df[df['unit_price'] > tail].index.tolist()
# df.drop(delete_list)
# # use cap method  to reduce noise
df.to_csv(r'C:\\Users\\Allen\\Desktop\\test\\data.csv', encoding='gb18030')
df.to_csv(r'C:\\Users\\Allen\\Desktop\\test\\data-utf8.csv', encoding='utf-8')
f1.close()
