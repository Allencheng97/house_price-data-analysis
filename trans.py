import json
import pandas as pd

f = open('data.csv','rb')
content=f.read()

ucontent=content.decode('gb2312').encode('utf-8-sig')
with open('data-utf8.csv','w') as f2:
    f2.write(ucontent)
f.close