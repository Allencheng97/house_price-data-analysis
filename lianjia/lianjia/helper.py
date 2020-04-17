import json
import pandas as pd
def jsontranscsv(sourcepath,newpath):
    f=open(sourcepath)
    records = [json.loads(line) for line in f.readlines()]
    df = pd.DataFrame(records)
    df.to_csv(newpath,encoding='gb18030')
    f.close()

def cap(x, quantile=[0.01, 0.99]):  # cap method helper function
    delete_list = list()
    head, tail = x.quantile(quantile).values.tolist()
    return head, tail