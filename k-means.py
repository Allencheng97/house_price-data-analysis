# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
records = pd.read_csv('data-utf8.csv')
data=records[['total_price','area']]
data=np.array(data)
model=KMeans(n_clusters=3,n_init=20)
model.fit(data)
y=model.predict(data)
a=[]
b=[]
c=[]
d=[]
e=[]
# f=[]
for i in range(len(data)):
    if y[i]==0:
        a.append(data[i,:])
        a1=np.array(a)
    if y[i]==1:
        b.append(data[i,:])
        b1=np.array(b)
    if y[i]==2:
        c.append(data[i,:])
        c1=np.array(c)
    if y[i]==3:
        d.append(data[i,:])
        d1=np.array(d)
    if y[i]==4:
        e.append(data[i,:])
        e1=np.array(e)
    # if y[i]==5:
    #     f.append(data[i,:])
    #     f1=np.array(f)

plt.scatter(a1[:,1],a1[:,0],s=55,c='red',marker='.')
plt.scatter(b1[:,1],b1[:,0],s=55,c='green',marker='.')
plt.scatter(c1[:,1],c1[:,0],s=55,c='blue',marker='.')
# plt.scatter(d1[:,1],d1[:,0],s=55,c='yellow',marker='.')
# plt.scatter(e1[:,1],e1[:,0],s=55,c='purple',marker='.')
# plt.scatter(f1[:,1],f1[:,0],s=55,c='grey',marker='.')
# plt.savefig('test.jpg')
plt.show()