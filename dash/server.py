import pandas as pd
import plotly.graph_objs as go
import dash
import dash_core_components as dcc                  # 交互式组件
import dash_html_components as html                 # 代码转html
from dash.dependencies import Input, Output         # 回调
from jupyter_plotly_dash import JupyterDash
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
records = pd.read_csv('data-utf8-loc.csv')
df = pd.DataFrame(records)
# df.drop(['id'], axis=1)
# 各区平均面积
groups_area_jzmj = df['area'].groupby(df['district'])
mean_area = groups_area_jzmj.mean()
# 各区平均单价
groups_area_unitprice = df['unit_price'].groupby(df['district'])
mean_unitprice = groups_area_unitprice.mean()
# 各区单价箱线图
box_unitprice_district = df['unit_price'].groupby(df['district'])
box_data_u = pd.DataFrame(list(range(14000)), columns=["start"])
for district, price in box_unitprice_district:
    box_data_u[district] = price
del box_data_u['start']
# 各区总价箱线图
box_totalprice_district = df['total_price'].groupby(df['district'])
box_data_t = pd.DataFrame(list(range(14000)), columns=["start"])
for district, price in box_totalprice_district:
    box_data_t[district] = price
del box_data_t['start']
# 面积分布
area_level = [0, 50, 100, 150, 200, 250, 300, 500, 1000]
label_level = ['0-50', '50-100', '100-150', '150-200',
               '200-250', '250-300', '300-500', '500-1000']
area_cut = pd.cut(df['area'], area_level, labels=label_level)
area_result = area_cut.value_counts()
t = np.random.randn(20000)
# #聚类1 总价
data1=records[['total_price','area']]
data1=np.array(data1)
model1=KMeans(n_clusters=3,n_init=20)
model1.fit(data1)
y1=model1.predict(data1)
cluster1=[]
cluster2=[]
cluster3=[]
for i in range(len(data1)):
    if y1[i]==0:
        cluster1.append(data1[i,:])
        cluster1_array=np.array(cluster1)
    if y1[i]==1:
        cluster2.append(data1[i,:])
        cluster2_array=np.array(cluster2)
    if y1[i]==2:
        cluster3.append(data1[i,:])
        cluster3_array=np.array(cluster3)
# #聚类2 单价
data2=records[['unit_price','area']]
data2=np.array(data2)
model2=KMeans(n_clusters=3,n_init=20)
model2.fit(data2)
y2=model2.predict(data2)
cluster4=[]
cluster5=[]
cluster6=[]
for i in range(len(data1)):
    if y2[i]==0:
        cluster4.append(data2[i,:])
        cluster4_array=np.array(cluster4)
    if y2[i]==1:
        cluster5.append(data2[i,:])
        cluster5_array=np.array(cluster5)
    if y2[i]==2:
        cluster6.append(data2[i,:])
        cluster6_array=np.array(cluster6)
#聚类3 方位
# data3=records[['lat','lng','unit_price']]
# data3=np.array(data3)
# model3=KMeans(n_clusters=3,n_init=20)
# model3.fit(data3)
# y3=model3.predict(data3)
# cluster7=[]
# cluster8=[]
# cluster9=[]
# for i in range(len(data3)):
#     if y3[i]==0:
#         cluster7.append(data3[i,:])
#         cluster7_array=np.array(cluster7)
#     if y3[i]==1:
#         cluster8.append(data3[i,:])
#         cluster8_array=np.array(cluster8)
#     if y3[i]==2:
#         cluster9.append(data3[i,:])
#         cluster9_array=np.array(cluster9)
# 装修类型与单价
fixture_unitprice = df['unit_price'].groupby(df['fixture'])
# 电梯存在于单价
ele_unitprice = df['unit_price'].groupby(df['elevator_exist'])
#随机森林模型
le = preprocessing.LabelEncoder()
X1=le.fit(df['district'].unique()).transform(df['district']) #X1表示房屋所在区
X2=le.fit(df['house_type'].unique()).transform(df['house_type']) #X2表示房屋的户型
X3=df['area']  #X3表示房屋的面积
X4=le.fit(df['fixture'].unique()).transform(df['fixture'])
X5=le.fit(df['elevator_exist'].unique()).transform(df['elevator_exist']) #X5表示房屋有无电梯
X6=le.fit(df['xiaoqu'].unique()).transform(df['xiaoqu']) 
X7=le.fit(df['weizhi'].unique()).transform(df['weizhi'])
X=np.mat([X1,X2,X3,X4,X5,X6,X7]).T
Y=df['unit_price']
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.1, random_state=0)
criterion=['mse','mae']
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1200, num = 50)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'criterion':criterion,
                'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
clf= RandomForestRegressor()
clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                              n_iter = 10,  
                              cv = 3, verbose=2, random_state=42, n_jobs=1)
rf=RandomForestRegressor(criterion='mse',bootstrap=True,max_features='sqrt', max_depth=70,min_samples_split=10, n_estimators=506,min_samples_leaf=4)
rf.fit(X_train, Y_train) 
Y_train_pred=rf.predict(X_train)
Y_test_pred=rf.predict(X_test)
print()


app = dash.Dash()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c574b']
app.layout = html.Div(
    id='main',
    style={'backgroundColor': '#111111'},
    children=[
        html.H1(children='威海商品房数据分析',
                style={'textAlign': 'center', 'color': '#7FDBFF', 'font-size': '70px'}),
        html.Div(
            id='pic1',
            style={'backgroundColor': '#111111'},
            children=[
                html.H2(children='威海房产高频词',
                        style=dict(textAlign='center', color='#7FDBFF')),
               html.Img(id='wordcloud_img',
                        src='https://s1.ax1x.com/2020/04/17/JEKHde.jpg',
                        style={'margin-left': '35%', 'margin-right': '35', 'width': '30%'}),


            ]

        ),
        html.Div(
            id='pic2',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(id='district_housenum',
                          style={'height': '50vh'},
                          figure=dict(
                              data=[go.Bar(
                                  x=df['district'].sort_values().unique(),
                                  y=df['district'].sort_values().value_counts(),
                                  marker=dict(color=colors, line=dict(
                                      color='white', width=1)),
                                  textposition='auto',
                                  opacity=0.7,
                                  text=df['district'].sort_values(
                                  ).value_counts()
                              )],

                              layout=go.Layout(title='各区房源数量',
                                               paper_bgcolor="#111111",
                                               plot_bgcolor='#111111',
                                               font=dict(family="Times New Roman",
                                                         size=20,
                                                         color='#7FDBFF'),
                                               yaxis=dict(title='房源数量',
                                                          titlefont=dict(
                                                              color='rgb(148, 103, 189)', size=24),
                                                          tickfont=dict(
                                                              color='#7FDBFF', size=24,),
                                                          tickwidth=4,
                                                          tickcolor='#7FDBFF',
                                                          showline=True,
                                                          linecolor='#7FDBFF',
                                                          linewidth=2,
                                                          showticklabels=True,
                                                          autorange=True,
                                                          type='linear',
                                                          ),
                                               xaxis=dict(title='区域',
                                                          titlefont=dict(
                                                              color='rgb(148, 103, 189)', size=24),
                                                          showline=True,
                                                          linecolor='#7FDBFF',
                                                          linewidth=2,
                                                          autorange=True
                                                          )

                                               )

                          )
                          ),
                html.P(
                    id='pic2_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=['从图中可以看出，乳山市和文登区房源数量最多，环翠区和经区房源数中等且数量较为接近',
                              '荣成市和高区房源数较少，所以后面的分析关于荣成市和高区的结论可能较其他区域存在一定误差']
                )



            ]
        ),
        html.Div(
            id='pic3',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='district_housearea',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Bar(
                            x=mean_area.keys(),
                            y=mean_area.unique(),
                            marker=dict(color=colors, line=dict(
                                color='white', width=1)),
                            textposition='auto',
                            opacity=0.7
                        )],

                        layout=go.Layout(title='各区在售房平均面积',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='平均面积 单位m²',
                                                          titlefont=dict(
                                                              color='rgb(148, 103, 189)', size=24),
                                                          tickfont=dict(
                                                              color='#7FDBFF', size=24,),
                                                          tickwidth=4,
                                                          tickcolor='#7FDBFF',
                                                          showline=True,
                                                          linecolor='#7FDBFF',
                                                          linewidth=2,
                                                          showticklabels=True,
                                                          autorange=True,
                                                          type='linear',
                                                    ),
                                         xaxis=dict(title='区域',
                                                          titlefont=dict(
                                                              color='rgb(148, 103, 189)', size=24),
                                                          showline=True,
                                                          linecolor='#7FDBFF',
                                                          linewidth=2,
                                                          autorange=True
                                                    )
                                         )

                    )
                ),
                html.P(
                    id='pic3_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '市区的三个区域：高区，经区，环翠区以及文登区，平均房源面积较大，约为100m²。乳山市和荣成市的平均房源面积总体较小，约为80—90m²。']
                )
            ]

        ),
        html.Div(
            id='pic4',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='district_houseprice',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Bar(
                            x=mean_unitprice.keys(),
                            y=mean_unitprice.unique(),
                            marker=dict(color=colors, line=dict(
                                color='white', width=1)),
                            textposition='auto',
                            opacity=0.7
                        )],

                        layout=go.Layout(title='各区在售房平均单价',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='平均单位面积价格 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='区域',
                                                          titlefont=dict(
                                                              color='rgb(148, 103, 189)', size=24),
                                                          showline=True,
                                                          linecolor='#7FDBFF',
                                                          linewidth=2,
                                                          autorange=True
                                                    )

                                         )

                    )
                ),
                html.P(
                    id='pic4_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '高区，经区，环翠区单位面积房价较高，均超过11000元/m²，其中经区最高为12600元/m²。乳山市，荣成市和文登区的房价较低，为4000-5000元/m²。威海市房价最高的区域均价超过最低的区域均价3倍']
                )
            ]

        ),
        html.Div(
            id='pic5',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='district_boxu',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[
                            go.Box(
                                y=box_data_u['乳山市'],
                                name='乳山市',
                            ),
                            go.Box(
                                y=box_data_u['文登区'],
                                name='文登区',
                            ),
                            go.Box(
                                y=box_data_u['环翠区'],
                                name='环翠区',
                            ),
                            go.Box(
                                y=box_data_u['经区'],
                                name='经区',
                            ),
                            go.Box(
                                y=box_data_u['荣成市'],
                                name='荣成市',
                            ),
                            go.Box(
                                y=box_data_u['高区'],
                                name='高区',
                            )
                        ],

                        layout=go.Layout(title='各区在售房单位面积价格箱线图',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='总价 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='区域',
                                                          titlefont=dict(
                                                              color='rgb(148, 103, 189)', size=24),
                                                          showline=True,
                                                          linecolor='#7FDBFF',
                                                          linewidth=2,
                                                          autorange=True
                                                    )
                                         )

                    )
                ),
                html.P(
                    id='pic5_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '经区和环翠区两个区域房源单价正常值分布都不是太集中，50%的单价分布在10000—12000元/m²的区间内，区间跨度比其他区都要大，同时上下极值相差极大，说明该区域房源质量存在多个档次。由于当前信息不包含具体的地址信息，推测是由于地理位置（学区房）与小区建成时间影响。乳山市和荣成市房价相对集中，更接近正态分布。']
                )

            ]

        ),
        html.Div(
            id='pic6',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='district_boxt',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[
                            go.Box(
                                y=box_data_t['乳山市'],
                                name='乳山市',
                            ),
                            go.Box(
                                y=box_data_t['文登区'],
                                name='文登区',
                            ),
                            go.Box(
                                y=box_data_t['环翠区'],
                                name='环翠区',
                            ),
                            go.Box(
                                y=box_data_t['经区'],
                                name='经区',
                            ),
                            go.Box(
                                y=box_data_t['荣成市'],
                                name='荣成市',
                            ),
                            go.Box(
                                y=box_data_t['高区'],
                                name='高区',
                            )
                        ],

                        layout=go.Layout(title='各区在售房总价箱线图',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='单位面积价格 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='区域',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    )
                                         )

                    )
                ),
                html.P(
                    id='pic6_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '高区，经区，环翠区这三个高价区域内，200万元的二手房以分布在正常值范围内。威海市市区二手房价格大部分都集中在80万元至200万元之间。同时超过1000万元的豪宅主要分布在高区和经区。乳山市和荣成市房价相对集中，在60万元至100万元之间。']

                )

            ]

        ),
        html.Div(
            id='pic7',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='district_houseuse',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Bar(
                            x=df['house_use'].unique(),
                            y=df['house_use'].sort_values().value_counts(),
                            marker=dict(
                                color=['#7fdbff', '#fd9900', '#fd00b2']),
                            textposition='auto',
                            opacity=0.7,
                            text=df['house_use'].sort_values().value_counts()
                        )],

                        layout=go.Layout(title='威海市房屋用途',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='房屋用途',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='区域',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    )
                                         )

                    )
                ),
                html.P(
                    id='pic7_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '威海市现有在售二手房中，普通住宅占据绝对主体，剩余少量商住两用房与别墅。在数据清洗过程中，部分过低的类型如车位，车库被清洗掉。']

                )
            ]

        ),
        html.Div(
            id='pic8',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='area_distribute',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Scatter(
                            x=['0-50', '50-100', '100-150', '150-200',
                                '200-250', '250-300', '300-500', '500-1000'],
                            y=[area_result['0-50'], area_result['50-100'], area_result['100-150'], area_result['150-200'], area_result['200-250'],
                                area_result['250-300'], area_result['300-500'], area_result['500-1000']],
                            mode='lines',
                            connectgaps=True,
                        )],

                        layout=go.Layout(title='威海市住房面积分布',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='面积区间商品房套数',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='面积区间 单位：m²',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    )

                                         )
                    )
                ),
                html.P(
                    id='pic8_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=['威海市现有在售二手房中，绝大部分房源面积集中在50m²-200m²，少部分超过200m²']

                )

            ]

        ),

        html.Div(
            id='pic11',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='house_type_precentage',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[
                            go.Pie(
                                labels=df['house_type'].value_counts().keys(),
                                values=df['house_type'].value_counts())
                        ],
                        layout=go.Layout(title='威海市房型占比饼状图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(family="Times New Roman", size=20, color='#7fdbff'))



                    )
                ),
                html.P(
                    id='pic11_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '威海市现有在售房中，主流的房型共三种：2室2厅1厨1卫，2室1厅1厨1卫，3室2厅1厨1卫。这三种房型占据超过50%的现有房源。']

                )

            ]

        ),
        html.Div(
            id='pic12',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='fixture_precentage',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[
                            go.Pie(
                                labels=df['fixture'].value_counts().keys(),
                                values=df['fixture'].value_counts())
                        ],
                        layout=go.Layout(title='威海市装修占比饼状图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(family="Times New Roman", size=20, color='#7fdbff'))



                    )
                ),
                html.P(
                    id='pic12_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=['威海市现有在售房中,超过一半为精装房，30%为简装房，剩下少数为毛胚房。']
                )
            ]

        ),
        html.Div(
            id='pic23',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[

                dcc.Graph(
                    id='fixture_unitprice',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Bar(
                            x=fixture_unitprice.mean().keys(),
                            y=fixture_unitprice.mean(),
                            marker=dict(color=colors, line=dict(
                                color='white', width=1)),
                            textposition='auto',
                            opacity=0.7
                        )],

                        layout=go.Layout(title='各装修类型与单位面积价格均价',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='单位面积价格 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='装修类型',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    )

                                         )

                    )
                ),
                html.P(
                    id='pic23_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '虽然精装房与简装房占据绝大多数，但从平均单位面积价格来看，毛坯房的价格最高，高于精装房，简装房单位面积价格最低。']
                )


            ]

        ),
        html.Div(
            id='pic13',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='building_type_precentage',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[
                            go.Pie(
                                labels=df['building_type'].value_counts().keys(),
                                values=df['building_type'].value_counts())
                        ],
                        layout=go.Layout(title='威海市房源建筑类型占比饼状图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(family="Times New Roman", size=20, color='#7fdbff'))



                    )
                ),
                html.P(
                    id='pic13_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '刨除暂无数据的房源，房源的建筑类型87.9%都是板楼，几乎不存在平房，明显属于长尾分布类型（严重偏态）。这与威海市大部分楼盘为建造日期较近的事实相符合。']
                )

            ]

        ),
        html.Div(
            id='pic14',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='building_structure_precentage',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[
                            go.Pie(
                                labels=df['building_structure'].value_counts(
                                ).keys(),
                                values=df['building_structure'].value_counts())
                        ],
                        layout=go.Layout(title='威海市房源建筑结构占比饼状图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(family="Times New Roman", size=20, color='#7fdbff'))



                    )
                ),
                html.P(
                    id='pic14_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '刨除未知结构的房源，房源的建筑结构40.5%为现在2000年-2010年主流的钢筋混凝土结构，其次为占据31.7%的框架结构，主要以高层住宅为主，部分砖混结构主要位于文登区与乳山市，极少数为钢结构，主要为商住两用房。']
                )
            ]

        ),
        html.Div(
            id='pic20',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='elevator_exist',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[
                            go.Pie(
                                labels=df['elevator_exist'].value_counts().keys(),
                                values=df['elevator_exist'].value_counts())
                        ],
                        layout=go.Layout(title='威海市房源有无电梯占比饼状图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(family="Times New Roman", size=20, color='#7fdbff'))



                    )
                ),
                html.P(
                    id='pic20_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '刨除暂无数据的房源，房源中37.8%有电梯，62.2%无电梯，约为1比2的比例。']
                )
            ]

        ),
        html.Div(
            id='pic24',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[

                dcc.Graph(
                    id='ele_unitprice',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Bar(
                            x=ele_unitprice.mean().keys(),
                            y=ele_unitprice.mean(),
                            marker=dict(color=colors, line=dict(
                                color='white', width=1)),
                            textposition='auto',
                            opacity=0.7
                        )],

                        layout=go.Layout(title='有无电梯与单位面积价格均价',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='单位面积价格 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='有无电梯',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    )

                                         )

                    )
                ),

                html.P(
                    id='pic24_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '可以从图中看出，有电梯的房源均价为11.1k元明显高于无电梯的均价7.6k元。这可能与存在电梯的房源楼层较高，建造日期较新有关。']
                )


            ]

        ),
        html.Div(
            id='pic15',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='elevator_precentage',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[
                            go.Pie(
                                labels=df['elevator_ratio'].value_counts().keys(),
                                values=df['elevator_ratio'].value_counts())
                        ],
                        layout=go.Layout(title='威海市房源梯户占比饼状图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(family="Times New Roman", size=20, color='#7fdbff'))



                    )
                ),
                html.P(
                    id='pic15_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '可以从图中看出，目前59%房源为一梯两户，占据主流，其次是一梯三户和一梯四户。多梯房型成本较高，数量较少。']
                )
            ]

        ),
        html.Div(
            id='pic16',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='overall_floor',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Bar(
                            x=df['overall_floor'].value_counts().keys(),
                            y=df['overall_floor'].value_counts(),
                            marker=dict(colorscale='Viridis', color=t,
                                        showscale=True),
                            textposition='auto',
                            opacity=0.7
                        )],

                        layout=go.Layout(title='各区房源楼层数柱状图',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='该层数房源套数',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='层数',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    )

                                         )

                    )
                ),
                
                html.P(
                    id='pic16_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '二手房中，楼层为6或7的房屋最多，分别为4253套与3844套。超过20层高层房屋和低于4层的房屋数量较少。']
                )
            ]

        ),
        html.Div(
            id='pic17',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Dropdown(id='district_choose',
                             options=[
                                 {'label': '乳山市', 'value': '乳山市'},
                                 {'label': '文登区', 'value': '文登区'},
                                 {'label': '环翠区', 'value': '环翠区'},
                                 {'label': '经区', 'value': '经区'},
                                 {'label': '荣成市', 'value': '荣成市'},
                                 {'label': '高区', 'value': '高区'}

                             ],
                             value='环翠区'
                             ),
                dcc.Graph(id='top20house', style={'height': '50vh'}),
                
            ]

            

        ),
        html.Div(
            id='pic18',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Slider(
                    id='slider2',
                    min=0,
                    max=20,
                    value=5,
                    step=1
                ),
                dcc.Graph(id='top20xiaoqu', style={'height': '50vh'}),
                



            ]



        ),
        html.Div(
            id='pic19',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[

            ]

        ),
        html.Div(
            id='pic9',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='area_price',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Scatter(
                            x=df['area'],
                            y=df['total_price'],
                            mode='markers',
                            marker=dict(
                                color=t,  # set color equal to a variable
                                colorscale='Viridis',  # one of plotly colorscales
                                showscale=True
                            )
                        )],
                        layout=go.Layout(title='威海市住房总价与建筑面积散点图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='总价 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='面积 单位：m²',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    ))
                    )
                ),
                html.P(
                    id='pic17_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '总价与建筑面积这两个变量符合正相关关系。数据点分布比较集中，大多数都在总价0~500万元与建筑面积0~300平米这个区域内。']
                )
            ]

        ),
        html.Div(
            id='pic10',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='area_unit_price',
                    style={'height': '50vh'},
                    figure=dict(
                        data=[go.Scatter(
                            x=df['area'],
                            y=df['unit_price'],
                            mode='markers',
                            marker=dict(
                                color=t,  # set color equal to a variable
                                colorscale='Viridis',  # one of plotly colorscales
                                showscale=True
                            )
                        )],
                        layout=go.Layout(title='威海市住房单价与建筑面积散点图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='单位面积价格 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='面积 单位：m²',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    ))
                    )
                ),
                html.P(
                    id='pic18_test',
                    style={'width': '100%', 'height': '20vh', 'fontSize': '25px',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF'},
                    children=[
                        '建筑面积与单价并无明显关系，同样样本点分布也较为集中，离散值不多，但单价特别高的房源，建筑面积都不是太大，因为这些房源一般都位于市中心。']
                )
            ]

        ),
        html.Div(
            id='pic21',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='kmeans_total',
                    style={'height': '50vh'},
                    figure=dict(

                        data=[
                                go.Scatter(x=cluster1_array[:,1], y=cluster1_array[:,0], mode='markers'),
                                go.Scatter(x=cluster2_array[:,1], y=cluster2_array[:,0], mode='markers'),
                                go.Scatter(x=cluster3_array[:,1], y=cluster3_array[:,0], mode='markers')
                        ],
                        layout=go.Layout(title='威海市住房总价与建筑面积聚类图', paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='总价 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='面积 单位：m²',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    ))
                    )
                ),
            ]

        ),
        html.Div(
            id='pic22',
            style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
            children=[
                dcc.Graph(
                    id='kmeans_unit',
                    style={'height': '50vh'},
                    figure=dict(

                        data=[
                                go.Scatter(x=cluster4_array[:,1], y=cluster4_array[:,0], mode='markers'),
                                go.Scatter(x=cluster5_array[:,1], y=cluster5_array[:,0], mode='markers'),
                                go.Scatter(x=cluster6_array[:,1], y=cluster6_array[:,0], mode='markers')
                        ],
                        layout=go.Layout(title='威海市住房单位面积价格与建筑面积聚类图', paper_bgcolor="#111111",
                                        margin={'b': 0},
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='单价 单位：元',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='面积 单位：m²',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    ))
                    )
                ),
            ]

        ),
    
        html.P(
                    id='cluster_test',
                    style={'width': '100%', 'height': '30vh', 'fontSize': '25px','padding-left': '0%', 'padding-right': '0%',
                           'float': 'left', 'align-items': 'center',  'background-color': '#111111', 'color': '#7FDBFF','position':'relative','z-index':'99999','margin-top':'0px','margin-bottom':'0px'},
                    children=[
                        '根据聚类结果和我们的经验分析，我们大致可以将这14000多套房源分为以下3类:①trace1（面积小、价格相对较低、房源多）：这类房源分布范围广，在威海市各个区域均有分布，在靠近市中心如经区环翠区较少，在高区偏远地区和荣成市乳山市文登区分布较多。',
                        '②trace0（面积相对第一类大，经济适用型）这种房源围绕医疗商业中心位置集中分布，地理位置极好，交通方便，为大多数市区购买者首选房源，主要分布在经区高区环翠区。',
                        '③trace2（豪宅类型，平均面积都在200平以上，这种大户型的房源相对数量较少）这种房源一般为地段优秀的市中心或为部分为面积小单价高的学区房，包含少量别墅']
                ),
        # html.Div(
        #     id='pic25',
        #     style={'width': '40%', 'height': '200%', 'padding-left': '5%', 'padding-right': '5%',
        #            'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111'},
        #     children=[
        #         dcc.Graph(
        #             id='kmeans_loc',
        #             style={'height': '50vh'},
        #             figure=dict(

        #                 data=[
        #                         go.Scatter(x=cluster7_array[:,1], y=cluster7_array[:,0], mode='markers'),
        #                         go.Scatter(x=cluster8_array[:,1], y=cluster8_array[:,0], mode='markers'),
        #                         go.Scatter(x=cluster9_array[:,1], y=cluster9_array[:,0], mode='markers')
        #                 ],
        #                 layout=go.Layout(title='威海市住房方位与单价聚类图', paper_bgcolor="#111111",
        #                                  plot_bgcolor='#111111',
        #                                  font=dict(
        #                                      family="Times New Roman", size=20, color='#7fdbff'),
        #                                  yaxis=dict(title='单价 单位：元',
        #                                             titlefont=dict(
        #                                                 color='rgb(148, 103, 189)', size=24),
        #                                             tickfont=dict(
        #                                                 color='#7FDBFF', size=24,),
        #                                             tickwidth=4,
        #                                             tickcolor='#7FDBFF',
        #                                             showline=True,
        #                                             linecolor='#7FDBFF',
        #                                             linewidth=2,
        #                                             showticklabels=True,
        #                                             autorange=True,
        #                                             type='linear',
        #                                             ),
        #                                  xaxis=dict(title='方位',
        #                                             titlefont=dict(
        #                                                 color='rgb(148, 103, 189)', size=24),
        #                                             showline=True,
        #                                             linecolor='#7FDBFF',
        #                                             linewidth=2,
        #                                             autorange=True
        #                                             ))
        #             )
        #         ),
        #     ]

        # ),
        html.Div(
            id='pic26',
            style={'width': '80%', 'height': '200%', 'padding-left': '10%', 'padding-right': '10%',
                   'float': 'left', 'align-items': 'center', 'justify-content': 'center', 'background-color': '#111111','margin-top':'0px','margin-bottom':'0px'},
            children=[
                dcc.Graph(
                    id='importance',
                    style={'height': '70vh'},
                    figure=dict(
                        data=[go.Bar(
                            x=['所在区','户型','面积','装修','有无电梯','小区名称','所在位置'],
                            y=rf.feature_importances_.tolist(),
                            marker=dict(colorscale='Viridis', color=colors,
                                        showscale=True),
                            textposition='auto',
                            opacity=0.7
                        )],

                        layout=go.Layout(title='各变量重要性',
                                         paper_bgcolor="#111111",
                                         plot_bgcolor='#111111',
                                         font=dict(
                                             family="Times New Roman", size=20, color='#7fdbff'),
                                         yaxis=dict(title='重要性指数',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    tickfont=dict(
                                                        color='#7FDBFF', size=24,),
                                                    tickwidth=4,
                                                    tickcolor='#7FDBFF',
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    showticklabels=True,
                                                    autorange=True,
                                                    type='linear',
                                                    ),
                                         xaxis=dict(title='变量',
                                                    titlefont=dict(
                                                        color='rgb(148, 103, 189)', size=24),
                                                    showline=True,
                                                    linecolor='#7FDBFF',
                                                    linewidth=2,
                                                    autorange=True
                                                    )

                                         )

                    )
                )
                
            ]

        ),



    ]
)


@app.callback(
    Output(component_id='top20house', component_property='figure'),
    [Input(component_id='district_choose', component_property='value')]
)
def district_update(district_name):
    unitprice_top = df.sort_values(by="unit_price", ascending=False)[:2000]
    unitprice_top = unitprice_top.sort_values(by="unit_price")
    unitprice_top.set_index(unitprice_top["xiaoqu"], inplace=True)
    x_value = unitprice_top[unitprice_top['district']
                            == district_name][: 10].index
    y_value = unitprice_top[unitprice_top['district']
                            == district_name][: 10]['unit_price'].values
    text_name = district_name+'单价TOP房源'
    traces = []
    traces.append(go.Bar(
        x=x_value,
        y=y_value,
        textposition='auto',
        opacity=0.7,
        marker=dict(colorscale=[[0.0, "rgb(165,0,38)"],
                                [0.1111111111111111, "rgb(215,48,39)"],
                                [0.2222222222222222, "rgb(244,109,67)"],
                                [0.3333333333333333, "rgb(253,174,97)"],
                                [0.4444444444444444, "rgb(254,224,144)"],
                                [0.5555555555555556, "rgb(224,243,248)"],
                                [0.6666666666666666, "rgb(171,217,233)"],
                                [0.7777777777777778, "rgb(116,173,209)"],
                                [0.8888888888888888, "rgb(69,117,180)"],
                                [1.0, "rgb(49,54,149)"]], color=t,
                    showscale=True)


    ))
    fig = dict(
        data=traces,
        layout=go.Layout(title=text_name,
                         paper_bgcolor="#111111",
                         plot_bgcolor='#111111',
                         font=dict(family="Times New Roman",
                                   size=20, color='#7fdbff'),
                         yaxis=dict(title='单位面积价格',
                                    titlefont=dict(
                                        color='rgb(148, 103, 189)', size=24),
                                    tickfont=dict(
                                        color='#7FDBFF', size=24,),
                                    tickwidth=4,
                                    tickcolor='#7FDBFF',
                                    showline=True,
                                    linecolor='#7FDBFF',
                                    linewidth=2,
                                    showticklabels=True,
                                    autorange=True,
                                    type='linear',
                                    ),
                         xaxis=dict(title='房源',
                                    titlefont=dict(
                                        color='rgb(148, 103, 189)', size=24),
                                    showline=True,
                                    linecolor='#7FDBFF',
                                    linewidth=2,
                                    autorange=True
                                    )

                         )
    )
    return fig


@app.callback(
    Output(component_id='top20xiaoqu', component_property='figure'),
    [Input(component_id='slider2', component_property='value')]
)
def xiaoqu_update(slider_value):
    top_xiaoqu = df['unit_price'].groupby(
        df['xiaoqu']).mean().sort_values(ascending=False)[:20]
    x_value = top_xiaoqu.index[:slider_value]
    y_value = top_xiaoqu.values[:slider_value]
    text_name = '单价TOP'+str(slider_value)+'小区'
    traces = []
    traces.append(go.Bar(
        x=x_value,
        y=y_value,
        textposition='auto',
        opacity=0.7,
        marker=dict(colorscale='Viridis', color=t,
                    showscale=True)

    ))
    fig = dict(
        data=traces,
        layout=go.Layout(title=text_name,
                         paper_bgcolor="#111111",
                         plot_bgcolor='#111111',
                         font=dict(family="Times New Roman",
                                   size=20, color='#7fdbff'),
                         yaxis=dict(title='单位面积价格',
                                    titlefont=dict(
                                        color='rgb(148, 103, 189)', size=24),
                                    tickfont=dict(
                                        color='#7FDBFF', size=24,),
                                    tickwidth=4,
                                    tickcolor='#7FDBFF',
                                    showline=True,
                                    linecolor='#7FDBFF',
                                    linewidth=2,
                                    showticklabels=True,
                                    autorange=True,
                                    type='linear',
                                    ),
                         xaxis=dict(title='小区',
                                    titlefont=dict(
                                        color='rgb(148, 103, 189)', size=24),
                                    showline=True,
                                    linecolor='#7FDBFF',
                                    linewidth=2,
                                    autorange=True
                                    )

                         )
    )
    return fig


if __name__ == '__main__':
    app.run_server(host="0.0.0.0")
