# -*- coding: utf-8 -*-
from wordcloud import WordCloud
import jieba
from scipy.misc import imread

filename='data-utf8.csv'
backpicture='house2.jpg'
savepicture='wordcloud.jpg'
fontpath = "simhei.ttf"
stopwords=["null","暂无","数据","上传","照片","房本"]

comment_text = open(filename,encoding="utf-8").read()
color_mask = imread(backpicture) 

key_words = jieba.cut(comment_text)
key_words = [word for word in key_words if word not in stopwords]
cut_text = " ".join(key_words) 
cloud = WordCloud(
    font_path=fontpath,
    background_color='white',
    mask=color_mask,
    max_words=2000,
    max_font_size=60
   )
word_cloud = cloud.generate(cut_text)
word_cloud.to_file(savepicture)