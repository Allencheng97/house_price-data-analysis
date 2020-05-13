# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class LianjiaItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    house_id=scrapy.Field()#房屋编号
    xiaoqu=scrapy.Field()  #小区名称
    district=scrapy.Field()#所在区域
    total_price=scrapy.Field()#总价
    unit_price=scrapy.Field()#单价
    house_type=scrapy.Field()#户型
    floor=scrapy.Field()#所在楼层
    area=scrapy.Field()#建筑面积
    house_struct=scrapy.Field()#户型结构
    in_area=scrapy.Field()#套内面积
    building_type=scrapy.Field()#建筑类型
    direction=scrapy.Field()#房屋朝向
    building_structure=scrapy.Field()#建筑结构
    fixture=scrapy.Field()#装修情况
    elevator_ratio=scrapy.Field()#梯户比例
    elevator_exist=scrapy.Field()#配备电梯
    yearlimit=scrapy.Field()#产权年限
    list_time=scrapy.Field()#挂牌时间
    trade_type=scrapy.Field()#交易类型
    last_tradetime=scrapy.Field()#上次交易时间
    house_use=scrapy.Field()#房屋用途
    house_time=scrapy.Field()#房屋年限
    owner_attribute=scrapy.Field()#产权所属
    mortage=scrapy.Field()#抵押信息
    property_status=scrapy.Field()#房本信息
    overall_floor=scrapy.Field()#总楼层
    villa_type=scrapy.Field()
    weizhi=scrapy.Field()






    # housename = scrapy.Field()          #房屋名称
    # yearlimit = scrapy.Field()          #产权年限
    # houselink = scrapy.Field()          # 链接
    # totalprice=scrapy.Field()           #总价
    # unitprice=scrapy.Field()            #每平米单位价格
    # housetype=scrapy.Field()            #房屋户型
    # housearea=scrapy.Field()            #套内面积
    # housefloor=scrapy.Field()           #楼层
    # house_use=scrapy.Field()            #房屋用途
    # houseproperty=scrapy.Field()        #交易属性
    # follow=scrapy.Field()               #关注次数
    # view=scrapy.Field()                 #观看次数
    # district=scrapy.Field()             #所属区域
    # sold_tprice=scrapy.Field()          #销售总价
    # sold_aprice=scrapy.Field()          #销售均价
    # sold_time=scrapy.Field()            #销售时间
    # community_aprice=scrapy.Field()     #小区均价
    # community_time=scrapy.Field()       #小区年份
    # direction=scrapy.Field()            #房屋朝向
    # elevator=scrapy.Field()             #有无电梯

    


