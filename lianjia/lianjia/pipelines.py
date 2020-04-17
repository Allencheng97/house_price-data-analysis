# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import csv
import json
import os
import pandas as pd
import re
from scrapy.exceptions import DropItem
from lianjia.items import LianjiaItem
from lianjia.sql import Sql
from lianjia import helper


class LianjiaPipeline(object):
    def __init__(self):
        self.f = open('C:\\Users\\Allen\\Desktop\\test\\data.json', 'w')
        self.id_set = set()

    def open_spider(self,spider):
        pass

    def process_item(self, item, spider):
        # data cleaning:
        # uniform the data format and data type
        # delete all replicate data
        # fill in some none attribute
        house_id = item['house_id']
        if house_id in self.id_set:
            raise DropItem("Duplicate information found:%s" % item)
        xiaoqu = item['xiaoqu']
        if xiaoqu:
            item['xiaoqu'] = xiaoqu.strip().replace('\n', '').replace(' ', '')
        district = item['district']
        if district:
            item['district'] = district.strip().replace('\n', '').replace(' ', '')
        total_price = item['total_price']
        if total_price:
            if type(total_price) != float:
                total_price = float(total_price)
            else:
                pass
        item['total_price'] = total_price * 10000
        # uniform the unit
        # increase the length of total price in sql or there will be out of range error
        unit_price = item['unit_price']
        if unit_price:
            if type(unit_price) != float:
                item['unit_price'] = float(unit_price)
            else:
                pass
        floor = item['floor']
        if floor:
            item['overall_floor'] = float(re.sub('\D', '', floor))
            item['floor'] = floor[:3]
            # subtract the number from str
        house_struct = item['house_struct']
        if house_struct == '暂无数据':
            item['house_struct'] = '平层'
        area = item['area']
        if type(area) != float:
            item['area'] = float(area)
        direction = item['direction']
        if direction:
            direction = direction.strip().replace('\n', '').replace(' ', '')
            if direction == "南":
                direction = "南北"
            if len(direction) > 2:
                direction = direction[1:]
            item['direction'] = direction
        # mortage=item['mortage']
        # pattern=re.compile('\d+')
        # exist_list=['无抵押','暂无数据','有抵押']
        # if mortage:
        #     if mortage=='暂无数据':
        #         mortage='无抵押'
        #     elif mortage not in exist_list:
        #         mortage_num=int(re.findall(pattern,mortage)[0])
        #         mortage=mortage_num
        #     item['mortage'] = mortage
        year_limit = item['yearlimit']
        if year_limit:
            year_limit = str(year_limit)
        item['yearlimit'] = year_limit
        overall_floor = item['overall_floor']
        if overall_floor:
            overall_floor = int(overall_floor)
        item['overall_floor'] = overall_floor

        content = json.dumps(dict(item), ensure_ascii=False) + '\n'
        self.f.write(content)
        if isinstance(item, LianjiaItem):
            house_id = item['house_id']
            ret = Sql.select_raw_house_id(house_id)
            if ret[0] == 1:
                Sql.drop_raw_house_id(item['house_id'])
                xiaoqu = item['xiaoqu']
                district = item['district']
                total_price = item['total_price']
                unit_price = item['unit_price']
                house_type = item['house_type']
                floor = item['floor']
                area = item['area']
                house_struct = item['house_struct']
                in_area = item['in_area']
                building_type = item['building_type']
                direction = item['direction']
                building_structure = item['building_structure']
                fixture = item['fixture']
                elevator_ratio = item['elevator_ratio']
                elevator_exist = item['elevator_exist']
                yearlimit = item['yearlimit']
                list_time = item['list_time']
                trade_type = item['trade_type']
                last_tradetime = item['last_tradetime']
                house_use = item['house_use']
                house_time = item['house_time']
                owner_attribute = item['owner_attribute']
                mortage = item['mortage']
                property_status = item['property_status']
                villia_type = item['villa_type']
                overall_floor = item['overall_floor']
                Sql.insert_raw_data(house_id, xiaoqu, district, total_price, unit_price, house_type, floor, area,
                                    house_struct, in_area,
                                    building_type, direction, building_structure, fixture, elevator_ratio,
                                    elevator_exist, yearlimit,
                                    list_time, trade_type, last_tradetime, house_use, house_time, owner_attribute,
                                    mortage, property_status, villia_type, overall_floor)

            else:
                xiaoqu = item['xiaoqu']
                district = item['district']
                total_price = item['total_price']
                unit_price = item['unit_price']
                house_type = item['house_type']
                floor = item['floor']
                area = item['area']
                house_struct = item['house_struct']
                in_area = item['in_area']
                building_type = item['building_type']
                direction = item['direction']
                building_structure = item['building_structure']
                fixture = item['fixture']
                elevator_ratio = item['elevator_ratio']
                elevator_exist = item['elevator_exist']
                yearlimit = item['yearlimit']
                list_time = item['list_time']
                trade_type = item['trade_type']
                last_tradetime = item['last_tradetime']
                house_use = item['house_use']
                house_time = item['house_time']
                owner_attribute = item['owner_attribute']
                mortage = item['mortage']
                property_status = item['property_status']
                villia_type = item['villa_type']
                overall_floor = item['overall_floor']
                Sql.insert_raw_data(house_id, xiaoqu, district, total_price, unit_price, house_type, floor, area,
                                    house_struct, in_area,
                                    building_type, direction, building_structure, fixture, elevator_ratio,
                                    elevator_exist, yearlimit,
                                    list_time, trade_type, last_tradetime, house_use, house_time, owner_attribute,
                                    mortage, property_status, villia_type, overall_floor)

        return item
    def close_spider(self, spider):
        self.f.close()
        self.data_process()



    def data_process(self):
        f1 = open('C:\\Users\\Allen\\Desktop\\test\\data.json','r')
        records = [json.loads(line) for line in f1.readlines()]
        # read file
        df = pd.DataFrame(records)
        df.drop_duplicates()
        print(df.describe())
        print(df.apply(lambda col: sum(col.isnull()) / col.size))
        delete_list = df[(df.building_type.isnull())].index.tolist()
        delete_list += df[(df.building_structure.isnull())].index.tolist()
        delete_set = set(delete_list)
        delete_list = list(delete_set)
        df = df.drop(delete_list)
        # delete all rows which building_structure or building_type is null
        head, tail = helper.cap(df['unit_price'])
        delete_list = df[df['unit_price'] < head].index.tolist() + df[df['unit_price'] > tail].index.tolist()
        df.drop(delete_list)
        # use cap method  to reduce noise
        df.to_csv(r'C:\\Users\\Allen\\Desktop\\test\\data.csv', encoding='gb18030')
        df.to_csv(r'C:\\Users\\Allen\\Desktop\\test\\data-utf8.csv', encoding='utf-8')
        f1.close()
        # write back
