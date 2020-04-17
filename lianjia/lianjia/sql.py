import mysql.connector
from lianjia import  settings
MYSQL_HOSTS = settings.MYSQL_HOSTS
MYSQL_USER = settings.MYSQL_USER
MYSQL_PASSWORD = settings.MYSQL_PASSWORD
MYSQL_PORT = settings.MYSQL_PORT
MYSQL_DB = settings.MYSQL_DB

cnx=mysql.connector.connect(user=MYSQL_USER,password=MYSQL_PASSWORD,host=MYSQL_HOSTS,database=MYSQL_DB)
cur=cnx.cursor(buffered=True)

class Sql:
    @classmethod
    def insert_dd_data(cls,house_id,xiaoqu,district,total_price,
                       unit_price,house_type,floor,area,house_struct,
                       in_area,building_type,direction,building_structure,
                       fixture,elevator_ratio,elevator_exist,
                       yearlimit,list_time,trade_type,last_tradetime,
                       house_use,house_time,owner_attribute,mortage,
                       property_status,villa_type,overall_floor):
        sql ='INSERT INTO houseinfo (`house_id`, `xiaoqu`, `district`, `total_price`,`unit_price`,`house_type`,`floor`,`area`,`house_struct`,`in_area`,`building_type`,`direction`,`building_structure`,`fixture`,`elevator_ratio`,`elevator_exist`,`yearlimit`,`list_time`,`trade_type`,`last_tradetime`,`house_use`,`house_time`,`owner_attribute`,`mortage`,`property_status`,`villa_type`,`overall_floor`) VALUES(%(house_id)s, %(xiaoqu)s, %(district)s, %(total_price)s, %(unit_price)s, %(house_type)s, %(floor)s, %(area)s, %(house_struct)s, %(in_area)s, %(building_type)s, %(direction)s, %(building_structure)s, %(fixture)s, %(elevator_ratio)s,%(elevator_exist)s,%(yearlimit)s, %(list_time)s, %(trade_type)s, %(last_tradetime)s, %(house_use)s, %(house_time)s, %(owner_attribute)s, %(mortage)s, %(property_status)s, %(villa_type)s, %(overall_floor)s)'
        value = {
            'house_id':house_id,
            'xiaoqu':xiaoqu,
            'district':district,
            'total_price':total_price,
            'unit_price':unit_price,
            'house_type':house_type,
            'floor':floor,
            'area':area,
            'house_struct':house_struct,
            'in_area':in_area,
            'building_type':building_type,
            'direction':direction,
            'building_structure':building_structure,
            'fixture':fixture,
            'elevator_ratio':elevator_ratio,
            'elevator_exist':elevator_exist,
            'yearlimit':yearlimit,
            'list_time':list_time,
            'trade_type':trade_type,
            'last_tradetime':last_tradetime,
            'house_use':house_use,
            'house_time':house_time,
            'owner_attribute':owner_attribute,
            'mortage':mortage,
            'property_status':property_status,
            'villa_type':villa_type,
            'overall_floor':overall_floor
        }
        cur.execute(sql, value)
        cnx.commit()
    @classmethod
    def drop_house_id(cls,house_id):
        sql='DELETE FROM houseinfo WHERE house_id=%(house_id)s;'
        value ={
            'house_id': house_id
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_house_id(cls, house_id):
        sql = "SELECT EXISTS(SELECT 1 FROM houseinfo WHERE house_id=%(house_id)s)"
        value = {
            'house_id': house_id
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_xiaoqu(cls, xiaoqu):
        sql = "SELECT EXISTS(SELECT 1 FROM houseinfo WHERE xiaoqu=%(xiaoqu)s)"
        value = {
            'xiaoqu': xiaoqu
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_district(cls, district):
        sql = "SELECT EXISTS(SELECT 1 FROM houseinfo WHERE district=%(district)s)"
        value = {
            'district': district
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_housetype(cls, housetype):
        sql = "SELECT EXISTS(SELECT 1 FROM houseinfo WHERE house_type=%(house_type)s)"
        value = {
            'house_type': housetype
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_houseuse(cls, house_use):
        sql = "SELECT EXISTS(SELECT 1 FROM houseinfo WHERE house_use=%(house_use)s)"
        value = {
            'house_use': house_use
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def insert_raw_data(cls, house_id, xiaoqu, district, total_price,
                       unit_price, house_type, floor, area, house_struct,
                       in_area, building_type, direction, building_structure,
                       fixture, elevator_ratio, elevator_exist,
                       yearlimit, list_time, trade_type, last_tradetime,
                       house_use, house_time, owner_attribute, mortage,
                       property_status, villa_type, overall_floor):
        sql = 'INSERT INTO rawhouseinfo (`house_id`, `xiaoqu`, `district`, `total_price`,`unit_price`,`house_type`,`floor`,`area`,`house_struct`,`in_area`,`building_type`,`direction`,`building_structure`,`fixture`,`elevator_ratio`,`elevator_exist`,`yearlimit`,`list_time`,`trade_type`,`last_tradetime`,`house_use`,`house_time`,`owner_attribute`,`mortage`,`property_status`,`villa_type`,`overall_floor`) VALUES(%(house_id)s, %(xiaoqu)s, %(district)s, %(total_price)s, %(unit_price)s, %(house_type)s, %(floor)s, %(area)s, %(house_struct)s, %(in_area)s, %(building_type)s, %(direction)s, %(building_structure)s, %(fixture)s, %(elevator_ratio)s,%(elevator_exist)s,%(yearlimit)s, %(list_time)s, %(trade_type)s, %(last_tradetime)s, %(house_use)s, %(house_time)s, %(owner_attribute)s, %(mortage)s, %(property_status)s, %(villa_type)s, %(overall_floor)s) '
        value = {
            'house_id': house_id,
            'xiaoqu': xiaoqu,
            'district': district,
            'total_price': total_price,
            'unit_price': unit_price,
            'house_type': house_type,
            'floor': floor,
            'area': area,
            'house_struct': house_struct,
            'in_area': in_area,
            'building_type': building_type,
            'direction': direction,
            'building_structure': building_structure,
            'fixture': fixture,
            'elevator_ratio': elevator_ratio,
            'elevator_exist': elevator_exist,
            'yearlimit': yearlimit,
            'list_time': list_time,
            'trade_type': trade_type,
            'last_tradetime': last_tradetime,
            'house_use': house_use,
            'house_time': house_time,
            'owner_attribute': owner_attribute,
            'mortage': mortage,
            'property_status': property_status,
            'villa_type': villa_type,
            'overall_floor': overall_floor
        }
        cur.execute(sql, value)
        cnx.commit()

    @classmethod
    def drop_raw_house_id(cls,house_id):
        sql='DELETE FROM rawhouseinfo WHERE house_id=%(house_id)s'
        value ={
            'house_id': house_id
        }
        cur.execute(sql, value)


    @classmethod
    def select_raw_house_id(cls, house_id):
        sql = "SELECT EXISTS(SELECT 1 FROM rawhouseinfo WHERE house_id=%(house_id)s)"
        value = {
            'house_id': house_id
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_raw_xiaoqu(cls, xiaoqu):
        sql = "SELECT EXISTS(SELECT 1 FROM rawhouseinfo WHERE xiaoqu=%(xiaoqu)s)"
        value = {
            'xiaoqu': xiaoqu
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_raw_district(cls, district):
        sql = "SELECT EXISTS(SELECT 1 FROM rawhouseinfo WHERE district=%(district)s)"
        value = {
            'district': district
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_raw_housetype(cls, housetype):
        sql = "SELECT EXISTS(SELECT 1 FROM rawhouseinfo WHERE house_type=%(house_type)s)"
        value = {
            'house_type': housetype
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]

    @classmethod
    def select_raw_houseuse(cls, house_use):
        sql = "SELECT EXISTS(SELECT 1 FROM rawhouseinfo WHERE house_use=%(house_use)s)"
        value = {
            'house_use': house_use
        }
        cur.execute(sql, value)
        return cur.fetchall()[0]
