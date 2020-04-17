# -*- coding: utf-8 -*-
import scrapy
from scrapy.http import Request
from lxml import etree
from bs4 import BeautifulSoup
import requests
import json
from lianjia.items import LianjiaItem
import re


class Simplespider(scrapy.Spider):
    name = 'lianjia'
    domins = 'weihai.lianjia.com'
    front_url = 'https://weihai.lianjia.com/ershoufang'

    # front_url = 'https://bj.lianjia.com/ershoufang/pg'
    def start_requests(self):
        yield Request(self.front_url, self.parse)

    def parse(self, response):
        end_urls = response.xpath('/html/body/div[3]/div/div[1]/dl[2]/dd/div/div/child::a/@href').extract()
        front = self.front_url[:-11]
        for end_url in end_urls:
            url = front + end_url
            yield Request(url, callback=self.parse2)

    def parse2(self, response):
        data = etree.HTML(response.text).xpath("//div[@class='page-box house-lst-page-box']/@page-data")[0]
        temp_dic = json.loads(data)
        maxnum = int(temp_dic['totalPage'])
        for num in range(1, maxnum+1):#maxnum + 1
            url = response.request.url + 'pg' + str(num) + '/'
            yield Request(url, callback=self.get_link)

    def get_link(self, response):
        # bs=BeautifulSoup(response.text,'lxml')
        # links=bs.find_all(class_='title')
        # for l in links:
        #     link=l.get('href')
        link_list = response.xpath("//div[@class='info clear']/div[@class='title']/a")
        for link in link_list:
            link = link.xpath("./@href").extract()[0]
            yield Request(link, self.get_data)


    def get_data(self, response):
        item = LianjiaItem()
        pattern = re.compile('(?<=>)\w+(?=<)')
        content = BeautifulSoup(response.text, 'lxml')
        house_use = response.xpath(
            "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[4]/span[2]/text()").extract()[
            0]
        if house_use == '普通住宅':
            item['house_id'] = str(content.find_all('span', class_='info')[2].get_text()[:-2])
            item['xiaoqu'] = content.find_all('a', class_='info')[0].get_text()
            item['district'] = response.xpath('/html/body/div[5]/div[2]/div[4]/div[2]/span[2]/a[1]/text()').extract()[0]
            # item['district'] = content.find_all('a', target='_blank')[3].get_text()
            item['total_price'] = float(content.find_all('span', class_='total')[0].get_text())
            item['unit_price'] = float(content.find_all('span', class_='unitPriceValue')[0].get_text()[:-4])
            item['house_type'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[1]/text()').extract()[0]
            item['floor'] = response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[2]/text()').extract()[0]
            item['area'] = float(
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[3]/text()').extract()[0][:-1])
            item['house_struct'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[4]/text()').extract()[0]
            item['in_area'] = response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[5]/text()').extract()[
                0]
            item['building_type'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[6]/text()').extract()[0]
            item['direction'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[7]/text()').extract()[0]
            item['building_structure'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[8]/text()').extract()[0]
            item['fixture'] = response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[9]/text()').extract()[
                0]
            item['elevator_ratio'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[10]/text()').extract()[0]
            item['elevator_exist'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[11]/text()').extract()[0]
            item['yearlimit'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[2]/div[2]/ul/li[5]/span[2]/text()').extract()[0]
            item['list_time'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[1]/span[2]/text()").extract()[
                0]
            item['trade_type'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[2]/span[2]/text()").extract()[
                0]
            item['last_tradetime'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[3]/span[2]/text()").extract()[
                0]
            item['house_use']=house_use
            item['house_time'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[5]/span[2]/text()").extract()[
                0]
            item['owner_attribute'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[6]/span[2]/text()").extract()[
                0]
            item['mortage'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[7]/span[2]/text()").extract()[
                0].replace('\n', '').replace(' ', '')
            item['property_status'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[8]/span[2]/text()").extract()[
                0]
            item['villa_type']='暂无数据'
        elif house_use =='别墅':
            item['house_id'] = str(content.find_all('span', class_='info')[2].get_text()[:-2])
            item['xiaoqu'] = content.find_all('a', class_='info')[0].get_text()
            item['district']=response.xpath('/html/body/div[5]/div[2]/div[4]/div[2]/span[2]/a[1]/text()').extract()[0]
            # item['district'] = content.find_all('a', target='_blank')[3].get_text()
            item['total_price'] = float(content.find_all('span', class_='total')[0].get_text())
            item['unit_price'] = float(content.find_all('span', class_='unitPriceValue')[0].get_text()[:-4])
            item['house_type'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[1]/text()').extract()[0]
            item['floor'] = response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[2]/text()').extract()[0]
            item['area'] = float(
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[3]/text()').extract()[0][:-1])
            item['in_area'] = response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[4]/text()').extract()[
                0]
            item['direction']=response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[5]/text()').extract()[
                0]
            item['house_struct'] = \
            response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[6]/text()').extract()[
                0]
            item['fixture']=response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[7]/text()').extract()[
                0]
            item['villa_type']=response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[8]/text()').extract()[
                0]
            item['yearlimit']=response.xpath('//*[@id="introduction"]/div/div/div[2]/div[2]/ul/li[5]/span[2]/text()').extract()[0]
            item['list_time']=response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[1]/span[2]/text()").extract()[
                0]
            item['trade_type']=response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[2]/span[2]/text()").extract()[
                0]
            item['last_tradetime'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[3]/span[2]/text()").extract()[
                0]
            item['house_use']=house_use
            item['house_time'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[5]/span[2]/text()").extract()[
                0]
            item['owner_attribute'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[6]/span[2]/text()").extract()[
                0]
            item['mortage'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[7]/span[2]/text()").extract()[
                0].replace('\n', '').replace(' ', '')
            item['property_status'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[8]/span[2]/text()").extract()[
                0]
            item['house_struct']='暂无数据'
            item['elevator_exist']='无'
            item['elevator_ratio']='无'
        elif house_use == '商住两用':
            item['house_id'] = str(content.find_all('span', class_='info')[2].get_text()[:-2])
            item['xiaoqu'] = content.find_all('a', class_='info')[0].get_text()
            item['district'] = response.xpath('/html/body/div[5]/div[2]/div[4]/div[2]/span[2]/a[1]/text()').extract()[0]
            # item['district'] = content.find_all('a', target='_blank')[3].get_text()
            item['total_price'] = float(content.find_all('span', class_='total')[0].get_text())
            item['unit_price'] = float(content.find_all('span', class_='unitPriceValue')[0].get_text()[:-4])
            item['house_type'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[1]/text()').extract()[0]
            item['floor'] = response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[2]/text()').extract()[0]
            item['area'] = float(
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[3]/text()').extract()[0][:-1])
            item['house_struct'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[4]/text()').extract()[0]
            item['in_area'] = response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[5]/text()').extract()[
                0]
            item['building_type'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[6]/text()').extract()[0]
            item['direction'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[7]/text()').extract()[0]
            item['building_structure'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[8]/text()').extract()[0]
            item['fixture'] = response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[9]/text()').extract()[
                0]
            item['elevator_ratio'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[10]/text()').extract()[0]
            item['elevator_exist'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[11]/text()').extract()[0]
            item['yearlimit'] = \
                response.xpath('//*[@id="introduction"]/div/div/div[2]/div[2]/ul/li[5]/span[2]/text()').extract()[0]
            item['list_time'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[1]/span[2]/text()").extract()[
                0]
            item['trade_type'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[2]/span[2]/text()").extract()[
                0]
            item['last_tradetime'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[3]/span[2]/text()").extract()[
                0]
            item['house_use'] = house_use
            item['house_time'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[5]/span[2]/text()").extract()[
                0]
            item['owner_attribute'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[6]/span[2]/text()").extract()[
                0]
            item['mortage'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[7]/span[2]/text()").extract()[
                0].replace('\n', '').replace(' ', '')
            item['property_status'] = response.xpath(
                "//div[@class='introContent']/div[@class='transaction']/div[@class='content']/ul/li[8]/span[2]/text()").extract()[
                0]
            item['villa_type'] = '暂无数据'
        else:
            pass
        yield item
