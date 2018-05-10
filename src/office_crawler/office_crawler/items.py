# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class LineItem(scrapy.Item):
    conversation = scrapy.Field()
    url = scrapy.Field()


