# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class LineItem(scrapy.Item):
    character = scrapy.Field()
    episode = scrapy.Field()
    line = scrapy.Field()
    season = scrapy.Field()

