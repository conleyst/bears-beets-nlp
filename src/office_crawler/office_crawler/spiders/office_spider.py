import scrapy
from office_crawler.items import LineItem
from bs4 import BeautifulSoup


class OfficeSpider(scrapy.Spider):
    name = 'office_crawler'
    allowed_domains = ['officequotes.net']
    start_urls = [
        'http://officequotes.net/index.php'
    ]

    # start at home page and retrieve links, parse linked pages with parse_quotes
    def parse(self, response):
        links = response.xpath('//a[contains(@href, "no")]/@href').extract()
        for link in links:
            url = "http://officequotes.net/" + link
            yield scrapy.Request(url, callback=self.parse_quotes)

    # define what to output at each page
    def parse_quotes(self, response):
        soup = BeautifulSoup(response.text, 'lxml')
        all_quotes = soup.find_all('div', class_='quote')  # retrieve all div elements with tag 'quote'
        current_url = response.request.url
        items = []
        for quote in all_quotes:
            text = list(quote.strings)
            text = list(filter(lambda s: s != '\n', text))
            item = LineItem(
                conversation=text,
                url=current_url
            )
            items.append(item)
        return items
