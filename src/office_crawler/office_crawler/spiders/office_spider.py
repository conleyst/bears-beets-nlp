import scrapy
from office_crawler.items import LineItem
from bs4 import BeautifulSoup


# create helpers to parse text on web-pages
def clean(s):
    """Remove described actions, whitespace padding, and colons from strings."""
    while s.find("[") > -1:
        start = s.find("[")
        end = s.find("]") + 1
        s = s[:start] + s[end:]
    s = s.strip()
    s = s.replace(':', '')

    return s


def extract_conversation(element):
    """Return list of character-line pairs from element."""
    element_strs = list(element.strings)
    conversation = list(filter(lambda s: '\n' not in s, element_strs))
    cleaned = list(map(clean, conversation))
    lines = [(cleaned[i], cleaned[i + 1]) for i in range(0, len(cleaned), 2)]

    return lines


def extract_season_episode(s):
    """Return season and episode number in a tuple."""
    season = s[-8]
    episode = s[-6:-4]
    return season, episode


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
            item = LineItem(
                conversation=quote.contents,
                url=current_url
            )
            items.append(item)
        return items
