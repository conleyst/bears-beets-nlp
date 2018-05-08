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
    conversation = list(filter(lambda s: s != '\n', element_strs))
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
        'http://officequotes.net/no1-01.php'
    ]

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'lxml')
        all_quotes = soup.find_all('div', class_='quote')
        current_url = response.request.url
        season_ep = extract_season_episode(current_url)
        lines = []
        for element in all_quotes:
            for pair in extract_conversation(element):
                item = LineItem(
                    character=str(pair[0]),
                    line=str(pair[1]),
                    season=season_ep[0],
                    episode=season_ep[1]
                )
                lines.append(item)
        return lines
