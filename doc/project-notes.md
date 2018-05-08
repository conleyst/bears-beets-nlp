
- All of the lines are in boxes. The div elements of the boxes are tagged `quote`. This means that we can get all of the html `div` elements with the tag `quote` by using the xpath command `response.xpath('//div[contains(@class, "quote")]')`. Then extract the first using `response.xpath('//div[contains(@class, "quote")]').extract_first()`, etc. I think this exactly returns all the quote boxes, meaning all of the lines.
- We can do the exact same thing with css using `response.css('div.quote').extract_first()`. Again, I think it returns all of the correct things.
- Each div element will have to be dealt with in turn using something like BeautifulSoup.

- I can get the links in a list using the xpath `response.xpath('//a[contains(@href, "no")]/@href').extract()`. Takes advantage of the fact that no other element with the `a` tag has an `href` attribute containing the string `no`.
    - If I'm starting at 1-1, then should use `response.xpath('//a[contains(@href, "no")]/@href').extract()[1:]` so I don't revisit the same page twice.


#### Creating the Crawler With Scrapy

There's a nice outline of all of this in the scrapy documentation [here](https://doc.scrapy.org/en/0.10.3/intro/tutorial.html), and that's where I'm drawing a lot of this from.

The general strategy for a web crawler is:

1. Set-up the project structure
2. Define the actual items you want to return
3. Create the spider and define how it will parse a single page
4. Create the item pipeline
5. Define how your spider should follow links

##### Create the Folder Structure

- Scrapy will initialize everything using the command `scrapy startproject <project_name>` from within the directory where you want the crawler files to be stored. Created mine with `scrapy startproject office_crawler`.

##### Create the Item to be returned

- Ignore for now the sctual crawling from page to page. Once we're on a page, we're going to select the things that we want (the lines) using BeautifulSoup, and we're going to store the lines in a csv, along with the characters who said them, and the season and episode the line is from.
- For convenience, scrapy has a notion of an item, and items are what is actually returned.  Think of items as a specific example from the broader class of what you want to return. For example, we're looking for lines said by characters in the office, so one of the items we want our scraper to return is,

    > ("No you cannot. It has to be official, and it has to be urine.", "Dwight", 2, 20)

    This is a line that Dwight says in the 20th episode of season two and it's going to be one of the lines returned by the web crawler we're creating. Using a tuple here as the format is somewhat arbitrary, which is why we use items. Items standardize the way that info will be returned. Items are implemented as a class, and every line is going to be an instance of the class. Each of those entries in the tuple above will be a class attribute. The item class we'll create is basic, but if you want a refresher on Python classes, a nice source is @@here@@. We'll call the class `LineItem` and add the following to `items.py`,

    ```{python}
    from scrapy.item import Item, Field

    class LineItem(Item):
        self.line = Field()
        self.character = Field()
        self.season = Field()
        self.episode = Field()
    ```

    You can read more about items in the [scrapy documentation](https://doc.scrapy.org/en/latest/topics/items.html).

#### Create the Spider

- Still focusing on one page and not traversing the website.
- The spider is the thing that actually crawls around the website, going to links that we specify and deploying the BeautifulSoup code that we're going to write to parse the HTML there.
- The spider we're creating is defined as a class (specifically, as a subclass of the `BaseSpider` class in scrapy). When we want to send our crawler out to explore a website, we do so with an instance of the defined class. In defining the class, we need to give it:
    1.  `name`: Spiders need a unique name. We'll use this name to call it from the shell.
    2. `start_url`: A list of URLs to start the crawler at
    2. `parse()` method: When the spider attempts to connect to a webpage, it needs some way to parse the response that it gets. This method is what it uses. Ours is going to contain the BeautifulSoup code that extracts the lines and info we want and returns them in the form of a `LineItem`.


    ```{python}
    from scrapy.spider import BaseSpider

    class OfficeSpider(BaseSpider):
        name = 'office_spider'
        allowed_url = 'http://officequotes.net'
        start_url = ['http://officequotes.net/no1-01.php']

        def parse(self, response):
            ...
    ```
- Websites are all different. This means that depending on what you're going to have to spend some time looking at the page HTML in order to figure out exactly how you can pull out the info you're interested in. There is no universal way, and if the page layout changes, then you'll probably need to rewrite your crawler. You can specify specific elements in the page using CSS or XPath. By inspecting the HTML for the pages we're interested in, we can see that the lines are all contained in elements within the page with the tag `<div class='quote'>`, and they seem to be the only elements in the page with these tags. We can specify these using XPath as `response.xpath('//div[contains(@class, "quote")]').extract()`, which returns a list of the elements with the tags. We can then handle the items in the list using BeautifulSoup.
- Parse a given page and extract the quotes using the code
- to write to csv, use `scrapy crawl <spider name> -o file.csv -t csv`
- in settings, set `DOWNLOAD_DELAY=1`. Waits a second before requesting a new page.
