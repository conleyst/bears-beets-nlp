### Source Directory

This directory contains all code used in the project.

- `office_crawler`<br/>
Contains all code used to scrape the data from [here](http://officequotes.net/index.php). Uses `Scrapy` and `BeautifulSoup` for the scraping. Can be run from within the top-level `office_crawler` directory using the command,

    ```
    scrapy crawl office_crawler -o raw_lines.json
    ```

    This will return a JSON file in the same directory containing the results.

- `clean_lines.py`<br/>
Contains the script used to clean the raw JSON returned by the web-crawler. Should be run from the `src` directory and expects that the output from the crawler is in the `data` directory. Can be run using the command,

    ```
    python clean_lines.py
    ```

    This will return CSV file `lines.csv` in the `data` directory. It contains a row for every line, with information on the who said it and which season and episode it is from.
