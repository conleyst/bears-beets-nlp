### Source Directory

This directory contains all code used in the project. See instruction in main project README for instructions for running it.

- `office_crawler`<br/>
Contains all code used to scrape the data from [here](http://officequotes.net/index.php). Uses `Scrapy` and `BeautifulSoup` for the scraping.

- `clean_lines.py`<br/>
Contains the script used to clean the raw JSON returned by the web crawler.
 
- `create_train_test_set.py`<br/>
Contains the script used to create a training and test set from the cleaned data.

- `bow_random_forest.py`<br/>
Contains the script used to create and train a random forest model on the data. Uses bag-of-word features.

- `tfidf_log_reg.py`<br/>
Contains the script used to create and train a logistic regression model on the data. Uses tf-idf features.

- `tf_convnet.py`<br/>
Contains the script used to create and train a convolutional neural net, implemented in tensorflow.
