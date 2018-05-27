# Bears, Beets, NLP

This repository contains all data, code, and results used to scrape data and create a series of classifiers to predict speakers of lines from The Office. Each classifier uses different NLP techniques to handle the data.

The models used include a random forest model using a bag-of-words model for text, logistic regression using tf-idf features, and a convolutional neural net as described in the paper [here](https://arxiv.org/abs/1408.5882), implemented in `tensorflow`. For the convnet, embeddings were learned, but were initialized as previously learned, existing Glove embeddings. The data downloaded for the Glove embeddings was the file `glove.6B.zip` from [here](https://nlp.stanford.edu/projects/glove/), and the 100-dimensional embeddings were used.

### Project Structure

The project is contained in:

- `data`: Contains the raw, cleaned data, and training & test sets used in the project. Scripts for producing these can be found in the `src` directory.

- `results`: Contains the outputs from the project, including the EDA as a Jupyter notebook, `sklearn` models in `pkl` format, and parameters for a convolutional neural net trained with Tensorflow.

- `src`: Contains all the cleaning and processing scripts, as well as the code used to train the models. Details can be found in the `README` of the directory.

- `environment.yml`: A list of packages used in the project, suitable for use with `conda` for creating an environment to run the project in.

### Project Requirements

The entire project was written in Python, version 3.6. All required packages can be found in `environment.yml`. With Anaconda installed, a virtual environment suitable for running the project can be installed with the command,

```
conda env create -f environment.yml
```

To activate the environment, use `source activate theoffice` and to deactivate the environment, use `source deactivate theoffice`.

### Running the Project

To run and repeat the project from scratch with just the code, the steps are as follows, assuming you're in the virtual environment created with `environment.yml`,

1. Scrape the data and store in JSON format  
From within `src/office_crawler`, use the command,
    ```
    scrapy crawl office_crawler -o ../data/raw_lines.json
    ```

2. Clean the raw data  
From within the `src` directory, use the command,
    ```
    python clean_lines.py
    ```
    This will output a file `lines.csv` in the `data` directory.  

3. Create the training and test set  
From within the `src` directoryCode, use the command,
    ```
    python create_train_test_set.py -dat "../data/raw_lines.json" -d "../data"
    ```
    This will output two files in `data`, `train.csv` and `test.csv`.

4. Perform the EDA  
This can be run from within the `EDA.ipynb` notebook. Save the figures created by uncommented the relevant code.

5. Create, train, and save the random forest model  
From within the `src` directory, use the command,

    ```
    python bow_random_forest.py
    ```
    This will output a `pkl` file in `results` saving the trained model. Note that for reproducibility, a random seed was set. Change or remove if you would like to experiment.

6. Create, train, and save the logistic regression model  
From within the `src` directory, use the command,

    ```
    python tfidf_log_red.py
    ```
    This will output a `pkl` file in `results` saving the trained model. Note that for reproducibility, a random seed was set. Change or remove if you would like to experiment.

7. Create, train, and save the convnet model and model predictions
    - Download the Glove embeddings from the link above. Extract the file `glove.6B.100d.txt`. From within the directory containing the file, open the Python interpreter and use the command,

        ```
        import gensim
        gensim.scripts.glove2word2vec.glove2word2vec("glove.6B.100d.txt", "word2vec_words")
        ```
        This should output a file `word2vec_words` containing the embeddings in the word2vec format that `gensim` uses. Store `word2vec_words` in whichever directory you would like, but you will need the path to the file.

    - From within the `src` directory, use the command,

        ```
        python tf_convnet.py -wv <path_to_word2vec_words>
        ```
        where `<path_to_word2vec_words>` is a string giving the path to `word2vec_words`, e.g. `"/.../word2vec_words"`.

        Note that the model uses 60 epochs and 100 filters. If the model is taking too long to train on your machine, try reducing the number of epochs or the number of filters used.

        This will output saved parameters to restore the model in `results/data` as well as the model softmax predictions in `convnet/data` as `numpy` arrays.
