#!/usr/bin/env python

import argparse
import pandas as pd
import re
import nltk
import string
from sklearn.model_selection import train_test_split

# handle args
parser = argparse.ArgumentParser()
parser.add_argument("-dat", "--data", required=True, type=str, help="Path to CSV of lines to be split.")
parser.add_argument("-d", "--dir", required=True, type=str, help="Directory to output train/test sets into")
args = vars(parser.parse_args())

# import data
lines = pd.read_csv(args['data'])

# filter to characters that spoke at least 1000 lines
# total lines spoken over the course of the series
total_lines = lines.groupby('character').count()[['line']].sort_values('line', ascending=False).reset_index()

# filter to characters that had at least 1000 lines
gt1000 = total_lines.loc[total_lines.line >= 1000]
char_classes = gt1000.character.tolist()  # list of characters that spoke >=1000 lines
filtered_lines = lines.loc[lines.character.isin(char_classes)]
filtered_lines = filtered_lines.reset_index(drop=True)


# filter so each unique line only spoken by one character
def get_words(s, keep_stopwords=False):
    """Return words from sentence s with with punctuation removed & words lower-case.
    Args:
        s --- string, to be tokenized
        keep_stopwords --- if False, stopwords will be removed
    Returns a list of strings.
    """
    eng_stopwords = nltk.corpus.stopwords.words('english')
    punctuation = set(string.punctuation) - set("\'")

    s = re.sub("[^A-Za-z0-9(),!?:\'`]", " ", s)  # remove any special characters
    s = ''.join(list(filter(lambda x: x not in punctuation, s)))  # remove punctuation
    s = s.split()
    s = list(map(lambda x: x.lower(), s))
    if not keep_stopwords:
        s = list(filter(lambda x: x not in eng_stopwords, s))

    return s


# turn each line into a list of words, lowercase, without punctuation
filtered_lines['words'] = filtered_lines['line'].apply(lambda x: get_words(x, keep_stopwords=True))
filtered_lines['words'] = filtered_lines['words'].apply(lambda x: tuple(x))  # to make hashable

# group by line and speaker
filtered_grouped = filtered_lines.groupby(['words', 'character']).count().reset_index()
# take just most frequent speaker of each line
top_speaker = filtered_grouped.sort_values('line', ascending=False).groupby('words').first().reset_index()

# inner join on (character, words)
lines_joined = pd.merge(filtered_lines, top_speaker[['character', 'words']], on=['character', 'words'])
lines_joined = lines_joined[['character', 'line', 'season', 'episode']]

# create train test split, 20% test set, split by character
lines_train, lines_test, \
    char_train, char_test = train_test_split(lines_joined.iloc[:, 1:], lines_joined['character'],
                                             test_size=0.2, random_state=77, stratify=lines_joined['character'])
lines_train['character'] = char_train
lines_test['character'] = char_test

# save training/test sets
lines_train.to_csv(args['dir'] + "/train.csv", index=False)
lines_test.to_csv(args['dir'] + "/test.csv", index=False)
