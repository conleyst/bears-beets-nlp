import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle


# import training and test sets
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# encode characters as integers
unique_chars = train_df.character.unique().tolist()
char_dict = dict(zip(unique_chars, range(len(unique_chars))))
train_df.character = train_df.character.apply(lambda x: char_dict[x])
test_df.character = test_df.character.apply(lambda x: char_dict[x])

# create tf-idf matrix of features
print("Creating tf-idf features from training data...")
tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
tfidf_vectorizer.fit(train_df.line.tolist())

X_train = tfidf_vectorizer.transform(train_df.line.tolist())
X_test = tfidf_vectorizer.transform(test_df.line.tolist())

y_train = train_df.character.tolist()
y_test = train_df.character.tolist()

