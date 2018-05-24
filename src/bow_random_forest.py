import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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

# create bag-of-words representation of the features
print("Creating a bag-of-words representation of the training data...")
vectorizer = CountVectorizer(lowercase=True)
vectorizer.fit(train_df.line.tolist())  # train the vectorizer on the lines from the training set to make vocabulary

X_train = vectorizer.transform(train_df.line.tolist())  # transform training lines as BOW
X_test = vectorizer.transform(test_df.line.tolist())  # transform test lines as BOW

y_train = train_df.character.tolist()
y_test = test_df.character.tolist()

print("{} words in the vocabulary.".format(len(vectorizer.vocabulary_)))

# perform cross-validation to find optimal hyperparameters
params = {'n_estimators': [10, 15, 20, 25, 30], 'max_depth': [5, 10, 15, 20]}
print("Performing cross-validation...")
print("Using n_estimators in {}".format(params['n_estimators']))
print("Using max_depth in {}".format(params['max_depth']))
forest = RandomForestClassifier(random_state=123)
cv = GridSearchCV(forest, params)
cv.fit(X_train, y_train)

print('The optimal parameters found are:')
print('max_depth={}'.format(cv.best_params_['max_depth']))
print('n_estimators={}'.format(cv.best_params_['n_estimators']))

# find training and test accuracy
mod = cv.best_estimator_
print("The best model had a training accuracy of {}".format(mod.score(X_train, y_train)))
print("The best model had a test accuracy of {}".format(mod.score(X_test, y_test)))

print("Saving the best model found...")
with open('../results/random_forest_model.pkl', 'wb') as f:
    pickle.dump(mod, f)
print("Done")
