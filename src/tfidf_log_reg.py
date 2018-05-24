import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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
tfidf_vectorizer = TfidfVectorizer(lowercase=True)
tfidf_vectorizer.fit(train_df.line.tolist())

X_train = tfidf_vectorizer.transform(train_df.line.tolist())
X_test = tfidf_vectorizer.transform(test_df.line.tolist())
y_train = train_df.character.tolist()
y_test = test_df.character.tolist()

# perform cross-validation to select regularization strength
print("Performing cross-validation...")
params = {'C': [10**i for i in range(-2, 3)]}
print("Using C in {}".format(params['C']))
log_reg = LogisticRegression(multi_class="multinomial", solver='sag', max_iter=300)
cv = GridSearchCV(log_reg, params)
cv.fit(X_train, y_train)
mod = cv.best_estimator_

# report outcome
print('The optimal parameter was:')
print('C={}'.format(cv.best_params_['C']))

print("The best model had a training accuracy of {}".format(mod.score(X_train, y_train)))
print("The best model had a test accuracy of {}".format(mod.score(X_test, y_test)))

print("Saving the best model found...")
with open('../results/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(mod, f)
print("Done")
