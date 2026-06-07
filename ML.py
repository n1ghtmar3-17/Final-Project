import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#results
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dfFake = pd.read_csv("Fake.csv")
dfTrue = pd.read_csv("True.csv")

#0 = fake; 1 = real
dfFake["type"] = 0
dfTrue["type"] = 1

#total + cleaning
df = pd.concat([dfFake, dfTrue], axis = 0)
df = df.drop(["subject", "date"], axis = 1)
df["text"] = df["title"] + " " + df["text"]
df = df.drop(["title"], axis = 1)
df = df.sample(frac = 1) #randomize
df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

#text cleaning
import re
import string
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) #removes brackets
    text = re.sub(r"\\W"," ",text) #removes symbols
    text = re.sub(r'https?://\S+|www\.\S+', '', text) #removes links
    text = re.sub(r'<.*?>+', '', text) #removes html tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) #removes punctuation
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text) #removes numbers in words?
    return text
df["text"] = df["text"].apply(wordopt)

#Create the variables for ML
X = df["text"]
Y = df["type"]

#convert words into numbers for ML (use TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7) #what does this do??
X_vec = vectorizer.fit_transform(X)

#split train test + hyperparameter tuning
X_train, X_test, Y_train, Y_test = train_test_split(X_vec, Y, test_size = 0.25, random_state = 42)
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, Y_train)

Y_pred = LR.predict(X_test)
print("Accuracy", accuracy_score(Y_test, Y_pred))
print("Classification Report \n", classification_report(Y_test, Y_pred))


#hyperparameter tuning
# param_grid = {
#     'max_depth': [3, 5, 7, 10, 15], #depth of tree
#     'min_samples_split': [2, 5, 10], #Minimum samples required to split a node
#     'min_samples_leaf': [1, 2, 4], #minimun number of samples to be at a node
#     'max_features': ['sqrt', 'log2', None], #number of features to consider before making the best split: sqrt = sqrt of total # of data, log2 = log base 2(total features), None = use all
#     'criterion': ['gini', 'entropy']
# }
# tuned_DTC = DecisionTreeClassifier(random_state = 25)
# GS = GridSearchCV(estimator = tuned_DTC, param_grid = param_grid, cv = 5, scoring = "accuracy", n_jobs = -1)
# GS.fit(X_train, Y_train)
# best_params = GS.best_params_
# print(best_params)

# GS_Y_pred = best_params.predict(X_test)
# print("GridSearch Accuracy", accuracy_score(Y_test, GS_Y_pred))
# print("GridSearchClassification Report \n", classification_report(Y_test, GS_Y_pred))


#save the model
import joblib
joblib.dump(LR, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
