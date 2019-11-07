#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import re
import nltk
import pandas as pd
import numpy as np
import collections
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten, Dense, Dropout, Convolution1D, MaxPooling1D, SpatialDropout1D, Input
import timeit

start = timeit.default_timer()

nltk.data.path.append("/modules/cs918/nltk_data/")
TRAINING = "twitter-training-data.txt"

#Preprocess a single tweet
def preprocess(tweet):

    urls = re.compile(r"\b(https?://)?([a-z_\-0-9]+)(\.[a-z_\-0-9]+)+(/\S*)*\b")
    tweet = re.sub(urls, r"_URLLINK", tweet)

    #Must be at least 2 chars long, alphachars only
    allcaps = re.compile(r"\b(([A-Z0-9]*[A-Z][A-Z0-9]){2,})\b")
    tweet = re.sub(allcaps, r"_ALLCAPS_\1", tweet)

    mentions = re.compile(r"(@[A-Za-z0-9_]+)") 
    tweet = re.sub(mentions, r"_USERMENTION", tweet)

    #Hastags must begin with a alphabetical char, followed by alphachars + _
    hashtag = re.compile(r"(#[A-Za-z_][A-Za-z0-9_]*)") 
    tweet = re.sub(hashtag, r"_HASHTAG", tweet)

    alpha = re.compile(r"[^a-zA-Z0-9_ ]")
    tweet = re.sub(alpha, "", tweet)

    #Reduce words with 3 or more consecutive chars to 2 chars
    multi = re.compile(r"(\w)\1{3,}")
    tweet = re.sub(multi, r"\1\1", tweet)

    return tweet.lower()

#Take the name of some data file, extract and preprocess it
def extract(name):
    
    data = pd.read_csv(name, sep="\t", names=["id", "class", "text"], header=None, dtype={"id":str})
    data["text"] = data["text"].map(preprocess)

    return data

train = extract(TRAINING)
twtkn = TweetTokenizer()

tf_vect = CountVectorizer(twtkn.tokenize, min_df=3, max_df=0.8)
tfidf_vect = TfidfVectorizer(tokenizer=twtkn.tokenize, strip_accents=None, ngram_range=(1, 3), min_df=3, max_df=0.8, sublinear_tf=False)

x_train = train["text"].values
y_train = train["class"].values

"""
Classifier 1
"""

print("=" * 80)
classifier = "naive bayes"

#Tranform the text to the tf and tfidf features
tf_train = tf_vect.fit_transform(x_train)
tfidf_train = tfidf_vect.fit_transform(x_train)

clf = MultinomialNB()
clf.fit(tf_train, y_train)

clf2 = MultinomialNB()
clf2.fit(tfidf_train, y_train)

for testset in testsets.testsets:

    test = extract(testset)
    
    x_test = test["text"].values
    y_test = test["class"].values

    tf_test = tf_vect.transform(x_test)
    tfidf_test = tfidf_vect.transform(x_test)
    
    acc = clf.score(tf_test, y_test)
    print("accuracy for multinomial naive bayes: ", acc*100, "%")

    preds = clf.predict(tf_test)
    predictions = {row["id"] : preds[i] for i, row in test.iterrows()}

    evaluation.evaluate(predictions, testset, classifier)
    evaluation.confusion(predictions, testset, classifier)

    acc2 = clf2.score(tfidf_test, y_test)
    print("accuracy for multinomial naive bayes with tfidf: ", acc2*100, "%")

    preds = clf2.predict(tfidf_test)
    predictions2 = {row["id"] : preds[i] for i, row in test.iterrows()}

    evaluation.evaluate(predictions2, testset, classifier)
    evaluation.confusion(predictions2, testset, classifier)


"""
Classifier 2
"""

print("=" * 80)
classifier = "SVM"

#If the model is already trained, then load it, otherwise train again
#NOTE: Training the SVM model can take multiple hours!!!!!
try:
    grid_svm = joblib.load("svm.pkl")
except:
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    np.random.seed(1)
    pipeline_svm = make_pipeline(tfidf_vect, SVC(probability=True, kernel="linear", class_weight="balanced"))

    grid_svm = GridSearchCV(pipeline_svm, param_grid = {"svc__C": [0.01, 0.1, 1]}, 
    cv = kfolds, scoring="f1_macro", verbose=1, n_jobs=-1) 

    grid_svm.fit(x_train, y_train)
    joblib.dump(grid_svm, "svm.pkl")

for testset in testsets.testsets:

    test = extract(testset)
    
    x_test = test["text"].values
    y_test = test["class"].values

    preds = grid_svm.predict(x_test)
    predictions = {row["id"] : preds[i] for i, row in test.iterrows()}
    acc = (preds == y_test).mean()
    print("accuracy for svm with tfidf: ", acc*100, "%")

    print("Precision: {}, Recall: {}".format(*(evaluation.evaluate(predictions, testset, classifier))))
    evaluation.confusion(predictions, testset, classifier)


"""
Classifier 3
"""

print("=" * 80)
classifier = "CNN"

#Tokenise the data and compute the vocabulary
def transform(data):
    def update_vocab_counter(row):
        for word in row:
            vocab_counter[word] += 1

    def transform_to_ids(row):
        return [w2id[w] for w in row]

    vocab_counter = collections.Counter()
    w2id = dict()

    data["tokenized"] = data["text"].apply(twtkn.tokenize)
    data["tokenized"].apply(update_vocab_counter);
    vocab = sorted(vocab_counter, key=vocab_counter.get, reverse=True)

    w2id = {w:i for i, w in enumerate(vocab)}
    data["tokenized"] = data["tokenized"].apply(lambda x: transform_to_ids(x))

    return data["tokenized"].values, vocab

maxlen = 100
x_train, vocab = transform(train)
x_train = pad_sequences(x_train, maxlen=maxlen)

#One hot encoding
classes = ["negative", "neutral", "positive"]
transformed = LabelEncoder().fit_transform(classes).tolist()
labels = dict(zip(classes, transformed))
labels.update(dict(zip(transformed, classes)))
y_train = train["class"].map(lambda x:labels[x]).values
y_train_enc = to_categorical(y_train, nb_classes=3)

#Create the CNN
model = Sequential()
model.add(Embedding(input_dim=len(vocab), output_dim=32, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(Dropout(0.25))
model.add(Convolution1D(64, 5, activation="relu"))
model.add(Dropout(0.25))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.85))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["fmeasure"])
model.fit(x_train, y_train_enc, validation_split=0.2, nb_epoch=4, batch_size=32)

for testset in testsets.testsets:

    test = extract(testset)
    x_test, _ = transform(test)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    y_test = test["class"].map(lambda x:labels[x]).values
    y_test_enc = to_categorical(y_test, nb_classes=3)

    preds = model.predict_classes(x_test, batch_size=32, verbose=1)
    predictions = {row["id"] : (lambda x:labels[x])(preds[i]) for i, row in test.iterrows()}

    acc = (preds == y_test).mean()
    print("\naccuracy for CNN: ", acc*100, "%")
    evaluation.evaluate(predictions, testset, classifier)
    evaluation.confusion(predictions, testset, classifier)


end = timeit.default_timer()
print ("time taken: ", end - start)
