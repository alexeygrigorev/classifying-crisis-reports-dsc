# coding: utf-8

import json
import re
from glob import glob
from time import time

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.cross_validation import KFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


# reading the data
# for training use only data from 2005 onwards

all_topics = list(pd.read_csv('data/topicDictionary.txt', header=None)[0])
all_topics_set = set(all_topics)
all_topics_idx = {t: i for (i, t) in enumerate(all_topics)}

train_files = sorted(glob('data/TrainingData/2*.json'))
train_files = [f for f in train_files if f >= 'data/TrainingData/2005a_TrainingData']

df_train = []

for file in tqdm(train_files):
    with open(file, 'r') as f:
        content = json.load(f)
        content = content['TrainingData']

        for k in sorted(content.keys()):
            article = content[k]

            body = article['bodyText']
            topics = set(article['topics']) & all_topics_set
            if len(topics) > 0:
                topics = sorted(topics)
                df_train.append((body, topics))

df_train = pd.DataFrame(df_train, columns=['body', 'topics'])


df_test = []

with open('data/TestData.json', 'r') as f:
    content = json.load(f)
    content = content['TestData']

    for k in sorted(content.keys()):
        article = content[k]
        body = article['bodyText']
        df_test.append((k, body))

df_test = pd.DataFrame(df_test, columns=['id', 'body'])


# reading labels

y_all = []

for topics in df_train.topics:
    row_labels = np.zeros(len(all_topics), dtype='uint8')
    for topic in topics:
        row_labels[all_topics_idx[topic]] = 1
    y_all.append(row_labels)

y_all = np.array(y_all)


# keeping only topics with at least 30 documents

all_topics = np.array(all_topics)
mask = y_all.sum(axis=0) >= 30

selected_topics = list(all_topics[mask])
y = y_all[:, mask]


# vectorization

t = time()

vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), min_df=10)
X_train = vec.fit_transform(df_train.body)
X_test = vec.transform(df_test.body)

print('vectorization took %.3fs' % (time() - t))


# fitting svm for each label

print('training the models...')

train_preds = {}
test_preds = {}

t = time()

svm_params = {
    'penalty': 'l1',
    'dual': False,
    'C': 1.0,
    'random_state': 1,
}

cv = KFold(n=len(df_train), n_folds=3, shuffle=True, random_state=1)

for i, topic in enumerate(selected_topics):
    t0 = time()

    y_train = y[:, i]
    train_pred = np.zeros(len(y_train), dtype='float32')

    for train_idx, val_idx in cv:
        svm = LinearSVC(**svm_params)
        svm.fit(X_train[train_idx], y_train[train_idx])
        train_pred[val_idx] = svm.decision_function(X_train[val_idx])

    train_preds[topic] = train_pred

    svm = LinearSVC(**svm_params)
    svm.fit(X_train, y_train)
    test_pred = svm.decision_function(X_test)
    test_preds[topic] = test_pred.astype('float32')

    print('%s, took %.3fs' % (topic, time() - t0))

print('overall took %.3fs' % (time() - t))


pred_total = [train_preds[t] for t in selected_topics]
pred_total = np.array(pred_total).T


# using oof predictions to find the best threshold

f1s = []

for t in np.linspace(-1, 0, 11):
    f1 = f1_score(y, pred_total >= t, average='micro')
    print('t=%0.2f, f1=%.4f' % (t, f1))
    f1s.append((f1, t))

best_f1, best_t = max(f1s)
print('best threshold=%0.2f, best f1=%.4f' % (best_t, best_f1))


# creating the submission

all_zeros = np.zeros(X_test.shape[0], dtype='uint8')

df_final_pred = pd.DataFrame()
df_final_pred['id'] = df_test['id']

for t in all_topics:
    if t in test_preds:
        pred = test_preds[t]
        pred = (pred >= best_t).astype('uint8')
        df_final_pred[t] = pred
    else:
        df_final_pred[t] = all_zeros


# setting labels for new topics (not present in training) 

topics_to_match = {
    'bastilledaytruckattack': {'bastille', 'day', 'truck', 'attack'},
    'berlinchristmasmarketattack': {'berlin', 'christmas', 'market', 'attack'},
    'charliehebdoattack': {'charlie', 'hebdo', 'attack'},
    'munichshooting': {'munich', 'shooting'},
    'zikavirus': {'zika', 'virus'},
}

re_tokens = re.compile('\\w+')

for t in df_test.itertuples():
    idx = t.Index
    bow = set(re_tokens.findall(t.body.lower()))

    for col, vals in topics_to_match.items():
        if len(bow & vals) == len(vals):
            df_final_pred.loc[idx, col] = 1


df_final_pred.to_csv('final_sub.csv', index=False)


