# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:15:22 2019

@author: mankup
"""

!pip install -q -U tensorflow>=1.8.0
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
from sys import stdout
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense
import tensorflow_hub as hub
import pickle

test = pd.read_csv('test.tsv',header=None, sep='\t')
train = pd.read_csv('train.tsv',header = None, sep='\t')
valid = pd.read_csv('valid.tsv',header = None, sep='\t')

df_train = pd.read_table(train,
                             names = ['id',	'label'	,'statement',	'subject',	'speaker', 	'job', 	'state',	'party',	'barely_true_c',	'false_c',	'half_true_c',	'mostly_true_c',	'pants_on_fire_c',	'venue'])
    
df_valid = pd.read_table(valid,
                             names =['id',	'label'	,'statement',	'subject',	'speaker', 	'job', 	'state',	'party',	'barely_true_c',	'false_c',	'half_true_c',	'mostly_true_c',	'pants_on_fire_c',	'venue'])

df_test = pd.read_csv(test, sep='\t', 
                            names =['id',	'label'	,'statement',	'subject',	'speaker', 	'job', 	'state',	'party',	'barely_true_c',	'false_c',	'half_true_c',	'mostly_true_c',	'pants_on_fire_c',	'venue']) 

df = pd.concat([df_train, df_valid])

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
def elmo_vectors(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

elmo_train = [df_train.iloc[i:i+100] for i in range(0,df_train.shape[0],100)]
elmo_valid = [df_valid.iloc[i:i+100] for i in range(0,df_valid.shape[0],100)]
elmo_test = [df_test.iloc[i:i+100] for i in range(0,df_test.shape[0],100)]

# Extract ELMo embeddings
elmo_train = [elmo_vectors(x['statement']) for x in elmo_train]
elmo_valid = [elmo_vectors(x['statement']) for x in elmo_valid]
elmo_test = [elmo_vectors(x['statement']) for x in elmo_test]

elmo_train_new = np.concatenate(elmo_train, axis = 0)
elmo_valid_new = np.concatenate(elmo_valid, axis = 0)
elmo_test_new = np.concatenate(elmo_test, axis = 0)

pickle.dump(elmo_train_new ,open("elmo_train_new.pickle","wb"))
pickle.dump(elmo_valid_new ,open("elmo_valid_new.pickle","wb"))
pickle.dump(elmo_test_new ,open("elmo_test_new.pickle","wb"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

x_train = elmo_train_new
y_train = np.array(df_train['label'])
x_valid = elmo_valid_new
y_valid = np.array(df_valid['label'])
x_test = elmo_test_new
y_test = np.array(df_test['label'])

lreg = LogisticRegression()
lreg.fit(x_train, y_train)

preds_valid = lreg.predict(x_test)

f1_score(y_test, preds_valid, average='micro')

#XGboost Classifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score

model_xgb = XGBClassifier()
model_xgb.fit(x_train, y_train)
y_pred = model_xgb.predict(x_valid)
accuracy = accuracy_score(y_valid, y_pred )
print(f"Validation Accuracy: {round(accuracy * 100, 4)}")
