import pandas as pd
import numpy as np
import torch
import codecs
import os
import spacy
import itertools as it


train_data_dir = os.path.join('..', 'data', 'train_data')
test_data_dir = os.path.join('..', 'data', 'test_data')

english_train = os.path.join(train_data_dir, 'train.en')
german_train = os.path.join(train_data_dir, 'train.de')

en_nlp = spacy.load('en_core_web_trf')
de_nlp = spacy.load('de_dep_news_trf')

with codecs.open(english_train, encoding='utf_8') as f:
    sample_review = list(it.islice(f, 8, 9))[0]
    sample_review = sample_review.replace('\\n', '\n')

print(sample_review)
