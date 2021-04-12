import pandas as pd
import numpy as np
import codecs
import os
import spacy
import itertools as it

import torch
import torch.nn as nn

from models import PreProcess

train_data_dir = os.path.join('..', 'data', 'train_data')
test_data_dir = os.path.join('..', 'data', 'test_data')

english_train = os.path.join(train_data_dir, 'train.en')
# german_train = os.path.join(train_data_dir, 'train.de')

en_nlp = spacy.load('en_core_web_trf')
de_nlp = spacy.load('de_dep_news_trf')

f = open(english_train, 'r')

data = PreProcess('en_core_web_trf', f)

print(data.create_vocab())


