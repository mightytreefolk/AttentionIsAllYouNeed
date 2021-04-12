import pandas as pd
import numpy as np
import codecs
import os
import spacy
import itertools as it

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torchtext


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/preprocess.py

class PreProcess(object):
    def __init__(self, language, text, ):
        self.nlp = spacy.load(language)
        self.text = text
        self.counter = Counter()

    def tokenize(self):
        tokens = []
        for i in self.text:
            tok = self.nlp.tokenizer(i)
            tokens.append(tok.text)
        return tokens

    def create_vocab(self):
        for token in self.tokenize():
            self.counter.update(token)
        vocab = Vocab(self.counter, min_freq=1)
        return vocab






