# Import external packages
import math
import argparse
import time
import os

import pandas as pd
from tqdm import tqdm
import pickle
import copy
import torch
import spacy
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, BucketIterator, TabularDataset, Dataset
from torch.autograd import Variable

# Import internal functions and models
from encoder import Encoder, EncoderLayer
from decoder import DecoderLayer, Decoder
from sublayer import MultiHeadAttention, PositionWiseFeedForward
from models import EncoderDecoder, Generator
from training import Batch, LabelSmoothing, MyIterator, SimpleLossCompute, run_epoch, batch_size_fn, rebatch, greedy_decode
from optimizer import ScheduledOptim, NoamOpt


spacy_de = spacy.load('en_core_web_trf')
spacy_en = spacy.load('de_dep_news_trf')

def tokenize_eng(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, 0))

    x = torch.tensor(pred, requires_grad=True).permute((0, 2, 1))
    #    y=torch.Tensor(real, dtype=torch.float32)

    loss_ = torch.nn.CrossEntropyLoss(reduction="none")(x, real)

    mask = torch.tensor(mask, dtype=loss_.dtype)
    loss_ = torch.mul(loss_, mask)

    return torch.sum(loss_) / torch.sum(mask)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-dropout', type=float, default=0.1)
    # parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)

    opt = parser.parse_args()

    english = Field(sequential=True,
                    use_vocab=True,
                    tokenize=tokenize_eng,
                    lower=True,
                    pad_token='<blank>',
                    init_token='<s>',
                    eos_token='</s>')

    german = Field(sequential=True,
                   use_vocab=True,
                   tokenize=tokenize_ger,
                   lower=True,
                   pad_token='<blank>',
                   init_token='<s>',
                   eos_token='</s>')

    fields = {'English': ('eng', english), 'German': ('ger', german)}
    train_data, test_data = TabularDataset.splits(path='',
                                                  train='train.json',
                                                  test='test.json',
                                                  format='json',
                                                  fields=fields)

    english.build_vocab(train_data, max_size=1000, min_freq=1)
    print('[Info] Get source language vocabulary size:', len(english.vocab))

    german.build_vocab(train_data, max_size=1000, min_freq=1)
    print('[Info] Get target language vocabulary size:', len(german.vocab))

    batch_size = opt.batch_size
    train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True)
    test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device)


    # data = pickle.load(open(opt.data_file, 'rb'))

    opt.src_pad_idx = english.vocab.stoi['<blank>']
    opt.trg_pad_idx = german.vocab.stoi['<blank>']

    opt.src_vocab_size = len(english.vocab)
    opt.trg_vocab_size = len(german.vocab)

    criterion = LabelSmoothing(size=opt.trg_vocab_size, padding_idx=0, smoothing=0.0)
    model = make_model(opt.src_vocab_size, opt.trg_vocab_size, N=6)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        print(train_iterator)
        run_epoch((rebatch(opt.trg_pad_idx, b) for b in train_iterator), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(train_iterator, model, SimpleLossCompute(model.generator, criterion, None)))





if __name__ == "__main__":
    main()
