# Import external packages
import math
import argparse
import dill as pickle
import copy

import pandas as pd
import torch
import time
import spacy
import numpy as np
import torch.nn as nn
from torchtext.legacy.data import Field, BucketIterator, TabularDataset, Dataset
from torch.autograd import Variable


# Import internal functions and models
from encoder import Encoder, EncoderLayer
from decoder import DecoderLayer, Decoder
from sublayer import MultiHeadAttention, PositionWiseFeedForward
from models import EncoderDecoder, Generator
from training import LabelSmoothing, run_epoch, rebatch, SimpleLossCompute
from optimizer import NoamOpt
from plotting import plot


spacy_en = spacy.load('en_core_web_trf')
spacy_de = spacy.load('de_dep_news_trf')

def tokenize_eng(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=512, h=8, dropout=0.1):
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


def stream_data(train, test, max_vocab_size, opt):
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
                   lower=False,
                   pad_token='<blank>',
                   init_token='<s>',
                   eos_token='</s>')

    fields = {'English': ('eng', english), 'German': ('ger', german)}
    train_data, test_data = TabularDataset.splits(path='',
                                                  train=train,
                                                  test=test,
                                                  format='json',
                                                  fields=fields)

    english.build_vocab(train_data, max_size=max_vocab_size, min_freq=1)
    print('[Info] Get source language vocabulary size:', len(english.vocab))

    german.build_vocab(train_data, max_size=max_vocab_size, min_freq=1)
    print('[Info] Get target language vocabulary size:', len(german.vocab))

    opt.src_pad_idx = english.vocab.stoi['<blank>']
    opt.trg_pad_idx = german.vocab.stoi['<blank>']

    opt.src_vocab_size = len(english.vocab)
    opt.trg_vocab_size = len(german.vocab)

    return train_data, test_data


def load_data(data, opt):
    data = pickle.load(open(data, 'rb'))
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi['<blank>']
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi['<blank>']

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    print('[Info] Get source language vocabulary size:', opt.src_vocab_size)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)
    print('[Info] Get target language vocabulary size:', opt.trg_vocab_size)
    fields = {'eng': data['vocab']['src'], 'ger': data['vocab']['trg']}
    train = Dataset(examples=data['train'], fields=fields)
    test = Dataset(examples=data['test'], fields=fields)
    return train, test



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=500)
    parser.add_argument('-b', '--batch_size', type=int, default=30)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=256)
    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('--model_file', default=None)

    opt = parser.parse_args()

    # train_data, test_data = stream_data('train.json', 'test.json', 1000, opt)
    train_data, test_data = load_data('data.obj', opt)

    batch_size = opt.batch_size
    train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True)
    test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device)


    criterion = LabelSmoothing(size=opt.trg_vocab_size, padding_idx=0, smoothing=0.0)
    model = make_model(src_vocab=opt.src_vocab_size,
                       tgt_vocab=opt.trg_vocab_size,
                       N=opt.n_layers,
                       d_model=opt.d_model,
                       d_ff=opt.d_inner_hid,
                       h=opt.n_head,
                       dropout=opt.dropout)

    if opt.model_file is not None:
        model.load_state_dict(torch.load(opt.model_file))

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    accuracies_train = []
    losses_train = []
    accuracies_test = []
    losses_test = []
    for epoch in range(opt.epoch):
        print(f"We are on epoch: {epoch}")
        model.train()
        loss_train, accuracy_train, total_losses_train = run_epoch((rebatch(opt.trg_pad_idx, b) for b in train_iterator),
                                               model,
                                               SimpleLossCompute(generator=model.generator,
                                                                 criterion=criterion,
                                                                 opt=opt,
                                                                 optim=model_opt,),
                                               opt,)

        model.eval()
        accuracies_train.append(np.array(accuracy_train))
        losses_train.append(loss_train)
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'model-{epoch}-{time.time()}.pt')
            print("BEGIN VALIDATE EPOCH")
            loss_test, accuracy_test, total_losses_test = run_epoch((rebatch(opt.trg_pad_idx, b) for b in test_iterator), model,
                                        SimpleLossCompute(generator=model.generator,
                                                          criterion=criterion,
                                                          opt=opt,
                                                          optim=model_opt,
                                                          backprop=False), opt)
            accuracies_test.append(np.array(accuracy_test))
            losses_test.append(loss_test)
            print("END VALIDATE EPOCH")

    test_acc_df = pd.DataFrame(accuracies_test)
    test_loss_df = pd.DataFrame(losses_test)

    train_acc_df = pd.DataFrame(accuracies_train)
    train_loss_df = pd.DataFrame(losses_train)

    test_acc_df.to_csv(f'test_acc-{opt.epoch}.csv')
    test_loss_df.to_csv(f'test_loss-{opt.epoch}.csv')
    train_acc_df.to_csv(f'train_acc-{opt.epoch}.csv')
    train_loss_df.to_csv(f'train_loss-{opt.epoch}.csv')
    """Plot results"""
    plot(test_loss_df, 'Loss of test data', 'Epochs', 'Mean Loss', opt)
    plot(test_acc_df, 'Accuracy of test data', 'Epochs', 'Mean accuracy', opt)
    plot(train_loss_df, 'Average train loss per epoch', 'Epochs', 'Average Loss', opt)
    plot(train_acc_df, 'Average train accuracy per epoch', 'Epochs', 'Average Accuracy', opt)



if __name__ == "__main__":
    main()
