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


class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion,
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                          devices=self.devices)
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss,
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


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
    parser.add_argument('-output_dir', type=str, default=None)
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
    # data = pickle.load(open(opt.data_file, 'rb'))

    opt.src_pad_idx = english.vocab.stoi['<blank>']
    opt.trg_pad_idx = german.vocab.stoi['<blank>']

    opt.src_vocab_size = len(english.vocab)
    opt.trg_vocab_size = len(german.vocab)



    devices = [0, 1, 2, 3]
    pad_idx = opt.trg_vocab_size
    model = make_model(len(english.vocab), len(german.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(german.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train_data, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.eng), len(x.ger)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(test_data, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.eng), len(x.ger)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion,
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model_par,
                         MultiGPULossCompute(model.generator, criterion,
                                             devices=devices, opt=None))
        print(loss)

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != english.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=german.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = german.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = german.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break

if __name__ == "__main__":
    main()
