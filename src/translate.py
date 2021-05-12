import numpy as np
import torch
import pandas as pd
import spacy
import dill as pickle
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import argparse
from torch.autograd import Variable
from torchtext.legacy.data import Field, BucketIterator, TabularDataset, Dataset

from models import subsequent_mask
from main import make_model



spacy_en = spacy.load('en_core_web_trf')
spacy_de = spacy.load('de_dep_news_trf')

def tokenize_eng(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def preprocess_vald_data(src, trg):
    english_valid_text = open(src, encoding='utf8').read().split('\n')
    german_valid_text = open(trg, encoding='utf8').read().split('\n')
    raw_valid = {'English': [line for line in english_valid_text],
                'German': [line for line in german_valid_text]}
    valid_df = pd.DataFrame(raw_valid, columns=['English', 'German'])
    valid_df.to_json('valid.json', orient='records', lines=True)

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
    fields = {'English': ('eng', english),
              'German': ('ger', german)}

    _, valid_data = TabularDataset.splits(path='',
                                             test='test.json',
                                             validation='valid.json',
                                             format='json',
                                             fields=fields)
    print(valid_data)
    data = {'vocab': {'src': english, 'trg': german},
            'valid': valid_data.examples}
    english.build_vocab(valid_data, min_freq=1)
    print('[Info] Get source language vocabulary size:', len(english.vocab))

    german.build_vocab(valid_data, min_freq=1)
    print('[Info] Get target language vocabulary size:', len(german.vocab))

    pickle.dump(data, open('valid.obj', 'wb'))


def load_data(data):
    data = pickle.load(open(data, 'rb'))

    src_vocab_size = len(data['vocab']['src'].vocab)
    print('[Info] Get source language vocabulary size:', src_vocab_size)
    trg_vocab_size = len(data['vocab']['trg'].vocab)
    print('[Info] Get target language vocabulary size:', trg_vocab_size)
    fields = {'eng': data['vocab']['src'], 'ger': data['vocab']['trg']}
    valid = Dataset(examples=data['valid'], fields=fields)
    src_pad_idx = data['vocab']['src'].vocab.stoi['<blank>']
    trg_pad_idx = data['vocab']['trg'].vocab.stoi['<blank>']
    return valid, data['vocab']['src'], data['vocab']['trg']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--trg', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=12000)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('--model_file', default=None)

    opt = parser.parse_args()

    preprocess_vald_data(src=opt.src, trg=opt.trg)

    valid_data, english, german = load_data('valid.obj')

    valid_iter = BucketIterator(valid_data, batch_size=10, repeat=False)

    model = make_model(src_vocab=9797,
                       tgt_vocab=18669,
                       N=opt.n_layers,
                       d_model=opt.d_model,
                       d_ff=opt.d_inner_hid,
                       h=opt.n_head,
                       dropout=opt.dropout)

    model.load_state_dict(torch.load(opt.model_file))

    for i, batch in enumerate(valid_iter):
        src = batch.eng.transpose(0, 1)[:1]
        src_mask = (src != english.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=german.vocab.stoi["<blank>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = german.vocab.itos[out[0, i]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.ger.size(0)):
            sym = german.vocab.itos[batch.ger.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break

if __name__ == "__main__":
    main()
