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
from models import EncoderDecoder, Generator, Batch
from optimizer import ScheduledOptim


spacy_de = spacy.load('en_core_web_trf')
spacy_en = spacy.load('de_dep_news_trf')

def tokenize_eng(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def prepare_dataloaders(opt, device):
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

    #========= Preparing Model =========#
    train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True)
    test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device)

    return train_iterator, test_iterator


def cal_performance(pred, gold, trg_pad_idx):
    loss = cal_loss(pred, gold, trg_pad_idx)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def patch_src(src):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.eng).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.ger))

        # forward
        data = Batch(src_seq, trg_seq, 0)
        optimizer.zero_grad()
        pred = model(data.src, data.trg, data.src_mask, data.trg_mask)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(pred, gold, opt.trg_pad_idx)
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg))
            data = yield Batch(src_seq, trg_seq, 0)
            # forward
            pred = model(data.src, data.trg, batch.src_mask, batch.trg_mask)
            loss, n_correct, n_word = cal_performance(pred, gold, opt.trg_pad_idx)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, opt, device)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'model': model.state_dict()}

        model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
        torch.save(checkpoint, os.path.join(opt.output_dir, model_name))

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))

        tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
        tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
        tb_writer.add_scalar('learning_rate', lr, epoch_i)


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


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    # parser.add_argument('-vocab_data', default=None)
    # parser.add_argument('-training_data', default=None)
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

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    training_data, validation_data = prepare_dataloaders(opt, device)

    print(opt)


    transformer = make_model(src_vocab=opt.src_vocab_size,
                            tgt_vocab=opt.trg_vocab_size,)

    optimizer = ScheduledOptim(optimizer=optim.Adam(transformer.parameters(),
                                                    betas=(0.9, 0.98),
                                                    eps=1e-09),
                               lr_mul=opt.lr_mul,
                               d_model=opt.d_model,
                               n_warmup_steps=opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)


if __name__ == "__main__":
    main()
