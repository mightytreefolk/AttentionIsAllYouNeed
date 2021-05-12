import os
import dill as pickle
import pandas as pd
from torchtext.legacy.data import Field, TabularDataset, LabelField
import argparse
from transformers import BertTokenizer, BertModel, BertForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument('--lines', '-l', type=int, default=30000)
parser.add_argument('--vocab_size', '-vs', type=int, default=None)
opt = parser.parse_args()

en_train = os.path.join('data', 'train_data', 'train.en')
ger_train = os.path.join('data', 'train_data', 'train.de')

en_test = os.path.join('data', 'test_data', 'newstest2014.en')
ger_test = os.path.join('data', 'test_data', 'newstest2014.de')

english_train_text = open(en_train, encoding='utf8').read().split('\n')
german_train_text = open(ger_train, encoding='utf8').read().split('\n')

english_test_text = open(en_test, encoding='utf8').read().split('\n')
german_test_text = open(ger_test, encoding='utf8').read().split('\n')

raw_train = {'English': [line for line in english_train_text[1:opt.lines]],
                 'German': [line for line in german_train_text[1:opt.lines]]}

raw_test = {'English': [line for line in english_test_text[1:1000]],
                'German': [line for line in german_test_text[1:1000]]}

train_df = pd.DataFrame(raw_train, columns=['English', 'German'])
test_df = pd.DataFrame(raw_test, columns=['English', 'German'])

train_df.to_json('train.json', orient='records', lines=True)
test_df.to_json('test.json', orient='records', lines=True)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

english = Field(sequential=True,
                use_vocab=True,
                tokenize=tokenizer.encode,
                pad_token=pad_index,
                init_token=tokenizer.cls_token,
                eos_token=tokenizer.sep_token)

german = Field(sequential=True,
               use_vocab=True,
               tokenize=tokenizer.encode,
               pad_token=pad_index,
               init_token=tokenizer.cls_token,
               eos_token=tokenizer.sep_token)

fields = {'English': ('eng', english),
          'German': ('ger', german)}
train_data, test_data = TabularDataset.splits(path='',
                                              train='train.json',
                                              test='test.json',
                                              format='json',
                                              fields=fields)

data = {'vocab': {'src': english, 'trg': german},
        'train': train_data.examples,
        'test': test_data.examples}

english.build_vocab(train_data, max_size=opt.vocab_size, min_freq=1)
print('[Info] Get source language vocabulary size:', len(english.vocab))

german.build_vocab(train_data, max_size=opt.vocab_size, min_freq=1)
print('[Info] Get target language vocabulary size:', len(german.vocab))

pickle.dump(data, open('data.obj', 'wb'))
