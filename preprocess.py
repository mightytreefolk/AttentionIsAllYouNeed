
import pickle
import spacy
import os
import pandas as pd
from torchtext.legacy.data import Field, BucketIterator, TabularDataset

spacy_de = spacy.load('en_core_web_trf')
spacy_en = spacy.load('de_dep_news_trf')

def tokenize_eng(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def main():
    en_train = os.path.join('data', 'train_data', 'train.en')
    ger_train = os.path.join('data', 'train_data', 'train.de')

    en_test = os.path.join('data', 'test_data', 'newstest2014.en')
    ger_test = os.path.join('data', 'test_data', 'newstest2014.de')

    english_train_text = open(en_train, encoding='utf8').read().split('\n')
    german_train_text = open(ger_train, encoding='utf8').read().split('\n')

    english_test_text = open(en_test, encoding='utf8').read().split('\n')
    german_test_text = open(ger_test, encoding='utf8').read().split('\n')

    raw_train = {'English': [line for line in english_train_text[1:1000]],
                 'German': [line for line in german_train_text[1:1000]]}

    raw_test = {'English': [line for line in english_test_text[1:1000]],
                'German': [line for line in german_test_text[1:1000]]}

    train_df = pd.DataFrame(raw_train, columns=['English', 'German'])
    test_df = pd.DataFrame(raw_test, columns=['English', 'German'])

    train_df.to_json('train.json', orient='records', lines=True)
    test_df.to_json('test.json', orient='records', lines=True)

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

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

    english.build_vocab(train_data, max_size=100000, min_freq=1)
    print('[Info] Get source language vocabulary size:', len(english.vocab))

    german.build_vocab(train_data, max_size=100000, min_freq=1)
    print('[Info] Get target language vocabulary size:', len(german.vocab))

    # train_interator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=32, device='cuda')

    data = {
        'vocab': {'src': english, 'trg': german},
        'train': train_data,
        'test': test_data}

    pickle.dump(data, open('data.obj', 'wb'))


if __name__ == '__main__':
    main()
