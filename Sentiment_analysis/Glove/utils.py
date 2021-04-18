
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import sys
import os
import string, nltk
#nltk.download('stopwords')
nltk.data.path.append("/home/ubuntu/nltk_data")
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
#nltk.download('punkt')
#nltk.download('wordnet')
import numpy as np
import torch
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import re
import matplotlib.pyplot as plt
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english') + list(string.punctuation)) 
device="cuda"

def Norm(text):
    text = text.lower().strip()
    text =  re.sub(' +', ' ', text)
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            w = wordnet_lemmatizer.lemmatize(w, pos="v")
            filtered_sentence.append(w) 
    return " ".join(filtered_sentence)


def LoadData():

    MAX_SEQ_LEN = 50

    TEXT = Field(sequential=True, batch_first=True, pad_first=True)#,use_vocab=True, batch_first=True,fix_length=MAX_SEQ_LEN, pad_first=True ,tokenize=Norm)
    TAGS = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

    fields = [('label', TAGS), ('text', TEXT)]

    train, valid, test = TabularDataset.splits(path="Sentiment_analysis/data", train='train.tsv', validation='dev.tsv',
                                            test='test.tsv', format='TSV', fields=fields, skip_header=True)
#
    device = "cuda"
    train_iter = BucketIterator(train, batch_size=64, device=device, train=True)
    valid_iter = BucketIterator(valid, batch_size=64, device=device, train=True)
    test_iter = Iterator(test, batch_size=1, device=device, train=False, shuffle=False, sort=False)

    embed_len = 300
    TEXT.build_vocab(train, vectors="glove.6B.300d")

    return train_iter, valid_iter, test_iter, TAGS, TEXT, fields

def save_checkpoint(save_path, model):

    if save_path == None:
        return
    
    save_path = "checkpoints/" + save_path

    state_dict = {'model_state_dict': model.state_dict()}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return

    load_path = "checkpoints/" + load_path
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])

if __name__ == '__main__':
    print(Norm("Load at "))