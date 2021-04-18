
import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import BertTokenizer
import functools

device = "cuda"

def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens

def cut_to_max_length(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens

def LoadData():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #MAX_SEQ_LEN = tokenizer.max_model_input_sizes['bert-base-uncased']
    MAX_SEQ_LEN = 128

    text_preprocessor = functools.partial(cut_and_convert_to_id,
                                      tokenizer = tokenizer,
                                      max_input_length = MAX_SEQ_LEN)

    tag_preprocessor = functools.partial(cut_to_max_length,
                                        max_input_length = MAX_SEQ_LEN)


    INIT_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    TEXT = Field(use_vocab = False,
                  lower = True,
                  preprocessing = text_preprocessor,
                  init_token = INIT_INDEX,                      # Bert pretrain need start token
                  pad_token = PAD_INDEX,
                  unk_token = UNK_INDEX,

                  include_lengths=False, batch_first=True
                  )

    TAGS = Field(unk_token = None,
                        batch_first=True,
                        init_token = '<pad>',                       # Them vao cho deu voi tren, nhung khong tinh loss cho cls
                        preprocessing = tag_preprocessor)

    fields = [('text', TEXT), ('tag', TAGS)]
    train, valid, test = TabularDataset.splits(path="../data", train='train.tsv', validation='dev.tsv',
                                            test='test.tsv', format='TSV', fields=fields, skip_header=True)

    TAGS.build_vocab(train)
        
    device = "cuda"
    train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=1, device=device, train=False, shuffle=False, sort=False)

    return train_iter, valid_iter, test_iter, TAGS


def NormData(filetext, filetag, fileout):
    with open(fileout, "w+", encoding="utf-8") as fw:
        fw.write("text\ttag\n")
        with open(filetext, "r",encoding="ISO-8859-1") as f:
            texts = f.read().splitlines()
            with open(filetag, "r",encoding="ISO-8859-1") as f1:
                tags = f1.read().splitlines()
                for text,tag in zip(texts,tags):
                    fw.write( text+ "\t"+tag+"\n")


def save_checkpoint(save_path, model):

    if save_path == None:
        return
    
    save_path = "../../checkpoints/" + save_path

    state_dict = {'model_state_dict': model.state_dict()}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return

    load_path = "../../checkpoints/" + load_path
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])

if __name__ == '__main__':
    filetext = "../data/train.text"
    filetag = "../data/train.label"
    fileout = "../data/train.tsv"
    NormData(filetext,filetag,fileout)

    filetext = "../data/dev.text"
    filetag = "../data/dev.label"
    fileout = "../data/dev.tsv"
    NormData(filetext,filetag,fileout)

    filetext = "../data/test.text"
    filetag = "../data/test.label"
    fileout = "../data/test.tsv"
    NormData(filetext,filetag,fileout)
    
                