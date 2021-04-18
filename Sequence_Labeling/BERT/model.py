import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertGenerationEncoder, BertModel
device="cuda"

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertGenerationEncoder, BertModel

device="cuda"

# Additive Attention LSTM using Pretrain BERT
class BERT_Seqence(nn.Module):

    def __init__(self, n_embed, n_output):
        super(BERT_Seqence, self).__init__()

        self.encoder = BertModel.from_pretrained("bert-base-uncased")

        self.fc = nn.Linear(n_embed,  n_output)        
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        text_fea = self.encoder(text).last_hidden_state                                    # (batch, seq, emb)

        hidden_stack = []
        batch = text_fea.shape[0]
        seq_lenght= text_fea.size(1)

        embedded_words = text_fea.permute(1,0,2)                                      # => (seq_length,batch_size,  n_embed)
        
                    
        #embedded = [sent len, batch size, emb dim]
        
        predictions = self.fc(self.dropout(embedded_words))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions






