import torch
import torch.nn as nn

device="cuda"

from torchcrf import CRF

class AttentionBiLSTM_Seqence(nn.Module):
    
    def __init__(self, text_field, label_field, n_embed, n_hidden):
        super(AttentionBiLSTM_Seqence, self).__init__()
        
        self.n_hidden = n_hidden
        
        # LSTM1
        self.linear_hidden_r = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o = nn.Linear(n_embed, n_hidden)
        
        # LSTM2
        self.linear_hidden_r2 = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r2 = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f2 = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f2 = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g2 = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g2 = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o2 = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o2 = nn.Linear(n_embed, n_hidden)


        self.n_labels = len(label_field.vocab)       

        voc_size = len(text_field.vocab)
        self.embedding = nn.Embedding(voc_size, n_embed)
        if text_field.vocab.vectors is not None:
            self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
                                                       requires_grad=False)

        self.fc_out = nn.Linear(2*n_hidden, self.n_labels)
 
        self.pad_word_id = text_field.vocab.stoi[text_field.pad_token]
        self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
                
    def forward(self, input, tags):
        embedded = self.embedding(input)    # seq - batch - 300

        hidden_stack = []
        hidden_stack2 = []
        batch = embedded.size(1)
        seq_lenght = embedded.size(0)
        
        hidden = torch.zeros(batch, self.n_hidden).to(device)              # batch-node
        
        c = torch.zeros(batch, self.n_hidden).to(device)
        c2 = torch.zeros(batch, self.n_hidden).to(device)
        
        for i in range(seq_lenght):                                                         #for i in seq_length

            ir=self.linear_input_r(embedded[i])
            hr=self.linear_hidden_r(hidden)
            r= ir.add(hr)
            rt = self.sigmoid(r)
            
            iff=self.linear_input_f(embedded[i])
            hff=self.linear_hidden_f(hidden)
            ff= iff.add(hff)
            fft = self.sigmoid(ff)
            
            ig=self.linear_input_g(embedded[i])
            hg=self.linear_hidden_g(hidden)
            g= ig.add(hg)
            gt = self.tanh(g)
            
            io=self.linear_input_o(embedded[i])
            ho=self.linear_hidden_o(hidden)
            o= io.add(ho)
            ot = self.sigmoid(o)
            
            c = fft*c + rt*gt
            hidden = ot*self.tanh(c)
            
            hidden_stack.append(hidden)
            
        for i in range(seq_lenght-1, -1, -1):                                                         #for i in seq_length

            ir2=self.linear_input_r2(embedded[i])
            hr2=self.linear_hidden_r2(hidden)
            r2= ir2.add(hr2)
            rt2 = self.sigmoid(r2)
            
            iff2=self.linear_input_f2(embedded[i])
            hff2=self.linear_hidden_f2(hidden)
            ff2= iff2.add(hff2)
            fft2 = self.sigmoid(ff2)
            
            ig2=self.linear_input_g2(embedded[i])
            hg2=self.linear_hidden_g2(hidden)
            g2= ig2.add(hg2)
            gt2 = self.tanh(g2)
            
            io2=self.linear_input_o2(embedded[i])
            ho2=self.linear_hidden_o2(hidden)
            o2= io2.add(ho2)
            ot2 = self.sigmoid(o2)
            
            c2 = fft2*c2 + rt2*gt2
            hidden2 = ot2*self.tanh(c2)
            
            hidden_stack2.insert(0,hidden2)
        
        outputs1 = torch.stack(hidden_stack)                                   #[seq-len, batch, hidden-node]
        outputs2 = torch.stack(hidden_stack2)                                       #[seq-len,batch,  hidden-node]

        outputs = torch.cat((outputs1,outputs2),2)                                                   #[seq-len,batch,  hidden-node*2]

        # seq - batch - n_hidden*2
        out = self.fc_out(outputs)
        
        pad_mask = (input == self.pad_word_id).float()
        out[:, :, self.pad_label_id] += pad_mask*10000

        return out #-self.crf(out, labels)