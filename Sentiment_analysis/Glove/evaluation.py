
import numpy as np
import torch
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import sys
from utils import LoadData
from model.cnn import CNN
from model.rnn import NetworkGRU_reimplement, NetworkLSTM_reimplement, NetworkRNN_reimplement
from model.attention import AttentionLSTM_Additive, AttentionLSTM_Multiplicative, AttentionLSTM_Dot, AttentionBiLSTM_Additive
from utils import load_checkpoint, save_checkpoint
device="cuda"
train_iter, valid_iter, test_iter, TAGS, TEXT, fields = LoadData()

def evaluation(net,model_name):

    load_checkpoint(model_name,net)

    net.eval().to(device)
    count = 0
    sums = 0

    for (v_labels, v_inputs), _ in test_iter:
        
        sums = sums + len(v_inputs)
        
        v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

        v_output, v_h = net(v_inputs)
            

        output = torch.round(v_output.squeeze()).detach().cpu().numpy().astype(int)

        ground = v_labels.detach().cpu().numpy().astype(int)

        count = count + np.sum(output == ground)
        
    print("Accuracy: " + str(count/sums))

if __name__ == '__main__':

    model_name = str(sys.argv[1])

    n_embed= 300
    n_hidden = 512
    n_hidden_decode = 512
    v_attention_dimensionality = 512
    n_output = 1   # 1 ("positive") or 0 ("negative")
    layers = 1

    if model_name =="cnn":
        net = CNN(TEXT, TAGS, n_embed, n_hidden, n_output, layers).cuda()

    elif model_name=="rnn":   
        net = NetworkRNN_reimplement( TEXT, TAGS, n_embed, n_hidden, n_output, layers).cuda()

    elif model_name=="lstm":
        net = NetworkLSTM_reimplement( TEXT, TAGS, n_embed, n_hidden, n_output, layers).cuda()

    elif model_name=="gru":
        net = NetworkGRU_reimplement( TEXT, TAGS, n_embed, n_hidden, n_output, layers).cuda()

    elif model_name=="att_dot":
        net = AttentionLSTM_Dot(TEXT, TAGS, n_embed, n_hidden, n_output, layers).cuda()

    elif model_name=="att_mul":
        net = AttentionLSTM_Multiplicative(TEXT, TAGS, n_embed, n_hidden, n_hidden_decode, n_output, layers).cuda()

    elif model_name=="att_add":
        net = AttentionLSTM_Additive(TEXT, TAGS, n_embed, n_hidden, n_hidden_decode, n_output, layers, v_attention_dimensionality).cuda()

    elif model_name=="att_bilstm_add":
        net = AttentionBiLSTM_Additive(TEXT, TAGS, n_embed, n_hidden, n_hidden_decode, n_output, layers, v_attention_dimensionality).cuda()
    
    evaluation(net,"sen_"+model_name+".pt")