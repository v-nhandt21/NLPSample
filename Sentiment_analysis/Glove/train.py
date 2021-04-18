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

def train(net,model_name):

    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    optimizer = optim.Adam(net.parameters(), lr = 0.01, weight_decay=0.01)

    step = 0
    n_epochs = 10
    clip = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net.to(device)
    net.train()

    for epoch in range(n_epochs):
        
        for (labels, inputs), _ in train_iter:

            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            net.zero_grad()
            output, h = net(inputs)
            loss = criterion(output.squeeze(), labels.float())

            loss.backward()
            
            #To prevent exploding gradients
            nn.utils.clip_grad_norm(net.parameters(), clip)
            optimizer.step()
            
    #         if loss.item() < 0.02:
    #             break
            
            
            if (step % 50) == 0:            
                net.eval()
                valid_losses = []
                num_val_batch =0 
                for (v_labels, v_inputs), _ in valid_iter:
                    num_val_batch += 1
                    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

                    
                    v_output, v_h = net(v_inputs)
                    v_loss = criterion(v_output.squeeze(), v_labels.float())
                    valid_losses.append(v_loss.item())
                
                valid_losses = sum(valid_losses)/len(valid_losses)
                    
                print("Epoch: {}/{}".format((epoch+1), n_epochs),
                    "Step: {}".format(step),
                    "Training Loss: {:.4f}".format(loss.item()),
                    "Validation Loss: {:.4f}".format(valid_losses),
                    )

                if valid_losses - loss.item() > 0.2:
                    break
                    
                net.train()
    save_checkpoint(model_name,net)

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

    train(net,"sen_"+model_name+".pt")