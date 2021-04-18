from model import AttentionBiLSTM_Seqence
from utils import LoadData, save_checkpoint, save_metrics,loss_visual
import torch
import torch.optim as optim
import torch.nn as nn
from torchcrf import CRF

train_iter, valid_iter, test_iter, TAGS, TEXT, fields = LoadData()


device = "cuda"

def train(model,optimizer,train_loader = train_iter,valid_loader = valid_iter,
        num_epochs = 70, eval_every = len(train_iter)):

    best_valid_loss = float("Inf")
    crf = CRF(len(TAGS.vocab)).to(device)
    train_loss, valid_loss = [],[]

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.to(device)
    model.train()
    for epoch in range(num_epochs):

        for (text, tags), _ in train_loader:
            
            output = model(text, tags)

            loss = -crf(output, tags)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1


            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                    for (text, tags), _ in valid_loader:
                        
                        output = model(text, tags)

                        loss = -crf(output, tags)

                        valid_running_loss += loss.item()

                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))

                train_loss.append(average_train_loss)
                valid_loss.append(average_valid_loss)

                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint('Seq_BiLSTM.pt', model)
               
    loss_visual("loss",train_loss, valid_loss)




if __name__ == '__main__':
    n_output = len(TAGS.vocab)

    model = AttentionBiLSTM_Seqence(TEXT, TAGS, n_embed=300, n_hidden=128)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    train(model=model, optimizer=optimizer)
