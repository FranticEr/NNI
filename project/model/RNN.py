import torch
import torch.nn as nn
class Rec_RNN(nn.Module):
    def __init__(self,inpput_size,hidden_size,output_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.input_size=inpput_size
        self.output_size=output_size

        self.rnn=nn.GRU(input_size=inpput_size,hidden_size=hidden_size,num_layers=1)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(hidden_size,output_size)
    def init_state(self,batch_size):
        self.state=torch.zeros(1,batch_size,self.hidden_size)
        return self.state
    def forward(self,x):
        y_hat,state=self.rnn(x)
        y_hat=self.linear(y_hat.reshape((-1,y_hat.shape[-1])))
        return y_hat,state