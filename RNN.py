#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.nn.functional as F


# In[ ]:


df = pd.read_csv('Dataset2nd.csv').values
df.shape


# In[ ]:


df1 = pd.read_csv('Dataset_df.csv').iloc[:,1:]
df1 = df1.values


# In[ ]:


df.shape


# In[ ]:


X = df[:,1:1033].reshape(100,1,1032)
Y = df[:,-1]


# In[ ]:


X.shape


# In[ ]:


TrainXTensor = torch.from_numpy(X).type(torch.FloatTensor)
TestXTensor = torch.from_numpy(X[70:]).type(torch.FloatTensor)

TrainYTensor = torch.from_numpy(Y).type(torch.LongTensor)
TestYTensor = torch.from_numpy(Y[70:]).type(torch.LongTensor)


# In[ ]:


TrainXTensor


# In[ ]:


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=1032,
            hidden_size=4,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,
            bidirectional=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(8, 2)

    def forward(self, x):
        # x (time_step, batch_size, input_size)
        # h (n_layers, batch, hidden_size)
        # out (time_step, batch_size, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out
rnn = RNN()


# In[ ]:


out= rnn(TrainXTensor)
prediction = torch.max(F.softmax(out), 1)[1]
prediction


# In[ ]:


optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)   # optimize all parameters
loss_func = nn.CrossEntropyLoss() 


# In[ ]:


for step in range(300):
    out = rnn(TrainXTensor)
    loss = loss_func(out, TrainYTensor)
    prediction = torch.max(F.softmax(out), 1)[1]

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step() 
    print(loss)
#print(out)


# In[ ]:


out_test = rnn(TestXTensor)
prediction_test = torch.max(F.softmax(out_test), 1)[1]

