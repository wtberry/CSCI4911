# Lab 12 RNN
from string import ascii_uppercase
from collections import OrderedDict
from itertools import count

from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable

import pre_process as pre

'''
THis is a practice for using torch to implement simple, one cell RNN.
It takes sequence of alphabet letters as input (here, my name) and 
outputs/predicts next letter based on the input. 
So... Name: Wataru Takahashi, Input: Wataru Takahashi, Label: ataru Takahashi
'''

'''
Input is encoded as one-hot matrix based on index number from 0-26
'''


torch.manual_seed(777)  # reproducibility


idx2char = [letter for letter in ascii_uppercase]
idx2char.append(' ')

cipher = pre.data() # importing the dataframe of cipher txt
key_cipher = cipher.iloc[0].key + cipher.iloc[0].cipher
key_length = cipher.iloc[0].key
plain = cipher.iloc[0].key + cipher.iloc[0].plantxt

od = OrderedDict(zip(ascii_uppercase, count(0)))
od.update({'-': 26}) # adding white space as leters

def encode(text):
    ## strip white space and "'" in the string, and convert it into numbers
    return  [od[letter] for letter in text.replace(' ', '').strip("").upper()]

def to_np(x):
    return x.data.cpu().numpy()

# Wataru Takahashi

x_cipher = [encode(key_cipher)]
y_plain = [encode(plain)]

def one_hot(data):
    # creating empty nested list to store one-hot
    empty = [[0 for n in range(0, len(od))] for N in range(0, len(data[0]))]

    for n in enumerate(data[0]):
        empty[n[0]][n[1]] = 1
    return empty
# Teach Wataru Takahash -> ataru Takahashi


cipher_one_hot = one_hot(x_cipher)

## enclosing the data inside of list, for dimensionality of input

cipher_one_hot = [cipher_one_hot]
y_plain = y_plain[0]

# As we have one batch of samples, we will change them to variables only once

inputs_cipher = Variable(torch.Tensor(cipher_one_hot))
labels_plain = Variable(torch.LongTensor(y_plain))

num_classes = 27
input_size = 27  # one-hot size
hidden_size = 27  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = inputs_cipher.shape[1]  # |ihello| == 6
num_layers = 3  # one-layer lstm


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # layer1 lstm
        self.lstm1= nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#        # layer2 lstm
#        self.lstm2= nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)


    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch hidden_size) for batch_first=True
        h_1 = (Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size)), Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size)))

#        h_2 = (Variable(torch.randn(x.size(0), self.num_layers, self.hidden_size)), Variable(torch.randn(x.size(0), self.num_layers, self.hidden_size)))

        #print('h_o', h_0)

        # Reshape input
        x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (batch, num_layers * num_directions, hidden_size)

        out, _ = self.lstm1(x, h_1)
#        out2, _ = self.lstm2(out, h_2)
        return out.view(-1, num_classes)


# Instantiate RNN model
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
print(lstm)

# set logger directory
LOG_PATH = '/home/wataru/Uni/4911/cipher/new_logs/'
LOG_DIR= 'Layer_'+ str(num_layers)+'_lr_'+str(lr)+'_epoch_'+str(num_iter)
# set logger
logger = Logger(LOG_PATH+LOG_DIR)


# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(lstm.parameters(), lr=0.1)

# Train the model
for epoch in range(1500):
    outputs = lstm(inputs_cipher)
    optimizer.zero_grad()
    loss = criterion(outputs, labels_plain)
    loss.backward()
    optimizer.step()
    if epoch%10 == 0:
        _, idx = outputs.max(1)
        idx = idx.data.numpy()
        result_str = [idx2char[c] for c in idx.squeeze()]
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
        print("Predicted string: ", ''.join(result_str))

print("Learning finished!")

##### Tensorboard logging
# 1. log the scalar values
info = {
        'loss': loss.data[0]
        }

for tag, value in info.items():
    logger.scalar_summary(tag, value, step+1)


# 2. Log values and gradients of the parameters (histogram)
for tag, value in net.named_parameters():
    tag = tag.replace('.', '/')
    logger.histo_summary(tag, to_np(value), step+1)
    logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)