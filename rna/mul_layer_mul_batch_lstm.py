# Lab 12 RNN
import os
from string import ascii_uppercase
from collections import OrderedDict
from itertools import count

# In order to use Tensorboard with pyTorch...
from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import pre_process as pre

'''
LSTM to learn how to decrypt vigenere cipher based on training data

Make sure to set your own PATH to save Tensorboard (the visualization tool) event
file, as well as PATH to load training and test data, in pre_process.py

To run this script, 2 other scripts
- pre_process.py
- logger.py
are required. It'll be on the google drive.


### Tuning Parameters
There are some adjustable parameters when training the model.
- Learning rate: determines amount of 'updates' the model applies on weights on each iteration, higher the model might converge faster, but risk lots of spikes.

- num_iter: number of training iterations

- N: number of characters included in ciphered&plain text

- input size: 27, for alphabet & '-' for key padding

- batch_size: number of key_text pairs in training dataset

- sequence_length: total number of letters in the input/output sequence, 28 here.

- num_layers: how many LSTM layers are used in the model

After choosing values for the parameters, now you can train the model.
Happy Machine Leaerning :)
'''

'''
Input is encoded as one-hot matrix based on index number from 0-26

About adaptive learning rate....
http://pytorch.org/docs/master/optim.html
'''


torch.manual_seed(777)  # reproducibility


#### Setting up parameters
# Sequece number N, how long the texts are.. for now, 28
N = 70
num_classes = 27
input_size = 27  # one-hot size
hidden_size = 27  # output from the LSTM. 5 to directly predict one-hot
batch_size = 500# one sentence
sequence_length = N+8# text + key
num_layers = 3  # one-layer lstm

lr = 0.01
num_iter = int(15000)

### Prepping the data
idx2char = [letter for letter in ascii_uppercase]
idx2char.append(' ')


cipher = pre.data_train() # importing the dataframe of cipher txt, as Pandas DataFrame

#key_cipher = cipher.iloc[0].key + cipher.iloc[0].cipher

## creating list of strings of all the texts
key_cipher = [[cipher.iloc[i].key + cipher.iloc[i].cipher.replace(' ', '')[:N]] for i in range(batch_size)]
#plain = cipher.iloc[0].key + cipher.iloc[0].plantxt
plain = [[cipher.iloc[i].key + cipher.iloc[i].plantxt[:N]] for i in range(batch_size)]

key_length = len(cipher.iloc[0].key)

## Creating dictionary of alphabet and numbers
od = OrderedDict(zip(ascii_uppercase, count(0)))
od.update({'-': 26}) # adding '-' as leters

def encode(text):
    ## strip white space and "'" in the string, and convert it into numbers
    return  [[od[letter] for letter in line[0].replace(' ', '').strip("").upper()] for line in text]

def to_np(x):
    return x.data.cpu().numpy()

# encoding strings
x_cipher = encode(key_cipher)
y_plain = encode(plain)

## One hot encoding...
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
def one_hot(data):
    # creating empty nested list to store one-hot
    empty = [[[0 for n in range(0, len(od))] for l in range(N+key_length)] for i in range(len(key_cipher))]

    for i, one_batch in enumerate(data):
        for c, n in enumerate(one_batch):
            empty[i][c][n] = 1
    return empty

# One Hot encoding the cipher input, no need to one_hot labels
cipher_one_hot = one_hot(x_cipher)

# wrapping the data as pyTorch Variable
inputs_cipher = Variable(torch.Tensor(cipher_one_hot))
labels_plain = Variable(torch.LongTensor(y_plain))


###### Building the LSTM model

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.lstm1= nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Linear/ normal Neural Network layer on top of LSTM
        self.linear = nn.Linear(num_classes, num_classes)


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
        out = out.contiguous().view(-1, num_classes)

        # Neural Net/Fully Connected layer
        p = self.linear(out)
        return p


# Instantiate RNN model
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
print(lstm)

# set logger directory
# set your own directory to store training run record file for tensorboard
# the orange visualization tool for training runs
abspath = os.path.abspath(os.curdir)
root_name = 'rna-capstone'
root_path = os.path.join(abspath[:abspath.index(root_name)], root_name)

LOG_PATH = os.path.join(root_path, 'logs')

# Name each event (training run) file according to different training parameters
LOG_DIR= 'Layer_'+ str(num_layers)+'_lr_'+str(lr)+'_epoch_'+str(num_iter)

# set logger
logger = Logger(os.path.join(LOG_PATH, LOG_DIR))

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

# adaptive learning rate
# here, I'm using adaptive learning rate that decreases based on 'milestone',
# preset iteration number.
# THere are other adaptive learning rate algorithms avalable at...
#  http://pytorch.org/docs/master/optim.html  under how to adjust learning rate
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8500, 12000])

# accuracy function to compute prediction accuracy
def accuracy(prediction, labels):
    _, idx = prediction.max(1) # LongTensor containing predicted labels in idx
    idx_label = labels.view(sequence_length*batch_size) # Resizing labels
    compare = idx == idx_label # Getting 1/0 true false values
    compare = compare.float() # converting bol to float of 1/0
    return compare.mean() * 100

# Train the model
for epoch in range(num_iter):
    outputs = lstm(inputs_cipher)
    optimizer.zero_grad()
    # "Flattening the labels vector so that it can be fed into loss function"
    loss = criterion(outputs, labels_plain.view(sequence_length*batch_size))
    loss.backward()
    optimizer.step()
    #if epoch%10 == 0:

    if (epoch+1)%10==0:

        ## Printing out predicted and label string
        _, idx = outputs.max(1)
        idx = idx.data.numpy()[:sequence_length] # printing out only one sentence
        idx_label = labels_plain[0].data.numpy()
        result_str = [idx2char[c] for c in idx.squeeze()]
        label_str = [idx2char[c] for c in idx_label.squeeze()]

        # printing out accuracyy
        acc_train = accuracy(outputs, labels_plain)
        print()
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
        print("Training_accuracy: %1.3f" % (acc_train))
        print("Predicted string: ", ''.join(result_str))
        print("Label string:     ", ''.join(label_str))


        ##### Tensorboard logging
        # 1. log the scalar values
        info = {
                'loss': loss.data[0],
                'accuracy': acc_train
                }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch+1)


        # 2. Log values and gradients of the parameters (histogram)
        for tag, value in lstm.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), epoch+1)
            logger.histo_summary(tag+'/grad', to_np(value.grad), epoch+1)
print("Learning finished!")



#### Compute testing accuracy

prediction = lstm(inputs_cipher)
acc = accuracy(prediction, labels_plain)
print("Training set Accuracy: %1.1f" % (acc), "%")

## Loading test set data

cipher_test = pre.data_train() # importing the dataframe of cipher txt
#key_cipher = cipher.iloc[0].key + cipher.iloc[0].cipher
## creating list of strings of all the texts
key_cipher_test = [[cipher_test.iloc[i].key + cipher_test.iloc[i].cipher.replace(' ', '')[:N]] for i in range(batch_size)]
#plain = cipher.iloc[0].key + cipher.iloc[0].plantxt
plain_test = [[cipher_test.iloc[i].key + cipher_test.iloc[i].plantxt[:N]] for i in range(batch_size)]

key_length_test = len(cipher_test.iloc[0].key)
od = OrderedDict(zip(ascii_uppercase, count(0)))
od.update({'-': 26}) # adding white space as leters


x_cipher_test = encode(key_cipher_test)
y_plain_test = encode(plain_test)


cipher_one_hot_test = one_hot(x_cipher_test)

## enclosing the data inside of list, for dimensionality of input

# As we have one batch of samples, we will change them to variables only once

inputs_cipher_test = Variable(torch.Tensor(cipher_one_hot_test))
labels_plain_test = Variable(torch.LongTensor(y_plain_test))

pred_test = lstm(inputs_cipher_test)
acc_test = accuracy(pred_test, labels_plain_test)

