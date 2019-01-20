# Lab 12 RNN
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
BLSTM to learn how to decrypt vigenere cipher based on training data

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

- N: numbers of characters included in ciphered&plain text

- input size: 27, for alphabet & '-' for key padding

- batch_size: number of key_text pairs in training dataset

- sequence_length: total number of letters in the input/output sequence, 28 here.

- num_layers: how many BLSTM layers are used in the model


This version will add a column to the one-hot matrix, indicating if the input 
letter is key/not.

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
N = 20
num_classes = 27
input_size = 28
hidden_size = 27  # output from the BLSTM. 5 to directly predict one-hot
batch_size = 500# one sentence
sequence_length = N+8# text + key
num_layers = 1  # one-layer blstm

lr = 0.01
num_iter = int(50)

### Printing out the parameters
def param_print():
    print('number of classes: ', num_classes)
    print('input_size: ', input_size)
    print('hidden_sizs: ', hidden_size)
    print('batch size: ', batch_size)
    print('length of sequence: ', sequence_length)
    print('Number of layers: ', num_layers)
    print('Training details.....')
    print('Learning rate: ', lr)
    print('number of iterations: ', num_iter)

param_print()

### Prepping the data
idx2char = [letter for letter in ascii_uppercase]
idx2char.append(' ')


def data_load(dtype, batch_size, N):
    '''
    This function import data as pandas dataframe from csv, and 
    encode it, and create one hot matrix, representing alphabets and key_indicator.
    dtype: 'train'/'test'
    batch_size: int, more than 2 and less than the datasize of the whole dataset.
    '''

    if dtype == 'train':
        cipher = pre.data_train() # importing the dataframe of cipher txt, as Pandas DataFrame
    elif dtype == 'test': 
        cipher = pre.data_test() # importing the dataframe of cipher txt, as Pandas DataFrame
    #key_cipher = cipher.iloc[0].key + cipher.iloc[0].cipher
    
    ### Sample random data
    cipher = cipher.sample(batch_size) # sampling random data, batch_size amout
    # DataFrame.sample() http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.sample.html 
    # Pandas DataFrame.iterrows() https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    
    key_cipher = [[row[-1].key + row[-1].cipher.replace(' ', '')[:N]] for row in cipher.sample(batch_size).iterrows()]
    #plain = cipher.iloc[0].key + cipher.iloc[0].plantxt
    plain = [[row[-1].key + row[-1].plantxt[:N]] for row in cipher.sample(batch_size).iterrows()]

    ## create input text for backwards RNN units
    key_cipher_b = [[row[0][::-1]] for row in key_cipher]
    plain_b = [[row[0][::-1]] for row in plain]

    ## creating list of strings of all the texts
    
    key_length = len(cipher.iloc[0].key)
    
    ## Creating dictionary of alphabet and numbers
    od = OrderedDict(zip(ascii_uppercase, count(0)))
    od.update({'-': 26}) # adding '-' as leters
    
    def encode(text):
        ## strip white space and "'" in the string, and convert it into numbers 
        return  [[od[letter] for letter in line[0].replace(' ', '').strip("").upper()] for line in text]
    
    
    # encoding forward strings
    x_cipher = encode(key_cipher)
    y_plain = encode(plain)
    # encoding backword strings
    x_cipher_b = encode(key_cipher_b)
    y_plain_b = encode(plain_b)
    
    ## One hot encoding...  
    # https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/ 
    def one_hot(data):
        # creating empty nested list to store one-hot
        # len(od)+1: adding 1 more coulumn for key/cipher indicator
        empty = [[[0 for n in range(0, len(od)+1)] for l in range(N+key_length)] for i in range(len(key_cipher))]
    
        for batch_count, one_batch in enumerate(data):
            for sequence_count, idx_count in enumerate(one_batch):
                empty[batch_count][sequence_count][idx_count] = 1
                
                # if sequence count/number of letters in a sequence is less than 
                # key length (within key length) & the letter is not equal to '-', 

                # the key column value == 1, otherwise left equal to 0
                if sequence_count < key_length and idx_count != 26:
                    empty[batch_count][sequence_count][-1] = 1
        return empty
    
    # One Hot encoding the cipher input, no need to one_hot labels
    cipher_one_hot = one_hot(x_cipher)
    cipher_one_hot_b = one_hot(x_cipher_b)
    
    # wrapping the data as pyTorch Variable
    inputs_cipher = Variable(torch.Tensor(cipher_one_hot))
    labels_plain = Variable(torch.LongTensor(y_plain))
    # wrapping the backward data as Pytorch Variable
    inputs_cipher_b = Variable(torch.Tensor(cipher_one_hot_b))
    labels_plain_b = Variable(torch.LongTensor(y_plain_b))

    data_dic = {'input_f':inputs_cipher, 'label_f':labels_plain, 'input_b':inputs_cipher_b, 'label_b':labels_plain_b}

    return data_dic
    


def to_np(x):
    return x.data.cpu().numpy()

###### Building the BLSTM model

class BLSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(BLSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.lstm_f = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_b = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Linear/ normal Neural Network layer on top of BLSTM
        self.linear = nn.Linear(num_classes, num_classes)
        self.linear2 = nn.Linear(num_classes, num_classes)


    def forward(self, x_f, x_b):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch hidden_size) for batch_first=True
        h_f = (Variable(torch.randn(self.num_layers, x_f.size(0), self.hidden_size)), Variable(torch.randn(self.num_layers, x_f.size(0), self.hidden_size)))
        h_b = (Variable(torch.randn(self.num_layers, x_b.size(0), self.hidden_size)), Variable(torch.randn(self.num_layers, x_b.size(0), self.hidden_size)))

#        h_2 = (Variable(torch.randn(x.size(0), self.num_layers, self.hidden_size)), Variable(torch.randn(x.size(0), self.num_layers, self.hidden_size)))

        #print('h_o', h_0)

        # Reshape input
        x_f.view(x_f.size(0), self.sequence_length, self.input_size)
        x_b.view(x_b.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (batch, num_layers * num_directions, hidden_size)
        out_f, _ = self.lstm_f(x_f, h_f)
        out_b, _ = self.lstm_b(x_b, h_b)
        out_f = out_f.contiguous().view(-1, num_classes)
        out_b = out_b.contiguous().view(-1, num_classes)

        ## concatenate/sum the output from forward and backward nets
        out = out_f + out_b

        # Neural Net/Fully Connected layer
        p = self.linear(out)
        pred = self.linear2(p)
        return pred


# Instantiate RNN model
lstm = BLSTM(num_classes, input_size, hidden_size, num_layers)
print(lstm)

# set logger directory
# set your own directory to store training run record file for tensorboard 
# the orange visualization tool for training runs
LOG_PATH = '/home/wataru/Uni/4911/cipher/new_logs/'

# Name each event (training run) file according to different training parameters
LOG_DIR= 'Key_cipher_indicator_Bidirectional_BLSTM'+'Layer_'+ str(num_layers)+'_lr_'+str(lr)+'_epoch_'+str(num_iter)

# set logger
logger = Logger(LOG_PATH+LOG_DIR)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

# adaptive learning rate
# here, I'm using adaptive learning rate that decreases based on 'milestone', 
# preset iteration number.
# THere are other adaptive learning rate algorithms avalable at...
#  http://pytorch.org/docs/master/optim.html  under how to adjust learning rate
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8000, 10000, 12000])

# accuracy function to compute prediction accuracy
def accuracy(prediction, labels):
    _, idx = prediction.max(1) # LongTensor containing predicted labels in idx
    idx_label = labels.view(sequence_length*batch_size) # Resizing labels
    compare = idx == idx_label # Getting 1/0 true false values
    compare = compare.float() # converting bol to float of 1/0
    return compare.mean() * 100

# Loading testing dataset
data_dict = data_load('test', batch_size, N)
inputs_cipher_test, labels_plain_test, inputs_cipher_test_b, labels_plain_test_b = data_dict['input_f'], data_dict['label_f'], data_dict['input_b'], data_dict['label_b']

# Train the model
for epoch in range(num_iter):

    # loading the data using the function, training data
    data_dict = data_load('test', batch_size, N)
    inputs_cipher, labels_plain, inputs_cipher_b, labels_plain_b = data_dict['input_f'], data_dict['label_f'], data_dict['input_b'], data_dict['label_b']
    outputs = lstm(inputs_cipher, inputs_cipher_b)
    # laoding test data
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

    if (epoch+1)%100==0:
        pred_test = lstm(inputs_cipher_test, inputs_cipher_test_b)
        acc_test = accuracy(pred_test, labels_plain_test)
        print("Testing set Accuracy: %1.1f" % (acc_test), "%")
print("Learning finished!")


# printing out the model's parameter again...
param_print()

#### Compute testing accuracy

prediction = lstm(inputs_cipher_test, inputs_cipher_test_b)
acc = accuracy(prediction, labels_plain)
print("Training set Accuracy: %1.1f" % (acc), "%")


pred_test = lstm(inputs_cipher_test, inputs_cipher_test_b)
acc_test = accuracy(pred_test, labels_plain_test)
print("Testing set Accuracy: %1.1f" % (acc_test), "%")

