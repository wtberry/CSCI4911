import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

'''
RNN Practice using PyTorch.
nn.Module and RNN class are used, to create own RNN class to train
Example data of 'hello' string is used after one hot encoded.
'''

torch.manual_seed(777) # reproducibility

idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [1, 0, 2, 3, 3, 4] # ihello

# As we have one batch of samples, we will change them to variables
inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5 # one-hot size
hidden_size = 5 # output from the LSTM. 5 to directly predict one-hot
batch_size = 1 # one sentence
sequence_length = 6 # |ihello| == 6
num_layers = 1 # One layer rnn


class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        ## constructor as in java...??? 

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length= sequence_length

        self.rnn = nn.RNN(input_size=5,
                          hidden_size=5, batch_first=True)

        def forward(self, x):
            
            # Initialize hidden and cell states
            # (bath, num_layers*num_directions, hidden_size) for batch_first=True
            h_0 = Variable(torch.zeros(
                    x.size(0), self.num_layers, self.hidden_size))

            # Reshape input
            x.view = (x.size(0), self.sequence_length, self.input_size)

            # Propagate input through RNN
            # Input: (batch x seq_len x input_size)
            # Hidden: (batch x num_layers * num_directions x hidden_size)

            # Passing the param to the model, no need to keep hidden state, 
            # since not iterating through the each sequence
            out, _ = self.rnn(x, h_o)
            return out.view(-1, num_classes)



## Instantiate the RNN model
rnn = RNN(num_classes, input_size, hidden_size, num_layers)
print(rnn)

# Set loss and optimizer function
# Cross entropyloss = logsoftmax + NLLoss
criteron = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

for epoch in range(100):
    outputs = rnn(inputs)
    optimizer.zero_grad()
    loss = criteron(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    print('Predicted string: ', ''.join(result_str))

print("learning finished!")
