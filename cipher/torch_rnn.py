import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd

##
'''
Practice RNN using pytorch from youtube tutorial,
link: https://www.youtube.com/watch?v=ogZi5oIo4fI&index=12&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m 

This script don't use class RNN(nn.Module) to define RNN model like torch_rnn_seq.py
'''
## First, feed the sequence of letters, 'hello' one by one to observe outputs

# defining the RNN cell...
cell = nn.RNN(input_size = 4, hidden_size=2, batch_first=True)
# input_size = input data's dimension, here one hot encoded matrix, 4 letters
# hidden_size = output dimension from hidden layer, here, 2

### Defining inputs
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# Create input sequence...
input1 = autograd.Variable(torch.Tensor([[h]]))
input2 = autograd.Variable(torch.Tensor([[e]]))
input3 = autograd.Variable(torch.Tensor([[l]]))
input4 = autograd.Variable(torch.Tensor([[o]]))

input_list = [input1, input2, input3, input4]

## Initializing the first hidden state to feed into the hidden layer, random values
hidden = autograd.Variable(torch.randn(1, 1, 2))

# in a for loop, we'll feed the input and hidden state from 1 iter before into 
# the cell, and print out the hidden and output
for inputs in input_list:
    out, hidden = cell(inputs, hidden)
    print('output', out.data)
    print('hidden state', hidden.data)


'''
Now, we'll create one big input sequence matrix to feed into the RNN at 'once'.
Meaning we only have to feed into the network once, as one whole matrix, but it
is actually computing output values based on sequence.
Therefore no need of for-loop
We can reuse the same cell, and same hidden layer size. Only the output size 
changes.
'''

# creating new input, all letters in one matrix
inputs = autograd.Variable(torch.Tensor([[h, e, l, l, o]]))

print('input: ', inputs.data)

# Initialize the hidden state
hidden = autograd.Variable(torch.randn(1, 1, 2))

# Feeding the all of the sequence at once
out, hidden = cell(inputs, hidden)

'''
Sizes of output and hidden...
output: (batch x #of sequences x output dimension)
hidden: (#of layers x batch x output/hidden size)
'''
print('output... in different size now, as (1x5x4)', out)
print()
print('hidden... ', hidden)
