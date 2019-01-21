#nn uses autograd to define models and differentiate them
#An nn.Module contains layers & a method forward(input) that returns the output
#Typical procedure:
#1) Define neural network that has learnable parameters or weights
#2) Iterate over a dataset of inputs
#3) Process input through the network
#4) Compute loss (how far away is output from being correct)
#5) Propagate gradients back into the network's parameters
#6) Update the weights of the network, use this simple rule: weight=weight - learning_rate * gradient

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
    # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self,x):
        #Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        #if the size is square you can only specify 1 number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = Variable(torch.randn(1,1,32,32))
out = net(input)
print(out)

net.zero_gra()
out.backward(torch.randn(1,10))

#Recap
# torch.tensor - a multi dimensional array
# autograd.Variable - wraps a Tensor and records the history of operations applied to it
# nn.Module - Neural network module
# autograd.Function - implements forward and backward definitions of an autograd operation

# Loss Function
output = net(input)
target = Variable(torch.arrange(1,11)) #dummy target 1,11
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

#Following a few backward steps
print(loss.grad_fn)  #MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(locc.grad_fn.next_functions[0][0].next_functions[0][0])  #ReLU

#Backprop
# Backpropagate the error by loss.backward()
# Clear the existing gradients or else they accumulate
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

# Update the weights
# simplest rule: Stochastic GRadient Descent (SGD)
#weight = weight - learning_rate * gradient

learninf_rate = 0.01
for f in net.paraeters():
    f.data.sub_(f.grad.data * learning_rate)


# use various update ruls in addition to SGD in torch.optim
import torch.optim

#create optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

#Training loop
optimizer.zero_grad()  #zero gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  #does the update


