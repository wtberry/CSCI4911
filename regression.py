import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 10)
        self.output = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

plt.ion() # dynamically change the graph in runtime

for t in range(100):
    
    x = torch.unsqueeze(torch.linspace(-2, 2, 100), dim=1)
    y = x.pow(2) + 1 * torch.rand(x.size())

    x, y = Variable(x), Variable(y)
    
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla() # clear the current axis
        plt.scatter(x, y)
        # Plot for prediction line (the curve)
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-') 
        plt.ylim([-1, 6])
        # pause to make time to plot
        plt.pause(0.05)

plt.ioff() # turn interactive mode off
plt.show()

