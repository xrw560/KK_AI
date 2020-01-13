import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from visdom import Visdom
import torch
import torch.nn as nn

num_step = 50
input_size = 1
output_size = 1
hidden_size = 16
lr = 0.01
device = 'cpu'


class RNNnet(nn.Module):

    def __init__(self):
        super(RNNnet, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden


if __name__ == '__main__':
    rnnnet = RNNnet().to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(rnnnet.parameters(), lr=lr)
    hidden = torch.zeros(1, 1, hidden_size)

    global_step = 0
    vis = Visdom()
    # win 表征该env下的窗口句柄，一个win代表一个窗口，窗口标题由title决定
    vis.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

    for epoch in range(5000):
        start = np.random.randint(3, size=1)[0]
        time_steps = np.linspace(start, start + 10, num_step)
        data = np.sin(time_steps)
        data = data.reshape(num_step, 1)
        x = torch.tensor(data[:-1]).float().view(1, num_step - 1, 1).to(device)
        y = torch.tensor(data[1:]).float().view(1, num_step - 1, 1).to(device)

        output, hidden = rnnnet(x, hidden)
        hidden = hidden.detach()
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        vis.line([loss.item()], [global_step], win='train_loss', update='append')

        if epoch % 100 == 0:
            print("Iteration: {} loss: {}".format(epoch, loss.item()))
