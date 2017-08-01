import random
import torch
from torch.autograd import Variable


class WeirdNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WeirdNet, self).__init__()
        self.input_linear = torch.nn.Linear(input_dim, hidden_dim)
        self.middle_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Randomly choose 0, 1, 2, or 3 hidden layers. use weight sharing
        """

        h_relu = self.input_linear(x).clamp(min=0)
        for i in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu.clamp(min=0))
        y_pred = self.output_linear(h_relu)
        return y_pred

batch_size = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

x = Variable(torch.randn(batch_size, input_dim))
y = Variable(torch.randn(batch_size, output_dim), requires_grad=False)
model = WeirdNet(input_dim, hidden_dim, output_dim)
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for it in range(5000):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(it, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
