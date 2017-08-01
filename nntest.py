import torch
from torch.autograd import Variable

batch_size = 1
dim_in = 1000
dim_hid = 100
dim_out = 10

X = Variable(torch.randn(batch_size, dim_in))
Y = Variable(torch.randn(batch_size, dim_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, dim_hid),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_hid, dim_out)
)

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for it in range(500):
    y_pred = model(X)

    loss = loss_fn(y_pred, Y)
    print(it, loss.data[0])

    optimizer.zero_grad() # Let optimizer solve
    # model.zero_grad() # Manually update params

    loss.backward()

    optimizer.step() # Let optimizer solve
    # for param in model.parameters(): # Manually update params
    #     param.data -= lr * param.grad.data