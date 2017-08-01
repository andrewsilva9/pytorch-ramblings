import torch
from torch.autograd import Variable

class meReLU(torch.autograd.Function):
    """
    custom autograd function
    """
    def forward(self, input):
        """
        cache tensors for backward pass using save_for_backward
        """
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor

batch_size = 64
dim_in = 1000
dim_hid = 100
dim_out = 10

x = Variable(torch.randn(batch_size, dim_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(batch_size, dim_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(dim_in, dim_hid).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(dim_hid, dim_out).type(dtype), requires_grad=True)

lr = 1e-6
for it in range(500):
    relu = meReLU()

    y_pred = relu(x.mm(w1)).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(it, loss.data[0])

    # autograd to compute backward pass.
    loss.backward()

    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data

    # Set gradients to 0 after updating
    w1.grad.data.zero_()
    w2.grad.data.zero_()
