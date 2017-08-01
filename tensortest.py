import torch

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # use for GPU
batch_size = 64
dim_in = 1000
dim_hid = 100
dim_out = 10

X = torch.randn(batch_size, dim_in).type(dtype)
Y = torch.randn(batch_size, dim_out).type(dtype)

w1 = torch.randn(dim_in, dim_hid).type(dtype)
w2 = torch.randn(dim_hid, dim_out).type(dtype)

lr = 1e-6
for it in range(500):
    h = X.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Loss
    loss = (y_pred - Y).pow(2).sum()
    print(it, loss)

    grad_y_pred = 2.0 * (y_pred - Y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = X.t().mm(grad_h)

    w1 -= lr * grad_w1
    w2 -= lr * grad_w2