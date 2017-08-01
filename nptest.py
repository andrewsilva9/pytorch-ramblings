import numpy as np

batch_size = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

X = np.random.randn(batch_size, input_dim)
Y = np.random.randn(batch_size, output_dim)

w1 = np.random.randn(input_dim, hidden_dim)
w2 = np.random.randn(hidden_dim, output_dim)

learning_rate = 1e-6
for it in range(500):
    h = X.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred - Y).sum()
    print(it, loss)

    grad_y_pred = 2.0 * (y_pred - Y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = X.T.dot(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
