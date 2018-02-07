import torch
import matplotlib.pyplot as plt
import lstm_utils
import fclstm
import random

# Learning Rate
learning_rate = 1e-3
running_loss = []
BATCH_SIZE = 64
fc = True
logs_per_epoch = 10
num_epochs = 18
# MAX
_LEN = 500
PAD_BATCH_FLAG = True
train_loss = []
test_loss = []
use_gpu = torch.cuda.is_available()

train_dir = '/home/asilva/Data/aclImdb/train'
test_dir = '/home/asilva/Data/aclImdb/test'

dataset, word_to_id = lstm_utils.build_dataset_imdb(train=True)
vocab = len(word_to_id)
test_dataset, word_to_id_t = lstm_utils.build_dataset_imdb(train=False)

# Sort for more efficient batching
dataset.sort(key=lambda x: len(x[0]), reverse=True)
test_dataset.sort(key=lambda x: len(x[0]), reverse=True)

# random.shuffle(dataset)
# random.shuffle(test_dataset)
if fc:
    model = fclstm.FCLSTM(vocab_size=vocab,
                           hidden_dim=512,
                           embed_dim=128,
                           num_layers=1,
                           num_classes=2,
                           dropout=0.4,
                           batch_size=BATCH_SIZE)
else:
    model = lstm_utils.Net(vocab_size=vocab,
                           hidden_dim=512,
                           embed_dim=128,
                           num_layers=1,
                           num_classes=2,
                           dropout=0.4,
                           batch_size=BATCH_SIZE)

loss_fn = torch.nn.NLLLoss()
if use_gpu:
    model = model.cuda()
    loss_fn = loss_fn.cuda()

# Lists to keep track of average train / test losses over time (for plotting)
step_size = BATCH_SIZE if PAD_BATCH_FLAG else 1
train_log_interval = len(dataset)/step_size/logs_per_epoch
test_log_interval = len(test_dataset)/step_size/logs_per_epoch

for epoch in range(1, num_epochs+1):
    # Training over all training data
    print('EPOCH:', epoch)
    if epoch < num_epochs/4:
        lr = learning_rate
    elif epoch < num_epochs/2:
        lr = learning_rate * 0.5
    else:
        lr = learning_rate * 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.7, 0.99))
    train_loss.append(lstm_utils.pass_through(model,
                                              loss_fn,
                                              optimizer,
                                              dataset,
                                              batch=PAD_BATCH_FLAG,
                                              batch_size=BATCH_SIZE,
                                              train=True,
                                              log_every=train_log_interval))
    test_loss.append(lstm_utils.pass_through(model,
                                             loss_fn,
                                             optimizer,
                                             test_dataset,
                                             batch=PAD_BATCH_FLAG,
                                             batch_size=BATCH_SIZE,
                                             train=False,
                                             log_every=test_log_interval))
    # Checkpoint the model with the state_dict, optimizer, and current epoch number
    lstm_utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename='sentiment_epoch'+str(epoch+1))

plt.plot(train_loss, 'r', test_loss, 'b')
plt.savefig('avg_losses.png')
