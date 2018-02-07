import torch
import matplotlib.pyplot as plt
import lstm_utils

# Learning Rate
learning_rate = 1e-3
running_loss = []
BATCH_SIZE = 1
logs_per_epoch = 100
num_epochs = 14
PAD_BATCH_FLAG = True
use_gpu = torch.cuda.is_available()
filename_str = 'yelp_stars'
json_file = '/home/asilva/Data/yelp_dataset_2017/review.json'
train_percent = 80

raw_data = lstm_utils.load_yelp_data(json_file)
dataset, test_dataset, word_to_id = lstm_utils.build_dataset_yelp(raw_data, train_percent)
vocab = len(word_to_id)

dataset.sort(key=lambda x: len(x[0]), reverse=True)
test_dataset.sort(key=lambda x: len(x[0]), reverse=True)
dataset = lstm_utils.clean_dataset(dataset)
test_dataset = lstm_utils.clean_dataset(test_dataset)
model = lstm_utils.Net(vocab_size=vocab, 
                       hidden_dim=1024,
                       embed_dim=128, 
                       num_layers=2,
                       num_classes=5, 
                       dropout=0.4, 
                       batch_size=BATCH_SIZE)

loss_fn = torch.nn.NLLLoss()
if use_gpu:
    model = model.cuda()
    loss_fn = loss_fn.cuda()

# Lists to keep track of average train / test losses over time (for plotting)
train_loss = []
test_loss = []
step_size = BATCH_SIZE if PAD_BATCH_FLAG else 1
train_log_interval = len(dataset)/step_size/logs_per_epoch
test_log_interval = len(test_dataset)/step_size/logs_per_epoch


for epoch in range(1, num_epochs+1):
    # Training over all training data
    if epoch < num_epochs/2:
        lr = learning_rate
    else:
        lr = learning_rate*0.1
    print('Epoch:', epoch)
    print('Learning rate:', lr)
    print('Batch size:', step_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.7, 0.99))
    train_loss.extend(lstm_utils.pass_through(model,
                                              loss_fn,
                                              optimizer,
                                              dataset,
                                              batch=PAD_BATCH_FLAG,
                                              batch_size=BATCH_SIZE,
                                              train=True,
                                              log_every=train_log_interval))
    test_loss.extend(lstm_utils.pass_through(model,
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
    }, filename=filename_str+str(epoch+1))


plt.plot(train_loss, 'r', test_loss, 'b')
plt.savefig('yelp_avg_losses.png')
