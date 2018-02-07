# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import string
import json


use_gpu = torch.cuda.is_available()
#
train_dir = '/home/asilva/Data/aclImdb/train'
test_dir = '/home/asilva/Data/aclImdb/test'


def save_checkpoint(state, filename='checkpoint_rel_small075.pth.tar'):
    """
    Helper function to save checkpoints of PyTorch models
    :param state: Everything to save with the checkpoint (params/weights, optimizer state, epoch, loss function, etc.)
    :param filename: Filename to save the checkpoint under
    :return: None
    """
    torch.save(state, filename)

############################################## BEGIN IMDB DATA IMPORT #################################################
def create_dataset_idmb(train=True):
    pos_label = 1
    neg_label = 0
    if train:
        pos_dir = os.path.join(train_dir, 'pos')
        neg_dir = os.path.join(train_dir, 'neg')
    else:
        pos_dir = os.path.join(test_dir, 'pos')
        neg_dir = os.path.join(test_dir, 'neg')
    samples = []
    for f in os.listdir(pos_dir):
        if f.endswith('.txt'):
            input_text = open(os.path.join(pos_dir, f), 'r').read()
            input_text = input_text.translate(None, string.punctuation).lower()
            samples.append((input_text, pos_label))
    for f in os.listdir(neg_dir):
        if f.endswith('.txt'):
            input_text = open(os.path.join(neg_dir, f), 'r').read()
            input_text = input_text.translate(None, string.punctuation).lower()
            samples.append((input_text, neg_label))
    return samples


def read_words_imdb():
    # words and vocab will always come from training directory
    pos_dir = os.path.join(train_dir, 'pos')
    neg_dir = os.path.join(train_dir, 'neg')
    words = []
    for f in os.listdir(pos_dir):
        if f.endswith('.txt'):
            input_text = open(os.path.join(pos_dir, f), 'r').read()
            input_text = input_text.translate(None, string.punctuation).lower()
            words.append(input_text)
    for f in os.listdir(neg_dir):
        if f.endswith('.txt'):
            input_text = open(os.path.join(neg_dir, f), 'r').read()
            input_text = input_text.translate(None, string.punctuation).lower()
            words.append(input_text)
    return words


def _build_vocab_imdb():
    data = read_words_imdb()
    word_to_ix = dict()
    word_to_ix['PAD'] = 0
    for review in data:
        for word in review.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def build_dataset_imdb(train=True):
    word_to_id = _build_vocab_imdb()
    raw_dataset = create_dataset_idmb(train)
    indexed_dataset = []
    for index in range(len(raw_dataset)):
        data = raw_dataset[index][0].split()
        data_to_inds = [word_to_id[word] for word in data if word in word_to_id]
        indexed_dataset.append([torch.LongTensor(data_to_inds), torch.LongTensor([raw_dataset[index][1]])])
    return indexed_dataset, word_to_id
############################################## END IMDB DATA IMPORT ###################################################


############################################## BEGIN YELP DATA IMPORT #################################################
def load_yelp_data(filename):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def create_dataset_yelp(raw_yelp_data, train_percent):
    """
    Takes in raw data (array of json dicts) and an integer percentage that should be training
    Convert the input yelp array of JSONs into tuples of (text, label)
    Text is the raw review text (punctuation removed, cast to lower case) and label is star rating
    Stars go from 0-4 instead of 1-5 for pytorch...
    Returns 2 arrays of tuples, TRAIN and TEST
    """
    last_ind = int(len(raw_yelp_data)*train_percent/100.0)

    train_samples = []
    test_samples = []
    for review in raw_yelp_data[:last_ind]:
            input_text = review['text'].encode('utf-8')
            input_text = input_text.translate(None, string.punctuation).lower()
            rating = review['stars'] - 1
            train_samples.append((input_text, rating))
    for review in raw_yelp_data[last_ind:]:
            input_text = review['text'].encode('utf-8')
            rating = review['stars'] - 1
            input_text = input_text.translate(None, string.punctuation).lower()
            test_samples.append((input_text, rating))
    return train_samples, test_samples


def read_words_yelp(raw_yelp_data, train_percent):
    # words and vocab will always come from training directory
    all_words = []
    last_ind = int(len(raw_yelp_data)*train_percent/100.0)
    for review in raw_yelp_data[:last_ind]:
            input_text = review['text'].encode('utf-8')
            input_text = input_text.translate(None, string.punctuation).lower()
            all_words.append(input_text)
    return all_words


def _build_vocab_yelp(raw_yelp_data, train_percent):
    data = read_words_yelp(raw_yelp_data, train_percent)
    word_to_ix = dict()
    word_to_ix['PAD'] = 0
    for review in data:
        for word in review.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def build_dataset_yelp(raw_yelp_data, train_percent):
    word_to_id = _build_vocab_yelp(raw_yelp_data, train_percent)
    raw_train_dataset, raw_test_dataset = create_dataset_yelp(raw_yelp_data, train_percent)
    indexed_train_dataset = []
    indexed_test_dataset = []
    for index in range(len(raw_train_dataset)):
        data = raw_train_dataset[index][0].split()
        data_to_inds = [word_to_id[word] for word in data if word in word_to_id]
        indexed_train_dataset.append([torch.LongTensor(data_to_inds),
                                      torch.LongTensor([raw_train_dataset[index][1]])])
    for index in range(len(raw_test_dataset)):
        data = raw_test_dataset[index][0].split()
        data_to_inds = [word_to_id[word] for word in data if word in word_to_id]
        indexed_test_dataset.append([torch.LongTensor(data_to_inds),
                                     torch.LongTensor([raw_test_dataset[index][1]])])
    return indexed_train_dataset, indexed_test_dataset, word_to_id
############################################## END YELP DATA IMPORT ###################################################



def clean_dataset(data):
    count_invalids = 0
    to_delete = []
    for index, element in enumerate(data):
        try:
            to_add = element[0][0]
        except:
            to_delete.append(index)
            count_invalids += 1
    print("There were", count_invalids, "invalid entries in the data")
    print("They are:", to_delete)
    for invalid in to_delete[::-1]:
        del data[invalid]
    return data


def pad_batch_data(data, max_length):
    """
    Function that takes in a list of (sample, label) and converts them to a tensor of samples
    of size BATCH_SIZE x MAX_LENGTH
    NOTE: CURRENTLY TAKES PRE-SORTED DATA (IN THE FUTURE, UNCOMMENT SORTING LINE IF DATA COMES UNSORTED)
    """
    sample = None
    label = None
    lengths = []
    #     data.sort(key=lambda x: len(x[0]), reverse=True)
    for element in data:
        to_add = element[0][:max_length]
        lengths.append(len(to_add))
        to_add = torch.cat((to_add, torch.zeros(max_length - len(to_add)).type(torch.LongTensor)), 0).expand(1,
                                                                                                             max_length)
        if sample is None:
            sample = to_add
            label = element[1]
        else:
            sample = torch.cat((sample, to_add))
            label = torch.cat((label, element[1]))
    return sample, label, lengths


def max_len_of_batch(data):
    max_len = 0
    for element in data:
        if len(element[0]) > max_len:
            max_len = len(element[0])
    return max_len


class Net(nn.Module):
    def __init__(self, vocab_size,
                 hidden_dim,
                 embed_dim,
                 num_layers=1,
                 num_classes=2,
                 dropout=0,
                 batch_size=64):
        """
        Initialize the Net
        :param vocab_size: length of vocab for generating embeds
        :param input_dim: Input dimension
        :param hidden_dim: Hidden dimension (of LSTM)
        :param embed_dim: Embedding dimension (word embeds for input to LSTM)
        :param output_dim: Num output classes (to log_softmax over for prediction)
        """
        super(Net, self).__init__()
        # Architecture is Input -> LSTM -> Linear -> Sigmoid -> Cross Entropy Loss
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.LSTM(embed_dim,
                           hidden_dim,
                           num_layers=self.num_layers,
                           dropout=dropout,
                           batch_first=True)

        self.lin_layer = nn.Linear(hidden_dim, num_classes)
        # Initialize hidden state (as a function so it can be called externally to clear the hidden state between passes
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        """
        Initialize hidden state so that it can be clean for each new series of inputs
        :return: Variable of zeros of shape (num_layers, minibatch_size, hidden_dim)
        """
        first_dim = self.num_layers
        second_dim = batch_size
        # Because last batches might be smaller and testing is 1by1
        self.batch_size = batch_size
        third_dim = self.hidden_dim
        if use_gpu:
            return (Variable(torch.zeros(first_dim, second_dim, third_dim)).cuda(),
                    Variable(torch.zeros(first_dim, second_dim, third_dim)).cuda())
        else:
            return (Variable(torch.zeros(first_dim, second_dim, third_dim)),
                    Variable(torch.zeros(first_dim, second_dim, third_dim)))

    def set_lengths(self, lengths_arr):
        """
        Hacky way of setting up a lengths array to pack padded sequence for forward passing
        """
        self.lengths = lengths_arr

    def forward(self, x):
        """
        Forward pass of the data
        Data is reshaped to (num_layers(of LSTM), minibatch size, input dimensions), then passed to LSTM
        Output of LSTM is passed to Linear embedding layer (and hidden layer of LSTM feeds next iteration)
        Output of embedding layer is passed to final output classes later, log softmax over last layer is returned
        :param x: Input sample to predict
        :return: Predicted class for given input
        """
        embeds = self.embedding(x)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, self.lengths, batch_first=True)
        lstm_out, self.hidden = self.rnn(embeds, self.hidden)
        lstm_out, sizes = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lin_ins = [lstm_out[i, sizes[i] - 1, :] for i in range(len(sizes))]
        lin_ins = torch.stack(lin_ins)
        pred_ready_x = self.lin_layer(lin_ins)
        probs = F.log_softmax(pred_ready_x, dim=1)
        return probs


def pass_through(model, loss_fn, opt, data, batch=True, batch_size=64, train=True, log_every=100):
    step_size = batch_size if batch else 1
    agg_loss = 0
    num_correct = 0
    num_wrong = 0
    total_guessed = 0
    step_size_loss = 0
    if train:
        model.train()
    else:
        model.eval()
    step = 0
    for i in range(0, len(data), step_size):
        step += 1

        if batch:
            batch_in = data[i:i + batch_size]
            max_length = max_len_of_batch(batch_in)
            sample, label, lengths = pad_batch_data(batch_in, max_length)
            minibatch_size = len(lengths)
            model.set_lengths(lengths)
        else:
            sample = data[i][0]
            label = data[i][1]
            minibatch_size = 1
            model.set_lengths(len(sample))
        if use_gpu:
            x_in, y_in = Variable(sample).cuda(), Variable(label).cuda()
        else:
            x_in = Variable(sample)
            y_in = Variable(label)

        model.hidden = model.init_hidden(minibatch_size)
        # Zero gradients from last pass
        if train:
            model.zero_grad()

        # Get prediction from this sample
        output = model(x_in)
        # Calculate loss
#        print output
#        y_in = y_in.view(-1)
#        print y_in
        loss = loss_fn(output, y_in)
        # Send loss back
        if train:
            loss.backward()
            opt.step()
        step_size_loss += loss.data[0]
        # Logging
        if step % log_every == 0:
            agg_loss += step_size_loss
            avg_loss = step_size_loss / log_every
            print 'Step', step, '|| Avg Loss:', avg_loss
            # avg_loss_agg.append(avg_loss)
            step_size_loss = 0
        if not train:
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == y_in.data)
            wrong = (predicted != y_in.data)
            num_correct += correct.sum(0)[0]
            num_wrong += wrong.sum(0)[0]
            total_guessed += correct.size(0)
    if not train:
        print('Accuracy:', 100.0 * num_correct / total_guessed)
        print('Inaccuracy:', 100.0 * num_wrong / total_guessed)
    return agg_loss / len(data)
