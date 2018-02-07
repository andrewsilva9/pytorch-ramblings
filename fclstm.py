import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import string
import json

use_gpu = torch.cuda.is_available()


class FCLSTM(nn.Module):
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
        super(FCLSTM, self).__init__()
        # Architecture is Input -> LSTM -> Linear -> Sigmoid -> Cross Entropy Loss
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # input gate
        self.input_to_hidden = nn.Linear(embed_dim, hidden_dim)
        # forget gate
        self.forget_hidden = nn.Linear(hidden_dim, hidden_dim)
        # cell gate
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        # output gate
        self.hidden_to_output = nn.Linear(hidden_dim, hidden_dim)

        # input gate activation layer
        self.input_activation = nn.ReLU()
        # forget gate activation layer
        self.forget_activation = nn.Sigmoid()
        # cell gate activation layer
        self.cell_activation = nn.Tanh()
        # output gate activation layer
        self.output_activation = nn.ReLU()

        self.dropout_layer = nn.Dropout(p=dropout)

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
            return Variable(torch.zeros(batch_size, third_dim)).cuda()
        else:
            return Variable(torch.zeros(batch_size, third_dim))

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
        outputs = []
        for index in range(self.lengths[0]):
            embedded_seq = embeds[:, index, :]
            input_gated = self.input_to_hidden(embedded_seq)
            input_squashed = self.input_activation(input_gated)
            input_squashed = self.dropout_layer(input_squashed)
            self.hidden = self.dropout_layer(self.forget_activation(self.forget_hidden(self.hidden))) + \
                          self.cell_activation(self.hidden_to_hidden(self.hidden)) * input_squashed

            self.hidden = self.dropout_layer(self.hidden)
            output_gated = self.hidden_to_output(self.hidden)
            outputs.append(output_gated)
        # lin_ins = [outputs[i, self.lengths[i] - 1, :] for i in range(len(self.lengths))]
        lin_ins = []
        for index, length in enumerate(self.lengths):
            lin_ins.append(outputs[length-1][index])
        lin_ins = torch.stack(lin_ins)
        pred_ready_x = self.lin_layer(lin_ins)
        probs = F.log_softmax(pred_ready_x, dim=1)
        # probs = probs.view(1, -1)
        return probs
