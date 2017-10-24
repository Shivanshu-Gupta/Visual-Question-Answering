import torch.nn as nn
import torch.nn.functional as F
from lstm import customRNN
from gru import CustomGRUCell


class POSTaggerModel(nn.Module):

    def __init__(self, rnn_class, embedding_dim, hidden_dim, vocab_size, target_size, use_gpu=True):
        super(POSTaggerModel, self).__init__()
        self.rnn_class = rnn_class
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.use_gpu = use_gpu
        if use_gpu:
            self.word_embeddings.cuda()
        self.num_layers = 1
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.rnn_class == 'lstm':
            self.rnn = customRNN(nn.LSTMCell, embedding_dim, hidden_dim, batch_first=False)
        elif self.rnn_class == 'gru':
            self.rnn = customRNN(nn.GRUCell, embedding_dim, hidden_dim, batch_first=False)
        elif self.rnn_class == 'rnn':
            self.rnn = customRNN(nn.RNNCell, embedding_dim, hidden_dim, batch_first=False)
        else:
            self.rnn = customRNN(CustomGRUCell, embedding_dim, hidden_dim, batch_first=False)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentences, ranges, lengths):
        embeds = self.word_embeddings(sentences)
        lstm_out, _ = self.rnn(embeds, ranges, lengths)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space)
        # do this if want 3D tensor (ref: https://github.com/pytorch/pytorch/issues/1020)
        # tag_scores = F.log_softmax(tag_space.transpose(0, 2)).transpose(0, 2)
        return tag_scores
