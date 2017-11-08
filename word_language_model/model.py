import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MoS_Decoder(nn.Module):
    
    def __init__(self, nhid, ntoken, nsoftmax=4):
        super(MoS_Decoder, self).__init__()
        self.nsoftmax = nsoftmax
        self.linear = nn.Linear(nhid, nhid * nsoftmax) 
        self.decoder = nn.Linear(nhid, ntoken)
        self.gater = nn.Linear(nhid, nsoftmax)

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.gater.bias.data.fill_(0)
        self.gater.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        # input size: batch size, nhid
        nbatch= input.size(0)
        hidden = F.tanh(self.linear(input)) # nbatch, nsoftmax * nhid
        hidden = hidden.view(nbatch * self.nsoftmax, -1) # nbatch * nsoftmax, nhid
        logits = F.softmax(self.decoder(hidden)) # nbatch * nsoftmax, ntoken
        logits = logits.view(nbatch, self.nsoftmax, -1) # batch, nsoftmax, ntoken
        weights = F.softmax(self.gater(input)) # batch, nsoftmax
        logits = weights.unsqueeze(2) * logits # batch, nsoftmax, ntoken
        logits = logits.sum(1) # batch, ntoken
        return logits

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
            tie_weights=False, mos=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.mos = mos
        if mos:
            self.decoder = MoS_Decoder(nhid, ntoken)
        else:
            self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            if mos:
                self.decoder.decoder.weight = self.encoder.weight
            else:
                self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.mos:
            self.decoder.init_weights()
        else:
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
