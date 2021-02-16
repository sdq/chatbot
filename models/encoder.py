######################################################################
# Encoder
# ~~~~~~~
#
# The encoder RNN iterates through the input sentence one token
# (e.g. word) at a time, at each time step outputting an “output” vector
# and a “hidden state” vector. The hidden state vector is then passed to
# the next time step, while the output vector is recorded. The encoder
# transforms the context it saw at each point in the sequence into a set
# of points in a high-dimensional space, which the decoder will use to
# generate a meaningful output for the given task.
#
# At the heart of our encoder is a multi-layered Gated Recurrent Unit,
# invented by `Cho et al. <https://arxiv.org/pdf/1406.1078v3.pdf>`__ in
# 2014. We will use a bidirectional variant of the GRU, meaning that there
# are essentially two independent RNNs: one that is fed the input sequence
# in normal sequential order, and one that is fed the input sequence in
# reverse order. The outputs of each network are summed at each time step.
# Using a bidirectional GRU will give us the advantage of encoding both
# past and future context.
#
# Bidirectional RNN:
#
# .. figure:: /_static/img/chatbot/RNN-bidirectional.png
#    :width: 70%
#    :align: center
#    :alt: rnn_bidir
#
# Image source: https://colah.github.io/posts/2015-09-NN-Types-FP/
#
# Note that an ``embedding`` layer is used to encode our word indices in
# an arbitrarily sized feature space. For our models, this layer will map
# each word to a feature space of size *hidden_size*. When trained, these
# values should encode semantic similarity between similar meaning words.
#
# Finally, if passing a padded batch of sequences to an RNN module, we
# must pack and unpack padding around the RNN pass using
# ``nn.utils.rnn.pack_padded_sequence`` and
# ``nn.utils.rnn.pad_packed_sequence`` respectively.
#
# **Computation Graph:**
#
#    1) Convert word indexes to embeddings.
#    2) Pack padded batch of sequences for RNN module.
#    3) Forward pass through GRU.
#    4) Unpack padding.
#    5) Sum bidirectional GRU outputs.
#    6) Return output and final hidden state.
#
# **Inputs:**
#
# -  ``input_seq``: batch of input sentences; shape=\ *(max_length,
#    batch_size)*
# -  ``input_lengths``: list of sentence lengths corresponding to each
#    sentence in the batch; shape=\ *(batch_size)*
# -  ``hidden``: hidden state; shape=\ *(n_layers x num_directions,
#    batch_size, hidden_size)*
#
# **Outputs:**
#
# -  ``outputs``: output features from the last hidden layer of the GRU
#    (sum of bidirectional outputs); shape=\ *(max_length, batch_size,
#    hidden_size)*
# -  ``hidden``: updated hidden state from GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden