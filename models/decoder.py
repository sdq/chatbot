######################################################################
# Decoder
# ~~~~~~~
#
# The decoder RNN generates the response sentence in a token-by-token
# fashion. It uses the encoder’s context vectors, and internal hidden
# states to generate the next word in the sequence. It continues
# generating words until it outputs an *EOS_token*, representing the end
# of the sentence. A common problem with a vanilla seq2seq decoder is that
# if we rely soley on the context vector to encode the entire input
# sequence’s meaning, it is likely that we will have information loss.
# This is especially the case when dealing with long input sequences,
# greatly limiting the capability of our decoder.
#
# To combat this, `Bahdanau et al. <https://arxiv.org/abs/1409.0473>`__
# created an “attention mechanism” that allows the decoder to pay
# attention to certain parts of the input sequence, rather than using the
# entire fixed context at every step.
#
# At a high level, attention is calculated using the decoder’s current
# hidden state and the encoder’s outputs. The output attention weights
# have the same shape as the input sequence, allowing us to multiply them
# by the encoder outputs, giving us a weighted sum which indicates the
# parts of encoder output to pay attention to. `Sean
# Robertson’s <https://github.com/spro>`__ figure describes this very
# well:
#
# .. figure:: /_static/img/chatbot/attn2.png
#    :align: center
#    :alt: attn2
#
# `Luong et al. <https://arxiv.org/abs/1508.04025>`__ improved upon
# Bahdanau et al.’s groundwork by creating “Global attention”. The key
# difference is that with “Global attention”, we consider all of the
# encoder’s hidden states, as opposed to Bahdanau et al.’s “Local
# attention”, which only considers the encoder’s hidden state from the
# current time step. Another difference is that with “Global attention”,
# we calculate attention weights, or energies, using the hidden state of
# the decoder from the current time step only. Bahdanau et al.’s attention
# calculation requires knowledge of the decoder’s state from the previous
# time step. Also, Luong et al. provides various methods to calculate the
# attention energies between the encoder output and decoder output which
# are called “score functions”:
#
# .. figure:: /_static/img/chatbot/scores.png
#    :width: 60%
#    :align: center
#    :alt: scores
#
# where :math:`h_t` = current target decoder state and :math:`\bar{h}_s` =
# all encoder states.
#
# Overall, the Global attention mechanism can be summarized by the
# following figure. Note that we will implement the “Attention Layer” as a
# separate ``nn.Module`` called ``Attn``. The output of this module is a
# softmax normalized weights tensor of shape *(batch_size, 1,
# max_length)*.
#
# .. figure:: /_static/img/chatbot/global_attn.png
#    :align: center
#    :width: 60%
#    :alt: global_attn
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attn

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden