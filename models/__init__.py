######################################################################
# Define Models
# -------------
#
# Seq2Seq Model
# ~~~~~~~~~~~~~
#
# The brains of our chatbot is a sequence-to-sequence (seq2seq) model. The
# goal of a seq2seq model is to take a variable-length sequence as an
# input, and return a variable-length sequence as an output using a
# fixed-sized model.
#
# `Sutskever et al. <https://arxiv.org/abs/1409.3215>`__ discovered that
# by using two separate recurrent neural nets together, we can accomplish
# this task. One RNN acts as an **encoder**, which encodes a variable
# length input sequence to a fixed-length context vector. In theory, this
# context vector (the final hidden layer of the RNN) will contain semantic
# information about the query sentence that is input to the bot. The
# second RNN is a **decoder**, which takes an input word and the context
# vector, and returns a guess for the next word in the sequence and a
# hidden state to use in the next iteration.
#
# .. figure:: /_static/img/chatbot/seq2seq_ts.png
#    :align: center
#    :alt: model
#
# Image source:
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
#

from .encoder import EncoderRNN
from .decoder import LuongAttnDecoderRNN
from .attention import Attn
from .search import GreedySearchDecoder

__all__ = ["EncoderRNN", "LuongAttnDecoderRNN", "Attn", "GreedySearchDecoder"]