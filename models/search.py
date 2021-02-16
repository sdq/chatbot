######################################################################
# Define Evaluation
# -----------------
#
# After training a model, we want to be able to talk to the bot ourselves.
# First, we must define how we want the model to decode the encoded input.
#
# Greedy decoding
# ~~~~~~~~~~~~~~~
#
# Greedy decoding is the decoding method that we use during training when
# we are **NOT** using teacher forcing. In other words, for each time
# step, we simply choose the word from ``decoder_output`` with the highest
# softmax value. This decoding method is optimal on a single time-step
# level.
#
# To facilite the greedy decoding operation, we define a
# ``GreedySearchDecoder`` class. When run, an object of this class takes
# an input sequence (``input_seq``) of shape *(input_seq length, 1)*, a
# scalar input length (``input_length``) tensor, and a ``max_length`` to
# bound the response sentence length. The input sentence is evaluated
# using the following computational graph:
#
# **Computation Graph:**
#
#    1) Forward input through encoder model.
#    2) Prepare encoder's final hidden layer to be first hidden input to the decoder.
#    3) Initialize decoder's first input as SOS_token.
#    4) Initialize tensors to append decoded words to.
#    5) Iteratively decode one word token at a time:
#        a) Forward pass through decoder.
#        b) Obtain most likely word token and its softmax score.
#        c) Record token and score.
#        d) Prepare current token to be next decoder input.
#    6) Return collections of word tokens and scores.
#

import torch
import torch.nn as nn

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores