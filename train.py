import torch
import torch.nn as nn
import os
import random
import itertools
from loss import maskNLLLoss
from utils import zeroPadding, binaryMatrix, indexesFromSentence

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


######################################################################
# Prepare Data for Models
# -----------------------
#
# Although we have put a great deal of effort into preparing and massaging our
# data into a nice vocabulary object and list of sentence pairs, our models
# will ultimately expect numerical torch tensors as inputs. One way to
# prepare the processed data for the models can be found in the `seq2seq
# translation
# tutorial <https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`__.
# In that tutorial, we use a batch size of 1, meaning that all we have to
# do is convert the words in our sentence pairs to their corresponding
# indexes from the vocabulary and feed this to the models.
#
# However, if you’re interested in speeding up training and/or would like
# to leverage GPU parallelization capabilities, you will need to train
# with mini-batches.
#
# Using mini-batches also means that we must be mindful of the variation
# of sentence length in our batches. To accomodate sentences of different
# sizes in the same batch, we will make our batched input tensor of shape
# *(max_length, batch_size)*, where sentences shorter than the
# *max_length* are zero padded after an *EOS_token*.
#
#
# .. figure:: /_static/img/chatbot/seq2seq_batches.png
#    :align: center
#    :alt: batches
#
# The ``inputVar`` function handles the process of converting sentences to
# tensor, ultimately creating a correctly shaped zero-padded tensor. It
# also returns a tensor of ``lengths`` for each of the sequences in the
# batch which will be passed to our decoder later.
#
# The ``outputVar`` function performs a similar function to ``inputVar``,
# but instead of returning a ``lengths`` tensor, it returns a binary mask
# tensor and a maximum target sentence length. The binary mask tensor has
# the same shape as the output target tensor, but every element that is a
# *PAD_token* is 0 and all others are 1.
#
# ``batch2TrainData`` simply takes a bunch of pairs and returns the input
# and target tensors using the aforementioned functions.
#

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, PAD_token)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, PAD_token)
    mask = binaryMatrix(padList, PAD_token)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

######################################################################
# Single training iteration
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train`` function contains the algorithm for a single training
# iteration (a single batch of inputs).
#
# We will use a couple of clever tricks to aid in convergence:
#
# -  The first trick is using **teacher forcing**. This means that at some
#    probability, set by ``teacher_forcing_ratio``, we use the current
#    target word as the decoder’s next input rather than using the
#    decoder’s current guess. This technique acts as training wheels for
#    the decoder, aiding in more efficient training. However, teacher
#    forcing can lead to model instability during inference, as the
#    decoder may not have a sufficient chance to truly craft its own
#    output sequences during training. Thus, we must be mindful of how we
#    are setting the ``teacher_forcing_ratio``, and not be fooled by fast
#    convergence.
#
# -  The second trick that we implement is **gradient clipping**. This is
#    a commonly used technique for countering the “exploding gradient”
#    problem. In essence, by clipping or thresholding gradients to a
#    maximum value, we prevent the gradients from growing exponentially
#    and either overflow (NaN), or overshoot steep cliffs in the cost
#    function.
#
# .. figure:: /_static/img/chatbot/grad_clip.png
#    :align: center
#    :width: 60%
#    :alt: grad_clip
#
# Image source: Goodfellow et al. *Deep Learning*. 2016. https://www.deeplearningbook.org/
#
# **Sequence of Operations:**
#
#    1) Forward pass entire input batch through encoder.
#    2) Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state.
#    3) Forward input batch sequence through decoder one time step at a time.
#    4) If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
#    5) Calculate and accumulate loss.
#    6) Perform backpropagation.
#    7) Clip gradients.
#    8) Update encoder and decoder model parameters.
#
#
# .. Note ::
#
#   PyTorch’s RNN modules (``RNN``, ``LSTM``, ``GRU``) can be used like any
#   other non-recurrent layers by simply passing them the entire input
#   sequence (or batch of sequences). We use the ``GRU`` layer like this in
#   the ``encoder``. The reality is that under the hood, there is an
#   iterative process looping over each time step calculating hidden states.
#   Alternatively, you ran run these modules one time-step at a time. In
#   this case, we manually loop over the sequences during the training
#   process like we must do for the ``decoder`` model. As long as you
#   maintain the correct conceptual model of these modules, implementing
#   sequential models can be very straightforward.
#
#

teacher_forcing_ratio = 1.0

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length, device):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


######################################################################
# Training iterations
# ~~~~~~~~~~~~~~~~~~~
#
# It is finally time to tie the full training procedure together with the
# data. The ``trainIters`` function is responsible for running
# ``n_iterations`` of training given the passed models, optimizers, data,
# etc. This function is quite self explanatory, as we have done the heavy
# lifting with the ``train`` function.
#
# One thing to note is that when we save our model, we save a tarball
# containing the encoder and decoder state_dicts (parameters), the
# optimizers’ state_dicts, the loss, the iteration, etc. Saving the model
# in this way will give us the ultimate flexibility with the checkpoint.
# After loading a checkpoint, we will be able to use the model parameters
# to run inference, or we can continue training right where we left off.
#

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename, max_length, device):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length, device)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}'.format(encoder_n_layers, decoder_n_layers))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))