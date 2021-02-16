# If we simply convert our English sentences to tensors by converting
# words to their indexes(\ ``indexesFromSentence``) and zero-pad, our
# tensor would have shape *(batch_size, max_length)* and indexing the
# first dimension would return a full sequence across all time-steps.
# However, we need to be able to index our batch along time, and across
# all sequences in the batch. Therefore, we transpose our input batch
# shape to *(max_length, batch_size)*, so that indexing across the first
# dimension returns a time step across all sentences in the batch. We
# handle this transpose implicitly in the ``zeroPadding`` function.

import itertools

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m