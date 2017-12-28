import tensorflow as tf

__author__ = "Mukil Kesavan"

def cross_entropy_loss(logits, y):
    """
    Computes the cross-entropy loss between predicted logits
    (after applying softmax) and actual output.

    params:
        logits: the predicted probability vector for each output class
        y: the actual output (ground truth - one-hot)

    returns:
        the cross-entropy loss vector
    """
    y_reshaped = tf.reshape(y, [-1])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, \
                                                            labels=y_reshaped)
    return losses

def sequence_loss(logits, y, configs):
    """
    Computes the weighted cross entropy loss or sequence loss between
    predicted logits and actual output.
    
    params:
    logits: the predicted probability vector for each output class
    y: the actual output (ground truth - one-hot)
    configs: the configuration parameters for the job
    
    returns:
        the sequence loss vector
    """
    logits_reshaped = tf.reshape(logits, [configs.batch_size, configs.num_steps, -1])
    #TODO: Allow loss weights to be passed in as argument to the functino
    losswts = tf.ones([configs.batch_size, configs.num_steps], dtype=tf.float32)
    losses = tf.contrib.seq2seq.sequence_loss(logits=logits_reshaped, \
                                               targets=y, \
                                               weights=losswts)
    return losses

