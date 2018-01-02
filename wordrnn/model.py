"""
   Word RNN Model: build graph

    Relevant hyperparameters:
        batch_size
        num_steps (optional)
        state_size 
        num_layers
        learning_rate
        dropout_rate
"""

__author__ = "Mukil Kesavan"

import tensorflow as tf
import logging
from common.constants import *
from wordrnn.loss_functions import cross_entropy_loss

class WordRNN(object):
    """
    A multi-layer recurrent neural network (RNN) that can
    work with word input for training and evaluation.
    """    
    def __init__(self, num_classes, configs):
        """
        Constructs a Word RNN model.

        params:
            num_classes: prediction class (max num of words in wordrnn)
            configs: all tunable hyperparameters
        """
        self.num_classes = num_classes  # data-dependent
        self.configs = configs
        self.logger = logging.getLogger(DEFAULT_LOGGER)

    def reset_graph(self):
        """
        Closes any open tensorflow sessions and safely resets
        the current computation graph.
        """
        self.logger.debug("resetting computation graph")
        sess = tf.get_default_session()
        if sess is not None:
            sess.close()
        tf.reset_default_graph()

    def build_computation_graph(self):
        """
        Builds the RNN model as a tensorflow computation graph.

        Returns:
            A dictionary of computation graph tensors that can
            be evaluated or run in the context of a tensorflow
            session.
        """
        self.logger.debug("building computation graph")
        self.reset_graph()

        # place holders -- starting nodes
        x = tf.placeholder(tf.int32, [self.configs.batch_size,self.configs.num_steps], name='inputs_placeholder')
        y = tf.placeholder(tf.int32, [self.configs.batch_size, self.configs.num_steps], name='labels_placeholder')

        embedding_init_op = None
        embed_placeholder = None
        # inputs -- embedding lets us map each input word to a higher dimensional vector.
        # We initialize each embedding using random real numbers via the default random
        # variable initializer (glorot) in tensorflow. For example, if we had 64 unique
        # word classes, one-hot input encoding for each word would require a
        # vector of size 64 whereas our embedding can project this down to a smaller one.
        # One intuitive way of thinking about it is if you want to encode 1024 in binary
        # you would only need 10 bits vs a 1024 bit one-hot encoded vector.
        if not self.configs.embedding:
            # use random embedding that is also learned during training
            embeddings = tf.get_variable('embeddings', [self.num_classes, self.configs.embed_sz])
        else:
            # use pre-trained embedding from external source
            embeddings = tf.get_variable('embeddings', trainable=False,
                                         initializer=tf.constant(0.0, shape=[self.num_classes, self.configs.embed_sz]))
            embed_placeholder = tf.placeholder(tf.float32, [self.num_classes, self.configs.embed_sz],
                                               name='embed_placeholder')
            embedding_init_op = embeddings.assign(embed_placeholder)

        # results a [self.configs.batch_size, self.configs.num_steps, self.configs.embed_sz tensor]
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        def build_cell():
            # cells -- has a state and performs some operation that takes a matrix of inputs
            cell = None
            if self.configs.model_type.lower() == "lstm":
                cell = tf.contrib.rnn.LSTMCell(self.configs.model_state_size)
            elif self.configs.model_type.lower() == "gru":
                cell = tf.contrib.rnn.GRUCell(self.configs.model_state_size)
            elif self.configs.model_type.lower() == "basic":
                cell = tf.contrib.rnn.BasicRNNCell(self.configs.model_state_size)
            else:
                raise ValueError("Unknown RNN Model Type - %s - Specified" % (self.configs.model_type.lower()))
            # dropout
            if self.configs.model_drop_out_rate > 0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=(1 - self.configs.model_drop_out_rate))
            return cell

        cell = [build_cell() for _ in range(self.configs.model_num_layers)]
        # if more than 1 layer, wrap them up in a single composite cell
        if self.configs.model_num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell(cell)
        else:
            cell = cell[0]

        # initial state
        init_state = cell.zero_state(self.configs.batch_size, tf.float32)
        # construct unrolled RNN graph on-demand
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

        # share softmax layer weights and biases if this function is called multiple times
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.configs.model_state_size, self.num_classes])
            b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.0))

        # compute loss
        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.configs.model_state_size])
        # note: bias should get broadcast added
        logits = tf.matmul(rnn_outputs, W) + b
        losses = cross_entropy_loss(logits, y)
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdamOptimizer(self.configs.model_learning_rate).minimize(total_loss)

        #self.logger.debug("Adding variable summary for tensorboard")
        # for v in tf.all_variables():
        #         print (v.name)
        #         tf.summary.histogram('%s' % v.name, v)
        for v in cell.variables:
            self.logger.debug(v.name)
            tf.summary.histogram('%s' % v.name, v)
        tf.summary.scalar('total_loss', total_loss)
        
        self.logger.debug("finished building computation graph")
        
        # return graph
        return dict(
            x=x,
            y=y,
            cell=cell,
            init_state=init_state,
            logits=logits,
            final_state=final_state,
            total_loss=total_loss,
            train_step=train_step,
            embed_placeholder=embed_placeholder,
            embedding_init_op=embedding_init_op,
            saver=tf.train.Saver(),
            params=self.configs.get_params(),
        )
