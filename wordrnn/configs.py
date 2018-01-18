"""
This file includes a class that encapsulates all the tunables.
"""

__author__ = "Mukil Kesavan"

import logging
from common.constants import *

class ModelParams(object):
    """
        The class encapsulation of all universal tunable hyper-parameters.
    """
    def __init__(self, batch_size=64, num_steps=24, data_steps_ahead=1, model_state_size=256,
                 model_learning_rate=1e-3, model_drop_out_rate=0.5, model_type="lstm",
                 model_num_layers=2, embed_sz=10, embedding=None, trained_epochs=0):
        """
        params:
            batch_size: the size of a batch (sequence length)
            num_steps: the number of back propagation steps
            data_steps_ahead: steps of predicted data ahead
            model_state_size: the number of hidden layer neurons
            model_learning_rate: use low values to prevent hitting local minima
            model_drop_out_rate: rate of updating only some neurons to prevent overfitting
            model_type: the rnn model architecture (basic, lstm or gru)
            model_num_layers: number of rnn layers
            embed_sz: character input embedding size
            embedding: type of pre-trained word embeddings to use
            trained_epochs: the number of epochs for which model has been trained (initially 0)
        """
        # settings
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.data_steps_ahead = data_steps_ahead
        self.model_state_size = model_state_size
        self.model_learning_rate = model_learning_rate
        self.model_drop_out_rate = model_drop_out_rate # 0.0 - 1.0
        self.model_type = model_type # options: "basic", "lstm", "gru" <- simplified version of "lstm"
        self.model_num_layers = model_num_layers
        self.embed_sz = embed_sz
        self.embedding = embedding
        self.trained_epochs = trained_epochs

        #logger
        self.logger = logging.getLogger(DEFAULT_LOGGER)

        # name associations
        self.params =\
            {"batch_size": self.batch_size,
             "num_steps": self.num_steps,
             "data_steps_ahead": self.data_steps_ahead,
             "model_state_size": self.model_state_size,
             "model_learning_rate": self.model_learning_rate,
             "model_drop_out_rate": self.model_drop_out_rate,
             "model_type": self.model_type,
             "model_num_layers": self.model_num_layers,
             "embed_sz": self.embed_sz,
             "embedding": self.embedding,
             "trained_epochs": self.trained_epochs,
        }

        #Dump parameters currently being used
        self.logger.debug(self)
        

    def __repr__(self):
        return "Model_config()"

    def __str__(self):
        return "\n".join("{0:<20}{1}".format(k,v) for k,v in self.params.items())

    def get_params(self):
        return self.params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            self.params[parameter] = value

    def set_params_from_dict(self, paramdict):
        for k, v in paramdict.items():
            setattr(self, k, v)
            self.params[k] = v

    def report_change(self, config_tag):
        try:
            return "{0:<20}{1}".format(
                config_tag, self.params[config_tag])
        except IndexError:
            return "Exception: config_tag not valid"
