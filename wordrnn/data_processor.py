"""
   Data-relevant handler: 
        num_classes: predictor output classes (words for wordrnn)
        idx_to_vocab: idx to vocab mapping
        vocab_to_idx: {vocab : idx}
   
   Relevant hyperparameters: 
        batch_size 
        num_steps (optional)
        steps_ahead (optional)
"""

__author__ = "Mukil Kesavan"

import os
import numpy as np
import logging
import pickle
#from nltk.tokenize import word_tokenize
from wordrnn.custom_word_tokenizer import CustomWordTokenizer
import common.utils as utils
from common.constants import *


class LocalDataProcessor(object):
    """
    Handles textual data loading, encoding into numeric vectors, decoding back to
    words and generating batches of data.
    """
    def __init__(self, configs, input_url=None, data_dir=None,
                 local_filename=None, num_classes=None):
        """
        Constructs a data processor instance for training on local machine.
        
        params:
            configs: tunable hyperparameters and other configurations
            input_url: url of text on which to train
            data_dir: local directory to store raw/processed  data
            local_filename: local filename of input data file
            num_classes: the maximum number of word classes
        """
        # logging
        self.logger = logging.getLogger(DEFAULT_LOGGER)

        # hardcode path
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.input_url = input_url or DEFAULT_INPUT_URL
        self.input_filename_local = local_filename or self.data_dir + 'traincorpus.txt'
        self.encoded_filename = self.input_filename_local + '-input_num_encoded.p'
        self.vocab_filename = self.input_filename_local + '-vocab.p'
        self.num_classes = num_classes

        # configurations
        self.configs = configs

        # data loader
        self.prepare_corpus()


    def prepare_corpus(self):
        """
        Download and encode the corpus if not already on disk.

        return:
            corpus: words in numeric encoding
            idx_to_vocab
            vocab_to_idx
        """
        # get vocab dictionaries
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.logger.debug("Reading corpus...")
        utils.maybe_download(self.input_filename_local, self.input_url)
        with open(self.input_filename_local, 'r', encoding='utf-8') as ff:
            raw_text = ff.read()
        tokenized_raw_text = CustomWordTokenizer().tokenize(raw_text)
        del raw_text

        if not os.path.exists(self.vocab_filename):
            # construct vocab look-up table
            self.logger.debug("constructing vocab lookup table")
            vocab = set(tokenized_raw_text)
            with open(self.vocab_filename, "wb") as f:
                pickle.dump(vocab, f)
        else:
            self.logger.debug("loading vocab lookup table")
            with open(self.vocab_filename, "rb") as f:
                vocab = pickle.load(f)
        self.num_classes = self.num_classes or len(vocab)
        self.idx_to_vocab = dict(enumerate(vocab))
        self.vocab_to_idx = {v: k for k, v in self.idx_to_vocab.items()}

        # corpus-dependent information
        self.num_batches = len(tokenized_raw_text) // self.configs.batch_size

        # get encoded corpus
        if not os.path.exists(self.encoded_filename):
            self.logger.debug("constructing numerically encoded corpus")
            self.corpus = np.empty(self.num_batches * self.configs.batch_size)
            for i, w in enumerate(tokenized_raw_text):
                if i >= self.num_batches * self.configs.batch_size:
                    break
                self.corpus[i] = min(self.vocab_to_idx[w], self.num_classes - 1)
            with open(self.encoded_filename, "wb") as f:
                pickle.dump(self.corpus, f)
        else:
            self.logger.debug("loading numerically encoded corpus")
            with open(self.encoded_filename, "rb") as f:
                self.corpus = pickle.load(f)

        del tokenized_raw_text
        return

    def gen_epoch_batch_data(self):
        """ Generates batches of input data for a single epoch.

            returns:
                a tuple containing a batch input and corresponding output
        """
        # TODO: Investigate batch shuffling based on tensorflow's backprop implementation
        # i.e., stateless vs stateful RNN.
        stacked_x = self.corpus.reshape((self.configs.batch_size, -1))
        num_batches_per_epoch = (self.num_batches - 1) // self.configs.num_steps
        for i in range(num_batches_per_epoch):
            batchx = stacked_x[:, i * self.configs.num_steps: (i + 1) * self.configs.num_steps]
            batchy = stacked_x[:, i * self.configs.num_steps + self.configs.data_steps_ahead:
                                        (i + 1) * self.configs.num_steps + self.configs.data_steps_ahead]
            yield (batchx, batchy)

    def gen_epoch_data(self, num_epochs):
        """
        Generates batches of input data for all epochs.
        
        params:
            num_epochs: number of trainins epochs
        """
        for _ in range(num_epochs):
            yield self.gen_epoch_batch_data()
