"""
utils.py: the general purpose python functions for
    downloading data;
    process;
"""

__author__ = "Mukil Kesavan"

import os
import sys
import pickle
import logging
from urllib.parse import urlparse
import tensorflow as tf
import numpy as np
from common.constants import *
from wordrnn.configs import ModelParams

def maybe_download(local_file_name, remote_url):
    """ Download file if needed """
    logger = logging.getLogger(DEFAULT_LOGGER)
    if not os.path.exists(local_file_name):
        logger.debug("downloading file...")
        parsed_url = urlparse(remote_url)
        if parsed_url.scheme.lower() == "http" or parsed_url.scheme.lower() == "https":
            if sys.platform == "darwin": # OS X
                cmd_str = "curl " + remote_url + " -o" + local_file_name
            else: # Linux or Windows
                cmd_str = "wget -O " + local_file_name + " " + remote_url
            os.system(cmd_str)
            logger.info("File: %s is on disk!", local_file_name)
            return True
        else:
            logger.error("Unsupported download url")
    return False

def save_as_text(new_file_name, list_to_save, sep):
    """ Save a list to as text file """
    logger = logging.getLogger(DEFAULT_LOGGER)
    if os.path.exists(new_file_name):
        logger.info("File %s exists!", new_file_name)
    else:
        with open(new_file_name, 'w+') as fw:
            fw.write(sep.join([str(i) for i in list_to_save]));
        logger.info( "list saved!")

def load_from_text(file_name, sep):
    """ Load a list from text file"""
    logger = logging.getLogger(DEFAULT_LOGGER)
    if os.path.exists(file_name):
        with open(file_name, "r+") as fr:
            return fr.read().split(sep)
    else:
        logger.warn("file %s not found", file_name)
        return list()

def reset_tensorboard_logs():
    """
    Reset tensorboard log directory data (usually before a training run)
    """
    if tf.gfile.Exists(DEFAULT_TENSORBOARD_LOG_DIR):
        tf.gfile.DeleteRecursively(DEFAULT_TENSORBOARD_LOG_DIR)
    tf.gfile.MakeDirs(DEFAULT_TENSORBOARD_LOG_DIR)


def maybe_download_embeddings(embedding_type):
    # Download word embeddings if required
    if embedding_type is not None and embedding_type.lower() == "glove":
        os.system("mkdir -p " + DEFAULT_EMBEDDINGS_DIR)
        maybe_download(DEFAULT_EMBEDDINGS_DIR + "/glove.zip", DEFAULT_GLOVE_EMBEDDINGS_URL)
        if not os.path.exists(DEFAULT_GLOVE_EMBEDDINGS_FILE):
            os.system("cd " + DEFAULT_EMBEDDINGS_DIR + " && unzip glove.zip")


def load_glove_embeddings():
    """
    Loads the glove word embeddings vectors
    in a python dictionary

    returns:
        a dictionary consisting of word to embed vector mappings
    """
    embeddings = dict()
    with open(DEFAULT_GLOVE_EMBEDDINGS_FILE, "r", encoding='utf-8') as f:
        for line in f:
            ld = line.strip().split()
            word = ld[0].strip()
            embedding = list(map(lambda x: float(x), ld[1:]))
            embeddings[word] = embedding
    return embeddings


def create_embeddings_matrix(vocab_to_idx, load_embeddings_func=load_glove_embeddings):
    """
    Creates an N x d embedding matrix where N is the
    number of unique words in our vocabulary
    and d is the embedding dimension. Words that
    don't have a pre-trained embedding in our data source
    will have a uniform random embedding.

    params:
        vocab_to_idx: dictionary mapping word to integer id
        load_embeddings_func: function to use to load original word embeddings

    returns:
        N x d embedding matrix for the words in vocabulary
    """
    logger = logging.getLogger(DEFAULT_LOGGER)
    logger.debug("loading embeddings from data source")
    embeddings = load_embeddings_func()
    logger.debug("creating embedding matrix")
    embedding_dim = len(next(iter(embeddings.values())))
    embedding_matrix = np.zeros((len(vocab_to_idx), embedding_dim))
    for word, idx in vocab_to_idx.items():
        embedding = embeddings.get(word)
        if embedding is not None:
            embedding_matrix[idx] = embedding
        else:
            embedding_matrix[idx] = np.random.uniform(-0.2, 0.2, embedding_dim)
    logger.debug("finished loading embeddings")
    return embedding_matrix

def load_saved_model_params(param_file=TRAINED_MODEL_CONFIGS):
    saved_params = None
    vocab_to_idx = None
    idx_to_vocab = None
    num_classes = None
    if os.path.exists(param_file):
        with open(param_file, "rb") as f:
            saved_params_dict = pickle.load(f)
            vocab_to_idx = pickle.load(f)
            idx_to_vocab = pickle.load(f)
            num_classes = pickle.load(f)
        saved_params = ModelParams()
        saved_params.set_params_from_dict(saved_params_dict)
    return saved_params, vocab_to_idx, idx_to_vocab, num_classes
