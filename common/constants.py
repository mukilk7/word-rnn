"""
Define constants here.
"""

__author__ = "Mukil Kesavan"

DEFAULT_LOGGER = ''
DEFAULT_INPUT_URL = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
DEFAULT_DATA_DIR = "./data/"
DEFAULT_LOG_DIR = "./logs/"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR + "run.log"
DEFAULT_TENSORBOARD_LOG_DIR = DEFAULT_LOG_DIR + "/tensorboard/"
DEFAULT_MAX_LOG_FILE_SIZE = 1e7 #10 MB in bytes
DEFAULT_MODEL_DIR = "./model/"
TRAINED_MODEL_NAME = DEFAULT_MODEL_DIR + "word-rnn"
TRAINED_MODEL_CONFIGS = DEFAULT_MODEL_DIR + "/configs.p"
DEFAULT_EMBEDDINGS_DIR = "./data/embeddings"
DEFAULT_GLOVE_EMBEDDINGS_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
DEFAULT_GLOVE_EMBEDDINGS_FILE = DEFAULT_EMBEDDINGS_DIR + "/glove.6B.50d.txt"
DEFAULT_GLOVE_EMBEDDINGS_DIM = 50
