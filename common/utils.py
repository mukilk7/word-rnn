"""
utils.py: the general purpose python functions for
    downloading data;
    process;
"""

__author__ = "Mukil Kesavan"

import os
import sys
import logging
from urllib.parse import urlparse
import tensorflow as tf
from common.constants import *

def maybe_download(local_file_name, remote_url):
    """ Download text corpus if needed """
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
        else:
            logger.error("Unsupported download url")
    return

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
