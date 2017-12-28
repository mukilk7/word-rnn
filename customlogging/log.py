"""
Sets up a custom logger for the project.
"""

__author__ = "Mukil Kesavan"

import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from common.constants import *

def setup_logging(logger_name=DEFAULT_LOGGER, logfile=DEFAULT_LOG_FILE,
                  max_log_file_sz=DEFAULT_MAX_LOG_FILE_SIZE,
                  loglevel=logging.DEBUG):
    logfmt = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt=logfmt)
    logger = logging.getLogger(logger_name)
    logger.setLevel(loglevel)

    # console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file logging
    logdir, _ = os.path.split(logfile)
    if not os.path.exists(logdir):
        os.system('mkdir -p ' + logdir)
    file_handler = RotatingFileHandler(logfile, maxBytes=(max_log_file_sz), backupCount=10)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
                                       
    return logger
