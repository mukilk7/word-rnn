#!/usr/bin/env python

"""
This should be the main and only entry point
to running all functionality implemented in
this project.
"""

__author__ = "Mukil Kesavan"

import os
import sys

#Setup project-wide logging
#NOTE: This should *ONLY* be done here!
from customlogging import log
logger = log.setup_logging()

from optparse import OptionParser
from wordrnn import master
from wordrnn.data_processor import LocalDataProcessor
from wordrnn.configs import ModelParams
from wordrnn.model import WordRNN
import common.utils as utils
from common.constants import *

_CMD_CHOICES = [
    "train",
    "generate",
    "anomaly-detect",
]

_EMBEDDING_CHOICES = [
    "glove"
]


def init_cmdline_args():
    """
    Initialize command line arguments.
    """
    parser = OptionParser(usage="Usage %prog [options]")

    parser.add_option("--command", "-c", help="command to run %s" % _CMD_CHOICES,
                      type="choice", choices=_CMD_CHOICES, action="store", dest="cmd")
    parser.add_option("--embedding", "-e", help="pre-trained embedding to use %s" % _EMBEDDING_CHOICES,
                      type="choice", choices=_EMBEDDING_CHOICES, action="store", dest="embedding")
    parser.add_option("--num-epochs", help="train model for specified epochs",
                      action="store", dest="num_epochs")
    parser.add_option("--num-words",
                      help="generate specified num of words using trained model",
                      action="store", dest="num_words_to_gen")
    parser.add_option("--input-file",
                      help="input file for train or test",
                      action="store", dest="input_file")
    parser.add_option("--input-url",
                      help="input data url for train or test (can be http or gs)",
                      action="store", dest="input_url")
    parser.add_option("--anomaly-threshold",
                      help="the percentile threshold for anomaly detection (0-100)",
                      action="store", dest="anomaly_threshold")
    return parser


def cmd_router(cmd, opts):
    """
    Call appropriate module/function based on
    command requested.
    """
    utils.reset_tensorboard_logs()
    utils.maybe_download_embeddings(opts.embedding)
    if cmd == "train":
        #look for saved params first to see if we're resuming training
        default_config, _, _, _ = utils.load_saved_model_params()
        if default_config is None:
            if opts.embedding is not None and opts.embedding.lower() == "glove":
                default_config = ModelParams(embed_sz=DEFAULT_GLOVE_EMBEDDINGS_DIM,
                                             embedding=opts.embedding.lower())
            else:
                default_config = ModelParams()
        data_processor = LocalDataProcessor(default_config,
                                            input_url=opts.input_url,
                                            local_filename=opts.input_file)
        num_epochs = int(opts.num_epochs)
        logger.info("Training epochs set to %s", num_epochs)
        model = WordRNN(data_processor.num_classes, default_config).build_computation_graph()
        master.train(data_processor, model, num_epochs)
        logger.info("Training complete. Model saved!")
    elif cmd == "generate":
        num_words = int(opts.num_words_to_gen)
        output = master.generate_text_from_model(num_words)
        logger.info("Generated Text\n---\n%s\n---\n", output)
    elif cmd == "anomaly-detect":
        logger.info("Average Line Loss is an Experimental Feature")
        anomaly_threshold = 95
        if opts.anomaly_threshold is not None:
            anomaly_threshold = int(opts.anomaly_threshold)
        anomaly_lines = master.compute_average_line_loss(local_filename=opts.input_file,
                                                         anomaly_percentile=anomaly_threshold,
                                                         input_url=opts.input_url)
        logger.info("*** Anomaly Lines***\n----\n%s----", "".join(anomaly_lines))


def main():
    parser = init_cmdline_args()
    (opts, args) = parser.parse_args()
    if len(sys.argv[1:]) <= 0:
        parser.print_help()
        sys.exit(0)
    cmd_router(opts.cmd.lower(), opts)

if __name__ == '__main__':
    main()
