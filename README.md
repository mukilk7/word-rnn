=== INTRODUCTION ===

This project builds a Word-level Recurrent Neural Network in Python3 using TensorFlow that can be trained to:
(i) generate text similar to the training corpus and (ii) find lines in a test file that are of not the same
"style" as the lines found in the training corpus. It supports optional pre-trained word embeddings from the Stanford Glove project.

https://nlp.stanford.edu/projects/glove/

If pre-trained word embeddings are not used, the embeddings will be learned as part of training.

Applications:

Case 1: Can be used to generate new poems, essays, source code etc. depending on the training set  
Case 2: Can be used to detect "fakes" that are similar in style to the training set on cursory glance

To run the project, execute main.py from a unix-style command line shell and follow the help section.

=== Basic Command Line Usage Examples ===

* Print help:

./main.py -h

* Train the Word RNN model (for 10 epochs):

./main.py -c train --num-epochs=10

* Generate text using a trained Word RNN model:

./main.py -c generate --num-words=100

* Compute anomaly lines for an input test file using trained model:

./main.py -c anomaly-detect --test-input-file="./myfile.txt" --anomaly-threshold=95

* Use pre-trained word embeddings with any of the above commands:

./main.py -c train -e glove --num-epochs=10  
./main.py -c generate -e glove --num-words=100  
./main.py -c anomaly-detect -e glove --test-input-file="./myfile.txt" --anomaly-threshold=95  
