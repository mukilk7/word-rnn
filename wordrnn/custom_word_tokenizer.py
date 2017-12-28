"""
    This file contains the definition of a custom
    word tokenizer class that can be tailored for
    processing specific types of input.
"""

__author__ = "Mukil Kesavan"

from nltk.tokenize import sent_tokenize, word_tokenize 

class CustomWordTokenizer(object):
    """
    A custom word tokenizer based on nltk's
    word_tokenize functionality that also
    treats newline character as a word.
    """
    def __init__(self):
        pass
    
    def tokenize(self, text):
        """
        Takes in a block of text and returns
        a list of word tokens.
        
        params:
            text: English language text
        
        returns:
            list of word tokens
        """
        if len(text.strip()) <= 0:
            return []
        tokens = []
        for sentence in sent_tokenize(text):
            tokens.extend(word_tokenize(sentence))
            tokens.append("\n")
        return tokens