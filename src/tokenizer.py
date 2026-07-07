import re


class SimpleTokenizerV1:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v:k for k,v in vocab.items()}

    def encode(self, text):
        """ Turn text into a list of id's"""
        split   = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        split   = [x for x in split if x.strip()]
        split   = [x if x in self.str_to_int else "<|unk|>" for x in split]
        ids     = [self.str_to_int[x] for x in split]
        return ids

    def decode(self, ids):
        """ Turn list of id's into text """
        text =  " ".join([self.int_to_str[x] for x in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        text = re.sub(r'(["(\'])\s+', r'\1', text)
        return text            # no space AFTER opening punctuation        return text



