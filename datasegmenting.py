import argparse
import os
import random
import json
import torch
import torch.nn as nn
from numpy import argmax
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk import sent_tokenize

class CreateDataset(Dataset):

    def __init__(self, training_file, init_sent):
        myfile = open('sometext.txt', "r")
        everything = myfile.read()
        abstracts = everything.split("\n\n")

        abstracts = abstracts[:-1]  # remove last line with nothing in it

        self.data = []
        for abstract in abstracts:
            sentences = sent_tokenize(abstract)
            if len(sentences) >= 3:
                prompt = init_sent + ' ' + sentences[0]
                rest = ' '.join(sentences[1:])
                datapoint = {
                    'prompt' : prompt,
                    'rest' : rest
                }
                self.data.append(datapoint)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    assert os.path.isfile('sometext.txt'), "Training file does not exist"
    init_sent = 'The following sentences are taken from the abstract of a scientific paper.'
    dataset = CreateDataset('sometext.txt', init_sent)

    # Print the first three datapoints for testing purposes
    for i, dp in enumerate(dataset):
        print(dp)
        if i > 3:
            raise Exception
