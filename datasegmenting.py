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

class CreateDataset(Dataset):
    def __init__(self, training_file):
        lol = ' '
        x = []
        y = []
        myfile = open('sometext.txt', "r")
        everything = myfile.read()
        sentences = everything.split("\n\n")
        print(sentences)
        sentences = sentences[:-1]  # remove last line with nothing in it
        #a, b = sentences[0].split('. ', 1)
        #print(b)
        for i in sentences:
            i = i + lol
            a, b = i.split('. ', 1)
            x.append(a)
            y.append(b)

        dictx = {'<PAD>': 0}
        dicty = {'<PAD>': 0}
        counterx = 1
        countery = 1

        for i in range(len(x)):
            for j in range(len(x[i])):
                dictx[x[i][j]] = counterx
                counterx += 1

        self.x = [torch.tensor([dictx[word] for word in sentence]) for sentence in x]

        for i in range(len(y)):
            for j in range(len(y[i])):
                dicty[y[i][j]] = countery
                countery += 1

        self.y = [torch.tensor([dicty[word] for word in sentence]) for sentence in y]


        #    for i in range(len(x)):
        #        for j in range(len(x[i])):
        #            if countwords[x[i][j]] < 4:
        #                x[i][j] = 'UNKA'

        self.dictx = dictx
        self.dicty = dicty
        self.totalabstracts = len(dictx.keys())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def main():
    assert os.path.isfile('sometext.txt'), "Training file does not exist"
    print(CreateDataset('sometext.txt'))
    #print(train('/Users/daniel/week09/12/EECS595HW3/wsj1-18.training'))

main()
