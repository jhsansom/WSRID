import argparse
import os
import random
import json

import torch
import torch.nn as nn
from numpy import argmax
from sklearn.metrics import accuracy_score
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import numpy as np

class CreateDataset(Dataset):
    def __init__(self, training_file):
        # Your code starts here
        x = []
        y = []
        myfile = open(training_file, "r")
        everything = myfile.read()
        sentences = everything.split("\n")
        sentences = sentences[:-1] #remove last line with nothing in it
        for i in sentences:
            words = i.split()
            tempx = []
            tempy = []
            for j in range(0, len(words), 2):
                tempx.append(words[j])
                tempy.append(words[j + 1])
            x.append(tempx)
            y.append(tempy)




        countwords = {}
        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] not in countwords:
                    countwords[x[i][j]] = 1
                else:
                    countwords[x[i][j]] += 1

        for i in range(len(x)):
            for j in range(len(x[i])):
                if countwords[x[i][j]] < 4:
                    x[i][j] = 'UNKA'


        dictx = {'<PAD>': 0}
        dicty = {'<PAD>': 0}
        counterx = 1
        countery = 1

        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] not in dictx:
                    dictx[x[i][j]] = counterx
                    counterx += 1

        self.x = [torch.tensor([dictx[word] for word in sentence]) for sentence in x]

        for i in range(len(y)):
            for j in range(len(y[i])):
                if y[i][j] not in dicty:
                    dicty[y[i][j]] = countery
                    countery += 1

        self.y = [torch.tensor([dicty[word] for word in sentence]) for sentence in y]

        self.dictx = dictx
        self.dicty = dicty
        self.totalwords = len(dictx.keys())
        self.tagsetsize = len(dicty.keys())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class RNNTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, totalwords, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.


        self.embedding = nn.Embedding(totalwords, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence, senlist):
        embed_out = self.embedding(sentence)
        b = torch.nn.utils.rnn.pack_padded_sequence(embed_out, senlist, batch_first=True, enforce_sorted=False)
        lstm_out , _ = self.lstm(b)
        c , _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, padding_value=0)
        tag_space = self.hidden2tag(c)
        #tag_scores = self.softmax(tag_space)
        return tag_space


def collate(stuff):
    xstuff = []
    ystuff = []
    for item in stuff:
        xstuff.append(item[0])
        ystuff.append(item[1])
    return xstuff, ystuff


def train(training_file):
    assert os.path.isfile(training_file), "Training file does not exist"

    dataset = CreateDataset(training_file)
    train_dataloader = DataLoader(dataset, collate_fn = collate, batch_size=64, shuffle=True)

    model = RNNTagger(8, 16, dataset.totalwords, dataset.tagsetsize)

    loss = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #test stuff
    for e in range(0, 1):
        for x,y in train_dataloader:
            optimizer.zero_grad()
            senlist = [len(sentence) for sentence in x]
            x = pad_sequence(x, True, 0)
            modeloutput = model.forward(x, senlist)
            modeloutput = torch.swapaxes(modeloutput, 1, 2)
            y = pad_sequence(y, True, 0)
            result = loss(modeloutput, y)
            result.backward()
            optimizer.step()
        print(result)


    with open('data.json', 'w') as fp:
        json.dump({0:dataset.dictx, 1:dataset.dicty, 2:dataset.totalwords, 3:dataset.tagsetsize}, fp, indent=6)



    # Your code ends here
    return model



def test(model_file, data_file, label_file):
    assert os.path.isfile(model_file), "Model file does not exist"
    assert os.path.isfile(data_file), "Data file does not exist"
    assert os.path.isfile(label_file), "Label file does not exist"

    # Your code starts here


    with open("data.json", "r") as fp:
        dict = json.load(fp)

    dictx = dict['0']

    dicty = dict['1']

    totalwords = dict['2']

    totaltags = dict['3']

    y1 = []
    myfile1 = open(label_file, "r")
    everything = myfile1.read()
    sentences = everything.split("\n")
    sentences = sentences[:-1]  # remove last line with nothing in it
    for i in sentences:
        words = i.split()
        tempy = []
        for j in range(0, len(words), 2):
            tempy.append(words[j + 1])
        y1.append(tempy)



    y1 = [torch.tensor([dicty[word] for word in sentence]) for sentence in y1]
    ypadded = pad_sequence(y1, True, 0)


    model = RNNTagger(8, 16, totalwords, totaltags)
    model.load_state_dict(torch.load(model_file))

    x = []
    myfile2 = open(data_file, "r")
    everything = myfile2.read()
    sentences = everything.split("\n")
    sentences = sentences[:-1] #remove empty sentence at end
    for i in sentences:
        words = i.split()
        tempx = []
        for j in range(0, len(words)):
            tempx.append(words[j])
        x.append(tempx)


    x = [torch.tensor([dictx[word] for word in sentence]) for sentence in x]

    xpadded = pad_sequence(x, True, 0)

    senlist = [len(sentence) for sentence in x]

    prediction = model(
        xpadded, senlist
    )  # replace with inference from the loaded model
    prediction = torch.argmax(prediction, -1).cpu()


    ground_truth = [
        ypadded
    ]  # replace with actual labels from the data files


    b = torch.nn.utils.rnn.pack_padded_sequence(prediction, senlist, batch_first=True, enforce_sorted=False)
    c, _ = torch.nn.utils.rnn.pad_packed_sequence(b, batch_first=True, padding_value=0)

    cstuff = c.numpy().flatten()
    tstuff = ypadded.numpy().flatten()

    clist = cstuff.tolist()
    trustlist = tstuff.tolist()

    totallen = len(clist)
    accuracy = 0.0
    for i in range(len(clist)):
        if ((clist[i] == 0) & (trustlist[i] == 0)):
            totallen -= 1
        elif(clist[i] == trustlist[i]):
            accuracy += 1
        else:
            accuracy = accuracy
    score = accuracy/totallen

    #score = accuracy_score(c.numpy().flatten(), ypadded.numpy().flatten())
    print(f"The accuracy of the model is {100*score:6.2f}%")


def main(params):
    #print(train('/Users/daniel/week09/12/EECS595HW3/wsj1-18.training'))
    if params.train:
        print(params.training_file)
        model = train(params.training_file)
        torch.save(model.state_dict(), params.model_file)
    else:
        test(params.model_file, params.data_file, params.label_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action="store_const", const=True, default=False)
    parser.add_argument("--model_file", type=str, default="file=model.torch")
    parser.add_argument("--training_file", type=str, default="wsj1-18.training")
    parser.add_argument("--data_file", type=str, default="wsj19-21.testing")
    parser.add_argument("--label_file", type=str, default="wsj19-21.truth")
    main(parser.parse_args())
