import json
import torch
import numpy as np
import nltk #type: ignore

from nltk.stem.porter import PorterStemmer #type: ignore
from nltk.tokenize import word_tokenize #type: ignore
from model import NeuralNet

device = torch.device('cuda')

nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    return word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype = np.float32)

    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

with open('C:/Users/Akshat/Desktop/Over Here/NLP/Space Chatbot/data/intents.json') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore = ['?', '.', '!']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore]))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train) #type: ignore
y_train = np.array(y_train) #type: ignore

model = NeuralNet(len(all_words), 8, len(tags)).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(1000):
    inputs = torch.from_numpy(X_train).to(device)
    labels = torch.from_numpy(y_train).long().to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save({
    "model_state": model.state_dict(),
    "input_size" : len(all_words),
    "hidden_size": 8,
    "output_size": len(tags),
    "words":all_words,
    "tags": tags
}, "C:/Users/Akshat/Desktop/Over Here/NLP/Space Chatbot/data/space_model.pth")

print("Model trained and saved.")