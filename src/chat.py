import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import torch
import numpy as np
import nltk #type: ignore
import random

from nltk.stem.porter import PorterStemmer #type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from model import NeuralNet

stemmer = PorterStemmer()
nltk.download('punkt')
nltk.download('punkt_tab')
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype = np.float32)

    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

device = torch.device('cpu')
data = torch.load("C:/Users/Akshat/Desktop/Over Here/NLP/Space Chatbot/data/space_model.pth", map_location = device)

model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

with open('C:/Users/Akshat/Desktop/Over Here/NLP/Space Chatbot/data/intents.json') as f:
    intents = json.load(f)

words = data['words']
tags = data['tags']

print("Chatbot active!! (type quit to exit) \n")

while True:
    sentence = input("You:  ")

    if sentence.lower() == 'quit':
        print("Bot: GoodBye!")
        break

    sentence = word_tokenize(sentence)
    X = bag_of_words(sentence, words)
    X = torch.from_numpy(X).to(device)

    output = model(X)
    probabilities = torch.softmax(output, dim = 0)
    confidence, predicted = torch.max(probabilities, dim = 0)

    tag = tags[predicted.item()]

    if confidence.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print("Bot: ", random.choice(intent["responses"]))
                found = True
                break
        if not found:
            print("I dont understand that yet.")
    else:
        print("Bot: I'm not sure about that. Try asking something about space")