# Space Chatbot using NLP

This project implements a simple space-themed chatbot using Natural Language Processing (NLP). The chatbot is trained on a custom intents dataset and can respond to user queries related to space, astronomy, and general conversation.

---

## Features

* Intent-based chatbot using NLP
* Custom dataset (`intents.json`)
* Pretrained model included for instant use
* Modular code structure (training, model, and chat separated)
* Easy to extend with new intents

---

## Project Structure

```
.
├── data/
│   ├── intents.json        # Training dataset
│   └── space_model.pth     # Trained model
├── src/
│   ├── chat.py             # Chat interface
│   ├── model.py            # Model architecture
│   └── train.py            # Training script
├── requirements.txt
└── .gitignore
```

---

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/Space_Chatbot.git
cd Space_Chatbot
```

2. Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

### Run the chatbot:

```
python src/chat.py
```

### Retrain the model (optional):

```
python src/train.py
```

---

## Model Details

* Framework: PyTorch
* Approach: Intent classification
* Input: User text
* Output: Predicted intent and response

---

## Dataset

* Custom intents dataset stored in `data/intents.json`
* Easily extendable by adding new intents and responses

---

## Notes

* Pretrained model is included for immediate use
* No external dataset download required
* Designed for learning and experimentation

---

## Future Improvements

* Add GUI (Tkinter / Web interface)
* Improve NLP using transformers
* Add context-aware responses
* Deploy as a web service

---

## Author

Akshat Sohni

---

## Acknowledgements

* PyTorch community
* NLP learning resources
