# Eusebio, the chatbot
This chatbot is built to score primary school essays. 

It is a pipeline system that:
  - Applies Natutal Language Processes (NLP) to clean and understand the data and connections between words and grades;
  - Applies Machine Learning models, specifically Reegression Algorithms, to predict the grade.
  
  
Eusebio is a chatbot that you should run on your own terminal. It is designed to be used as a personal tool while users write their own essays.

This repository has several files that the chatbot relies on to function:

  - test_nlp.csv -> csv file used to paste the user's essay
  - Chatbot.py -> Python programme with the main functions of the chatbot
  - random_forest_model.pk -> Regression Algorithm designed to predict the score of the essay
  - vectorizer.pk -> Tfidf Vectorizer, containing a pre-process of data structuring and cleaning
  

# How to install and run Eusebio

## Installation

There are important installation steps.

Python 3.6 is the recommended version.

Installation of the Chatterbot Library - version 0.7.4 recommended

!pip install chatterbot
!pip install chatterbot-corpus
!pip install --upgrade chatterbot==0.7.4 . 

Libraries used:

  - from chatterbot import ChatBot
  - from chatterbot.trainers import ListTrainer
  - from chatterbot.trainers import ChatterBotCorpusTrainer
  - import sys 
  - import pandas as pd
  - from sklearn.feature_extraction.text import TfidfVectorizer
  - import pickle
  - import spacy
  - import string
  - import en_core_web_sm
  - from spacy.lang.en import English
  - from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
  - from autocorrect import spell
  - from spellchecker import SpellChecker

Make sure that you have this libraries installed. To do so, you can use the command pip3 install <library_name>


## How to run Eusebio

To run Eusebio you have to:

  1. Clone the github repository into your local file
  2. Add your essay to the file 'test_nlp.csv' -> 'essay' columns -> save file
  3. Open Applications -> terminal
  4. Change the current directory to the folder cloned from github (use cd <name_directory> , e.g. cd Desktop)
  5. Run the chatbot with the command python3 Chatbot.py
  6. Enjoy the result!


