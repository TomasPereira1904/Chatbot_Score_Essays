
# coding: utf-8

# In[ ]:


#!pip install chatterbot
#!pip install chatterbot-corpus
#!pip install --upgrade chatterbot==0.7.4
#!pip3 install pandas
#!pip3 install spacy
#!python3 -m spacy download en_core_web_sm


# In[27]:


from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import sys 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import spacy
import string
import en_core_web_sm
from spacy.lang.en import English
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from autocorrect import spell
from spellchecker import SpellChecker


# In[29]:


spell = SpellChecker()


# In[30]:


parser = English()

def spacy_tokenizer_spell(sentence):
    tokens = nlp(sentence)
    #tokens = ["propn" if tok.pos_ == 'PROPN' else tok for tok in tokens]
    tokens = [(tok.lemma_.lower().strip(), tok.pos_) for tok in tokens]
    tokens = [tok for tok in tokens if tok[0] not in ["-pron-"] + list(string.ascii_letters)+list(string.digits)]
    tokens = [tok for tok in tokens if tok[1] not in ["SPACE", "PUNCT", "SYM"]]
    #tokens = [tok for tok in tokens if tok[1] not in ["PUNCT"]]
    #tokens = [tok for tok in tokens if tok[1] not in ["SYM"]]
    tokens = [tok for tok in tokens if tok[0] not in stopwords]
    #tokens = [tok for tok in tokens if tok not in list(punctuations)] 
    tokens = [(spell.correction(tok[0]), tok[1]) for tok in tokens]
    tokens = [("PROPN", tok[1]) if tok[1] in ["PROPN"] else tok for tok in tokens]
    return tokens

nlp = spacy.load('en_core_web_sm')


# In[3]:


# Create a new chat bot named Charlie
chatbot = ChatBot('Eusebio')


# In[4]:


chatbot.set_trainer(ListTrainer)


# In[5]:


print("Hi!! \n\n")
print("My name is Eusebio, the bot! Can I help you?\n")


# In[40]:


chatbot.train(["I want to grade an essay",'Sweet! I can help you with that! Can you specify which type of docuemnt?\n'])
chatbot.train(["yes",'Ok! I like to grade essays, can you specify the type of document?\n'])


# In[41]:


user_response1 = input()


# In[42]:


response1 = chatbot.get_response(user_response1)
print("\n")
print(response1)
print("\n")
print("I am sorry but at the moment I can only grade School essays...But I will learn more!")


# In[37]:


chatbot.train(['School essay','Sweet!\n'])
chatbot.train(['School','Sweet!\n'])


# In[38]:


user_response2 = input()


# In[39]:


response2 = chatbot.get_response(user_response2)
print("\n")
print(response2)
print("\n")


# In[14]:


fname = input('Could you write the name of the file? maybe....test_nlp.csv?:\n')

fileObject = open(fname, 'r') 


# In[15]:


data = pd.read_csv(fileObject)


# In[18]:


from sklearn.externals import joblib
model2 = joblib.load('random_forest_model.pk') 


# In[19]:


vectorizer = joblib.load('vectorizer.pk')


# In[31]:


tfidf_t = vectorizer.transform(data["essay"])


# In[35]:


pred = model2.predict(tfidf_t)


# In[36]:


print("Your Essay result is:\n\n")
print(pred)
print("\n")

