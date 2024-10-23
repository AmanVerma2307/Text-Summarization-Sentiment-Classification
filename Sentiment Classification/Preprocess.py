###### Importing Libraries
import os
import re
import torch
import torchtext
import nltk
import spacy
import html
import string
import numpy as np
import pandas as pd
from torchtext import data
from sklearn.model_selection import train_test_split

#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

###### Preprocess
def preprocess(csv_path):

    """
    Function to preprocess input .csv file into clean and tokenized sentences, and labels

    INPUTS:-
    1) csv_path: Path to the target .csv file 
    #2) train_flag: Boolean to 

    OUTPUTS:-
    1) X: (N,seq_len,1) dimensional list of cleaned and tokenized sentences
    2) y: Coressponding labels of dimension (N,)
    """

    def remove_punct(text):
        text = text.lower()
        text = "".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text))
        text = re.sub('[0-9]+', '', text)
        return text
    
    def tokenization(text):
        text = re.split('\W+', text)
        return text
    
    def remove_stopwords(text):
        stopword = nltk.corpus.stopwords.words('english')
        text = [word for word in text if word not in stopword]
        return text
    
    def lemmatizer(text):
        wn = nltk.WordNetLemmatizer()
        text = [wn.lemmatize(word) for word in text]
        return text

    df = pd.read_csv(csv_path,index_col=False)
    tweets = df['text'].to_list()
    labels = df['airline_sentiment'].to_list()

    X = []
    y = []

    for idx, item in enumerate(tweets):
        item = remove_punct(item)
        item = tokenization(item)
        item = remove_stopwords(item)
        item = lemmatizer(item)
        X.append(item)

    for idx, item in enumerate(labels):
        if(item=='positive'):
            y.append(0)
        elif(item=='neutral'):
            y.append(1)
        else:
            y.append(2)    
    
    return X, y

###### Testing
#X, y = preprocess('./train.csv')
#print(len(X),len(y))