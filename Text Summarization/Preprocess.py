###### Importing Libraries
import os
import re
import nltk
import torch
import torchtext
import numpy as np
import pandas as pd

###### Preprocessing
word_mapping = {"ain't": "is not","aint": "is not", "aren't": "are not",
                "arent": "are not","can't": "cannot","cant": "cannot", 
                "'cause": "because", "cause": "because", "could've": "could have", 
                "couldn't": "could not",
                "didn't": "did not", "doesn't": "does not", "don't": "do not", 
                "hadn't": "had not", 
                "hasn't": "has not", "haven't": "have not",
                "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                 "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                 "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                 "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 'mstake':"mistake",
                 "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                 "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                 "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                  "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                  "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                  "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                   "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                    "wasn't": "was not",'wasnt':"was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                    "you're": "you are", "you've": "you have", 'youve':"you have", 'goin':"going", '4ward':"forward", "shant":"shall not",'tat':"that", 'u':"you", 'v': "we",'b4':'before', "sayin'":"saying"
                    }

def clean(txt):

    """
    Function for cleaning text

    INPUTS:-
    1) txt: Input sentence
    2) label_flag: Boolean value for 

    OUTPUTS:-
    1) txt: Cleaned sentence
    """

    def word_contraction(txt):
        txt = txt.split()
        for i in range(len(txt)):
            word = txt[i]
            if word in word_mapping:
                txt[i] = word_mapping[word]
        return " ".join(txt)
    
    def remove_stopwords(text):
        text = text.split()
        stopword = nltk.corpus.stopwords.words('english')
        text = [word for word in text if word not in stopword]
        return " ".join(text)
    
    def remove_punct(text):
        text = "".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text))
        text = re.sub('[0-9]+', '', text)
        return text
    
    def remove_fullstop(text):
        txt = []
        for item in text.split():
            if(item != '.'):
                txt.append(item)
        return " ".join(txt)
    
    def lemmatizer(text):
        text = text.split()
        wn = nltk.WordNetLemmatizer()
        text = [wn.lemmatize(word) for word in text]
        return " ".join(text)

    txt = txt.lower() # Lower case
    txt = word_contraction(txt) # Word contraction
    txt = remove_punct(txt)  # Remove punctuations
    txt = remove_stopwords(txt) # Stop word removal
    #txt = re.sub(r'\(.*\)','',txt) # Remove (words)
    #txt = re.sub(r'[^a-zA-Z0-9. ]','',txt) # Remove punctuations
    #txt = re.sub(r'\.',' . ',txt)
    #txt = remove_fullstop(txt)
    #txt = txt.replace("'s",'') # Apostaphe removal
    #txt = remove_punct(txt) # Puntuation removal
    txt = lemmatizer(txt) # Lemmatization
    return txt

def preprocess(csv_path):

    """
    Function to preprocess input .csv file into clean sentences and summary

    INPUTS:-
    1) csv_path: Path to the target .csv file  

    OUTPUTS:-
    1) X: (N,) dimensional list of cleaned sentences
    2) y: Coressponding clean summary of dimension (N,)
    """
    X, y = [], []
    df = pd.read_csv(csv_path,index_col=False)
    
    for idx, item in enumerate(df['Text'].to_list()):
        x_curr = clean(item)
        y_curr = clean(str((df['Summary'].to_list())[idx]))

        if(y_curr != ""):
            X.append(x_curr)
            y.append(y_curr)

    return X,y

###### Testing
#X,y = preprocess('./train.csv')
#print(X[10],y[10])
#print(100*'=')
#print(X[-1],y[-1])
#print(y)
#print(len(X),len(y))
#for i in range(len(X)):
#    print(X[i])
#    print('------------')
#    print(y[i])
#    print('================================================')