# -*- coding: utf-8 -*-


import os
import json
import nltk
import gensim
import numpy as np
from gensim import corpora, models, similarities
import pickle

os.chdir('C:\\Users\\Yashwanth\\Desktop\\Chatbot')
model = gensim.models.KeyedVectors.load_word2vec_format('F:\\Data\\GoogleNews-vectors-negative300.bin', binary=True)
_file = open('C:\\Users\\Yashwanth\\Desktop\\Chatbot\\conversation.json')
data = json.load(_file)

cor=data["conversations"];

x=[]
y=[]

## Separately store question and answers  in x and y
for i in range(len(cor)):
    for j in range(len(cor[i])):
        if j<len(cor[i])-1:
            x.append(cor[i][j])
            y.append(cor[i][j+1])

tok_x=[]
tok_y=[]

### Tokenize words 
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))
    
    
## Create vector of 1*300 Dimensions ## Useful for padding
sentend=np.ones((300,),dtype=np.float32) 

## Convert into word2vec using google trained word2vec model
vec_x=[]
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)
    
vec_y=[]
for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_y.append(sentvec)           
    
## Restricting to only 14 words in a sentence     
for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    
## Padding to 300 Dimensions .. Google word2vec is a 300 dimensions vector
for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)    
            
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)             
            

## Pickle and save in local            
with open('conversation.pickle','wb') as f:
    pickle.dump([vec_x,vec_y],f)                
