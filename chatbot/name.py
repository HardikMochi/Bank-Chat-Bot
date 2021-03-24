import csv
import re
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply,Embedding, Dropout, Subtract,LSTM, Add, Conv2D

# Create your views here.
#def home(request):
   # return render(request,'index.html')


#stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

#create sentence lemitizer and clean sentence
def compute(sent): 
    words=word_tokenize(str(sent.lower())) 
    
    #Removing all single letter and and stopwords from question 
    sent1=' '.join(str(lemmatizer.lemmatize(j)) for j in words if j not in stop_words and (len(j)!=1)) 
    sent2=' '.join(str(j) for j in words if j not in stop_words and (len(j)!=1)) 
    sent1 = '<start> ' + sent1+ ' <end>'
    sent2 = '<start> ' + sent2+ ' <end>'
    return sent1, sent2

def index_review_words(text):
    review_word_list = []
    for word in text.lower().split():
        if word in word_index_dict.keys():
            review_word_list.append(word_index_dict[word])
        else:
            review_word_list.append(word_index_dict['<UNK>'])

    return review_word_list 

def create_question_pair(text,questions):
    que_1=[]
    que_2=[]
    for i in range(3):
        cs,cs1 = compute(text[i])
        cs = index_review_words(str(cs))
        que_1.append(cs)
        cs2,cs3 = compute(questions[i])
        cs2 = index_review_words(str(cs2))
        que_2.append(cs2)
    return que_1,que_2    

def Load_model():
    # load json and create model
    json_file = open('Data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Data/model.h5")
    print("Loaded model from disk")
    return loaded_model

def extract_transfer_entity(text):
    transfer_entity ={}
    entitym = re.findall('Rs[0-9]+',text)
    if entitym:
        money = re.findall('[0-9]+',entitym[0])
        transfer_entity['Money'] = money[0]

    entityt = re.findall('to\s{0,3}[a,an]?\s{1,3}\w+',text)
    if entityt:
        name_to = entityt[0].split()[-1]
        transfer_entity['Name_to'] = name_to
    else :transfer_entity['Name_to'] =None  

    entityf = re.findall('from\s{0,3}[a,an]?\s{1,3}\w+\s{0,3}\w+',text)
    if entityf:
        name_from = re.findall('(saving|current)',entityf[0])
        transfer_entity['Name_from'] = name_from[0]
    else :transfer_entity['Name_from'] =None     
    return transfer_entity

def extract_balance_entity(text):
    balance_entity ={}
    account = re.findall("(saving|current)",text)
    if account:
        balance_entity['account'] = account[0]
        return balance_entity

def classify():
    text = "how much money i have in my account"
    que2 = [random.choice(i) for i in [a,b,c]]
    que1 = [text for i in range(3)]

    que_1,que_2 = create_question_pair(que1,que2)
    question_5 = sequence.pad_sequences(que_1,value=word_index_dict['<PAD>'],padding='post',maxlen=7)
    question_6 = sequence.pad_sequences(que_2,value=word_index_dict['<PAD>'],padding='post',maxlen=7)
     
    model = Load_model()
    pred = model.predict((question_5,question_6))
    print(pred)
    print(np.argmax(pred))

classify()

