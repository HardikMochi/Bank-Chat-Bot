from django.shortcuts import render
from django.http import  HttpResponseRedirect
from django.contrib.sessions.models import Session
from django.http import HttpResponse 
from .models import Customers,Accounts
from .f2 import bert
import warnings
warnings.filterwarnings("ignore")
# Create your views here.
import csv
import json
import re
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from nltk.stem import WordNetLemmatizer
from django.http.response import JsonResponse
from transformers import BertTokenizer, TFBertForQuestionAnswering 
i,question_list =0,[]
entity,flag ={},0
user_id = 0
username = ""
#Create your views here.


word_index_dict = {'<start>': 4,
                    'balance': 5,
                    '<end>': 6,
                    'much': 7,
                    'money': 8,
                    'check': 9,
                    'account': 10,
                    'acc': 11,
                    'saving': 12,
                    'current': 13,
                    'show': 14,
                    'avilable': 15,
                    'move': 16,
                    '100': 17,
                    'rahul': 18,
                    '500': 19,
                    'ghanshyam': 20,
                    'send': 21,
                    'jay': 22,
                    'ghanshayam': 23,
                    '1000': 24,
                    'credit': 25,
                    'card': 26,
                    'transfer': 27,
                    'rahulm': 28,
                    'gahnshyam': 29,
                    'interest': 30,
                    'rate': 31,
                    'acoount': 32,
                    'loan': 33,
                    'approved': 34,
                    'home': 35,
                    'personal': 36,
                    'pas': 37,
                    '<PAD>': 0,
                    '<START>': 1,
                    '<UNK>': 2,
                    '<UNUSED>': 3}
#stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 
nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()

json_file = open('Bank/Data/chatbot_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("Bank/Data/chatbot_model.h5")
print("Loaded model from disk")


def home(request):
    global user_id,username
    print("in home")
    print( request.session.has_key('is_logged'))
    if request.session.has_key('is_logged'):
        user_id = request.session['userid']
        username = request.session['username']
        print(user_id)
        del request.session['is_logged']
        return render(request,'index.html') 
    return HttpResponseRedirect('Account/login')

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
    for i in range(4):
        cs,cs1 = compute(text[i])
        cs = index_review_words(str(cs))
        que_1.append(cs)
        cs2,cs3 = compute(questions[i])
        cs2 = index_review_words(str(cs2))
        que_2.append(cs2)
    return que_1,que_2    

def extract_transfer_entity(text):
    transfer_entity ={}
    entitym = re.findall('rs\s{0,2}[0-9]+',text)
    if entitym:
        money = re.findall('[0-9]+',entitym[0])
        transfer_entity['Money'] = money[0]
    else:transfer_entity['Money'] = None

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

def extract_interest_entity(text):
    interest_entity = {}
    entityr = re.findall("(saving|current|sort term|medium term|long term)",text)
    if entityr:
        interest_entity['type'] = entityr[0]
    else: interest_entity['type'] = None    
    return interest_entity

def extract_balance_entity(text):
    balance_entity ={}
    account = re.findall("(saving|current)",text)
    if account:
        balance_entity['account'] = account[0]
    else:balance_entity['account'] = None  
    ofa = re.findall('of\s{0,2}[a,an]?\s{0,2}\w+',text)
    if ofa:
        name = ofa[0].split()[-1]
        balance_entity['Name'] = name
    else :balance_entity['Name'] =None
    return balance_entity

def check_for_valid(text):
    valid = []
    print(text)
    words = text.split()
    for word in words:
        if word in word_index_dict:
            valid.append(1)
        else :
            valid.append(0)
    print(valid)        
    if sum(valid)>0:
        return True
    else:
        return False   

def transfer_money():
    global question_list,i,entity
    if entity['number_of_account']>1:                                                      # check if number of sender account is more than one
        sg = Accounts.objects.get(user_id_id = user_id,account_type=entity['Name_from'])          # fetch data of the sender user from database
    else:    
        sg = Accounts.objects.get(user_id_id = user_id)                                           # for sender account is 1
    print(sg.avilable_balance)    
    if int(sg.avilable_balance) < int(entity['Money']):                                    # check if the money that you want you send is higher than your avilable balance
            question_list,i,entity =[] ,0,{}
            return "Avilable balance is not sufficient"                                    
    else:
        cg = Customers.objects.get(user_id =entity['user_id'][0])                          
        if cg.number_of_account > 1:
            rg = Accounts.objects.get(user_id_id =entity['user_id'][0],account_type='saving')
        else: rg = Accounts.objects.get(user_id_id =entity['user_id'][0])
        
        new_sg = sg.avilable_balance - int(entity['Money'])
        new_rg = rg.avilable_balance + int(entity['Money'])
        sg.avilable_balance = new_sg
        rg.avilable_balance = new_rg
        sg.save()
        rg.save()
        return "Transaction is successful and you Avialble Blance is Rs {0}".format(new_sg)

def balance():
    if entity['number_of_account'] >1:
        sg = Accounts.objects.get(user_id_id = user_id,account_type=entity['account'])          # fetch data of the sender user from database
    else:    
        sg = Accounts.objects.get(user_id_id = user_id)    
    return "Your Avilable Blance is : Rs {0}".format(sg.avilable_balance)   


def fun_for_balance(text):
    global entity  
    print("un for balance")
    if entity['Name']:
        if entity['Name'] != username:
            return "Your Can Not see The Balance of Other Customer"
    Customer = Customers.objects.get(user_id= user_id)   
    if Customer.number_of_account >1:
        if entity['account'] == None:
            entity['number_of_account'] = Customer.number_of_account
            question_list.append("account")
            return "you have two account saving and current "
        else:
            return balance()     
    else:
        entity['number_of_account']=1
        return balance()        
             
def fun_for_loan_status(text):
    lg = Customers.objects.get(user_id= user_id)
    return "Your Loan status is : {}".format(lg.loan_status) 

def fun_for_interest(text):
    global question_list,i,entity
    if entity['type'] == None:
        question_list.append("type")
        return "Please Provide The Type of Account e.g saving or sort term or medium term"
    else:
        if len(question_list) > 2:
            if entity['type']  in ['sort term','medium term','long team']:
                q = question_list[0] + ' of '+entity['type'] + ' deposits ?'
            else:q = question_list[0] + ' of '+entity['type'] +' account ?'
        else:
            q = text   
        print(q)
        ans = bert("interest",q)
        question_list,i,entity =[] ,0,{}
        return ans
        


def fun_for_transfer(text,request):
    global entity
    if entity['Name_to'] ==None:
        question_list.append("Name_to")
        return "Plese provide the Use Name To send Money"
    if check_name(entity['Name_to']) == False:
        question_list.append("Name_to")
        return "please provide the valid user name"
    if entity['Name_from'] == None:
        print(request.session.items())
        Customer = Customers.objects.get(user_id= user_id)   
        if Customer.number_of_account >1:
            entity['number_of_account'] = Customer.number_of_account
            question_list.append("Name_from")
            return "you have two account saving and current "
        else:entity['number_of_account'] = 1   
    if entity['Money'] == None:
        question_list.append("Money")
        return "please write how much money you want to send  OR  please write RS before money"  
    if text != "yes":
        question_list.append("for assurance")
        return "are you sure to send Rs {0} to {1}".format(entity['Money'],entity['Name_to'])
    else:
        return transfer_money() 

# check for name is there in database
def check_name(word):
    global entity
    names = Customers.objects.all()
    uz = [i.name for i in names]                       # all the customer name
    ub =[i.user_id for i in names if i.name == word]   # for find the user id for customer
    print(ub)
    entity['user_id'] = ub
    print(entity)
    if word in uz:                                    # check if name is in the databse
        return True
    else:return False

#check the context    
def context(text):
    global question_list,i,entity
    if text in  ['saving','saving account','from savings','savings']:
        text = 'saving'
    elif text in ['current','current account','from current']:
        text = 'current' 
    elif check_name(text):
        return text
    elif text in ['yes','no']:
        text = text 
    elif text in ['sort term','medium term','long term']:
        text =text   
    else:
        print("this is a text")
        question_list,i,entity =[] ,0,{}
    return text


#this function is useful to generate the answetr   
def predict(request):
    global question_list,i,entity
    print("user_id")
    request.session.set_test_cookie()
    print(request.session.get('is_logged', False))
    byte = request.read()
    data= json.loads(byte.decode('utf-8')) 
    text = data['message']
    print(text)
    text = context(text)
    print("--------------------------new ---------------------------")
    print(text)
    print(question_list)
    if i==0:
        i=+1
        question_list.append(text)
        a = np.load('Bank/Data/a.npy')
        b = np.load('Bank/Data/b.npy')
        c = np.load('Bank/Data/c.npy')
        d = np.load('Bank/Data/d.npy')
        if check_for_valid(text):
            que2 = [random.choice(i) for i in [a,b,c,d]]
            que1 = [text for i in range(4)]
            
            que_1,que_2 = create_question_pair(que1,que2)
            question_5 = sequence.pad_sequences(que_1,value=word_index_dict['<PAD>'],padding='post',maxlen=7)
            question_6 = sequence.pad_sequences(que_2,value=word_index_dict['<PAD>'],padding='post',maxlen=7)
            print(question_5)
            print(question_6)
            pred = model.predict((question_5,question_6))
            print(pred)
            scores =[]
            for p in pred:
                if p>0.7:
                    scores.append(1)
                else :
                    scores.append(0)    
            print(scores)  
            if sum(scores) == 0:
                exam ="please write the good question"  
            else:     
                score = np.argmax(scores)
                print(score)
                if score == 0:
                    question_list.append(0)
                    entity = extract_balance_entity(text)
                    print(entity)
                    exam = fun_for_balance(text)
                
                if score == 1:
                    
                    question_list.append(1)
                    entity = extract_transfer_entity(text)
                    print(entity)
                    exam = fun_for_transfer(text,request) 
                    #exam = "this is for transfer" 
                if score == 2:
                    question_list.append(2)
                    entity = extract_interest_entity(text)
                    print(entity)
                    exam = fun_for_interest(text)
                if score == 3:
                    exam = fun_for_loan_status(text)     
        else:
             exam = "please write the good question"  
    else:
        print("text")
        print("ddgwehd",i)
        if question_list[1]==1:
            if question_list[-1] !="for assurance":
                print(question_list[-1])
                print(text)
                entity[question_list[-1]]=text
                print(question_list)
                print(entity)
                exam = fun_for_transfer(text,request)
            elif text == "yes" :
                exam =fun_for_transfer(text,request)   
            else:
                question_list,i=[],0 
        elif question_list[1] == 0:
            print(question_list[-1])
            print(text)
            entity[question_list[-1]]=text
            print(question_list)
            print(entity)
            exam = fun_for_balance(text)
        elif question_list[1] == 2:
            print(question_list[-1])
            print(text)
            entity[question_list[-1]]=text
            print(question_list)
            print(entity)
            exam = fun_for_interest(text)

           
    data =JsonResponse({'text':exam})
    return data 
      

