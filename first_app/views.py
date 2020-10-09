from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages

def index(request):
    return render(request, 'index.html')




import joblib
import pickle


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datadir = os.path.join(BASE_DIR, 'static/ml_files/SMSSpamCollection')
modeldir = os.path.join(BASE_DIR, 'static/ml_files/sms_spam.pickle')
save_cv_dir = os.path.join(BASE_DIR, 'static/ml_files/new_cv_fit.pickle')


############################ Save the CountVectorizer ####################################

#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns


#yelp = pd.read_csv(datadir, sep='\t', names=['label', 'message'])

#import string
#from nltk.corpus import stopwords

#def text_process(mess):

#    Takes in a string of text, then performs the following:
#    1. Remove all punctuation
#    2. Remove all stopwords
#    3. Returns a list of the cleaned text
#    """

    # Check characters to see if they are in punctuation
#    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
#    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
#    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


#X = yelp['message']

#from sklearn.feature_extraction.text import CountVectorizer

#cv = CountVectorizer(analyzer=text_process)

#cv_fit = cv.fit(X)
#X = cv_fit.transform(X)


########################### Save the CountVectorizer ###############################

#with open(save_cv_dir,'wb') as f:
#    pickle.dump(cv_fit, f)

########################### Load the Model ##########################

with open(modeldir, 'rb') as f:
    model = joblib.load(f)

########################### Load the CountVectorizer ##########################

import string
from nltk.corpus import stopwords

def text_process(mess):

    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """

    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


with open(save_cv_dir, 'rb') as f:
    new_cv_fit = joblib.load(f)
#############################################################################

import numpy as np
import pandas as pd
import pickle
import json
import joblib



def register(request):
    if request.method == 'POST':
        sms = request.POST['sms']

        message = new_cv_fit.transform([sms])

        answer = model.predict(message)

        final_ans = answer[0]


    return render(request, 'index.html', {'prediction' : final_ans})
