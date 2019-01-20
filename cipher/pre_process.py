from string import ascii_uppercase
from collections import OrderedDict
from itertools import count

import pandas as pd
import numpy as np

from sklearn import preprocessing

'''
This file load the csv datafile in as pandas DataFrame, 
and return it in a function called data_train() and data_test()

Make sure to change the path for train&test_file below for your own.
'''

train_file = '/home/wataru/Uni/spring_2018/4911/cipher/vigenere/data/train.csv'
test_file = '/home/wataru/Uni/spring_2018/4911/cipher/vigenere/data/test.csv'
## Creating ordered dictionary for encoding the letters
od = OrderedDict(zip(ascii_uppercase, count(0)))
od.update({'-':26})

## Defining the mapping function for encoding
def encode(text):
    ## strip white space and " ' " in the string, and convert it into numbers
    in_np = np.empty((1, 0))
    for letter in text.replace(' ', '').strip("'").upper():
        in_np = np.append(in_np, od[letter])
    return in_np

## Reading in csv file
column = ['key', 'plantxt', 'cipher']
df_text = pd.read_csv(test_file, names=column)

# Return the Dataframes 
def data_train():
    return pd.read_csv(train_file, names=column)

def data_test():
    return pd.read_csv(test_file, names=column)


##### No need to Worry about the rest
## Applying the funcftion to whole dataframe 
df_encoded_text = df_text.applymap(encode)

## convert it to numpy arrays
cipher = df_encoded_text.cipher.values.reshape(-1, 1)

def one_hot(arr):
    '''
    Takes in one array full of numbers (encoded one text) and convert it into 
    one hot matrix using Sklearn package.
    '''
    print(type(arr))
    arr = arr[0] # referencing the array inside of array
    oneHot = preprocessing.OneHotEncoder()
    y = arr.reshape(arr.shape[0], -1)
    oneHot.fit(y)
    y = oneHot.transform(y).toarray()
    return y
    ### Make a one hot encoder function for the encoded cipher/key text
