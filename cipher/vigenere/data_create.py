import random
import re
import csv
import os
import math
# script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
from itertools import cycle

maxKeylen = 8
minKeylen = 8

def vigenere(key, message, mode):
    cipherAlpha = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",range(26)))
    plainAlpha = dict(zip(range(26),"ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    translated = [] #stores encrypted/decrypted code

    keyIndex = 0
    key = key.upper()

    for symbol in message.upper(): # loop through each character in message
        if symbol.isalpha():
            if mode == "encrypt":
                translated.append(plainAlpha[ (cipherAlpha[symbol] + cipherAlpha[key[keyIndex]])%26])
            elif mode == "decrypt":
                translated.append(plainAlpha[ (cipherAlpha[symbol] - cipherAlpha[key[keyIndex]])%26])

            keyIndex += 1 # move to the next letter in the key
            if keyIndex == len(key):
                keyIndex = 0
        else:
            # The symbol was not a letter, so add it to translated as is.
            translated.append(symbol)

    # return ''.join(translated)
    return ''.join(w + " " if i%5==0 else w for i,w in enumerate(translated, 1)) if mode=="encrypt" else ''.join(w.strip() for w in translated)

def repeatedCombine(list1, list2):
    # print(list1[0],list2[0])
    return zip(list1, cycle(list2)) if len(list1) >= len(list2) else zip(cycle(list1), list2)

def translateKeyTexts(keys, msgs, mode="encrypt"):
    if mode not in ["encrypt","decrypt"]:   #if invalid mode input, return blank list
        return []

    keytext = repeatedCombine(keys, msgs)
    # [print(k,t) for k,t in keytext]

    data = []
    for key, in_msg in keytext:
        out_msg = vigenere(key,in_msg,mode)
        # print(vigenere(key,c_text,"decrypt"))     #ciphercheck
        d = [key.ljust(maxKeylen, '-'),in_msg,out_msg] if mode=="encrypt" else [key,out_msg,in_msg] if mode=="decrypt" else []

        if d:
            data.append(d)
        # print([key,p_text,c_text])

    return data

def writeToCSV(filename, data):
    csv_name="./data/"+filename+".csv"

    with open(csv_name, 'wt') as csvfile:
        # filewriter = csv.writer(csvfile, delimiter=',',
        #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer = csv.writer(csvfile)
        writer.writerows(data)
        ##write each data into csv
            # filewriter.writerow(['Name', 'Profession'])
        # [print(d) for d in data]
        # [filewriter.writerow(d) for d in data]

##      MAIN PROGRAM        ##

##grab 1000 random keys
key_file = open('./txt/corncob_wordlist.txt','r')
words = [word.strip() for word in key_file]
i_keys = [w.upper() for w in words if minKeylen <= len(w) <= maxKeylen]
random.shuffle(i_keys)
_70 = math.floor(len(i_keys) * .9)  #put aside 70% for training
print("keysLen: ", len(i_keys),"90%:", _70)
train_keys = i_keys[:_70]
test_keys = i_keys[_70:]
key_file.close()
print("Last training key:", train_keys[-1], len(train_keys))
print("First testing key:", test_keys[0], len(test_keys))

##grab sonnet cleartexts
msg_file = open('./txt/sonnets.txt', 'r')
lines = msg_file.read()
msg_file.close()
sonnets = re.split(r'Sonnet [IXVCL]+', lines)
# print("SONNET", sonnets)
plaintexts = [''.join(w.upper() for w in line if w.isalpha()) for line in sonnets]
plaintexts = list(filter(None, plaintexts)) #remove any empty strings
train_texts = [w[:20] for w in plaintexts]
# print("PLAIN:",plaintexts)

##pairing keys and cleartexts (of unequal count)
train_data = translateKeyTexts(train_keys,train_texts)
print(train_data[0],train_data[-1])
test_data = translateKeyTexts(test_keys,plaintexts)
print(test_data[0],test_data[-1])

##write data to csvfiles
writeToCSV("train",train_data)
writeToCSV("test",test_data)
