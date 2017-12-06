# algoritme voor testen van spam of geen spam
# -*- coding: utf-8 -*-

import os, sys
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing

assert os.path.exists('./Data/emails.train.csv'), "[Dataset File Not Found] Please download dataset first."
# Read in csv file as dataframe
df = pd.read_csv('./Data/emails.train.csv')

# Show a snippet of dataset.
# print (df.head())

# def checkSpam (text):
#   print (text)


# for text in df['text']:
#   checkSpam(text)

train_data = pd.read_csv('./Data/emails.train.csv')
test_data  = pd.read_csv('./Data/emails.test.csv')
trainPositive = {}
trainNegative = {}

positiveTotal = 0
negativeTotal = 0

pA = 0.0
pNotA = 0.0

def train (df):
  global pA, pNotA
  total = 0
  numSpam = 0
  for index, row in df.iterrows():
    if row['spam'] == 1:
        numSpam += 1
    total += 1
    processEmail(row['text'], row['spam'])
  pA = numSpam / (float(total))
  pNotA = (total - numSpam) / (float(total))

def processEmail(body, label):
  global positiveTotal, negativeTotal
  for word in body:
    if label == 1:
      trainPositive[word] = trainPositive.get(word, 0) + 1
      positiveTotal += 1
    else:
      trainNegative[word] = trainNegative.get(word, 0) + 1
      negativeTotal += 1

#gives the conditional probability p(B_i | A_x)

def conditionalWord(word, spam):
  global positiveTotal, negativeTotal
  if spam == 1:
    return trainPositive[word] / (float(positiveTotal))
  return trainNegative[word] / (float(negativeTotal))

#gives the conditional probability p(B | A_x)
def conditionalEmail(body, spam):
  result = 1.0
  for word in body:
    result *= conditionalWord(word, spam)
  return result

def classify(email):
  isSpam = pA * conditionalEmail(email, True) # P (A | B)
  notSpam = pNotA * conditionalEmail(email, False) # P(¬A | B)
  return isSpam > notSpam

train(train_data)
print ("pA", pA)
print ("pNotA", pNotA)
print (classify(train_data['text'][5]))




