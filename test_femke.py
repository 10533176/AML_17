# algoritme voor testen van spam of geen spam

import os, sys
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

assert os.path.exists('./Data/emails.train.csv'), "[Dataset File Not Found] Please download dataset first."
# Read in csv file as dataframe
df = pd.read_csv('./Data/emails.train.csv')

# Show a snippet of dataset.
print (df.head())

# def checkSpam (text):
#   print (text)


# for text in df['text']:
#   checkSpam(text)


