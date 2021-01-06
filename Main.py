#!/usr/bin/env python
# coding: utf-8

# In[12]:


##############################################
## Data Description ##########################


# In[13]:


import numpy as np
import pandas as pd
from IPython import get_ipython

train_df = pd.read_csv(".//Data//train.csv")

test_df = pd.read_csv(".//Data//test.csv")
test_df.head()


##################################################
## Data Preprocessing ###########################


# In[23]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

stemmer=PorterStemmer()
stop_words=set(stopwords.words("english"))
def tweet_preprocessing(tweets):
   list_tweets_words=[]
   for tweet in tweets:
    list_tweet_words=[]
    tweet=tweet.lower()
    ##Remove userName
    tweet=re.sub(r"@[a-z0-9_-]*","",tweet)
    ##Remove hyperlinks
    tweet=re.sub(r"https?://.*[\s]*","",tweet)
    ## Remove numbers and characters
    tweet=re.sub(r"[^a-z ]*","",tweet)
    ## Replace multiple spaces by single space
    tweet=re.sub(r"[\s]+"," ",tweet)
    ##Word Tokenization
    tweet_words=word_tokenize(tweet)
    for word in tweet_words:
        #if(word not in stop_words):
         # word=stemmer.stem(word)
          list_tweet_words.append(word)
    ## join : from list of words to string
    list_tweets_words.append(list_tweet_words)
   return list_tweets_words

# In[24]:


#Add 2 new columns to our dataframe content listes of splitted text end selected text
train_df["text_tokenize"]=tweet_preprocessing(train_df.text.astype(str))
train_df["selected_text_tokenize"]=tweet_preprocessing(train_df.selected_text.astype(str))

# In[42]:


# we can save the new dataframe in other file csv
#train_df.to_csv("preprocessed_train_data.csv")


# In[26]:


#preprocessed_df = pd.read_csv("preprocessed_train_data.csv",keep_default_na=False)
#del preprocessed_df['Unnamed: 0']
preprocessed_df = train_df

# In[30]:


import ast
# if we want first index we choose True, and False for the latest index
# we should use the next 2 lines one time (transform text to list)
#preprocessed_df["text_tokenize"] = preprocessed_df.text_tokenize.apply(lambda x: ast.literal_eval(x))
#preprocessed_df["selected_text_tokenize"] = preprocessed_df.selected_text_tokenize.apply(lambda x: ast.literal_eval(x))
#preprocessed_df.head()

def find_index (text_list,selectedText_list,i=True):  
    #find first word in selected_text
    try :
        if i == True :
            first_w = selectedText_list[0]
            #print(first_w)
            return (int(text_list.index(first_w))) 
            #find last word in selected_text
        else:
            last_w = selectedText_list[-1]
            # look for first_w index in text list
            return (int(text_list.index(last_w)))
    except :
        pass
preprocessed_df["first_index"] = preprocessed_df.apply(lambda row : find_index(row.text_tokenize,row.selected_text_tokenize,True),axis=1)
preprocessed_df["last_index"] = preprocessed_df.apply(lambda row : find_index(row.text_tokenize,row.selected_text_tokenize,False),axis=1)
#to convert float to int pandas
pd.options.display.float_format = '{:,.0f}'.format

##############################
######## tokenization


# In[33]:


df = preprocessed_df


# In[36]:


import nltk
import pandas as pd 
import ast
import tensorflow
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle


# In[37]:


## tokenize and make the index of words
tokenizer = Tokenizer(num_words=20000,oov_token="<OOV>")
tokenizer.fit_on_texts(df.text_tokenize)
tokenized_text = tokenizer.texts_to_sequences(df.text_tokenize)
tokenized_selected_text = tokenizer.texts_to_sequences(df.selected_text_tokenize)


#after tokenzation we can prepare input of the module  
pad_token_text = pad_sequences(tokenized_text,padding = "post")
pad_token_text

pd.DataFrame(pad_token_text).to_csv("pad_token_text.csv",header=None,index=None)
df.to_csv("tokenized.csv",index=None)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[40]:


###########################################
## and finally -----> the model


# In[41]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.python.keras.regularizers import l2, l1, l1_l2
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score


# In[45]:


targets = df[["first_index","last_index"]]


# In[48]:


training = pd.read_csv("pad_token_text.csv",header= None)


# In[49]:


# after preparing input and output we can split data for testing
x_train, x_test, y_train, y_test = train_test_split(training.values, targets.values, test_size=0.2, random_state=42)


# In[50]:


def first_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=33),
        Bidirectional(GRU(128, return_sequences=True, dropout=0.8, recurrent_dropout=0.8)),
        Bidirectional(GRU(128,return_sequences=True, dropout=0.8, recurrent_dropout=0.8)),
        BatchNormalization(),
        Dense(64, activation='elu',kernel_regularizer=l1_l2()),
        Dropout(0.8),
        Dense(2, activation='elu'),
        Flatten(),
        Dense(2, activation='elu')

    ])
    return model



vocab = 22000
model = first_model(vocab)
es = EarlyStopping(patience=5)
#tweet_sentiment.hdf5
mcp_save = ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_mse')
model.compile(loss="mse",optimizer="adam",metrics=['mse',"mae"])


# In[54]:


# epochs 100   
history = model.fit(x=x_train, y=y_train, batch_size = 32, epochs=100, validation_split = 0.2,callbacks=[es,mcp_save])
model.save('model.hdf5')


# In[ ]:




