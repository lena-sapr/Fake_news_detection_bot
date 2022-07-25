#!/usr/bin/env python
# coding: utf-8

# 
# # Неделя 7. Natural Language Processing 
# ## Классификация текста с помощью BERT

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# !rm -f /root/.kaggle/*
# !mkdir /root/.kaggle
# !cp /content/drive/MyDrive/kaggle.json /root/.kaggle


# In[ ]:


# иногда требуется для работы токенизатора
# !pip install sentencepiece 


# In[10]:




# In[2]:


# устанавливаем библиотеку со множеством моделей huggingface.co/
# get_ipython().system('pip install transformers')


# In[ ]:


# # скачиваем датасет
# !kaggle datasets download -d datatattle/covid-19-nlp-text-classification


# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch

# импортируем трансформеры
import transformers
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import re


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[11]:


# ## Загружаем модель
# 
# Список предобученных моделей: [https://huggingface.co/transformers/pretrained_models.html](https://huggingface.co/transformers/pretrained_models.html)


# DistilBERT:

## задаем саму модель
model_class = transformers.DistilBertModel

## токенайзер к ней (для некоторых моделей токенайзер будет отличаться, см.
## в документации к каждой модели конкретно)
tokenizer_class = transformers.DistilBertTokenizer

## загружаем веса для моделей
pretrained_weights = 'distilbert-base-uncased'

# AlBERT:
# model_class, tokenizer_class, pretrained_weights = (transformers.AlbertModel, 
#                                                     transformers.AlbertTokenizerFast,
#                                                     'albert-base-v1')

# BERT
# model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, 
#                                                     transformers.BertTokenizer, 
#                                                     'bert-base-uncased')

###########################################
# создаем объекты токенизатора для и модели
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

max_len = 200


model.to(device)


# batch_size = 100
# batch_num = len(all_news) // batch_size


# for i in tqdm.notebook.tqdm(range(batch_num +1)):
#   batch_ids = input_ids[i*batch_size:(i + 1)*batch_size, :]
#   batch_attn_msk = attention_mask[i*batch_size:(i + 1)*batch_size, :]
#   batch_ids, batch_attn_msk = batch_ids.to(device), batch_attn_msk.to(device)
  
# # Получаем выход модели (нам оттуда нужно не все)
#   with torch.inference_mode():
#       last_hidden_states = model(batch_ids, attention_mask=batch_attn_msk)
#   # features = last_hidden_states[0][:,0,:].cpu().numpy()
#   # np.savetxt('batch_feat.txt', features)

#   features = last_hidden_states[0][:,0,:].to(device)#.numpy()
#   np.savetxt((f'test/test_{i}.txt'), features.cpu().numpy())
  
#   # f = open('batch_feat.txt', 'r')
#   # f_feat.write(f.read())


# # In[98]:


# import os
# dirname = '/content/test/'
# files = os.listdir(dirname)
# i = 1
# full_df = pd.DataFrame()

# for file in files:
#   read_txt = pd.read_csv(f'/content/test/{file}', sep = ' ', header=None)
#   full_df = pd.concat([full_df, read_txt], axis=0)
#   i += 1

# full_df


# In[53]:


import io
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



mnb = MultinomialNB()
lrc = LogisticRegression()



import pickle
# # save the model to disk
# filename = 'linear_regression.sav'
# pickle.dump(model, open(filename, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open('linear_regression.sav', 'rb'))
# loaded_model = torch.load('linear_regression.sav', map_location='cpu')
  
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
          return super().find_class(module, name)

loaded_model = CPU_Unpickler(open('linear_regression.sav', 'rb')).load()#(open('linear_regression.sav', 'rb')) 
print(loaded_model)

# # Fake or not?



def check_fake(text):
  tokenized = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)
  padded = np.array([tokenized + [0]*(max_len-len(tokenized)) ])
  attention_mask = np.where(padded != 0, 1, 0)
  input_id = torch.tensor(padded)  
  attention_mask = torch.tensor(attention_mask)
  input_id, attention_mask = input_id.to(device), attention_mask.to(device)
  
  with torch.inference_mode():
      last_hidden_states = model(input_id, attention_mask=attention_mask)

  features = last_hidden_states[0][:,0,:].to(device)#.numpy()
  np.savetxt((f'test.txt'), features.cpu().numpy())
  
  vecorized = pd.read_csv(f'test.txt', sep = ' ', header=None)
  
  if loaded_model.predict(vecorized)[0]:
    return 'fake'
  else:
    return 'real'
  # return lrc.predict(vecorized)

