#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


# load packages
import pandas as pd
import numpy as np


# In[2]:


# load dataset
black_friday = pd.read_csv("black_friday.csv")

# create a dataset copy
df = black_friday.copy()


# ## Inicie sua análise a partir daqui

# In[3]:


# check if the dataset format is correct.
df.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[32]:


def q1():
    return df.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[7]:


df.Age.value_counts()


# In[36]:


def q2():
    women_age = black_friday[(black_friday.Gender == 'F') & (black_friday.Age == '26-35')].shape[0]
    return women_age


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[33]:


def q3():
    return len(df.User_ID.unique())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[11]:


def q4():
    return len(black_friday.dtypes.unique())


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[48]:


def q5():
    # get how many null values there are in each row
    rows_na = df.isna().sum(axis=1)
    
    # get how many rows have null values
    rows_na_sum = np.count_nonzero(rows_na)
    
    # get the %
    rows_na_perc = rows_na_sum/len(df)
    
    return rows_na_perc


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[14]:


def q6():
    return df.isna().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[125]:


def q7():
    return df.Product_Category_3.value_counts().index[0]


# In[126]:


q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[109]:


def q8():
    purch_norm = np.array(df["Purchase"]).reshape(-1, 1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    punch_normalized = scaler.fit_transform(purch_norm)
    df.Purchase = punch_normalized

    return df.Purchase.mean()


# In[79]:


q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[115]:


def q9():
    # standize the purchase column
    df.Purchase = (df.Purchase - df.Purchase.mean()) / df.Purchase.std()
    return len(df[(df.Purchase >= -1) & (df.Purchase <= 1)])


# In[112]:


q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[100]:


def q10():
    cat2_na = black_friday.loc[black_friday['Product_Category_2'].isna()]
    cat2_3 = bool(len(cat2_na) == cat2_na['Product_Category_3'].isna().sum())
    return cat2_3


# In[101]:


q10()

