#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[53]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[54]:


'''%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()'''


# In[55]:


athletes = pd.read_csv("athletes.csv")


# In[56]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[57]:


df = athletes.copy()
df.info()


# In[58]:


df.head()


# In[59]:


df.sex.value_counts()


# In[60]:


df.nationality.value_counts()


# In[61]:


df.describe()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[62]:


height_sample = get_sample(df, 'height', n = 3000)


# In[63]:


sns.distplot(height_sample)


# In[64]:


def q1():
    shap = sct.shapiro(height_sample)
    shap = shap[1]
    result = shap > 0.05
    return result
    pass 


# In[65]:


q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[66]:


import statsmodels.api as sm
sm.qqplot(height_sample, fit = True, line = "45");


# In[67]:


sns.distplot(height_sample, bins = 25)


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[68]:


sct.jarque_bera(height_sample)


# In[69]:


def q2():
    jb = sct.jarque_bera(height_sample)
    jb = jb[1]
    result = bool(jb >0.05)
    return result
    pass


# In[70]:


q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[71]:


weight_sample = get_sample(df, 'weight', n = 3000)


# In[72]:


sct.normaltest(weight_sample)


# In[73]:


def q3():
    normal = sct.normaltest(weight_sample)
    normal = normal[1]
    result = bool(normal > 0.05)
    return result
    pass


# In[74]:


q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[75]:


sns.distplot(weight_sample, bins = 25)


# In[76]:


sns.boxplot(weight_sample)


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[77]:


weight_log = np.log(weight_sample)


# In[78]:


def q4():
    normal = sct.normaltest(weight_log)
    normal = normal[1]
    result = bool(normal > 0.05)
    return result
    pass


# In[79]:


q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[80]:


sns.distplot(weight_log, bins =25)


# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[28]:


bra = df[df.nationality == 'BRA']
usa = df[df.nationality == 'USA']
can = df[df.nationality == 'CAN']


# In[81]:


sct.ttest_ind(bra.height, usa.height, equal_var = False, nan_policy = 'omit')


# In[82]:


def q5():
    ttest = sct.ttest_ind(bra.height, usa.height, equal_var = False, nan_policy = 'omit')
    ttest = ttest[1]
    result = bool(ttest > 0.05)
    return result
    pass


# In[83]:


q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[84]:


def q6():
    ttest = sct.ttest_ind(bra.height, can.height, equal_var = False, nan_policy = 'omit')
    ttest = ttest[1]
    result = bool(ttest > 0.05)
    return result
    pass


# In[33]:


q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[52]:


def q7():
    ttest = sct.ttest_ind(can.height, usa.height, equal_var = False, nan_policy = 'omit')
    pvalue = round(ttest[1], 8)
    return float(pvalue)
    pass


# In[51]:


q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
