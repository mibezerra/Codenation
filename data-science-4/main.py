#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)
from sklearn.datasets import load_digits, fetch_20newsgroups


# In[2]:


'''# Algumas configurações para o matplotlib.
%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()'''


# In[3]:


# countries = pd.read_csv("countries.csv", decimal= ',')


# In[4]:


# column_names = [
#     "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
#     "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
#     "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
#     "Industry", "Service"
# ]

# countries.columns = column_names


# In[5]:


# colunas_float = list(countries.columns[countries.dtypes == 'float64'])


# In[6]:


colunas_float = ['Pop_density',
 'Coastline_ratio',
 'Net_migration',
 'Infant_mortality',
 'GDP',
 'Literacy',
 'Phones_per_1000',
 'Arable',
 'Crops',
 'Other',
 'Climate',
 'Birthrate',
 'Deathrate',
 'Agriculture',
 'Industry',
 'Service']


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# In[7]:


countries =   pd.read_csv("countries.csv")


# In[8]:


column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = column_names


# In[9]:


df = countries.copy()
df['Country'] = df.Country.str.strip()
df['Region'] = df.Region.str.strip()


# In[10]:


df[colunas_float].columns


# In[11]:


for col in colunas_float:
    df[col] = df[col].replace(',', '.', regex = True)


# In[12]:


df[colunas_float] = df[colunas_float].astype(float)


# ## Inicia sua análise a partir daqui

# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[13]:


lista = list(df.Region.sort_values().unique())


# In[14]:


def q1():
    return lista
    pass


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[15]:


from sklearn.preprocessing import KBinsDiscretizer


# In[16]:


q2_df = df.copy()


# In[17]:


discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
discretizer.fit(df[['Pop_density']])
q2_df['score_bins'] = discretizer.transform(df[['Pop_density']])


# In[18]:


#conferindo o percentil 90
percentil = np.percentile(q2_df['score_bins'], 90)

# df com dados acima do percentil
q2_df = q2_df[q2_df['score_bins'] > percentil]

q2_df.head()


# In[19]:


def q2():
    result = int(len(q2_df.Country.unique()))
    return result
    pass


# In[20]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[21]:


'''one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int, handle_unknown="ignore")
region_encoded = one_hot_encoder.fit_transform(df[['Region']])'''


# In[22]:


# region_encoded.shape[1]


# In[23]:


# region_result = region_encoded.shape[1]
# climate_result = len(df.Climate.unique())
# result_q3 = int(region_result + climate_result)


# In[24]:


def q3():
    return int(18)
    pass


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[25]:


num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard_scaler", StandardScaler())
])


# In[26]:


df_numeric = df[list(df.columns[(df.dtypes == 'int64') | (df.dtypes == 'float64')])]


# In[27]:


df_numeric.head()


# In[28]:


num_pipeline.fit_transform(df_numeric)


# In[29]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[30]:


test_country = np.array(test_country[2:]).reshape(1,-1)


# In[31]:


country_transformation = num_pipeline.transform(test_country)


# In[32]:


country_transformation


# In[33]:


#criando um df com as colunas e os resultados
q4_df = pd.DataFrame(country_transformation, columns=df_numeric.columns)
q4_df.head()


# In[34]:


q4_df.Arable


# In[35]:


def q4():
    result = round(float(q4_df.Arable),3)
    return result
    pass


# In[36]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[37]:


net_copy = df.Net_migration.copy()


# In[38]:


sns.boxplot(net_copy, orient = 'vertical')


# In[39]:


q1 = net_copy.quantile(0.25)
q3 = net_copy.quantile(0.75)
iqr = q3 - q1

out_abaixo = q1 - 1.5 * iqr
out_acima = q3 + 1.5 * iqr
non_outlier_interval_iqr= [out_abaixo, out_acima]

print(f"Faixa considerada \"normal\": {non_outlier_interval_iqr}")


# In[40]:


outliers_abaixo = net_copy[net_copy < out_abaixo]
outliers_acima = net_copy[net_copy > out_acima]
outliers_remover = net_copy[(net_copy<out_abaixo) | (net_copy>out_acima)]


# In[41]:


outliers_perc = len(outliers_remover)/len(net_copy)


# In[42]:


# conferindo se a % de outliers removidos é menor do que 5%
outliers_perc < 0.05


# In[43]:


def q5():
    return(tuple((len(outliers_abaixo), len(outliers_acima), outliers_perc < 0.05)))
    pass


# In[44]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[45]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[46]:


contador = CountVectorizer()
newsgroups_counts = contador.fit_transform(newsgroup.data)


# In[47]:


word = u'phone'
n_phone = contador.vocabulary_.get(f"{word.lower()}")


# In[48]:


newsgroups_counts[:,n_phone].sum()


# In[49]:


def q6():
    result = int(newsgroups_counts[:,n_phone].sum())
    return result
    pass


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[50]:


tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit_transform(newsgroup.data)

newsgroups_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroup.data)


# In[51]:


word = tfidf_vectorizer.get_feature_names().index(word)


# In[52]:


tf_idf_phone = round(newsgroups_tfidf_vectorized[:,word].sum(),3)


# In[53]:


tf_idf_phone


# In[54]:


def q7():
    return float(tf_idf_phone)
    pass

