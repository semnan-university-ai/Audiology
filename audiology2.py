#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns


# In[2]:


audio = pd.read_csv('audiology2.csv',header=0,sep=',')
audio.columns
df=audio


# In[3]:



df=df.drop("number_row",axis = 1)
var_mod = ['age_gt_60','airBoneGap','boneAbnormal','history_buzzing','history_dizziness',
       'history_fluctuating', 'history_fullness', 'history_heredity',
       'history_nausea', 'history_noise', 'history_recruitment',
       'history_ringing', 'history_roaring', 'history_vomiting',
       'late_wave_poor', 'm_at_2k', 'm_cond_lt_1k', 'm_gt_1k', 'm_m_gt_2k',
       'm_m_sn', 'm_m_sn_gt_1k', 'm_m_sn_gt_2k', 'm_m_sn_gt_500',
       'm_p_sn_gt_2k', 'm_s_gt_500', 'm_s_sn', 'm_s_sn_gt_1k', 'm_s_sn_gt_2k',
       'm_s_sn_gt_3k', 'm_s_sn_gt_4k', 'm_sn_2_3k', 'm_sn_gt_1k', 'm_sn_gt_2k',
       'm_sn_gt_3k', 'm_sn_gt_4k', 'm_sn_gt_500', 'm_sn_gt_6k', 'm_sn_lt_1k',
       'm_sn_lt_2k', 'm_sn_lt_3k', 'middle_wave_poor', 'mod_gt_4k',
       'mod_mixed', 'mod_s_mixed', 'mod_s_sn_gt_500', 'mod_sn', 'mod_sn_gt_1k',
       'mod_sn_gt_2k', 'mod_sn_gt_3k', 'mod_sn_gt_4k', 'mod_sn_gt_500',
       'notch_4k', 'notch_at_4k', 's_sn_gt_1k',
       's_sn_gt_2k', 's_sn_gt_4k', 'static_normal','viith_nerve_signs',
      'wave_V_delayed', 'waveform_ItoV_prolonged']

le = LabelEncoder()

for i in var_mod:

    df[i] = le.fit_transform(df[i])

df.head(10)


# In[4]:


df.info()


# In[5]:


df['speech()'].value_counts()


# In[6]:


le = LabelEncoder()
data_cat=df['speech()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=["speech()"])
df['speech()']=data_cat_encoded
df['speech()'].value_counts()


# In[7]:


df['air()'].value_counts()


# In[8]:


le = LabelEncoder()
data_cat=df['air()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['air()'])
df['air()']=data_cat_encoded
df['air()'].value_counts()


# In[9]:


df['ar_c()'].value_counts()


# In[10]:


le = LabelEncoder()
data_cat=df['ar_c()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['ar_c()'])
df['ar_c()']=data_cat_encoded
df['ar_c()'].value_counts()


# In[11]:


le = LabelEncoder()
data_cat=df['ar_u()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['ar_u()'])
df['ar_u()']=data_cat_encoded
df['ar_u()'].value_counts()


# In[12]:


df['bone()'].value_counts()


# In[13]:


le = LabelEncoder()
data_cat=df['bone()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['bone()'])
df['bone()']=data_cat_encoded
df['bone()'].value_counts()


# In[14]:


df['bser()'].value_counts()


# In[15]:


le = LabelEncoder()
data_cat=df['bser()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['bser()'])
df['bser()']=data_cat_encoded
df['bser()'].value_counts()


# In[16]:


df['o_ar_c()'].value_counts()
le = LabelEncoder()
data_cat=df['o_ar_c()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['o_ar_c()'])
df['o_ar_c()']=data_cat_encoded
df['o_ar_c()'].value_counts()


# In[17]:


df['o_ar_u()'].value_counts()
le = LabelEncoder()
data_cat=df['o_ar_u()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['o_ar_u()'])
df['o_ar_u()']=data_cat_encoded
df['o_ar_u()'].value_counts()


# In[18]:


df['tymp()'].value_counts()
le = LabelEncoder()
data_cat=df['tymp()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['tymp()'])
df['tymp()']=data_cat_encoded
df['tymp()'].value_counts()


# In[19]:


df['tymp()'].value_counts()


# In[20]:



le = LabelEncoder()
data_cat=df['tymp()']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['tymp()'])
df['tymp()']=data_cat_encoded
df['tymp()'].value_counts()


# In[21]:


df.info()


# 

# In[22]:



le = LabelEncoder()
data_cat=df['classification']
data_cat_encoded= le.fit_transform(data_cat)
data_cat_encoded= pd.DataFrame(data_cat_encoded,columns=['classification'])
df['classification']=data_cat_encoded
df['classification'].value_counts()


# In[23]:


df_label=df["classification"].copy()


# In[24]:


df=df.drop("bser()",axis = 1)


# In[25]:


from sklearn.preprocessing import StandardScaler

#feature_scal = StandardScaler()
#df = pd.DataFrame(feature_scal.fit_transform(df), columns=df.columns)
#df.head()
y=df.classification
x = df.drop(columns=['classification'])


# In[26]:


df=df.drop("classification",axis = 1)
median = df['mod_sn'].median()
df['mod_sn'].fillna(median)
df.head(50)


# In[27]:


x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.4,random_state=400)


# In[28]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
clft=DecisionTreeClassifier()
clft=clft.fit(x_train,y_train)
y_predt = clft.predict(x_test)


from sklearn import tree
plt.figure(figsize=(50,80))
temp = tree.plot_tree(clft.fit(x,y),fontsize=24)
plt.show()


# In[29]:


from sklearn.naive_bayes import GaussianNB
clfb = GaussianNB()
clfb.fit(x_train,y_train.ravel())
y_predb = clfb.predict(x_test)
print(classification_report(y_test,clfb.predict(x_test)))


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
k=1
clfk= KNeighborsClassifier(n_neighbors=k)
clfk.fit(x_train,y_train.ravel())
y_predk=clfk.predict(x_test)
print("when k = {} neighbors , knn test acuracy : {}" .format(k,clfk.score(x_test,y_test)))
print("when k = {} neighbors , knn test acuracy : {}" .format(k,clfk.score(x_train,y_train))) 
print(classification_report(y_test,clfk.predict(x_test)))
ran = np.arange(1,30)
train_list = []
test_list = []
for i,each in enumerate(ran):
    clfk= KNeighborsClassifier(n_neighbors=each)
    clfk.fit(x_train,y_train.ravel())
    


# In[31]:


from sklearn.neural_network import MLPClassifier
clfm = MLPClassifier(hidden_layer_sizes=(5,),max_iter=1500)
clfm.fit(x_train,y_train.ravel())
y_predm = clfm.predict(x_test)
print ("acuracy:", metrics.accuracy_score (y_test,y_predm))


# In[ ]:




