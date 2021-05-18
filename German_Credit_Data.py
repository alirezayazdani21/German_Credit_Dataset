#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://thinkingneuron.com/german-credit-risk-classification-case-study-in-python/


# In[2]:


import pandas as pd
import numpy as np

path='C:/Users/HANNAH_SOPHIE/Desktop/MISCELANEOUS/MISCELANEOUS/ml_quantitative_Python/ml_quantitative/CreditRiskData.csv'

CRDF=pd.read_csv(path, encoding='latin')
print('Shape before deleting duplicate values:', CRDF.shape)

# Removing duplicate rows if any
CRDF=CRDF.drop_duplicates()
print('Shape After deleting duplicate values:', CRDF.shape)

CRDF.head(10)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
GroupedData=CRDF.groupby('GoodCredit').size()
GroupedData.plot(kind='bar', figsize=(4,3));
GroupedData


# In[4]:


CRDF.describe(include='all')


# In[5]:


#Any NA in target?
#pd.isnull(CRDF["GoodCredit"])

CRDF["GoodCredit"].isnull().sum()


# In[ ]:


#from pandas_profiling import ProfileReport
#ProfileReport(CRDF, title="CRDF Profiling Report")


# In[ ]:





# In[6]:


from sklearn.model_selection import *


# In[7]:


CRDF_train, CRDF_test = train_test_split(CRDF, test_size=0.2)


# In[8]:


CRDF_train.describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
GroupedData=CRDF_train.groupby('GoodCredit').size()
GroupedData.plot(kind='bar', figsize=(4,3));
GroupedData


# In[10]:


CRDF_test.describe()


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
GroupedData=CRDF_test.groupby('GoodCredit').size()
GroupedData.plot(kind='bar', figsize=(4,3));
GroupedData


# In[12]:


import seaborn as sn
sn.pairplot(CRDF_train);


# In[15]:


from pycaret.classification import *


# In[16]:


pycaret.classification.models


# In[20]:


from imblearn.under_sampling import RandomUnderSampler
RUS = RandomUnderSampler()


# In[21]:


clf=setup(CRDF_train,  target = 'GoodCredit',
          fold_strategy='kfold',fold=10, 
          fix_imbalance=True, fix_imbalance_method = RUS,
          session_id=123)


# In[22]:


clf_fits=compare_models(include = ['lr','dt','svm','rf','xgboost'], sort='AUC')


# In[23]:


rf_CRDF=create_model('rf')


# In[24]:


rf_CRDF


# In[25]:


tuned_rf_CRDF = tune_model(rf_CRDF)


# In[26]:


tuned_rf_CRDF


# In[45]:


plot_model(tuned_rf_CRDF)


# In[32]:


plot_model(tuned_rf_CRDF, plot='feature')


# In[46]:


plot_model(tuned_rf_CRDF,plot='confusion_matrix')


# In[47]:


plot_model(tuned_rf_CRDF, plot = 'boundary')


# In[48]:


eval_rf_CRDF = evaluate_model(tuned_rf_CRDF)


# In[42]:


interpret_model(tuned_rf_CRDF)


# In[50]:


interpret_model(tuned_rf_CRDF,plot = 'reason')


# In[53]:


interpret_model(tuned_rf_CRDF,plot = 'reason', observation=55) #chose an arbitrary observation for local contribution analysis


# In[71]:


interpret_model(tuned_rf_CRDF,plot = 'reason', observation=150) #chose an arbitrary observation for local contribution analysis


# In[72]:


predict_model(tuned_rf_CRDF)


# In[ ]:


#final_et_bos = finalize_model(tuned_et_bos)


# In[ ]:


#print(final_et_bos)


# In[73]:


rf_CRED_train_pred=predict_model(tuned_rf_CRDF,data=CRDF_train)
rf_CRED_test_pred=predict_model(tuned_rf_CRDF,data=CRDF_test)


# In[74]:


rf_CRED_train_pred.head()


# In[77]:


rf_CRED_test_pred.head()


# In[78]:


from sklearn import metrics


# In[89]:


[metrics.accuracy_score(rf_CRED_train_pred['GoodCredit'], rf_CRED_train_pred['Label']),
 metrics.precision_score(rf_CRED_train_pred['GoodCredit'], rf_CRED_train_pred['Label']),
 metrics.recall_score(rf_CRED_train_pred['GoodCredit'], rf_CRED_train_pred['Label']),
 metrics.f1_score(rf_CRED_train_pred['GoodCredit'], rf_CRED_train_pred['Label'])]


# In[90]:


[metrics.accuracy_score(rf_CRED_test_pred['GoodCredit'], rf_CRED_test_pred['Label']),
 metrics.precision_score(rf_CRED_test_pred['GoodCredit'], rf_CRED_test_pred['Label']),
 metrics.recall_score(rf_CRED_test_pred['GoodCredit'], rf_CRED_test_pred['Label']),
 metrics.f1_score(rf_CRED_test_pred['GoodCredit'], rf_CRED_test_pred['Label'])]


# In[ ]:




