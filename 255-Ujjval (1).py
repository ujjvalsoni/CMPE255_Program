
# coding: utf-8

# In[150]:


import pandas as pd
import numpy as np


# In[151]:


file = open('C:/Users/Ujjval/Downloads/train.txt', 'r') 
data = file.readlines()
list1 = []
label = []
values = []
for i in range (14437):
    list1.append(data[i])
    
for i in list1:
    label.append(i[0])
    values.append(i[2:])


# In[152]:


# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(values)
# X_train_counts.shape


# In[153]:


# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_train_tfidf.shape


# In[154]:


file = open('C:/Users/Ujjval/Downloads/test.txt', 'r') 
data = file.readlines()
values1 = []

for i in range (14441):
    values1.append(data[i])


# In[155]:


# X_test_counts = count_vect.fit_transform(list1)
# X_test_counts.shape
# X_test_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_test_tfidf.shape


# In[156]:


from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
text_clf_svm = Pipeline([('vect', CountVectorizer()),  ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='hinge',penalty='l2', alpha=1e-3, n_iter=5, random_state=42)), ])
text_clf_svm = text_clf_svm.fit(values, label)
predicted = text_clf_svm.predict(values1)


# In[157]:


file = open('C:/Users/Ujjval/Downloads/solution.txt', 'w')
for i in range (14437):
    file.write(predicted[i])
    file.write('\n')
file.close()

