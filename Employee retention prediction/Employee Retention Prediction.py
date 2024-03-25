#!/usr/bin/env python
# coding: utf-8

# ### Employee Retention Prediction
# 
# This project explores the application of deep learning to predict employee turnover and improve retention rates. The model utilizes KerasClassifier, a powerful deep learning library, to identify patterns in employee data that correlate with departure. To combat overfitting, a common challenge in machine learning, Dropout regularization is implemented. Dropout randomly drops out a percentage of neurons during training, preventing the model from becoming overly reliant on specific features and enhancing its generalizability to unseen data. This refined model aims to deliver more accurate predictions of employee flight risk, allowing HR departments to proactively implement targeted retention strategies.    

# In[3]:


#importing libraries

import pandas as pd 
import numpy as np
df=pd.read_csv('Downloads/hr.csv')


# In[5]:


df.head()


# In[7]:


#convert categorical to numeric values
feats=['department','salary']
df_final=pd.get_dummies(df,columns=feats,drop_first=True)


# In[9]:


df_final


# In[12]:


#Split data into training and test set
from sklearn.model_selection import train_test_split
X=df_final.drop(['left'],axis=1).values
y=df_final['left'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# ### Transforming the data

# In[16]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ###  Build the artificial neural network

# In[18]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[19]:


classifier=Sequential()


# In[20]:


classifier.add(Dense(9, kernel_initializer = "uniform",activation = "relu", input_dim=18))


# In[21]:


classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))


# In[22]:


classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])


# In[24]:


classifier.fit(X_train, y_train, batch_size = 10, epochs = 8)


# ### Improving the Model Accuracy

# In[27]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[28]:


def make_classifier():
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier


# In[29]:


classifier = KerasClassifier(build_fn = make_classifier, batch_size=10, nb_epoch=2)


# In[30]:


accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10,n_jobs = -1)


# In[31]:


mean = accuracies.mean()
mean


# ### Adding Dropout Regularization to Fight Over-Fitting

# In[32]:


from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])


# In[33]:


from sklearn.model_selection import GridSearchCV
def make_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= optimizer,loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier


# In[34]:


classifier = KerasClassifier(build_fn = make_classifier)


# In[35]:


params = {
    'batch_size':[20,35],
    'epochs':[2,3],
    'optimizer':['adam','rmsprop']
}


# In[36]:


grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring="accuracy",
                           cv=2)


# In[37]:


grid_search = grid_search.fit(X_train,y_train)


# In[38]:


best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[39]:


best_param


# In[40]:


best_accuracy


# In[ ]:




