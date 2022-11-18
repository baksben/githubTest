#!/usr/bin/env python
# coding: utf-8
Churn (target): 1 if customer cancelled service, 0 if not
AccountWeeks: number of weeks customer has had active account
ContractRenewal: 1 if customer recently renewed contract, 0 if not
DataPlan: 1 if customer has data plan, 0 if not
DataUsage: gigabytes of monthly data usage
CustServCalls: number of calls into customer service
DayMins: average daytime minutes per month
DayCalls: average number of daytime calls
MonthlyCharge: average monthly bill
OverageFee: largest overage fee in last 12 months
RoamMins: average number of roaming minutes
# In[1]:


import numpy as  np
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# In[2]:


df = pd.read_csv("churn_data.csv", sep = ';')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


cdf = df[['DataUsage','CustServCalls','MonthlyCharge','OverageFee','RoamMins','AccountWeeks', 'Churn']]
cdf.head(4)


# In[7]:


sns.pairplot(data = df, y_vars = ["Churn"], x_vars = ['DataUsage','CustServCalls','MonthlyCharge','OverageFee','RoamMins','AccountWeeks'],  hue = "Churn", palette = "Set2")


# In[8]:


sns.pairplot(data = df, y_vars = ["Churn"], x_vars = ['AccountWeeks','ContractRenewal','DataPlan','DayMins','DayCalls'],  hue = "Churn", palette = "Set2")


# In[9]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True);
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
# In[10]:


df.columns


# In[11]:


X = df[['Churn', 'AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage',
       'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 'OverageFee',
       'RoamMins']] .values  #.astype(float)
X[0:]


# In[12]:


y = df['Churn'].values
y[0:]


# In[13]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:]


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # Classification

# K nearest neighbor (KNN)

# In[15]:


from sklearn.neighbors import KNeighborsClassifier


# # Training

# Let's start the algorithm with k=4 for now:

# In[16]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# # Predicting

# We can use the model to make predictions on the test set:

# In[17]:


yhat = neigh.predict(X_test)
yhat[0:]


# # ### Accuracy evaluation
# 
# In multilabel classification, **accuracy classification score** is a function that computes subset accuracy. This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.

# In[18]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# # What about other K?
# K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the user. So, how can we choose right value for K? The general solution is to reserve a part of your data for testing the accuracy of the model. Then choose k =1, use the training part for modeling, and calculate the accuracy of prediction using all samples in your test set. Repeat this process, increasing the k, and see which k is the best for your model.
# 
# We can calculate the accuracy of KNN for different values of k.

# In[19]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[20]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[21]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:




