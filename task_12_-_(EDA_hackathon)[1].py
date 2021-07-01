#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np                # for mathematical computation
import pandas as pd               # for Data Manipulation
import matplotlib.pyplot as plt   # for Plotting 
import seaborn as sns             # for Plotting


# In[54]:


data=pd.read_csv(r'C:\Users\MANI\OneDrive\Desktop\data2.csv')
print(data)


# In[55]:


data.head()


# In[56]:


data.shape


# In[57]:


data.columns


# In[58]:


data.info()


# In[59]:


data.describe()


# In[60]:


numeric= ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_data= data.select_dtypes(include=numeric)
len(numeric_data.columns)


# In[61]:


data.isna().sum()


# In[62]:


data.drop(['Id','matchId','groupId'],axis=1, inplace=True)


# In[63]:


data.drop(['Unnamed: 0'],axis=1, inplace=True)


# In[64]:


data.head()


# In[65]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['matchType']=le.fit_transform(data["matchType"])


# In[66]:


data=round(data)
data.head()


# In[67]:


data.info()


# In[68]:


data.corr()


# In[72]:


corrmat =data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
## plot heatmap
g =sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")


# In[96]:


sns.distplot(data.assists)


# In[74]:


sns.distplot(data.boosts)


# In[75]:


sns.distplot(data.damageDealt)


# In[76]:


sns.distplot(data.headshotKills)


# In[77]:


sns.distplot(data.heals)


# In[78]:


sns.distplot(data.kills)


# In[79]:


sns.distplot(data.killStreaks)


# In[80]:


sns.distplot(data.revives)


# In[81]:


sns.distplot(data.rideDistance)


# In[82]:


sns.distplot(data.swimDistance)


# In[83]:


sns.distplot(data.walkDistance)


# In[84]:


sns.distplot(data.weaponsAcquired)


# In[85]:


# Bivariate Analysis


# In[86]:


plt.figure()
sns.pairplot(data, vars=['weaponsAcquired', 'winPlacePerc'])
plt.show()


# In[87]:


plt.figure()
sns.pairplot(data, vars=['assists', 'winPlacePerc'])
plt.show()


# In[88]:


plt.figure()
sns.pairplot(data, vars=['walkDistance', 'winPlacePerc'])
plt.show()


# In[89]:


plt.figure()
sns.pairplot(data, vars=['kills', 'winPlacePerc'])
plt.show()


# In[90]:


plt.figure()
sns.pairplot(data, vars=['boosts', 'winPlacePerc'])
plt.show()


# In[91]:


plt.figure()
sns.pairplot(data, vars=['headshotKills', 'winPlacePerc'])
plt.show()


# In[92]:


# Conclusions
# 1) Weaponsacquired is the most important column influencing winplaceperc(chicken dinner) alongside walk distance,boosts etc.
# 2) Average damage dealt is 130.
# 3) Average walk distance is 1115m.
# 4) Average assists is 1 to 2.
# 5) Average assists is 1 to 2.
# 6) Average ridedistance is 1800.
# 7) Average swimdistance is 100.
# 8) Average teamkills is 3.
# 9) Columns to include to obtain maximum accuracy

#* assists  
# * boosts  
# * damageDealt  
# * DBNOs  
# * headshotKills  
# * heals  
# * kills  
# * killStreaks  
# * longestKill  
# * revives  
# * ridedistance  
# * swimdistance  
# * walkdistance  
# * weaponsacquired


# In[ ]:





# In[ ]:





# In[ ]:




