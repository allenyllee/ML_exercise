
# coding: utf-8

# ## Umap

# In[1]:


import umap
from sklearn.datasets import load_digits

digits = load_digits()

embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(digits.data)


# In[2]:


digits


# In[4]:


import pandas as pd

df_umap = pd.DataFrame(digits.data).copy()
df_umap['label'] = digits.target.copy()
df_umap['label_str'] = df_umap['label'].apply(lambda x: str(x))


# In[5]:


df_umap


# In[7]:


from ggplot import *

df_umap['x-umap'] = embedding[:,0]
df_umap['y-umap'] = embedding[:,1]


chart = ggplot( df_umap, aes(x='x-umap', y='y-umap', color='label_str') )         + geom_point(size=70,alpha=0.5)         + ggtitle("umap dimensions colored by digit")
chart


# In[12]:





# In[23]:





# In[24]:


X[:,None,:] - X


# In[25]:





# In[26]:




