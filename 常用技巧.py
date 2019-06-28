
# coding: utf-8

# ### random seed的作用範圍

# In[1]:


import numpy as np

np.random.seed(44)

ll = [3.2,77,4535,123,4]

print(np.random.choice(ll))
print(np.random.choice(ll))


# In[3]:


# in [11]
import numpy as np
# even if I set the seed here the other cells don't see it
np.random.seed(333)
np.random.choice([1,23,44,3,2])


# ### 建立一個 dict 包含 array，且每個 array 是獨立的 copy，修改某一個不會改到其他的

# In[4]:


import numpy as np
my_grid =  np.zeros((5, 5))

parse = "max","min","avg"

from collections import defaultdict

GridMetric = defaultdict(list)
for arg in parse:
    GridMetric[arg].append(my_grid.copy())
    
GridMetric['max'][0][1][1]=5
GridMetric


# ### dict 中，如果key不存在時會報錯，因此要先檢查key是否存在

# In[5]:


key_values = [('even',2),('odd',1),('even',8),('odd',3),('float',2.4),('odd',7)]

multi_dict = {}
for key,value in key_values:
    if key not in multi_dict:
        multi_dict[key] = [value]
    else:
        multi_dict[key].append(value)


# In[6]:


multi_dict


# ### x[1:] 相當於 x[:-1] 往後shift一個，因此如果要做數列前後相加除以二，可用 0.5 * (x[:-1] + x[1:])

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)
dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative


# In[ ]:


x[:-1]


# In[ ]:


x[1:]

