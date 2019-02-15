
# coding: utf-8

# # Numba vs. Cython
# 
# - [Numba vs. Cython: Take 2 | Pythonic Perambulations](https://jakevdp.github.io/blog/2013/06/15/numba-vs-cython-take-2/)

# ## install numba

# In[7]:


get_ipython().system('pip install numba')


# ## generate random data

# In[8]:


import numpy as np
X = np.random.random((1000, 3))


# In[9]:


X


# In[10]:


X[:,None,:]


# In[11]:


X[:, None, :] - X


# ## Numpy Boardcasting version of pairwise distance

# In[12]:


def pairwise_numpy(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))
get_ipython().run_line_magic('timeit', 'pairwise_numpy(X)')


# ## Pure Python for-loop version of pairwise distance

# In[15]:


def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D
get_ipython().run_line_magic('timeit', 'pairwise_python(X)')


# ## Numba version of pairwise distance

# In[16]:


from numba import double
from numba.decorators import jit, autojit

pairwise_numba = autojit(pairwise_python)

get_ipython().run_line_magic('timeit', 'pairwise_numba(X)')


# ## Cython version of pairwise distance
# 
# - [Cython in Ipython: ERROR: Cell magic `%%cython` not found - Stack Overflow](https://stackoverflow.com/questions/36514338/cython-in-ipython-error-cell-magic-cython-not-found)

# In[20]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[21]:


get_ipython().run_cell_magic('cython', '', '\nimport numpy as np\ncimport cython\nfrom libc.math cimport sqrt\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef pairwise_cython(double[:, ::1] X):\n    cdef int M = X.shape[0]\n    cdef int N = X.shape[1]\n    cdef double tmp, d\n    cdef double[:, ::1] D = np.empty((M, M), dtype=np.float64)\n    for i in range(M):\n        for j in range(M):\n            d = 0.0\n            for k in range(N):\n                tmp = X[i, k] - X[j, k]\n                d += tmp * tmp\n            D[i, j] = sqrt(d)\n    return np.asarray(D)')


# In[22]:


get_ipython().run_line_magic('timeit', 'pairwise_cython(X)')


# In[35]:




