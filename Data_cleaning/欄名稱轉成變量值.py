
# coding: utf-8

# # 欄名稱轉成變量值
# 
# - [Pandas 与数据整理 | 张吉的博客](http://shzhangji.com/cnblogs/2017/09/30/pandas-and-tidy-data/)

# ## 列名称是数据值，而非变量名

# ### 宗教信仰与收入 - Pew 论坛

# #### 法1:使用stack

# In[1]:


import pandas as pd
df = pd.read_csv('data/pew.csv')
df.head(10)


# In[2]:


df = df.set_index('religion')
df = df.stack()
df.index = df.index.rename('income', level=1)
df.name = 'frequency'
df = df.reset_index()
df.head(10)


# #### 法2:使用melt

# In[4]:


df = pd.read_csv('data/pew.csv')
df = pd.melt(df, id_vars=['religion'], value_vars=list(df.columns)[1:],
             var_name='income', value_name='frequency')
df = df.sort_values(by='religion')
df.to_csv('data/pew-tidy.csv', index=False)
df.head(10)


# ### Billboard 2000

# In[26]:


df = pd.read_csv('data/billboard.csv')
df


# In[27]:


df = pd.melt(df, id_vars=list(df.columns)[:5], value_vars=list(df.columns)[5:],
             var_name='week', value_name='rank')

df['week'] = df['week'].str[2:].astype(int)
df['date.entered'] = pd.to_datetime(df['date.entered']) + pd.to_timedelta((df['week'] - 1) * 7, 'd')
df = df.rename(columns={'date.entered': 'date'})
df = df.sort_values(by=['track', 'date'])
df.to_csv('data/billboard-intermediate.csv', index=False)
df.head(10)


# ## 一列包含多个变量

# ### 结核病 (TB)

# In[7]:


df = pd.read_csv('data/tb.csv')
df


# In[8]:


df = pd.melt(df, id_vars=['country', 'year'], value_vars=list(df.columns)[2:],
             var_name='column', value_name='cases')
df = df[df['cases'] != '---']
df['cases'] = df['cases'].astype(int)
df['sex'] = df['column'].str[0]
df['age'] = df['column'].str[1:].map({
    '014': '0-14',
    '1524': '15-24',
    '2534': '25-34',
    '3544': '35-44',
    '4554': '45-54',
    '5564': '55-64',
    '65': '65+'
})
df = df[['country', 'year', 'sex', 'age', 'cases']]
df.to_csv('data/tb-tidy.csv', index=False)
df.head(10)


# ## 变量存储在行和列中

# ### 气象站

# In[28]:


df = pd.read_csv('data/weather.csv')
df


# In[29]:



df = pd.melt(df, id_vars=['id', 'year', 'month', 'element'],
             value_vars=list(df.columns)[4:],
             var_name='date', value_name='value')
df['date'] = df['date'].str[1:].astype('int')
df['date'] = df[['year', 'month', 'date']].apply(
    lambda row: '{:4d}-{:02d}-{:02d}'.format(*row),
    axis=1)
df = df.loc[df['value'] != '---', ['id', 'date', 'element', 'value']]
df = df.set_index(['id', 'date', 'element'])
df = df.unstack()
df.columns = list(df.columns.get_level_values('element'))
df = df.reset_index()
df.to_csv('data/weather-tidy.csv', index=False)
df


# ### 同一表中包含多种观测类型

# ### Billboard 2000

# In[30]:


df = pd.read_csv('data/billboard-intermediate.csv')
df


# In[31]:



df_track = df[['artist', 'track', 'time']].drop_duplicates()
df_track.insert(0, 'id', range(1, len(df_track) + 1))
df = pd.merge(df, df_track, on=['artist', 'track', 'time'])
df = df[['id', 'date', 'rank']]
df_track.to_csv('data/billboard-track.csv', index=False)
df.to_csv('data/billboard-rank.csv', index=False)
print(df_track, '\n\n', df)

