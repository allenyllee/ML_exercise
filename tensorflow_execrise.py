
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# $ y = 3x + b$

# In[27]:


X_in = np.linspace(0,1,200)
y_true = (3* x_in) + np.random.rand(len(x_in))
plt.plot(x_in, y_true, 'r.')
plt.show()


# In[44]:


tf.reset_default_graph()

w1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

y_pred = (w1 * X_in) + b1
loss = tf.reduce_mean(tf.square(y_pred - y_true))

optim = tf.train.GradientDescentOptimizer(0.1)
train_ops = optim.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in np.arange(500):
    sess.run(train_ops)
    if step % 25 == 0:
        print('step: ' + str(step) + ', weight: ' + str(sess.run(w1)[0]) + ', bias: ' + str(sess.run(b1)[0]))


# In[43]:


y_out = sess.run(y_pred)

plt.plot(X_in, y_true, 'r.')
plt.show()


plt.plot(X_in, y_true, 'r.')
plt.plot(X_in, y_out, 'b.')

plt.show()


# In[54]:


sess.close()


# ## Exercise
# ### 請依照上面的 case, 計算 $y = 2x^2 + 3x + c$
# 設定 x_in = np.linspace(0, 1, 100)

# In[71]:


x_in = np.linspace(0,1,100)
y_true = 2 * x_in**2 + 3 * x_in + np.random.rand(len(x_in))

plt.plot(x_in, y_true, 'r.')
plt.show()


# In[72]:


tf.reset_default_graph()

b1 = tf.Variable(tf.zeros([1]),dtype=tf.float32)
w1 = tf.Variable(tf.random_uniform([1],-1.0,1.0), dtype=tf.float32)
w2 = tf.Variable(tf.random_uniform([1],-1.0,1.0), dtype=tf.float32)
#b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

y_pred = b1 + w1 * x_in + w2 * x_in**2

loss = tf.reduce_mean(tf.square(y_pred - y_true))

optim = tf.train.GradientDescentOptimizer(0.1)

train_ops = optim.minimize(loss)


# In[73]:


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# In[74]:


for step in np.arange(500):
    sess.run(train_ops)
    if step % 25 == 0:
        print('step: ' + str(step) + ', b1: ' + str(sess.run(b1)) + ', w1: ' + str(sess.run(w1)) + ', w2: ' + str(sess.run(w2)))


# In[75]:


y_out = sess.run(y_pred)


# In[76]:


plt.plot(x_in, y_true, 'r.')
plt.plot(x_in, y_out, 'b.')

plt.show()


# In[77]:


sess.close()


# In[ ]:


### Exercise1: 
# 用自訂義的 variables and bias 做成 3 個 hidden layer (30, 25, 20) 的 Neural netowrk

### Exercise2: 
# 用 tf.layers 做成 3 個 hidden layer (25, 25, 25) 的 Neural netowrk


# In[8]:


from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt


# In[9]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--epochs', default = 100, type = int)
parser.add_argument('--lr', default = 0.001, type = float)
parser.add_argument('--train_ratio', default = 0.9, type = float)

FLAGS = parser.parse_args([]) # if not jupyter notebook, remove []


# In[55]:


datasize = 1000

x_train = np.random.rand(datasize,10)
y_train = np.random.randint(low=0, high=10, size=datasize)
y_train_one_hot = np.zeros((len(y_train),10))
y_train_one_hot[np.arange(len(y_train)), y_train]=1

x_valid = np.random.rand(int(datasize/10),10)
y_valid = np.random.randint(low=0, high=10, size=int(datasize/10))
y_valid_one_hot = np.zeros((len(y_valid),10))
y_valid_one_hot[np.arange(len(y_valid)),y_valid]=1


x_test = np.random.rand(int(datasize/10),10)
y_test = np.random.randint(low=0, high=10, size=int(datasize/10))
y_test_one_hot = np.zeros((len(y_test), 10))
y_test_one_hot[np.arange(len(y_test)), y_test]=1


# In[56]:


y_test_one_hot.shape


# In[57]:


tf.reset_default_graph()

with tf.name_scope('input'):
    x_input = tf.placeholder(shape=(None, x_train.shape[1]), 
                             name='x_input', 
                             dtype=tf.float32)
    y_out = tf.placeholder(shape=(None, y_train_one_hot.shape[1]), 
                           name='y_label', 
                           dtype=tf.float32)
    
with tf.variable_scope('hidden_layer1'):
    w1 = tf.get_variable(name='weight1', 
                         shape=[x_train.shape[1],30],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1))
    b1 = tf.Variable(tf.zeros(shape=[w1.shape[1]]))
    x_h1 = tf.nn.relu(tf.add(tf.matmul(x_input, w1), b1))
    
with tf.variable_scope('hidden_layer2'):
    w2 = tf.get_variable(name='weight2', 
                        shape=[x_h1.shape[1], 25], 
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.Variable(tf.zeros(shape=[w2.shape[1]]))
    
    x_h2 = tf.nn.relu(tf.add(tf.matmul(x_h1, w2),b2))
    

with tf.variable_scope('hidden_layer3'):
    #rint(type(x_h2.shape[1]))
    w3 = tf.get_variable(name='weight3', 
                        shape=[x_h2.shape[1],20],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b3 = tf.Variable(tf.zeros(shape=[w3.shape[1]]))
    x_h3 = tf.nn.relu(tf.add(tf.matmul(x_h2, w3), b3))
    
with tf.variable_scope('output_layer'):
    #rint(type(x_h3.shape[1]))
    # tf.Variable() 必須接受一個tensor 當作initial value, 所以不能傳入shape, 它不會幫你create
    # tf.get_variable() 可以傳入shape, initializer, 它會自動幫你create
    wo = tf.Variable(tf.truncated_normal(shape=[x_h3.shape[1].value, y_train_one_hot.shape[1]], stddev=0.1))
    bo = tf.Variable(tf.constant(0.0, shape=[wo.shape[1]]))
    output = tf.add(tf.matmul(x_h3, wo), bo)
    
with tf.name_scope('cross_entropy'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_out))
    
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(loss)
    
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(output),axis=1), tf.argmax(y_out,axis=1))
    compute_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[58]:


train_loss_list, valid_loss_list = [], []
train_acc_list, valid_acc_list = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in tqdm(range(FLAGS.epochs)):
        total_batch = int(np.floor(len(x_train)/FLAGS.batch_size))
        
        train_loss_collector, train_acc_collector = [], []
        
        for j in np.arange(total_batch):
            batch_idx_start = j*FLAGS.batch_size
            batch_idx_stop = (j+1)*FLAGS.batch_size
            
            x_batch = x_train[batch_idx_start:batch_idx_stop]
            y_batch = y_train_one_hot[batch_idx_start:batch_idx_stop]
            
            this_loss, this_acc, _ = sess.run([loss, compute_acc, train_step], 
                                                feed_dict={x_input:x_batch, y_out:y_batch})
            
            train_loss_collector.append(this_loss)
            train_acc_collector.append(this_acc)
            
        valid_loss, valid_acc = sess.run([loss, compute_acc],
                                        feed_dict={x_input:x_valid, y_out:y_valid_one_hot})
        
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        
        train_loss_list.append(np.mean(train_loss_collector))
        train_acc_list.append(np.mean(train_acc_collector))
        
        x_train, y_train = shuffle(x_train, y_train)
        
    test_loss, test_acc = sess.run([loss, compute_acc], feed_dict={x_input:x_test, y_out:y_test_one_hot})

print('---training done---')
print('testing accuracy: %.2f', test_acc)


# In[59]:


#train_loss_list


# In[60]:


plt.plot(np.arange(len(train_loss_list)), train_loss_list, 'b', label='train')
plt.plot(np.arange(len(valid_loss_list)), valid_loss_list, 'r', label='valid')

plt.legend()
plt.show()

plt.plot(np.arange(len(train_acc_list)), train_acc_list, 'b', label='train')
plt.plot(np.arange(len(valid_acc_list)), valid_acc_list, 'r', label='valid')

plt.legend(loc=4)
plt.show()


# In[28]:


get_ipython().system('pip install easydict')


# In[35]:


get_ipython().system('python -m visdom.server')


# In[41]:


import subprocess
import time

# create a new process for visdom.server
proc = subprocess.Popen(['python', '-m', 'visdom.server'])

# wait for visdom server startup
time.sleep(3)


# In[42]:


from visdom import Visdom
import numpy as np


# import argparse

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"

parser = argparse.ArgumentParser(description='Demo arguments')
parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                    help='port the visdom server is running on.')
parser.add_argument('-server', metavar='server', type=str,
                    default=DEFAULT_HOSTNAME,
                    help='Server address of the target to run the demo on.')
FLAGS = parser.parse_args(args=[])

# import easydict
# FLAGS = easydict.EasyDict({
#         'port': DEFAULT_PORT,
#         'server': DEFAULT_HOSTNAME
# })

print(FLAGS.port)
print(FLAGS.server)

#viz = Visdom()
viz = Visdom(port=FLAGS.port, server=FLAGS.server)


# In[43]:


assert viz.check_connection()
viz.close()


# In[44]:


## create a window
win=viz.line(
    X=np.array([0,1]),
    Y=np.array([0,1]),
    opts=dict(
        xtickmin=-2,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-1,
        ytickmax=10,
        ytickstep=1,
        markersymbol='dot',
        markersize=5,
    ),
    name="1"
)


# In[45]:


# loop for update line
import time
for a in range(10):
    viz.line(
        X=np.array([a]),
        Y=np.array([a]),
        win=win,
        name="1",
        update = 'append'
    )
    
    time.sleep(1)


# In[46]:


# kill visdom.server
time.sleep(3)
proc.kill()
print('server killed')

