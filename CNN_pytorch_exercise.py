
# coding: utf-8

# In[1]:


from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)


# In[2]:


import torch
torch.cuda.is_available()


# In[1]:


import torch as t

from torch.autograd import Variable

N, D, H = 3,4,5

x = Variable(t.randn(N,D))
w1 = Variable(t.randn(D,H))
w2 = Variable(t.randn(D,H))

z=10

if z<0:
    y = x.mm(w1)
else:
    y = x.mm(w2)

print(x)
print(y)


# In[2]:


x = t.Tensor(5,3)

x


# In[3]:


x = t.rand(5,3)

x


# In[4]:


print(x.size()[1])


# In[5]:


x


# In[6]:


y = t.rand(5,3)
y


# In[7]:


x+y


# In[8]:


t.add(x,y)


# In[9]:


result2 = t.Tensor(5,3)
t.add(x,y,out=result2)

result2


# In[10]:


y2 = y.add(x)
print(y)


# In[11]:


y2.add_(x)
print(y2)


# In[12]:


a = t.ones(5)

a


# In[13]:


b = a.numpy() # Tensor -> numpy

b


# In[14]:


import numpy as np

a = np.ones(5)
b = t.from_numpy(a)



print(a)
print(b)


# In[15]:


b.add_(1) # a,b 共用記憶體, 修改b,也會改到a


print(a)
print(b)


# In[18]:


if t.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    
    print(x+y)


# In[19]:


from torch.autograd import Variable

x = Variable(t.ones(2,2), requires_grad = True)

x


# In[22]:


y = x.sum()

y


# In[23]:


y.grad_fn


# In[46]:


y.backward()


# In[47]:


x.grad


# In[43]:


x.grad.zero_()


# In[48]:


x = Variable(t.ones(4,5))

y = t.cos(x)

x_tensor_cos = t.cos(x.data)

print(y)

print(x_tensor_cos)


# In[49]:


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
net = Net()
print(net)


# In[52]:


params = list(net.parameters())
print(len(params))
print(params)


# In[54]:


for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())


# In[68]:


input = Variable(t.randn(1,1,32,32))
out = net(input)
out.size()


# In[69]:


out


# In[70]:


net.zero_grad()


# In[71]:


out.backward(Variable(t.ones(1,10)))


# In[98]:


output = net(input)
target = Variable(t.Tensor(np.arange(0,10)))

print(output)
print(target)

criterion = nn.MSELoss()
loss = criterion(output, target)

loss


# In[99]:


loss.grad_fn


# In[100]:


net.zero_grad()

print(net.conv1.bias.grad)

loss.backward()

print(net.conv1.bias.grad)


# In[117]:


learning_rate = 0.01

for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    #rint(dir(f))
    print(f.grad)


# In[120]:


import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01)

optimizer.zero_grad()

output = net(input)
loss = criterion(output, target)

loss.backward()

optimizer.step()


# In[121]:


import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show = ToPILImage()


# In[201]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])


# In[202]:


get_ipython().system('ls .')


# In[203]:


trainset = tv.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform)


# In[152]:





# In[131]:


trainloader = t.utils.data.DataLoader(
                trainset,
                batch_size=4,
                shuffle=True,
                num_workers=2)


# In[133]:


testset = tv.datasets.CIFAR10(
            './data',
            train=False, 
            download=True, 
            transform=transform)


# In[136]:


testloader = t.utils.data.DataLoader(
                testset,
                batch_size=4,
                shuffle=False,
                num_workers=2)


# In[142]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[219]:


(data, label) = trainset[1]


# In[220]:


print(classes[label])

show(data).resize((100,100))


# In[221]:


show((data+1)/2).resize((100,100))


# In[222]:


data


# In[223]:


(data+1)/2


# In[230]:


dataiter = iter(trainloader)

images, labels = dataiter.next()

print(' '.join('%11s'%classes[labels[j]] for j in range(4)))


show(tv.utils.make_grid((images+1)/2)).resize((400,100))


# In[231]:


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
net = Net()
print(net)


# In[233]:


from torch import optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[242]:


for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        #print(loss.data)
        running_loss += loss.data
        
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f'                   % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')
        


# In[245]:


dataiter = iter(testloader)

images, labels = dataiter.next()

print('實際的label:', ' '.join(                           '%08s'%classes[labels[j]] for j in range(4)))

show(tv.utils.make_grid(images/2 -0.5)).resize((400,100))


# In[248]:


outputs = net(Variable(images))

_, predicted = t.max(outputs.data, 1)

print('預測結果:', ' '.join('%5s'                       % classes[predicted[j]] for j in range(4)) )


# In[249]:


t.max(outputs.data, 1)


# In[250]:


outputs.data


# In[255]:


correct = 0
total = 0

for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)
    
    #print(labels.size())
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('10000張測試中的準確率: %d %%' % (100 * correct / total))


# In[304]:


trainloader = t.utils.data.DataLoader(
                trainset,
                batch_size=4,
                shuffle=True,
                num_workers=2)

print(len(trainloader))

net = Net()
print(net)


# In[308]:


if t.cuda.is_available():
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            #print(loss.data)
            running_loss += loss.data

            num_iter = len(trainloader)/10
            if i % num_iter == num_iter - 1:
                print('[%d, %5d] loss: %.3f'                       % (epoch+1, i+1, running_loss/num_iter))
                running_loss = 0.0

    print('Finished Training')


# In[309]:


testloader = t.utils.data.DataLoader(
                testset,
                batch_size=4,
                shuffle=False,
                num_workers=2)

correct = 0
total = 0

for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)
    
    #print(labels.size())
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('10000張測試中的準確率: %d %%' % (100 * correct / total))


# In[311]:


a = t.Tensor(2,3)

a


# In[313]:


b = t.Tensor([[1,2,3],[4,5,6]])
b


# In[315]:


b.tolist()


# In[317]:


b_size = b.size()
b_size


# In[318]:


b.numel()


# In[320]:


b.nelement()


# In[322]:


c = t.Tensor(b_size)
c


# In[325]:


d = t.Tensor((2,3))
d


# In[326]:


c.shape


# In[2]:


import torch as t


# In[3]:



a = t.arange(0,6)

a.view(2,3)


# In[11]:


b = a.view(-1,3)
b


# In[16]:


b.unsqueeze(1)


# In[17]:


b.unsqueeze(-2)


# In[28]:


c = b.view(1,1,1,2,3)
c


# In[34]:


c.squeeze(0)


# In[5]:


import tensorflow as tf
import numpy as np

N, D, H = 3,4,5

x = tf.placeholder(tf.float32, shape=(N,D))
z = tf.placeholder(tf.float32, shape=None)
w1 = tf.placeholder(tf.float32, shape=(D,H))
w2 = tf.placeholder(tf.float32, shape=(D,H))

def f1():
    return tf.matmul(x,w1)

def f2():
    return tf.matmul(x,w2)

y = tf.cond(tf.less(z,0), f1, f2)

variables = {
    x: np.random.randn(N,D),
    z: 10,
    w1: np.random.randn(D,H),
    w2: np.random.randn(D,H)
}

with tf.Session() as sess:
    y_out = sess.run(y, feed_dict=variables)

print(y_out)


# In[12]:


get_ipython().run_line_magic('pinfo2', 't.abs')

