
# coding: utf-8

# # Multithreading

# ## without lock

# In[15]:


n = 0

def foo():
    global n
    n += 1


# In[16]:


import dis
dis.dis(foo)


# In[21]:


import threading

for _ in range(1000):
    n = 0
    threads = []
    for i in range(100):
        t = threading.Thread(target=foo)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(n)


# ## with lock

# In[12]:


n = 0
lock = threading.Lock()

def foo():
    global n
    with lock:
        n += 1


# In[14]:


import dis
dis.dis(foo)


# In[13]:


import threading

for _ in range(1000):
    n = 0
    threads = []
    for i in range(100):
        t = threading.Thread(target=foo)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(n)


# ## Parallelism with multiprocess (only unix)
# 
# - [os — Miscellaneous operating system interfaces — Python 3.7.2 documentation](https://docs.python.org/3/library/os.html#os.fork)
# 
#  os.fork()
# 
#     Fork a child process. Return 0 in the child and the child’s process id in the parent. If an error occurs OSError is raised.

# In[5]:


import os
import sys

nums =[1 for _ in range(1000000)]
chunk_size = len(nums) // 10
readers = []
pids = []

while nums:
    chunk, nums = nums[:chunk_size], nums[chunk_size:]
    reader, writer = os.pipe()
    pid = os.fork()
    if pid:
        pids.append(pid)
        readers.append(reader)  # Parent.
    else:
        # child
        subtotal = 0
        for i in chunk: # Intentionally slow code.
            subtotal += i

        print('subtotal %d' % subtotal)
        os.write(writer, str(subtotal).encode())
        #sys.exit(0) # must exit or child process will continue while loop
        break
        
if pid:
    # Parent.
    total = 0
    for reader in readers:
        subtotal = int(os.read(reader, 1000).decode())
        total += subtotal

    print("Total: %d" % total)

    # kill child process
    import os
    import signal
    for pid in pids:
        #Your Python code
        os.kill(pid, signal.SIGKILL)


# ## Parallelism with multiprocess (Unix & Windows)
# 
# - [python - how can i use fork() in Python3.3 - Stack Overflow](https://stackoverflow.com/questions/19547443/how-can-i-use-fork-in-python3-3)
# 
#     You should use the python's default multiprocessing package. It works with both Linux and Windows. 
# 
# - [multiprocessing — Process-based parallelism — Python 3.7.2 documentation](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.connection.Connection)
# 
# 

# In[22]:


from multiprocessing import Process, Pipe
import os

def Subtotal(chunk, writer):
    # child
    subtotal = 0
    for i in chunk: # Intentionally slow code.
        subtotal += i

    print('parent pid %d, pid %d, subtotal %d' % (os.getppid(), os.getpid(), subtotal))
    writer.send_bytes(str(subtotal).encode())


def total():
    # Parent.
    nums =[1 for _ in range(1000000)]
    chunk_size = len(nums) // 10
    readers = []
    pids = []

    # break down hard problem
    while nums:
        chunk, nums = nums[:chunk_size], nums[chunk_size:]
        reader, writer = Pipe()
        # spwan (windows) or fork(Unix) process
        p = Process(target=Subtotal, args=(chunk, writer))
        pids.append(p)
        readers.append(reader)

    # start child process
    for p in pids:
        p.start()

    # wait for child process until finish its job
    for p in pids:
        p.join()

    # get total
    total = 0
    for reader in readers:
        subtotal = int(reader.recv_bytes().decode())
        total += subtotal

    print("Total: %d" % total)

    # kill child process
    for p in pids:
        print('terminate pid', p.pid)
        p.terminate()
        print('pid %d is alive? %s' % (p.pid, p.is_alive()))


# In[23]:


total()

