#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt


# In[ ]:


df = pd.read_csv('EEG_data.csv')
df.head()


# In[ ]:


X = []
Y = []
for i in range(10):
    h= df[df['SubjectID'] == i]
    for j in range(10):
        x = h[h['VideoID'] == j]['Raw'][:100].values
        X.append(x.reshape(1,100))
        y = h[h['VideoID'] == j]['user-definedlabeln'].iloc[0]
        Y.append(y)


# In[ ]:


X = np.array(X).reshape(100,100)
Y = np.array(Y).reshape(100,1)
Y.shape


# In[ ]:


C0 = []
C1 = []
for i in range(Y.shape[0]):
    if Y[i] == 0:
        C0.append(X[i])
    else:
        C1.append(X[i])
C0,C1 = np.array(C0),np.array(C1)


# In[ ]:


C0.shape,C1.shape


# In[ ]:


def plot_signal_decomp(data,w):
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(1):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)
 
    rec_a = []
    rec_d = []
 
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))
 
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))
    return rec_a,rec_d


# In[ ]:


rec_a_c0,rec_d_c0 = plot_signal_decomp(C0,'sym5')
rec_a_c1,rec_d_c1 = plot_signal_decomp(C1,'sym5')


# In[ ]:


rec_a,rec_d = plot_signal_decomp(X,'sym5')


# In[ ]:


rec_d[9].shape


# In[ ]:


rec_d = np.array(rec_d)
rec_d.shape


# In[ ]:


plt.figure(figsize=(9,3))
plt.plot(rec_d[0][2],'k')


# In[ ]:


train = np.array(rec_d[9]).reshape(100,1032)


# In[ ]:


train.shape


# In[ ]:


data = pd.DataFrame(train)


# In[ ]:


data['Label'] = Y
data.head()
data.to_csv('Dataset2nd.csv')


# In[ ]:


plt.figure(figsize=(9,6))
plt.subplots_adjust(wspace =0, hspace =0.3)
plt.subplot(2,1,1)

plt.plot(rec_a_c0[0][0],color = 'k',label = 'Signal 1')
plt.xlabel('Time(s)',fontsize = 15)
plt.ylabel('Value',fontsize = 15)
plt.legend(loc = 'upper right',prop = {'size':15})

plt.subplot(2,1,2)
plt.plot(rec_a_c0[0][1],color = 'k',label = 'Signal 2')
plt.legend(loc = 'upper right',prop = {'size':15})
plt.xlabel('Time(s)',fontsize = 15)
plt.ylabel('Value',fontsize = 15)
plt.savefig('p2.eps',dpi=600,format='eps')
plt.show()


# In[ ]:


c0_mean = rec_d_c0[0].mean(axis = 0)
c1_mean = rec_d_c1[0].mean(axis = 0)


# In[ ]:


xa = np.arange(100)


# In[ ]:


overlap1 = []
for i in range(c0_mean.shape[0]):
    o = np.dot(c0_mean[:i],c1_mean[:i])
    overlap1.append(o)
overlap1 = np.array(overlap1)


# In[ ]:


overlap1


# In[ ]:


plt.figure(figsize = (9,6))
plt.plot(overlap1,c = 'k')

ax=plt.gca()
plt.axhline(y=0,c="k")
plt.xlabel('time(s)',fontsize = 15)
plt.ylabel('Dot product',fontsize = 15)
#ax.spines['bottom'].set_position(('data',0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('p3.eps',dpi=600,format='eps')
plt.show()


# In[ ]:


data.shape


# In[ ]:


data.to_csv('Dataset_df.csv')

