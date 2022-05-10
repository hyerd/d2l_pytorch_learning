import matplotlib.pyplot as plt
import numpy as np
import torch
from d2l import torch as d2l
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

x = np.arange(-5,5,0.02)
y = np.sin(x)
df=[]

for i in x:
    v = torch.tensor(i,requires_grad=True)
    y = torch.sin(v)
    y.backward()
    df.append(v.grad)

plt.plot(x,y, color="red")
plt.plot(x,df)
plt.show()