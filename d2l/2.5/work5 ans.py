import torch
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-3*np.pi, 3*np.pi, 100)
x=torch.tensor(x,requires_grad=True)
x1=x
y=torch.sin(x1)
y.sum().backward()

plt.plot(x.detach().numpy(),np.sin(x.detach().numpy()),label='sin(x)')
plt.plot(x.detach().numpy(),x.grad,label='df')
plt.show()
plt.legend()


