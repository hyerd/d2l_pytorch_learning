import torch
import numpy

def f(a):
    b = a ** 2 + a/2
    c = b
    return c

a = torch.randn(size=(),requires_grad=True)
d = f(a)
d.backward()
print(a.grad)

a.grad == d / a
print(a.grad)
