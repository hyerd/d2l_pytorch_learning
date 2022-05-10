import torch
import numpy

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

#题目要求修改a
a1 = torch.randn(25,dtype=torch.float32).reshape(5,5)
a1.requires_grad = True #使用backward的时候必须将属性requires_grad设置为True
d = f(a1)
v = d.backward()
print(d,'\n',v)

""""""
#在修改requires_grad = True后仍然出现了grad can be implicitly created only for scalar outputs的错误
#由查阅得知计算梯度时outputs需为标量(未指明grad_outputs或grad_outputs为None时)
#由此得出使用backward时候inputs只能是张量
""""""