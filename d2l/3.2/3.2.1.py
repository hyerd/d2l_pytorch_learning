import  random
import torch
from d2l import torch as d2l

def synthetic_date(w, b, num_examples):    #@save
    """"生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

ture_w = torch.tensor([2, -3.4])
ture_b = 4.2
features, labels = synthetic_date(ture_w, ture_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1) #绘制散点图
d2l.plt.show()