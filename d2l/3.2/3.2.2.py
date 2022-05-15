import  random
import torch
from d2l import torch as d2l

def date_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def synthetic_date(w, b, num_examples):    #@save
    """"生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

ture_w = torch.tensor([2, -3.4])
ture_b = 4.2
features, labels = synthetic_date(ture_w, ture_b, 1000)

batch_size = 10

for X, y in date_iter(batch_size, features, labels):
    print(X, '\n', y)
    break