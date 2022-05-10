import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
sp = multinomial.Multinomial(1, fair_probs).sample()  # 模拟1次投掷
sp2 = multinomial.Multinomial(10, fair_probs).sample()  # 模拟10次投掷
sp3 = multinomial.Multinomial(10000, fair_probs).sample()  # 模拟10000次投掷

print(sp3 / 10000)  # 根据大数定理，重复次数越多越接近真实值 0.167

counts = multinomial.Multinomial(10, fair_probs).sample((5000,))  # 模拟10次投掷,进行5000次实验
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die =" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()
