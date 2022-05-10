import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([10]) / 10 #十面骰子

counts = multinomial.Multinomial(10, fair_probs).sample((500,))  # 模拟10次投掷,进行500次实验
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))#图表尺寸
for i in range(10):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die =" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.100, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()
