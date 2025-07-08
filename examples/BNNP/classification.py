import numpy as np
from sklearn import datasets
from bayeformers import to_bayesian


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt


iris = datasets.load_iris()

X = iris.data
Y = iris.target 
SAMPLES = 5

x, y = torch.from_numpy(X).float(), torch.from_numpy(Y).long()
y = y.unsqueeze(0)

model = nn.Sequential(
    nn.Linear(in_features=4, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=3),
    nn.LogSoftmax(dim=-1)
)

bmodel = to_bayesian(model , delta = 0.05) 

optimizer = optim.Adam(bmodel.parameters(), lr=0.01)

for step in range(3000):
    optimizer.zero_grad()    
    
    predictions = torch.zeros(SAMPLES, 150, 3)
    log_prior = torch.zeros(SAMPLES, 150)
    log_variational_posterior = torch.zeros(SAMPLES, 150)

    for s in range(SAMPLES):
        pred = bmodel(x)
        predictions[s] = pred
        log_prior[s] = bmodel.log_prior()
        log_variational_posterior[s] = bmodel.log_variational_posterior()
    
    predictions = predictions.mean(dim=0)
    log_prior = log_prior.mean()
    log_variational_posterior = log_variational_posterior.mean()
    predictions = predictions.unsqueeze(0)
    loss = F.nll_loss(predictions.permute(0 , 2 , 1), y, reduction="sum") + (log_variational_posterior - log_prior)

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

predictions = predictions.squeeze(0)
y = y.squeeze(0)
_, predicted = torch.max(predictions.data, 1)
total = y.size(0)
correct = (predicted == y).sum()
print('- Accuracy: %f %%' % (100 * float(correct) / total))
print('- Loss : %2.2f, KL : %2.2f' % (loss.item(), log_variational_posterior.item() - log_prior.item()))

def draw_plot(predicted) :
    fig = plt.figure(figsize = (16, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    z1_plot = ax1.scatter(X[:, 0], X[:, 1], c = Y)
    z2_plot = ax2.scatter(X[:, 0], X[:, 1], c = predicted)

    plt.colorbar(z1_plot,ax=ax1)
    plt.colorbar(z2_plot,ax=ax2)

    ax1.set_title("REAL")
    ax2.set_title("PREDICT")

    pth = "./examples/BNNP/imgs/bnn_classification.png"
    fig.savefig(pth)


pre = model(x)
_, predicted = torch.max(pre.data, 1)
draw_plot(predicted)