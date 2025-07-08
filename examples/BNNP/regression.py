import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from torchhk.vis import plot_individual_weight


from bayeformers import to_bayesian

SAMPLES = 5

x = torch.linspace(-2, 2, 500)
y = x.pow(3) - x.pow(2) + 3*torch.rand(x.size())
x = torch.unsqueeze(x, dim=1)
y = torch.unsqueeze(y, dim=1)

model = nn.Sequential(
    nn.Linear(in_features=1, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=1)
)

bmodel = to_bayesian(model, delta=1)

optimizer = optim.Adam(bmodel.parameters(), lr=0.01)

for step in range(3000):
    optimizer.zero_grad()    
    
    predictions = torch.zeros(SAMPLES, 500, 1)
    log_prior = torch.zeros(SAMPLES, 500)
    log_variational_posterior = torch.zeros(SAMPLES, 500)

    for s in range(SAMPLES):
        pred = bmodel(x)
        predictions[s] = pred
        log_prior[s] = bmodel.log_prior()
        log_variational_posterior[s] = bmodel.log_variational_posterior()
    
    predictions = predictions.mean(dim=0)
    log_prior = log_prior.mean()
    log_variational_posterior = log_variational_posterior.mean()

    loss = nn.MSELoss()(predictions, y) + (log_variational_posterior - log_prior)

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, MSE: {nn.MSELoss()(predictions, y).item()}, KL: {log_variational_posterior.item() - log_prior.item()}")
    
print('- loss : %2.2f, KL : %2.2f' % (loss.item(), log_variational_posterior.item() - log_prior.item()))

x_test = torch.linspace(-2, 2, 500)
y_test = x_test.pow(3) - x_test.pow(2) + 3*torch.rand(x_test.size())

x_test = torch.unsqueeze(x_test, dim=1)
y_test = torch.unsqueeze(y_test, dim=1)

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.scatter(x_test.data.numpy(), y_test.data.numpy(), color='k', s=2) 

y_predict = bmodel(x_test)
plt.plot(x_test.data.numpy(), y_predict.data.numpy(), 'r-', linewidth=5, label='First Prediction')

y_predict = bmodel(x_test)
plt.plot(x_test.data.numpy(), y_predict.data.numpy(), 'b-', linewidth=5, label='Second Prediction')

y_predict = bmodel(x_test)
plt.plot(x_test.data.numpy(), y_predict.data.numpy(), 'g-', linewidth=5, label='Third Prediction')

plt.legend()

pth = "./examples/BNNP/imgs/regression.png"
plt.savefig(pth, dpi=300, bbox_inches='tight')

plot_individual_weight(model)

# Step 0, MSE: 11.350439071655273, KL: 518.9851837158203
# Step 100, MSE: 2.1530747413635254, KL: 523.9781799316406
# Step 200, MSE: 1.042975664138794, KL: 523.9576110839844
# Step 300, MSE: 1.0004814863204956, KL: 545.8030395507812
# Step 400, MSE: 1.3631280660629272, KL: 559.5182342529297
# Step 500, MSE: 0.77909255027771, KL: 566.0138244628906
# Step 600, MSE: 0.8384753465652466, KL: 572.8477783203125
# Step 700, MSE: 0.917655348777771, KL: 578.9312591552734
# Step 800, MSE: 0.8132080435752869, KL: 589.4316101074219
# Step 900, MSE: 0.8100005984306335, KL: 603.5462493896484
# Step 1000, MSE: 0.9011963605880737, KL: 609.846435546875
# Step 1100, MSE: 0.8330076336860657, KL: 615.4623107910156
# Step 1200, MSE: 0.7523210644721985, KL: 629.4016418457031
# Step 1300, MSE: 0.7458189129829407, KL: 631.9787902832031
# Step 1400, MSE: 0.7670785188674927, KL: 629.8126525878906
# Step 1500, MSE: 0.7316269874572754, KL: 643.3224487304688
# Step 1600, MSE: 0.7576956748962402, KL: 639.1321105957031
# Step 1700, MSE: 0.7528874278068542, KL: 658.5649108886719
# Step 1800, MSE: 0.7303372621536255, KL: 654.8385009765625
# Step 1900, MSE: 0.750735878944397, KL: 663.0758972167969
# Step 2000, MSE: 0.7252606749534607, KL: 664.7065124511719
# Step 2100, MSE: 0.7298727035522461, KL: 685.2449645996094
# Step 2200, MSE: 0.7683209776878357, KL: 680.6800231933594
# Step 2300, MSE: 0.7801461815834045, KL: 680.4489440917969
# Step 2400, MSE: 0.7186729907989502, KL: 682.1875
# Step 2500, MSE: 0.7235652804374695, KL: 690.3659973144531
# Step 2600, MSE: 0.7214584350585938, KL: 691.0361022949219
# Step 2700, MSE: 0.7212315797805786, KL: 708.2536315917969
# Step 2800, MSE: 0.7262303829193115, KL: 719.1326293945312
# Step 2900, MSE: 0.7156280279159546, KL: 709.3507690429688