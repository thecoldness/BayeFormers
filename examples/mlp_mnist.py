from torch import Tensor
from torch.optim import Adam,
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from bayeformers import to_bayesian
import bayeformers.nn as bnn
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb


# Model Class Definition (Frequentist Implementation)
class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, n_classes: int) -> None:
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden,      hidden), nn.ReLU(),
            nn.Linear(hidden,   n_classes), nn.LogSoftmax(dim=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp(input)

torch.cuda.set_device(7)


# Constants
EPOCHS     = 10
SAMPLES    = 10
BATCH_SIZE = 64
LR         = 1e-3
N_CLASSES  = 10
W, H       = 28, 28

# Dataset
transform = ToTensor()
dataset = MNIST(root="dataset", train=True, download=True, transform=transform)
loader = DataLoader(dataset,
    shuffle=True, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=True,
)

# Model and Optimizer
model = MLP(W * H, 512, N_CLASSES).cuda() # Frequentist
optim = Adam(model.parameters(), lr=LR)   # Adam Optimizer

run = wandb.init(
    entity = "3233822097-peking-university",
    project = "BayeFormers",
    name = "mlp_mnist_without_pretrained"
)
# Main Loop
# for epoch in tqdm(range(1), desc="Epoch"):
#     tot_loss = 0.0
#     tot_acc = 0.0

#     # Batch Loop
#     pbar = tqdm(loader, desc="Batch")
#     for img, label in pbar:
#         img, label = img.float().cuda(), label.long().cuda()

#         # Reset Grads
#         optim.zero_grad()
        
#         # Loss Computation
#         prediction = model(img.view(img.size(0), -1))
#         print(f"prediction shape{prediction.shape} , label shape{label.shape}")
#         loss = F.nll_loss(prediction, label, reduction="sum")
#         acc = (torch.argmax(prediction, dim=1) == label).sum()
        
#         # Weights Update
#         loss.backward()
#         optim.step()

#         # Display
#         tot_loss += loss.item() / len(loader)
#         tot_acc += acc.item() / len(dataset) * 100
        
#         pbar.set_postfix(loss=tot_loss, acc=f"{tot_acc:.2f}%")

# Convertion to Bayesian
bmodel = to_bayesian(model, delta=0.05)
boptim = Adam(bmodel.parameters(), lr=LR) 

tot_loss = 0.0
tot_nll = 0.0
tot_log_prior = 0.0
tot_log_variational_posterior = 0.0
tot_acc = 0.0

# Batch Loop Bayesian Eval
# pbar = tqdm(loader, desc="Bayesian Eval")
# for img, label in pbar:
#     img, label = img.float().cuda(), label.long().cuda()

#     # Setup Outputs
#     prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()
#     log_prior = torch.zeros(SAMPLES).cuda()
#     log_variational_posterior = torch.zeros(SAMPLES).cuda()

#     # Sample Loop (VI)
#     for s in range(SAMPLES):
#         prediction[s] = bmodel(img.view(img.size(0), -1))
#         log_prior[s] = bmodel.log_prior()
#         log_variational_posterior[s] = bmodel.log_variational_posterior()

#     # Loss Computation
#     log_prior = log_prior.mean()
#     log_variational_posterior = log_variational_posterior.mean()
#     nll = F.nll_loss(prediction.mean(0), label, reduction="sum")
#     # print(f"prediction shape{prediction.mean(0).shape} , label shape{label.shape}")
#     # print(f"log_prior shape{log_prior.shape} ; log_variational_posterior shape{log_variational_posterior.shape} , nll shape {nll.shape}")
#     loss = (log_variational_posterior - log_prior) / len(loader) + nll
#     acc = (torch.argmax(prediction.mean(0), dim=1) == label).sum()

#     # Display
#     nb = len(loader)
#     tot_loss += loss.item() / nb
#     tot_nll += nll.item() / nb
#     tot_log_prior += log_prior.item() / nb
#     tot_log_variational_posterior += log_variational_posterior.item() / nb
#     tot_acc += acc.item() / len(dataset) * 100

#     nb = len(loader)
#     pbar.set_postfix(
#         loss=tot_loss,
#         nll=tot_nll,
#         log_prior=tot_log_prior,
#         log_variational_posterior=tot_log_variational_posterior,
#         acc=f"{tot_acc:.2f}%"
#     )

# Main Loop
for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    tot_loss = 0.0
    tot_nll = 0.0
    tot_log_prior = 0.0
    tot_log_variational_posterior = 0.0
    tot_acc = 0.0

    # Batch Loop Bayesian Train
    pbar = tqdm(loader, desc="Bayesian")
    for img, label in pbar:
        img, label = img.float().cuda(), label.long().cuda()

        # Setup Outputs
        prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()
        log_prior = torch.zeros(SAMPLES).cuda()
        log_variational_posterior = torch.zeros(SAMPLES).cuda()

        boptim.zero_grad()

        # Sample Loop (VI)
        for s in range(SAMPLES):
            prediction[s] = bmodel(img.view(img.size(0), -1))
            log_prior[s] = bmodel.log_prior()
            log_variational_posterior[s] = bmodel.log_variational_posterior()

        # Loss Computation
        log_prior = log_prior.mean()
        log_variational_posterior = log_variational_posterior.mean()
        nll = F.nll_loss(prediction.mean(0), label, reduction="sum")
        loss = (log_variational_posterior - log_prior) / len(loader) + nll
        acc = (torch.argmax(prediction.mean(0), dim=1) == label).sum()
        acc_tmp = (torch.argmax(prediction.mean(0), dim=1) == label).float().mean()

        # Weights Update
        loss.backward()
        boptim.step()

        # Display
        nb = len(loader)
        tot_loss += loss.item() / nb
        tot_nll += nll.item() / nb
        tot_log_prior += log_prior.item() / nb
        tot_log_variational_posterior += log_variational_posterior.item() / nb
        tot_acc += acc.item() / len(dataset) * 100
        run.log({'loss' : loss.item() , 'nll' : nll.item() , 'log_prior' : log_prior.item() , 'log_variational_posterior' : log_variational_posterior.item() , 'acc' : acc_tmp.item()})
        nb = len(loader)
        pbar.set_postfix(
            loss=tot_loss,
            nll=tot_nll,
            log_prior=tot_log_prior,
            log_variational_posterior=tot_log_variational_posterior,
            acc=f"{tot_acc:.2f}%"
        )

