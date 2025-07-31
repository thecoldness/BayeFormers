from bayeformers import to_bayesian
from bayeformers.nn.parameters.gaussian import Gaussian


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import Logger
import weightwatcher as ww

from omegaconf import OmegaConf
from typing import List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import math
import wandb 

from dataclasses import dataclass
from examples.ihead.ihead_data import DataArgs, Dataset, ParallelDataLoader
from examples.ihead.ihead_basic_model import ModelArgs, Transformer

@dataclass
class OptimArgs:
    learning_rate: float = 1e-3  # for SGD
    weight_decay: float = 1e-4  # for SGD
    momentum: float = 0.9  # for SGD
    batch_size: int = 512
    use_sgd: bool = True  # otherwise use AdamW

@dataclass
class TrainerArgs:
    optim_args: OptimArgs
    data_args: DataArgs
    model_args: ModelArgs
    max_iters: Optional[int] = 10
    epoch:int=501
    pretrain:bool = True
    eval_delta: int = 5
    log_norms: bool = False
    log_probes: bool = False
    freeze_until: str = ''
    loss_head_only: bool = False
    bigram_outs_train: bool = False
    bigram_outs_test: bool = False
    num_data_workers: int = 60
    seed: int = 42
    load_model: bool = False
    load_dir: Optional[str] = "./examples/ihead/saved_models/"

# ---------------------------------------------------------------
# exact WW-layer style plot for one eigen-spectrum
# ---------------------------------------------------------------
def gram_eigs(weight: torch.Tensor) -> np.ndarray:
    """Return the sorted eigenvalues of WᵀW (CPU numpy)."""
    if weight.dim() > 2:              # conv weight -> flatten filters
        weight = weight.flatten(1)
    W = weight.detach().cpu().float()
    gram = W.t() @ W
    eigs = torch.linalg.eigvalsh(gram).cpu().numpy()
    return np.sort(eigs)

def hill_alpha(eigs: np.ndarray) -> float:
    eigs = eigs[eigs > 0]
    if eigs.size < 4:   # robust guard
        return math.nan
    k = eigs.size // 2
    tail = np.sort(eigs)[-k:]
    return 1.0 + k / np.sum(np.log(tail / tail.min()))

import powerlaw, numpy as np, matplotlib.pyplot as plt

if __name__ == '__main__':

    torch.cuda.set_device(2)
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs()
        )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())

    logger = Logger(f"./logs/{__name__}")

    ds = Dataset(cfg.data_args, train_test=None, bigram_outs=cfg.bigram_outs_train)
    ds_test = Dataset(cfg.data_args, train_test=None, bigram_outs=cfg.bigram_outs_test)
    ds_test.idxs = ds.idxs
    cfg.model_args.vocab_size = ds.num_tokens
    print(ds.num_tokens)
    output_dim = (DataArgs.seq_length , cfg.model_args.vocab_size)

    SAMPLES = 10
    EPOCHS = cfg.epoch

    model = Transformer(cfg.model_args)
    model.cuda()

    # initial param freezing
    freeze_until = defaultdict(list)
    to_freeze = []
    if cfg.freeze_until:
        for kv in cfg.freeze_until.split(','):
            k, v = kv.split(':')
            k = int(k)
            to_freeze.append(v)
            freeze_until[k].append(v)

        for name, p in model.named_parameters():
            if name in to_freeze:
                p.requires_grad_(False)
    
    optimizer = torch.optim.Adam(model.parameters() , lr = cfg.optim_args.learning_rate)

    # a test batch for experimentation
    x_exp, out_exp = ds.gen_batch(np.random.default_rng(0), 128)
    x_exp = x_exp[:,:ds.seq_length]

    bayesian_model = to_bayesian(model , delta=0.1)
    matrix_mu = {}
    matrix_sigma = {}

    load_dir = cfg.load_dir + f"/basic_bayesian_transformer.pth"
    bayesian_model.load_state_dict(torch.load(load_dir))
    print(f"Successfully loaded model from {load_dir}")
    bayesian_model.cuda()


    for name, module in bayesian_model.named_modules():
        if isinstance(module , Gaussian) and name.endswith("weight"):
            print("name : {name} , module : {module}".format(name=name, module=module))
            mu , sigma = module.mu.detach().clone() , F.softplus(module.rho).detach().clone()
            print(mu , sigma)