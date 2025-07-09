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
    epoch:int=1000
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
    load_dir: Optional[str] = "./examples/ihead/saved_models/basic_bayesian_transformer.pth"
    save_model: bool = False
    save_dir: Optional[str] = "./examples/ihead/saved_models"
    root_dir: str = ''

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

_blue  = '#91bfdb'
_red   = '#fc8d59'

def ww_style_esd(eigs: np.ndarray,
                 distribution: str = "power_law",
                 title_tag: str   = "",
                 xlim: tuple[float, float] = (1e-2, 1e2),
                 savefile: str | None = None,
                 show: bool = True) -> None:
    """
    Parameters
    ----------
    eigs         : 1-D array of eigenvalues (>0).
    distribution : one of {"power_law", "truncated_power_law",
                           "lognormal", "exponential"}.
    title_tag    : extra string to prepend to the automatic title.
    xlim         : x-axis limits.
    savefile     : if given, save figure to this path (pdf/png inferred).
    show         : whether to call plt.show().
    """

    eigs = np.asarray(eigs, dtype=float)
    eigs = eigs[eigs > 0]                           # keep positive only
    if eigs.size < 5:
        raise ValueError("Need at least 5 positive eigenvalues.")

    # ---------- power-law fit ------------------------------------------------
    EVALS_THRESH = 1e-5
    nz_eigs = eigs[eigs > EVALS_THRESH]
    xmax    = nz_eigs.max()

    # heuristic xmin range (same as notebook)
    h, edges = np.histogram(np.log10(nz_eigs), bins=100)
    xmin_guess = 10 ** edges[np.argmax(h)]
    xmin_range = (0.95 * xmin_guess, 1.05 * xmin_guess)

    if distribution == "power_law":
        fit = powerlaw.Fit(nz_eigs, xmax=xmax, verbose=False)
    else:
        fit = powerlaw.Fit(nz_eigs, xmin=xmin_range,
                           xmax=xmax, verbose=False,
                           xmin_distribution=distribution)

    xmin, xmax = fit.xmin, fit.xmax

    # ---------- plot ---------------------------------------------------------
    plt.figure(figsize=(10, 5))

    # blue log-binned histogram
    hist, bins = np.histogram(eigs, bins=100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(eigs, bins=logbins, color=_blue, density=True)

    # red vertical x_min
    plt.axvline(xmin, color=_red, linewidth=2, label=r"$x_{\min}$")

    # red solid empirical PDF returned by powerlaw
    fit.plot_pdf(color=_red, linewidth=2)

    # red dashed model PDF
    if   distribution == "power_law":
        fit.power_law.plot_pdf(color=_red, linestyle="--")
    elif distribution == "truncated_power_law":
        fit.truncated_power_law.plot_pdf(color=_red, linestyle="--")
    elif distribution == "lognormal":
        fit.lognormal.plot_pdf(color=_red, linestyle="--")
    elif distribution == "exponential":
        fit.exponential.plot_pdf(color=_red, linestyle="--")

    # axes & labels
    plt.xscale("log"); plt.yscale("log")
    plt.xlim(xlim);    plt.ylim(1e-4, 10)
    plt.xlabel("Eigenvalues of correlation matrix", fontsize=14)
    plt.ylabel("ESD", fontsize=14)

    # title exactly like the WW file
    alpha = f"{fit.alpha:.2f}"
    KS    = f"{fit.D:.2f}"
    if distribution == "power_law":
        title = f"{title_tag}alpha={alpha}, ks_distance={KS}"
    elif distribution == "truncated_power_law":
        lam = f"{fit.truncated_power_law.Lambda:.2f}"
        title = (f"{title_tag}E-TPL beta={alpha}, ks_distance={KS}\n"
                 f"E-TPL lambda={lam}")
    elif distribution == "lognormal":
        sigma = f"{fit.sigma:.2f}"
        title = f"{title_tag}sigma={sigma}, ks_distance={KS}"
    else:  # exponential
        lam = f"{fit.exponential.Lambda:.2f}"
        title = f"{title_tag}exponent={lam}, ks_distance={KS}"

    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':

    torch.cuda.set_device(1)

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

    # OOD test data
    x_test, out_test = ds_test.gen_batch(np.random.default_rng(0), 512)
    x_t = torch.from_numpy(x_test[:,:ds.seq_length]).cuda()
    y_t = torch.from_numpy(x_test[:,1:ds.seq_length + 1]).cuda()
    outs_t = torch.from_numpy(out_test[:,:ds.seq_length]).cuda()

    # ----- Read trained model -----
    bayesian_model = to_bayesian(model , delta=0.1)
    bayesian_model.load_state_dict(torch.load(cfg.load_dir))
    print(f"Successfully loaded model from {cfg.load_dir}")
    bayesian_model.cuda()

    print(bayesian_model.bayesian_children)
    print(bayesian_model.named_modules)

    for name, module in bayesian_model.named_modules():
        if isinstance(module , Gaussian) and name.endswith("weight"):
            print("name : {name} , module : {module}".format(name=name, module=module))
            mu , sigma = module.mu.detach() , F.softplus(module.rho).detach()
            # mu , sigma = mu.cpu().numpy(), sigma.cpu().numpy()
            # print(f"mu.shape: {mu.shape}, sigma.shape: {sigma.shape}")
            eigs = gram_eigs(mu)
            out = f"./examples/ihead/analyze/{name}_esd.pdf"
            ww_style_esd(eigs,
                distribution="power_law",
                title_tag=f"Layer:{name} ",
                xlim=(0.8, 100),
                savefile=out,
                show=True)