from bayeformers import to_bayesian

import bayeformers.nn as bnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from tqdm import tqdm
from logging import Logger
import wandb

from omegaconf import OmegaConf
from typing import List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

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
    load_dir: Optional[str] = "./examples/ihead/saved_models/basic_bayesian_transformer.pth"
    save_model: bool = False
    save_dir: Optional[str] = "./examples/ihead/saved_models"
    root_dir: str = ''


# save_model = True model_args.k=5 

if __name__ == '__main__':

    torch.cuda.set_device(1)

    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs()
        )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())

    run = wandb.init(
        entity = "3233822097-peking-university",
        project = "BayeFormers",
        # group="test",
        # name="test",
        group="changed dataset",
        name = "last acc , k=5",
        config = OmegaConf.to_container(cfg)
    )

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

    if cfg.load_model:
        bayesian_model = to_bayesian(model , delta=0.1)
        bayesian_model.load_state_dict(torch.load(cfg.load_dir))
        print(f"Successfully loaded model from {cfg.load_dir}")
        bayesian_model.cuda()
    else:
        if cfg.pretrain:
            for epoch in tqdm(range(100), desc="Epoch"):
                pbar = tqdm(ParallelDataLoader(ds, batch_size=cfg.optim_args.batch_size,num_workers=cfg.num_data_workers, seed=cfg.seed , max_iters = cfg.max_iters), 
                    total = cfg.max_iters, 
                    desc = "Basic Transformer Training...")
                tot_loss = 0.0
                tot_nll = 0.0
                tot_log_prior = 0.0
                tot_log_variational_posterior = 0.0
                tot_acc = 0.0
                tot_acc_start = 0.0
                tot_acc_end = 0.0
                tot_acc_bigram = 0.0
                for (x , y , outs) in pbar:
                    # print(f"x shape {x.shape} , y shape {y.shape} , outs shape{outs.shape}"

                    x = torch.from_numpy(x).cuda()
                    y = torch.from_numpy(y).cuda()
                    outs = torch.from_numpy(outs).cuda()

                    optimizer.zero_grad()

                    pred = model(x)

                    if cfg.loss_head_only:
                        pred_for_loss = pred[outs >= 2].permute(0 , 2 , 1)
                        loss = F.nll_loss(pred_for_loss, y[outs >= 2] , reduction = "sum")
                    else:
                        pred_for_loss = pred.permute(0 , 2 , 1)
                        loss = F.nll_loss(pred_for_loss, y , reduction = "sum")

                    loss.backward()

                    optimizer.step()
                    # acc = (pred.argmax(-1)[outs >= 1] == y[outs >= 1]).float().mean().item()
                    # print(f"pred.shape : {pred.shape} , outs shape : {outs.shape} , y shape : {y.shape}")
                    # print(pred.argmax(-1)[: , -1] , y[: , -1])
                    # input()
                    acc = (pred.argmax(-1)[: , -1] == y[: , -1]).float().mean().item()
                    sl = 10
                    acc_start = (pred[:,:sl].argmax(-1)[outs[:,:sl] >= 1] == y[:,:sl][outs[:,:sl] >= 1]).float().mean().item()
                    el = 500
                    acc_end = (pred[:,-el:].argmax(-1)[outs[:,-el:] >= 2] == y[:,-el:][outs[:,-el:] >= 2]).float().mean().item()
                    acc_bigram = (pred.argmax(-1)[outs == 0] == y[outs == 0]).float().mean().item()


                    nb = cfg.max_iters
                    tot_nll += loss.item() / nb
                    tot_acc += acc / nb * 100
                    tot_acc_start += acc_start / nb * 100
                    tot_acc_end += acc_end / nb * 100
                    tot_acc_bigram += acc_bigram / nb * 100

                run.log({'nll' : loss , 'acc(%)' : tot_acc,
                        'acc_start(%)' : tot_acc_start , 'acc_end(%)' : tot_acc_end , 'acc_bigram(%)' : tot_acc_bigram})
        bayesian_model = to_bayesian(model , delta=0.1)
        bayesian_model.cuda()


    boptim = torch.optim.Adam(bayesian_model.parameters() , lr = cfg.optim_args.learning_rate)

    for epoch in tqdm(range(EPOCHS), desc="Epoch"):

        tot_loss = 0.0
        tot_nll = 0.0
        tot_log_prior = 0.0
        tot_log_variational_posterior = 0.0
        tot_acc = 0.0
        tot_acc_start = 0.0
        tot_acc_end = 0.0
        tot_acc_bigram = 0.0
        pbar = tqdm(ParallelDataLoader(ds_test, batch_size=cfg.optim_args.batch_size,num_workers=cfg.num_data_workers, seed=cfg.seed , max_iters = cfg.max_iters),
                    total = cfg.max_iters,
                    desc = "Test Epoch")
        for (x , y , outs) in pbar:

            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()
            outs = torch.from_numpy(outs).cuda()

            SAMPLES = 10
            predictions = torch.zeros(SAMPLES , cfg.optim_args.batch_size , *output_dim).cuda()
            log_prior = torch.zeros(SAMPLES , cfg.optim_args.batch_size).cuda()
            log_variational_posterior = torch.zeros(SAMPLES , cfg.optim_args.batch_size).cuda()

            boptim.zero_grad()

            for s in range(SAMPLES):
                pred = bayesian_model(x)
                predictions[s] = pred
                log_prior[s] = bayesian_model.log_prior()
                log_variational_posterior[s] = bayesian_model.log_variational_posterior()
            
            predictions = predictions.mean(0)
            log_prior = log_prior.mean()
            log_variational_posterior = log_variational_posterior.mean()

            # acc = (predictions.argmax(-1)[outs >= 1] == y[outs >= 1]).float().mean().item()
            acc = (pred.argmax(-1)[: , -1] == y[: , -1]).float().mean().item()
            sl = 10
            acc_start = (predictions[:,:sl].argmax(-1)[outs[:,:sl] >= 1] == y[:,:sl][outs[:,:sl] >= 1]).float().mean().item()
            el = 500
            acc_end = (predictions[:,-el:].argmax(-1)[outs[:,-el:] >= 2] == y[:,-el:][outs[:,-el:] >= 2]).float().mean().item()
            acc_bigram = (predictions.argmax(-1)[outs == 0] == y[outs == 0]).float().mean().item()


            if cfg.loss_head_only:
                pred_for_loss = predictions[outs >= 2].permute(0 , 2 , 1)
                nll = F.nll_loss(pred_for_loss, y[outs >= 2] , reduction = "sum")
            else:
                pred_for_loss = predictions.permute(0 , 2 , 1)
                nll = F.nll_loss(pred_for_loss, y , reduction = "sum")
            
            loss = (log_variational_posterior - log_prior) / cfg.max_iters + nll

            loss.backward()
            boptim.step()

            nb = cfg.max_iters
            print(f"loss : {loss.item()} , nll : {nll.item()} , log_prior : {log_prior.item()} , log_variational_posterior : {log_variational_posterior.item()}")
            tot_loss += loss.item() / nb
            tot_nll += nll.item() / nb
            tot_log_prior += log_prior.item() / nb
            tot_log_variational_posterior += log_variational_posterior.item() / nb
            tot_acc += acc / nb * 100
            tot_acc_start += acc_start / nb * 100
            tot_acc_end += acc_end / nb * 100
            tot_acc_bigram += acc_bigram / nb * 100

        if cfg.save_model and epoch % 100 == 0:
            torch.save(bayesian_model.state_dict(), Path(cfg.save_dir) / f"basic_bayesian_transformer_epoch_{epoch}.pth")
            print("Successfully SAVED at {}".format(Path(cfg.save_dir) / f"basic_bayesian_transformer_epoch_{epoch}.pth"))
        pbar.set_postfix(
            loss = tot_loss,
            nll = tot_nll,
            log_prior = tot_log_prior,
            log_variational_posterior = tot_log_variational_posterior,
            acc = f"{tot_acc:.2f}%",
            acc_start = f"{tot_acc_start:.2f}%",
            acc_end = f"{tot_acc_end:.2f}%",
            acc_bigram = f"{tot_acc_bigram:.2f}%"
        )
        
        run.log({'loss' : tot_loss , 'nll' : tot_nll , 'log_prior' : tot_log_prior , 'log_variational_posterior' : tot_log_variational_posterior , 'acc(%)' : tot_acc,
                 'acc_start(%)' : tot_acc_start , 'acc_end(%)' : tot_acc_end , 'acc_bigram(%)' : tot_acc_bigram})
        # logger.log(msg = f"epoch : {epoch} , loss : {tot_loss} , nll : {tot_nll} , log_prior : {tot_log_prior} , log_variational_posterior : {tot_log_variational_posterior} , acc = {tot_acc:.2f}%\n")
    
    if cfg.save_model:
        torch.save(bayesian_model.state_dict(), Path(cfg.save_dir) / "basic_bayesian_transformer.pth")
        print("Successfully SAVED at {}".format(Path(cfg.save_dir) / "basic_bayesian_transformer.pth"))