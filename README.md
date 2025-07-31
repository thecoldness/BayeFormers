This project is to generate ESD for BNN Transformers.

# Initialize

```
cd BayeFormers
conda env create -f env.yml
conda activate BayeFormers
```

# Run

```
cd BayeFormers
python -m examples.ihead.ihead_basic_main save_model=True data_args.k=1 pretrain=False
```

# Load Model and Visualize
```
cd BayeFormers
python -m examples.ihead.load_models load_dir="./examples/ihead/saved_models/changed dataset k=1 unpretrain"
```

## References

**Papers**
- "Weight Uncertainty in Neural Networks", Blundell et al., ICML 2015, [Arxiv](https://arxiv.org/abs/1505.05424)
- "Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes", Krishnan et al., AAAI 2020, [Arxiv](https://arxiv.org/abs/1906.05323v3)
- BayesFormer: Transformer with Uncertainty Estimation [Arxiv](https://arxiv.org/pdf/2206.00826)

**Articles**
- "Bayesian inference: How we are able to chase the Posterior", Ritchie Vink, [Blog](https://www.ritchievink.com/blog/2019/06/10/bayesian-inference-how-we-are-able-to-chase-the-posterior/)
- "Weight Uncertainty in Neural Networks", Nitarshan Rajkumar, [Blog](https://www.nitarshan.com/bayes-by-backprop/)