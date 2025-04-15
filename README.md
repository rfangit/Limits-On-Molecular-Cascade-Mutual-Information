<h1 align='center'> Limitations On Mutual Information Between Steady-State Distributions In Molecular Cascades<br>(Placeholder)<br>
    [<a href="nonsense link">arXiv</a>, <a href="nonsense link 2">YouTube</a>] </h1>

<p align="center">
<img align="middle" src="./imgs/main.png" width="666" />
</p>

Building on the well-understood mathematical theory of _controlled differential equations_, we demonstrate how to construct models that:
+ Act directly on irregularly-sampled partially-observed multivariate time series.
+ May be trained with memory-efficient adjoint backpropagation - even across observations.
+ Demonstrate state-of-the-art performance.

They are straightforward to implement and evaluate using existing tools, in particular PyTorch and the [`torchcde`](https://github.com/patrick-kidger/torchcde) library.

----

### Example
We encourage looking at [example.py](https://github.com/rfangit/Limits-On-Molecular-Cascade-Mutual-Information/tree/main), which demonstrates 

A self contained short example:
```python

torchcde.cdeint(X=X, func=func, z0=z0, t=X.interval)
```

### Reproducing experiments
Everything to reproduce the experiments of the paper can be found in the [`experiments` folder](./experiments). Check the folder for details.

### Results
As an example (taken from the paper - have a look there for similar results on other datasets):

### Citation
```bibtex
@article{rfan,
    title={Limitations},
    author={Fan, Raymond},
    journal={Who knows.},
    year={2025}
}
```