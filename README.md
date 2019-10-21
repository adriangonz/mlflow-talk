# MLFlow and Seldon

End to end example integrating MLFlow and Seldon, with A/B testing between two
models.

The slides accompanying this demo can be [found here](https://docs.google.com/presentation/d/12UKDkyt3lLvdcYf8v5H3fzd1_78X3w61hByAsAhNOC4/edit?usp=sharing).

## Set up and requirements

The python requirements are listed in [requirements.txt](requirements.txt).
Besides these, we will also need a working installation of
[`kind`](https://kind.sigs.k8s.io/).

We also use a number of command line utils which the notebook expects to be on
your `$PATH`:

- [httpie](https://httpie.org/)
- [ccat](https://github.com/jingweno/ccat)
- [pygmentize](http://pygments.org/docs/cmdline/)

The rest of the set up (e.g. setting up the cluster, deploying Seldon Core,
etc.) is covered on the notebook.

## Notebook

A notebook with the entire demo can be found in
[mlflow-talk](./mlflow-talk.ipynb).
