# MLFlow and Seldon

End to end example integrating MLFlow and Seldon, with A/B testing
between two models.
We also cover how to explain the model's predictions with Alibi.

## Set up

The python requirements are listed in
[requirements.txt](requirements.txt).
Besides these, we will also need a working installation of
[`kind`](https://kind.sigs.k8s.io/).

The rest of the set up (e.g. setting up the cluster, deploying Seldon
Core, etc.) is covered on the notebook.

## Notebook

A notebook with the entire demo can be found in
[mlflow-talk](./mlflow-talk.ipynb).
