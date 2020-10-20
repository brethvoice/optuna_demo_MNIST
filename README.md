# How to run demo

NB: each of the following files is set to run over a 12-hour period as written; recommend changing timeout settings and/or number of trials for Optuna study if that is too long.

**simple_study_mnist_classifier_settings.py** : 
If using a docker, test the following commands first and make sure they run; if they do not then install `plotly` and `optuna`:

```
import optuna
import plotly
```

From the command line:

CUDA_VISIBLE_DEVICES="0", python3 study_mnist_classifier_settings.py

or if you want to use more than one GPU, use whichever you want (for example, if you have two GPUs and want to use the second, that is device 1).

The script has a commented-out import statement which will automatically choose a single GPU that is least utilized so you do not need to set this parameter:

`import setGPU`

If you choose to un-comment this import statement, run the following import first (warning, will not work on Windows):
`pip install setGPU`

If you have `setGPU` installed, this line avoid having to tell Python which GPU to use (by specifying `CUDA_VISIBLE_DEVICES`) because it will automatically select the one with least memory allocated.

**study_mnist_classifier_settings.py**
Implements median stopping rule pruner via `tf.keras` callback (a special "pruning hook" built by Optuna).This version also attempts to optimize the Adam settings, with a Beta(0.5, 0.5) Jeffreys prior for each of the `learning_rate`, `beta_1`, and `beta_2` parameters.

**EvaluateMNIST.py**
Not built for pruning; also sets Jeffreys priors for the Adam training optimizer hyper-parameters at instantiation.

**PruneableEvaluateMNIST.py**
Does the extra things `EvaluateMNIST.py` will not.
