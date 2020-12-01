# How to run demo

Change your timeout settings and/or number of trials for Optuna study before running.

**study_mnist_classifier_settings.py** : 
If using a docker, test the following commands first and make sure they run; if they do not then install `plotly` and `optuna`:

```
import optuna
import plotly
```

From the command line:

```
CUDA_VISIBLE_DEVICES="0", python3 study_mnist_classifier_settings.py
```

or if you want to use more than one GPU/a different one besides 0, use whichever you want (for example, if you have two GPUs and want to use the second, that is device 1).

The script has a commented-out import statement which will automatically choose a single GPU that is least utilized so you do not need to set this parameter:

`import setGPU`

If you choose to un-comment this import statement, run the following import first (warning, will not work on Windows):
`pip install setGPU`

If you have `setGPU` installed, this line avoid having to tell Python which GPU to use (by specifying `CUDA_VISIBLE_DEVICES`) because it will automatically select the one with least memory allocated.

**PruneableEvaluateMNIST.py**
Intended for use with `study_mnist_classifier_settings.py`.

# UPDATE

Added experimental code to compare performance of Jeffreys prior for uniformly distributed hyper-parameters.  The head to head matchup of using Jeffreys priors or not was with Adam as the optimizer and a learning rate schedule.  I am currently running the code with AdaBelief and no learning rate schedule to see if I can beat Adam; if I do, I will re-match Jeffreys priors and without using AdaBelief.
