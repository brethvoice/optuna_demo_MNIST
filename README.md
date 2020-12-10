# How to run demo

Change timeout settings and/or number of trials for Optuna study before running.

**tensorboard_example_code.py**:
This is posted to support a GitHub issue report on TensorBoard's "HPARAMS" plugin.  Please contribute there if you can:
https://github.com/tensorflow/tensorflow/issues/45384

(files beginning with) **study_mnist_classifier_settings.py**:
If using a docker, test the following commands first and make sure they run; if they do not then install `scikit-learn` and `optuna`:

```
import optuna
import sklearn
```

From the command line:

```
CUDA_VISIBLE_DEVICES="0", python3 study_mnist_classifier_settings.py
```

or if you want to use more than one GPU/a different one besides 0, use whichever you want (for example, if you have two GPUs and want to use the second, that is device 1).

The script has a commented-out import statement which will automatically choose a single GPU that is least utilized so you do not need to specify one:

`import setGPU`

If you choose to un-comment this import statement, run the following import first (warning, will not work on Windows):
`pip install setGPU`

If you have `setGPU` installed, this line avoids having to tell Python which GPU to use (by specifying `CUDA_VISIBLE_DEVICES`) because it will automatically select the one with least memory allocated.

**PruneableEvaluateMNIST.py**
Abstraction of some code for files beginning with `study_mnist_classifier_settings.py`.

# UPDATE

The Jeffreys prior experiments were flawed in that I continued using the prior after the first trial, so I am re-doing everything.  After the initial trial, there is an established posterior distribution, so a Jeffreys prior is no longer helpful.

I am also comparing AdaBelief to Adam (neither with amsgrad enabled nor with learning rate schedules).  I am hoping the results of the Jeffreys prior experiment will be the same for either choice of machine learning optimizer.
