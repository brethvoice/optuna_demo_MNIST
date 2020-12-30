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

Some of the scripts call `import setGPU` which will automatically choose a single GPU that is least utilized so you do not need to specify one.  If you choose to comment this import statement, set CUDA_VISIBLE_DEVICES as per the example command line call above.

If you have `setGPU` installed, this line avoids having to tell Python which GPU to use (by specifying `CUDA_VISIBLE_DEVICES`) because it will automatically select the one with least memory allocated.

**PruneableEvaluateMNIST.py**
Abstraction of some code for files beginning with `study_mnist_classifier_settings.py`.

# UPDATE

I have re-started the experiments examing use of a Jeffreys prior on the first trial (this time using GPU for Tensorflow as that is what the library was designed to use).  This is in connection with the following feature request for Optuna:
https://github.com/optuna/optuna/issues/1972

I am also comparing AdaBelief to Adam (neither with amsgrad enabled, but with rectify=True for AdaBelief as recommended at https://github.com/juntang-zhuang/Adabelief-Optimizer).  I am eager to see whether the results of the Jeffreys prior experiment will be the same for either choice of machine learning optimizer.
