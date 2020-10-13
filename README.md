# How to run demo

**simple_study_mnist_classifier_settings.py** : 
If using a docker, test the following commands first and make sure they run; if they do not then install `plotly` and `optuna`:

```
import optuna
import plotly
```

From the command line:

CUDA_VISIBLE_DEVICES="0", python3 study_mnist_classifier_settings.py

or if you want to use more than one GPU, use whichever you want (for example, if you have two GPUs and want to use the second, that is device 1).

The script has an import statement which will automatically choose a single GPU that is least utilized so you do not need to set this parameter:

`import setGPU`

This import statement is enabled by cloning and installing the following repo:

https://github.com/bamos/setGPU

If you have `setGPU` installed, you can avoid having to tell Python which GPU to use because it will automatically select the one with least memory allocated.

**study_mnist_classifier_settings.py**
This version also attempts to optimize the Adam settings.