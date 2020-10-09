# How to run demo

**study_mnist_classifier_settings.py** : If using a multi-GPU server, type the following on the command line:

CUDA_VISIBLE_DEVICES="0", python3 study_mnist_classifier_settings.py

or if you have more than one GPU, use whichever you want (for example, if you have two GPUs and want to use the second, that is device 1).

Also, the script has an import statement:

`import setGPU`

which is enabled by cloning and installing the following repo:

https://github.com/bamos/setGPU
