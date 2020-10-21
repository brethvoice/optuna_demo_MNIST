# -*- coding: utf-8 -*-
"""
Created on Thu Oct 8 2020
Adaptation of code at:
https://www.analyticsvidhya.com/blog/2020/07/how-to-train-an-image-classification-model-in-pytorch-and-tensorflow/
@author: swagenman
"""

import ssl

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session
from numpy import log2, floor
import optuna

from pdb import set_trace

from EvaluateMNIST import EvaluateMNIST

# import setGPU  # Find and make visible the GPU with least memory allocated


# Specify length and nature of study; depending on batch size some trials can take minutes
MAXIMUM_NUMBER_OF_TRIALS_TO_RUN = 10  # For the Optuna study itself
NUMBER_OF_TRIALS_BEFORE_PRUNING = int(0.2 * MAXIMUM_NUMBER_OF_TRIALS_TO_RUN)
MAXIMUM_SECONDS_TO_CONTINUE_STUDY = 1 * 3600  # 3600 seconds = one hour
MAXIMUM_EPOCHS_TO_TRAIN = 500  # Each model will not train for more than this many epochs
EARLY_STOPPING_PATIENCE_PARAMETER = int(0.1 * MAXIMUM_EPOCHS_TO_TRAIN)  # For tf.keras' EarlyStopping callback
VERBOSITY_LEVEL_FOR_TENSORFLOW = 2  # One verbosity for both training and EarlyStopping callback

# Establish some MNIST-specific constants used below
MNIST_TRAINING_AND_VALIDATION_SET_SIZE = 60000
JUN_SHAO_TRAINING_PROPORTION = (
    (MNIST_TRAINING_AND_VALIDATION_SET_SIZE**(3/4)) / MNIST_TRAINING_AND_VALIDATION_SET_SIZE
)
MAXIMUM_BATCH_SIZE_POWER_OF_TWO = floor(
    log2(JUN_SHAO_TRAINING_PROPORTION * MNIST_TRAINING_AND_VALIDATION_SET_SIZE)
)

# Prevent any weird authentication errors when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context

# Load MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path='mnist.npz')

# Normalize image pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Instantiate standard object outside of objective function
standard_object = EvaluateMNIST(
    train_images=train_images,
    test_images=test_images,
    train_labels=train_labels,
    test_labels=test_labels,
    validation_data_proportion=(1-JUN_SHAO_TRAINING_PROPORTION),
    early_stopping_patience=EARLY_STOPPING_PATIENCE_PARAMETER,
)


def objective(trial):
    standard_object.number_hidden_conv_layers = trial.suggest_int(
        'number_hidden_conv_layers',
        0,
        2,
    )
    standard_object.hidden_layers_activation_func = trial.suggest_categorical(
        'hidden_layers_activation_func',
        [
            'relu',
            'sigmoid',
            'softplus',
        ]
    )
    standard_object.batch_size_base_two_logarithm = trial.suggest_int(
        'batch_size_base_two_logarithm',
        0,
        MAXIMUM_BATCH_SIZE_POWER_OF_TWO,
    )
    standard_object.append_early_stopper_callback()
    keras_pruner = optuna.integration.TFKerasPruningCallback(
        trial,
        'val_categorical_accuracy',
    )
    standard_object.callbacks.append(keras_pruner)  # Append to callbacks list
    standard_object.stratified_split_for_training_and_validation()
    clear_session()
    classifier_uncompiled_model = standard_object.build_variable_depth_classifier()
    standard_object.optimizer = 'adam'
    classifier_compiled_model = standard_object.compile_classifier(classifier_uncompiled_model)
    standard_object.set_batch_size(standard_object.batch_size_base_two_logarithm)

    # Train and validate using hyper-parameters generated above, then evaluate on test data and report score
    classifier_compiled_model.fit(
        standard_object.train_split_images,
        standard_object.train_split_labels,
        epochs=MAXIMUM_EPOCHS_TO_TRAIN,
        validation_data=standard_object.validate_split_data,
        verbose=standard_object.verbosity,
        callbacks=standard_object.callbacks,
        batch_size=standard_object.batch_size,
    )
    test_metrics = classifier_compiled_model.evaluate(
        standard_object.test_images,
        standard_object.test_labels,
        batch_size=standard_object.batch_size,
    )
    test_results_dict = {out: test_metrics[i] for i, out in enumerate(classifier_compiled_model.metrics_names)}
    return(test_results_dict['categorical_accuracy'])


# Create and run Optuna study
sampler_multivariate = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(
    sampler=sampler_multivariate,
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=NUMBER_OF_TRIALS_BEFORE_PRUNING,
        n_warmup_steps=EARLY_STOPPING_PATIENCE_PARAMETER,
    ),
)
study.optimize(
    objective,
    n_trials=MAXIMUM_NUMBER_OF_TRIALS_TO_RUN,
    timeout=MAXIMUM_SECONDS_TO_CONTINUE_STUDY,
    gc_after_trial=True,
    )
set_trace()  # Before taking any more steps, pause execution

# Report completed study results:
print('Study statistics:  ')
print('Number of trials was {}'.format(len(study.trials)))
print('\n\nBest trial number was {}\n\n'.format(study.best_trial))
print('\n\nBest categorical accuracy was {}\n\n...'.format(study.best_trial.value))
print('\n\nParameters: ')
for key, value in study.best_trial.params.items():
    print('{}: {}'.format(key, value))

# This does not work on DGX currently, but also does not throw error
fig = optuna.visualization.plot_param_importances(study)
fig.show()
