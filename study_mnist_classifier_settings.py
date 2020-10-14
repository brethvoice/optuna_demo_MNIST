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
from numpy import log2, floor
import optuna

from pdb import set_trace

from PrunableEvaluateMNIST import PrunableEvaluateMNIST

import setGPU  # Find and make visible the GPU with least memory allocated


# Specify length and nature of study; depending on batch size some trials can take minutes
MAXIMUM_NUMBER_OF_TRIALS_TO_RUN = 10  # For the Optuna study itself
NUMBER_OF_TRIALS_BEFORE_PRUNING = int(0.2 * MAXIMUM_NUMBER_OF_TRIALS_TO_RUN)
MAXIMUM_SECONDS_TO_CONTINUE_STUDY = 0.5 * 3600  # 3600 seconds = one hour
EARLY_STOPPING_PATIENCE_PARAMETER = 10  # For tf.keras' EarlyStopping callback
VERBOSITY_LEVEL_FOR_TENSORFLOW = 2  # One verbosity for both training and EarlyStopping callback
MAXIMUM_EPOCHS_TO_TRAIN = 100  # Each model will not train for more than this many epochs

# Establish MNIST-specific constants used in code below
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


def objective(trial):
    # Instantiate class
    evaluator = PrunableEvaluateMNIST(
        train_images=train_images,
        test_images=test_images,
        train_labels=train_labels,
        test_labels=test_labels,
        validation_data_proportion=(1-JUN_SHAO_TRAINING_PROPORTION),
        adam_learn_rate=trial.suggest_uniform(
            'adam_learn_rate',
            0,
            1,
        ),
        adam_beta_1=trial.suggest_uniform(
            'adam_beta_1',
            0,
            1,
        ),
        adam_beta_2=trial.suggest_uniform(
            'adam_beta_2',
            0,
            1,
        ),
        adam_epsilon_multiplier=trial.suggest_int(
            'adam_epsilon_multiplier',
            0,
            1000000,
        ),
        adam_amsgrad_bool=trial.suggest_categorical(
            'adam_amsgrad_bool',
            [
                False,
                True,
            ]
        ),
        number_hidden_conv_layers=trial.suggest_int(
            'number_hidden_conv_layers',
            0,
            2,
        ),
        hidden_layers_activation_func=trial.suggest_categorical(
            'hidden_layers_activation_func',
            [
                'relu',
                'sigmoid',
                'softplus',
            ]
        ),
        early_stopping_patience=EARLY_STOPPING_PATIENCE_PARAMETER,
        verbosity=VERBOSITY_LEVEL_FOR_TENSORFLOW,
        max_epochs=MAXIMUM_EPOCHS_TO_TRAIN,
        batch_size_power_of_two=trial.suggest_int(
            'batch_size_power_of_two',
            0,
            MAXIMUM_BATCH_SIZE_POWER_OF_TWO,
        )
    )
    evaluator.specify_early_stopper()
    keras_pruner = optuna.integration.TFKerasPruningCallback(
        trial,
        'val_acc',
    )
    evaluator.callbacks.append(keras_pruner)  # Append to callbacks list
    evaluator.split_training_data_for_training_and_validation()
    classifier_uncompiled_model = evaluator.build_variable_depth_classifier()
    evaluator.specify_optimizer()
    classifier_compiled_model = evaluator.compile_classifier(classifier_uncompiled_model)
    val_loss, test_metrics = evaluator.train_test_and_delete_classifier(
        classifier_compiled_model
    )
    return(test_metrics['categorical_accuracy'])


# Create and run the study
sampler_multivariate = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(
    sampler=sampler_multivariate,
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=NUMBER_OF_TRIALS_BEFORE_PRUNING),
)
study.optimize(
    objective,
    n_trials=MAXIMUM_NUMBER_OF_TRIALS_TO_RUN,
    timeout=MAXIMUM_SECONDS_TO_CONTINUE_STUDY,
    gc_after_trial=True,
    )

# Report out on study results:
print('Study statistics:  ')
print('\n\nBest trial number was {}\n\n'.format(study.best_trial))
print('\n\nBest categorical accuracy was {}\n\n...'.format(study.best_trial.value))
print('\n\nParameters: ')
for key, value in study.best_trial.params.items():
    print('{}: {}'.format(key, value))
set_trace()  # Before taking any more steps, pause execution
completed_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
print('Number of completed trials is {}'.format(len(completed_trials)))
pruned_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
]
print('Number of pruned trials is {}'.format(len(pruned_trials)))
fig = optuna.visualization.plot_param_importances(study)
fig.show()
