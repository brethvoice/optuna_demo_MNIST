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
from numpy.random import default_rng as random_generator_instantiator
from tensorflow.keras.backend import epsilon
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from pdb import set_trace

from PrunableEvaluateMNIST import PrunableEvaluateMNIST

# import setGPU  # Find and make visible the GPU with least memory allocated


# Specify length and nature of study; depending on batch size some trials can take minutes
MAXIMUM_NUMBER_OF_TRIALS_TO_RUN = 100  # For the Optuna study itself
NUMBER_OF_TRIALS_BEFORE_PRUNING = int(0.2 * MAXIMUM_NUMBER_OF_TRIALS_TO_RUN)
MAXIMUM_SECONDS_TO_CONTINUE_STUDY = 14 * 3600  # 3600 seconds = one hour
MAXIMUM_EPOCHS_TO_TRAIN = 500  # Each model will not train for more than this many epochs
EARLY_STOPPING_PATIENCE_PARAMETER = int(0.1 * MAXIMUM_EPOCHS_TO_TRAIN)  # For tf.keras' EarlyStopping callback
VERBOSITY_LEVEL_FOR_TENSORFLOW = 2  # One verbosity for both training and EarlyStopping callback

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

# Instantiate base model outside of objective function
base_model = PrunableEvaluateMNIST(
    train_images=train_images,
    test_images=test_images,
    train_labels=train_labels,
    test_labels=test_labels,
    validation_data_proportion=(1-JUN_SHAO_TRAINING_PROPORTION),
    early_stopping_patience=EARLY_STOPPING_PATIENCE_PARAMETER,
    verbosity=VERBOSITY_LEVEL_FOR_TENSORFLOW,
    max_epochs=MAXIMUM_EPOCHS_TO_TRAIN,
)

def objective(trial):
    base_model.number_hidden_conv_layers = trial.suggest_int(
        'number_hidden_conv_layers',
        0,
        2,
    )
    base_model.hidden_layers_activation_func = trial.suggest_categorical(
        'hidden_layers_activation_func',
        [
            'relu',
            'sigmoid',
            'softplus',
        ]
    )
    base_model.batch_size_power_of_two = trial.suggest_int(
        'batch_size_power_of_two',
        0,
        MAXIMUM_BATCH_SIZE_POWER_OF_TWO,
    )
    rg_learn_rate = random_generator_instantiator()
    base_model.adam_learn_rate = rg_learn_rate.beta(0.5, 0.5) * trial.suggest_uniform(
        'adam_learn_rate',
        0,
        1,
    )
    del rg_learn_rate
    rg_beta_1 = random_generator_instantiator()
    base_model.adam_beta_1 = rg_beta_1.beta(0.5, 0.5) * trial.suggest_uniform(
        'adam_beta_1',
        0,
        1,
    )
    del rg_beta_1
    rg_beta_2 = random_generator_instantiator()
    base_model.adam_beta_2 = rg_beta_2.beta(0.5, 0.5) * trial.suggest_uniform(
        'adam_beta_2',
        0,
        1,
    )
    del rg_beta_2
    base_model.adam_amsgrad_bool = trial.suggest_categorical(
        'adam_amsgrad_bool',
        [
            False,
            True,
        ]
    )
    base_model.specify_early_stopper()
    keras_pruner = optuna.integration.TFKerasPruningCallback(
        trial,
        'val_categorical_accuracy',
    )
    base_model.callbacks.append(keras_pruner)  # Append to callbacks list
    base_model.split_training_data_for_training_and_validation()
    
    # Build, compile, evaluate, and delete model
    classifier_model = models.Sequential()
    classifier_model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    classifier_model.add(layers.MaxPooling2D((2, 2), strides=2))
    for level in range(base_model.number_hidden_conv_layers):
        classifier_model.add(layers.Conv2D(4, (3, 3), activation=base_model.hidden_layers_activation_func))
        classifier_model.add(layers.MaxPooling2D((2, 2), strides=2))
    classifier_model.add(layers.Flatten())
    classifier_model.add(layers.Dense(10, activation='softmax'))
    base_model.optimizer = Adam(
        learning_rate=base_model.adam_learn_rate,
        beta_1=base_model.adam_beta_1,
        beta_2=base_model.adam_beta_2,
        epsilon=epsilon(),
        amsgrad=base_model.adam_amsgrad_bool,
    )
    classifier_model.compile(
        optimizer=base_model.optimizer,
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()],
    )
    classifier_model.fit(
        base_model.train_split_images,
        base_model.train_split_labels,
        epochs=base_model.max_epochs,
        validation_data=base_model.validate_split_data,
        verbose=base_model.verbosity,
        callbacks=base_model.callbacks,
        batch_size=base_model.batch_size,
    )
    test_results = classifier_model.evaluate(
        base_model.test_images,
        base_model.test_labels,
        batch_size=base_model.batch_size,
    )
    test_results = {out: test_results[i] for i, out in enumerate(classifier_model.metrics_names)}
    del classifier_model
    return(test_results['categorical_accuracy'])


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
