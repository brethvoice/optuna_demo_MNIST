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
from tensorflow.keras.backend import clear_session
from tensorflow.keras import layers, models
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from pdb import set_trace
from adabelief_tf import AdaBeliefOptimizer

from PrunableEvaluateMNIST import PrunableEvaluateMNIST

import setGPU  # Find and make visible the GPU with least memory allocated

# Specify length and nature of study; depending on batch size some trials can take minutes
MAXIMUM_NUMBER_OF_TRIALS_TO_RUN = 500  # For the Optuna study itself
NUMBER_OF_TRIALS_BEFORE_PRUNING = int(0.2 * MAXIMUM_NUMBER_OF_TRIALS_TO_RUN)
MAXIMUM_SECONDS_TO_CONTINUE_STUDY = 96 * 3600  # 3600 seconds = one hour
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
train_images = train_images.reshape((MNIST_TRAINING_AND_VALIDATION_SET_SIZE, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# This is what Optuna will optimize (minimize or maximize)
def objective(trial):
    standard_object = PrunableEvaluateMNIST(
        train_images=train_images,
        test_images=test_images,
        train_labels=train_labels,
        test_labels=test_labels,
        validation_data_proportion=(1-JUN_SHAO_TRAINING_PROPORTION),
        early_stopping_patience=EARLY_STOPPING_PATIENCE_PARAMETER,
        verbosity=VERBOSITY_LEVEL_FOR_TENSORFLOW,
    )

    # Instantiate a random number generator
    rg = random_generator_instantiator()

    # Generate hyper-parameters
    standard_object.number_of_conv_2d_filters = trial.suggest_int(
        'number_of_conv_2d_filters',
        4,
        8,
    )
    standard_object.first_conv2d_kernel_dim = trial.suggest_int(
        'first_conv2d_kernel_dim',
        1,
        2,
    )
    standard_object.second_conv2d_kernel_dim = trial.suggest_int(
        'second_conv2d_kernel_dim',
        1,
        2,
    )
    standard_object.conv2d_layers_activation_func = trial.suggest_categorical(
        'conv2d_layers_activation_func',
        [
            'relu',
            'sigmoid',
            'softplus',
        ]
    )
    standard_object.number_hidden_conv_layers = trial.suggest_int(
        'number_hidden_conv_layers',
        0,
        2,
    )
    standard_object.batch_size_base_two_logarithm = trial.suggest_int(
        'batch_size_base_two_logarithm',
        0,
        MAXIMUM_BATCH_SIZE_POWER_OF_TWO,
    )
    standard_object.adam_learning_rate = rg.beta(0.5, 0.5) * trial.suggest_uniform(
        'adam_learning_rate',
        0,
        2,
    )
    standard_object.adam_beta_1 = rg.beta(0.5, 0.5) * trial.suggest_uniform(
        'adam_beta_1',
        0,
        1,
    )
    standard_object.adam_beta_2 = rg.beta(0.5, 0.5) * trial.suggest_uniform(
        'adam_beta_2',
        0,
        1,
    )

    # Add early stopping callback
    standard_object.append_early_stopper_callback()

    # Append tf.keras pruner for later use during study
    keras_pruner = optuna.integration.TFKerasPruningCallback(
        trial,
        'val_categorical_accuracy',
    )
    standard_object.callbacks.append(keras_pruner)  # Append to callbacks list

    # Train and validate using hyper-parameters generated above
    clear_session()
    classifier_model = models.Sequential()
    classifier_model.add(layers.Conv2D(
        standard_object.number_of_conv_2d_filters,
        (standard_object.first_conv2d_kernel_dim, standard_object.second_conv2d_kernel_dim),
        activation=standard_object.conv2d_layers_activation_func,
        input_shape=(28, 28, 1),
    ))
    classifier_model.add(layers.MaxPooling2D((2, 2)))
    for level in range(standard_object.number_hidden_conv_layers):
        classifier_model.add(layers.Conv2D(
        standard_object.number_of_conv_2d_filters,
        (standard_object.first_conv2d_kernel_dim, standard_object.second_conv2d_kernel_dim),
        activation=standard_object.conv2d_layers_activation_func,
    ))
        classifier_model.add(layers.MaxPooling2D((2, 2), strides=2))
    classifier_model.add(layers.Flatten())
    classifier_model.add(layers.Dense(10, activation='softmax'))
    standard_object.optimizer = AdaBeliefOptimizer(
        learning_rate=standard_object.adam_learning_rate,
        beta_1=standard_object.adam_beta_1,
        beta_2=standard_object.adam_beta_2,
        epsilon=epsilon(),
        rectify=False,  # to compare with Adam (unrectified)
        amsgrad=False,  # this was just another attempt to make Adam converge
    )
    classifier_model.compile(
        optimizer=standard_object.optimizer,
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()],
    )
    standard_object.set_batch_size(standard_object.batch_size_base_two_logarithm)
    standard_object.stratified_split_for_training_and_validation()
    classifier_model.fit(
        standard_object.train_split_images,
        standard_object.train_split_labels,
        epochs=MAXIMUM_EPOCHS_TO_TRAIN,
        validation_data=standard_object.validate_split_data,
        verbose=VERBOSITY_LEVEL_FOR_TENSORFLOW,
        callbacks=standard_object.callbacks,
        batch_size=standard_object.batch_size,
        # run_eagerly=True,
    )

    # Evaluate performance on test data and report score
    test_metrics = classifier_model.evaluate(
        standard_object.test_images,
        standard_object.test_labels,
        batch_size=standard_object.batch_size,
    )
    test_results_dict = {out: test_metrics[i] for i, out in enumerate(classifier_model.metrics_names)}
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

# Report completed study results:
print('Study statistics:  ')
print('\n\nBest trial number was {}\n\n'.format(study.best_trial))
print('\n\nBest categorical accuracy was {}\n\n...'.format(study.best_trial.value))
print('\n\nParameters: ')
for key, value in study.best_trial.params.items():
    print('{}: {}'.format(key, value))
completed_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
print('Number of completed trials is {}'.format(len(completed_trials)))
pruned_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
]
print('Number of pruned trials is {}'.format(len(pruned_trials)))
set_trace()

# This does not work for some reason
fig = optuna.visualization.plot_param_importances(study)
fig.show()
