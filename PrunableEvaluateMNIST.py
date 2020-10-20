# -*- coding: utf-8 -*-
"""
Created on Wed Oct 7 2020
Adaptation of code at:
https://www.analyticsvidhya.com/blog/2020/07/how-to-train-an-image-classification-model-in-pytorch-and-tensorflow/
@author: swagenman
"""

from sklearn.model_selection import train_test_split as trn_val_split
from numpy.random import RandomState
from tensorflow.keras.backend import clear_session
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from numpy import min
from numpy.random import default_rng as random_generator_instantiator
from gc import collect as take_out_trash


class PrunableEvaluateMNIST:
    __instance__ = None

    def __init__(
        self,
        train_images,
        test_images,
        train_labels,
        test_labels,
        validation_data_proportion=0.06147143,
        adam_learn_rate=0.001,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        adam_amsgrad_bool=False,
        number_hidden_conv_layers=1,
        hidden_layers_activation_func='relu',
        early_stopping_patience=10,
        verbosity=2,
        max_epochs=100,
        batch_size_power_of_two=4,
    ):
        if PrunableEvaluateMNIST.__instance__ is None:
            self.train_images = train_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.test_labels = test_labels
            self.validation_data_proportion = validation_data_proportion
            self.callbacks = []  # Empty list to which one may append any number of callbacks
            self.early_stopping_patience = early_stopping_patience
            self.verbosity = verbosity  # 1, 2, or 3 (2)
            self.max_epochs = max_epochs  # 1 to 500 (50)
            self.batch_size = 2 ** batch_size_power_of_two  # powers of 2 (2**5=32)

            PrunableEvaluateMNIST.__instance__ = self
        else:
            raise Exception("This is a singleton; PrunableEvaluateMNIST has already been created.")

    def split_training_data_for_training_and_validation(self):
        instance_of_random_state = RandomState()
        self.train_split_images, validate_split_images, self.train_split_labels, validate_split_labels = trn_val_split(
            self.train_images,
            self.train_labels,
            test_size=self.validation_data_proportion,
            random_state=instance_of_random_state,
            shuffle=True,
            stratify=self.train_labels,
        )
        self.validate_split_data = (validate_split_images, validate_split_labels)

    def specify_early_stopper(self):  # Set patience parameter for early stopping callback
        early_stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            verbose=self.verbosity,
        )
        self.callbacks.append(early_stopper)  # Append to callbacks list

    def train_test_and_delete_classifier(self, model):  # Train model
        history = model.fit(
            self.train_split_images,
            self.train_split_labels,
            epochs=self.max_epochs,
            validation_data=self.validate_split_data,
            verbose=self.verbosity,
            callbacks=self.callbacks,
            batch_size=self.batch_size,
        )
        test_results = model.evaluate(
            self.test_images,
            self.test_labels,
            batch_size=self.batch_size,
        )
        test_results = {out: test_results[i] for i, out in enumerate(model.metrics_names)}
        take_out_trash()
        del model
        validation_losses = history.history['val_loss']
        return min(validation_losses), test_results
