# -*- coding: utf-8 -*-
"""
Created on Wed Oct 7 2020
Adaptation of code at:
https://www.analyticsvidhya.com/blog/2020/07/how-to-train-an-image-classification-model-in-pytorch-and-tensorflow/
@author: swagenman
"""

from sklearn.model_selection import train_test_split as trn_val_split
from numpy.random import RandomState
from tensorflow.keras.callbacks import EarlyStopping
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class PrunableEvaluateMNIST(object):

    def __init__(
        self,
        train_validate_images,
        train_validate_labels,
        validation_data_proportion=0.93611667,  # (60,000 - 56,167) / 60,000
        early_stopping_significant_delta=1e-6,
        early_stopping_patience=10,
        verbosity=2,
        seed=None,
    ):
        self.train_validate_images = train_validate_images
        self.train_validate_labels = train_validate_labels
        self.validation_data_proportion = validation_data_proportion
        self.early_stopping_significant_delta = early_stopping_significant_delta
        self.early_stopping_patience = early_stopping_patience
        self.verbosity = verbosity  # 1, 2, or 3 (2)
        self.callbacks = []  # Empty list to which one may append any number of callbacks
        self.seed = seed

    def set_batch_size(self, batch_size_base_two_logarithm):
        self.batch_size = 2 ** int(batch_size_base_two_logarithm)

    def stratified_split_for_training_and_validation(self):
        instance_of_random_state = RandomState(self.seed)
        self.train_split_images, validate_split_images, self.train_split_labels, validate_split_labels = trn_val_split(
            self.train_validate_images,
            self.train_validate_labels,
            test_size=self.validation_data_proportion,
            random_state=instance_of_random_state,
            shuffle=True,
            stratify=self.train_validate_labels,
        )
        self.validate_split_data = (validate_split_images, validate_split_labels)

    def append_early_stopper_callback(self):  # Implement patience for early stopping
        early_stopper = EarlyStopping(
            monitor='val_loss',
            min_delta=self.early_stopping_significant_delta,
            patience=self.early_stopping_patience,
            verbose=self.verbosity,
            mode='auto',
            baseline=None,
            restore_best_weights=True,
        )
        self.callbacks.append(early_stopper)  # Append to callbacks list
