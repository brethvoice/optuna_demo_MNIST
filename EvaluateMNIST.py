# -*- coding: utf-8 -*-
"""
Created on Wed Oct 7 2020
Adaptation of code at:
https://www.analyticsvidhya.com/blog/2020/07/how-to-train-an-image-classification-model-in-pytorch-and-tensorflow/
@author: swagenman
"""

from sklearn.model_selection import train_test_split as trn_val_split
from numpy.random import RandomState
from tensorflow.keras import layers, models
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping


class EvaluateMNIST:
    __instance__ = None  # Necessary for singleton class

    def __init__(
        self,
        train_images,
        test_images,
        train_labels,
        test_labels,
        validation_data_proportion=0.06147143,
        early_stopping_patience=50,
        verbosity=2,
    ):
        if EvaluateMNIST.__instance__ is None:  # Necessary for singleton class
            self.train_images = train_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.test_labels = test_labels
            self.validation_data_proportion = validation_data_proportion
            self.early_stopping_patience = early_stopping_patience
            self.verbosity = verbosity  # 1, 2, or 3 (2)
            self.callbacks = []  # This is a placeholder; see specify_early_stopper method below

            EvaluateMNIST.__instance__ = self  # Necessary for singleton class
        else:
            raise Exception("This is a singleton; EvaluateMNIST has already been created.")

    def set_batch_size(self, batch_size_base_two_logarithm):
        self.batch_size = 2 ** int(batch_size_base_two_logarithm)

    def stratified_split_for_training_and_validation(self):
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

    def append_early_stopper_callback(self):  # Implement patience parameter for early stopping
        earlystopper = EarlyStopping(
            patience=self.early_stopping_patience,
            verbose=self.verbosity,
        )
        self.callbacks.append(earlystopper)  # Append to callbacks list

    def build_variable_depth_classifier(self):  # Define model architecture
        uncompiled_model = models.Sequential()
        uncompiled_model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        uncompiled_model.add(layers.MaxPooling2D((2, 2), strides=2))
        for level in range(self.number_hidden_conv_layers):
            uncompiled_model.add(layers.Conv2D(4, (3, 3), activation=self.hidden_layers_activation_func))
            uncompiled_model.add(layers.MaxPooling2D((2, 2), strides=2))
        uncompiled_model.add(layers.Flatten())
        uncompiled_model.add(layers.Dense(10, activation='softmax'))
        return uncompiled_model

    def compile_classifier(self, uncompiled_model):  # Compile model
        uncompiled_model.compile(
            optimizer=self.optimizer,
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()],
        )
        return uncompiled_model
