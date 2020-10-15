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
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.backend import epsilon
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from numpy import min
from gc import collect as take_out_trash


class EvaluateMNIST:

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
        batch_size_power_of_two=0,
    ):
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.validation_data_proportion = validation_data_proportion
        self.optimizer = 'adam'  # This is a placeholder; see specify_optimizer method below
        self.callbacks = []  # This is a placeholder; see specify_callbacks method below
        self.adam_learn_rate = adam_learn_rate  # float between 0.001 and 0.999 (0.001)
        self.adam_beta_1 = adam_beta_1  # float between 0.001 and 0.999 (0.9)
        self.adam_beta_2 = adam_beta_2  # float between 0.001 and 0.99 (0.999)
        self.adam_amsgrad_bool = adam_amsgrad_bool  # False or True (False)
        self.number_hidden_conv_layers = number_hidden_conv_layers  # 0 or 1 or 2 (1)
        self.hidden_layers_activation_func = hidden_layers_activation_func  # 'relu' 'sigmoid' 'softplus' ('relu')
        self.early_stopping_patience = early_stopping_patience
        self.verbosity = verbosity  # 1, 2, or 3 (2)
        self.max_epochs = max_epochs  # 1 to 500 (50)
        self.batch_size = 2 ** batch_size_power_of_two  # powers of 2 (2**5=32)

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

    def specify_optimizer(self):  # Set optimizer parameters
        opt = Adam(
            learning_rate=self.adam_learn_rate,
            beta_1=self.adam_beta_1,
            beta_2=self.adam_beta_2,
            epsilon=epsilon(),
            amsgrad=self.adam_amsgrad_bool,
        )
        self.optimizer = opt  # Replace placeholder from __init__

    def specify_early_stopper(self):  # Set patience parameter for early stopping callback
        earlystopper = EarlyStopping(
            patience=self.early_stopping_patience,
            verbose=self.verbosity,
        )
        self.callbacks.append(earlystopper)  # Append (empty?) list from __init__
        return self.callbacks

    def build_variable_depth_classifier(self):  # Define model architecture
        clear_session()
        model = models.Sequential()
        model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2), strides=2))
        for level in range(self.number_hidden_conv_layers):
            model.add(layers.Conv2D(4, (3, 3), activation=self.hidden_layers_activation_func))
            model.add(layers.MaxPooling2D((2, 2), strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
        return model

    def compile_classifier(self, model):  # Compile model
        model.compile(
            optimizer=self.optimizer,
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()],
        )
        return model

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
