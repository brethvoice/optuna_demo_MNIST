from datetime import datetime
from numpy import floor, log2
from numpy.random import RandomState
from pdb import set_trace
import optuna
import os
from sklearn.model_selection import train_test_split as trn_val_split
import shutil
import ssl
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class PrunableEvaluateMNIST(object):

    def __init__(
        self,
        train_images,
        test_images,
        train_labels,
        test_labels,
        validation_data_proportion=0.06147143,
    ):
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.validation_data_proportion = validation_data_proportion
        self.callbacks = []  # Empty list to which one may append any number of callbacks

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


class TbCallback(TensorBoard):
    """
    Tensorboard callback that cleans the directory after training is completed.
    """

    def on_train_end(self, logs=None):
        if os.path.exists(os.path.join(self.log_dir, "train", "plugins")):
            shutil.rmtree(os.path.join(self.log_dir, "train", "plugins"))


# Specify length and nature of study; depending on batch size some trials can take minutes
MAXIMUM_NUMBER_OF_TRIALS_TO_RUN = 50  # For the Optuna study itself
MAXIMUM_SECONDS_TO_CONTINUE_STUDY = 10 * 3600  # 3600 seconds = one hour
NUMBER_OF_TRIALS_BEFORE_PRUNING = int(0.2 * MAXIMUM_NUMBER_OF_TRIALS_TO_RUN)
MAXIMUM_EPOCHS_TO_TRAIN = 100  # Each model will not train for more than this many epochs
WARMUP_EPOCHS_BEFORE_PRUNING = int(0.2 * MAXIMUM_EPOCHS_TO_TRAIN)
PERCENTILE_FOR_PRUNING = 100 - (100 / WARMUP_EPOCHS_BEFORE_PRUNING)
EARLY_STOPPING_SIGNIFICANT_DELTA = 1e-6
EARLY_STOPPING_PATIENCE_PARAMETER = int(0.1 * MAXIMUM_EPOCHS_TO_TRAIN)  # For tf.keras' EarlyStopping callback
VERBOSITY_LEVEL_FOR_TENSORFLOW = 2  # One verbosity for both training and EarlyStopping callback

# Prevent any weird authentication errors when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context

# Load MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path='mnist.npz')

# Normalize image pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Establish MNIST-specific constants used in code below
MNIST_TRAINING_AND_VALIDATION_SET_SIZE = 60000
JUN_SHAO_TRAINING_PROPORTION = (
    (MNIST_TRAINING_AND_VALIDATION_SET_SIZE**(3/4)) / MNIST_TRAINING_AND_VALIDATION_SET_SIZE
)
MAXIMUM_BATCH_SIZE_POWER_OF_TWO = floor(
    log2(JUN_SHAO_TRAINING_PROPORTION * MNIST_TRAINING_AND_VALIDATION_SET_SIZE)
)

# Reshape images
train_images = train_images.reshape((MNIST_TRAINING_AND_VALIDATION_SET_SIZE, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Set up directory stem for use by Tensorboard callback
stem_output_dir = os.path.abspath(os.getcwd())
print('Stem output directory for Tensorboard is {}'.format(stem_output_dir))


# OPTUNA'S OBJECTIVE FUNCTION
def objective(trial):
    standard_object = PrunableEvaluateMNIST(
        train_images=train_images,
        test_images=test_images,
        train_labels=train_labels,
        test_labels=test_labels,
        validation_data_proportion=(1-JUN_SHAO_TRAINING_PROPORTION),
    )

    # Generate hyper-parameters
    standard_object.number_of_conv_2d_filters = trial.suggest_int(
        'number_of_conv_2d_filters',
        4,
        8,
    )
    standard_object.first_conv2d_kernel_dim = trial.suggest_int(
        'first_conv2d_kernel_dim',
        1,
        4,
    )
    standard_object.second_conv2d_kernel_dim = trial.suggest_int(
        'second_conv2d_kernel_dim',
        1,
        4,
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
    standard_object.adam_learning_rate = trial.suggest_uniform(
        'adam_learning_rate',
        0,
        2,
    )
    standard_object.adam_beta_1 = trial.suggest_uniform(
        'adam_beta_1',
        0,
        1,
    )
    standard_object.adam_beta_2 = trial.suggest_uniform(
        'adam_beta_2',
        0,
        1,
    )
    standard_object.adam_epsilon = trial.suggest_uniform(
        'adam_epsilon',
        0,
        1
    )

    # Train and validate using hyper-parameters generated above
    clear_session()
    classifier_model = models.Sequential()
    classifier_model.add(layers.Conv2D(
        standard_object.number_of_conv_2d_filters,
        (standard_object.first_conv2d_kernel_dim, standard_object.second_conv2d_kernel_dim),
        activation=standard_object.conv2d_layers_activation_func,
        input_shape=(28, 28, 1),
    ))
    classifier_model.add(layers.MaxPooling2D((2, 2), strides=2))
    for level in range(standard_object.number_hidden_conv_layers):
        classifier_model.add(layers.Conv2D(
            standard_object.number_of_conv_2d_filters,
            (standard_object.first_conv2d_kernel_dim, standard_object.second_conv2d_kernel_dim),
            activation=standard_object.conv2d_layers_activation_func,
        ))
        classifier_model.add(layers.MaxPooling2D((2, 2), strides=2))
    classifier_model.add(layers.Flatten())
    classifier_model.add(layers.Dense(10, activation='softmax'))
    standard_object.optimizer = Adam(
        learning_rate=standard_object.adam_learning_rate,
        beta_1=standard_object.adam_beta_1,
        beta_2=standard_object.adam_beta_2,
        epsilon=standard_object.adam_epsilon,
        amsgrad=0,  # False
    )
    classifier_model.compile(
        optimizer=standard_object.optimizer,
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()],
    )

    output_dir = os.path.join(stem_output_dir, ('trial_' + datetime.now().strftime('%Y.%m.%d_%H.%M.%S')))
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    tb_callback = TbCallback(
        log_dir=output_dir,
        histogram_freq=1,
        write_graph=1,  # True
    )
    standard_object.callbacks.append(tb_callback)  # Append to callbacks list

    es_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=EARLY_STOPPING_SIGNIFICANT_DELTA,
        patience=EARLY_STOPPING_PATIENCE_PARAMETER,
        verbose=VERBOSITY_LEVEL_FOR_TENSORFLOW,
        mode='auto',
        baseline=None,
        restore_best_weights=1,  # True
    )
    standard_object.callbacks.append(es_callback)  # Append to callbacks list

    # Append tf.keras pruner for later use during study
    keras_pruner = optuna.integration.TFKerasPruningCallback(
        trial,
        'val_categorical_accuracy',
    )
    standard_object.callbacks.append(keras_pruner)  # Append to callbacks list

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
        use_multiprocessing=1,  # True
        workers=4,
    )

    # Evaluate performance on test data and report score
    test_metrics = classifier_model.evaluate(
        standard_object.test_images,
        standard_object.test_labels,
        batch_size=standard_object.batch_size,
    )
    test_results_dict = {output: test_metrics[i] for i, output in enumerate(classifier_model.metrics_names)}
    return(test_results_dict['categorical_accuracy'])


sampler_multivariate = optuna.samplers.TPESampler(multivariate=1)  # True
study_date_time = datetime.now().strftime('%Y.%m.%d_%H.%M')
study = optuna.create_study(
    sampler=sampler_multivariate,
    study_name=('MNIST_' + study_date_time),
    direction='maximize',
    pruner=optuna.pruners.PercentilePruner(
        percentile=PERCENTILE_FOR_PRUNING,
        n_startup_trials=NUMBER_OF_TRIALS_BEFORE_PRUNING,
        n_warmup_steps=WARMUP_EPOCHS_BEFORE_PRUNING,
    ),
)
study.optimize(
    objective,
    n_trials=MAXIMUM_NUMBER_OF_TRIALS_TO_RUN,
    timeout=MAXIMUM_SECONDS_TO_CONTINUE_STUDY,
    gc_after_trial=1,  # True
    )
set_trace()  # Before taking any more steps, pause execution


# Report completed study results:
print('Study statistics:  ')
print('\n\nBest trial number was {}\n\n'.format(study.best_trial))
print('\n\nBest score was {}\n\n...'.format(study.best_trial.value))
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
