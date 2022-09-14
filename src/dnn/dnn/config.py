from dnn.utils.constants import (
    F1, F2, F3, F4, F8, F9, DAY, MONTH, HOUR
)


class DNNConfig():
    CURRENT_PATH = None   # will be set to working directory by os.getcwd()

    # All directories below should be given with respect to working directory (CURRENT_DIR)
    # Data directory
    DATA_DIR = "data/"

    # Model directory for saving model
    MODEL_DIR = "model/"

    # Reports directory for saving artefacts
    REPORTS_DIR = "reports/"

    # ZIP File
    ZIP_FILE_NAME = "coms4771-spring-2022-regression-competition.zip"
    ZIP_FOLDER = "coms4771-spring-2022-regression-competition"
    ZIP_RELEVANT_FILES = {
        "train_examples": "train_examples.csv",
        "train_labels": "train_labels.csv",
        "test_examples": "test_examples.csv"
    }

    # Model selection
    SPLIT_VALIDATION_SIZE = .2
    SPLIT_RANDOM_SEED = 1024

    # Pre-processing
    PROCESSOR_FILE_NAME = "processor.pickle"
    NUMERIC_TRANSFORMER = "numeric"
    NUMERIC_SCALAR = "scalar"
    NUMERIC_FEATURES = [F2, F8, F9]
    NOMINAL_TRANSFORMER = "nominal"
    NOMINAL_FEATURES = [F1, F3, F4, HOUR, MONTH, DAY]
    NOMINAL_CATEGORIES = [
        [i for i in range(10)],
        [i for i in range(1, 266)],
        [i for i in range(1, 266)],
        [i for i in range(24)],
        [i for i in range(1, 13)],
        [i for i in range(1, 32)]
    ]
    NOMINAL_HANDLE_UNKNOWN = "ignore"

    # Model
    MODEL_STATE_DICT_NAME = "mlp.pth"
    MODEL_PARAMETERS_NAME = "mlp_params.json"
    MODEL_HIDDEN_DIMENSIONS = [784, 624, 312, 156, 78]
    MODEL_DROPOUT_SIZE = .2

    # Train
    TRAIN_BATCH_SIZE = 764
    TRAIN_LEARNING_RATE = 0.001
    TRAIN_EPOCHS = 10
    TRAIN_NUM_WORKERS = 2
