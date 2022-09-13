import os
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dnn.config import DNNConfig
from dnn.data_access.helpers import unzip
from dnn.utils import timing
from dnn.utils.constants import (
    ID, TARGET, SPLIT, TRAIN, VAL, HOLDOUT, TEST, RAW_DIR, CSV_NAME
)

logger = logging.getLogger(__name__)


class DataClass():
    def __init__(self, config: DNNConfig):
        self.config = config

        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, config.DATA_DIR))
        self.raw_path = Path(os.path.join(self.data_path, RAW_DIR))
        self.model_path = Path(os.path.join(self.current_path, config.MODEL_DIR))
        self.reports_path = Path(os.path.join(self.current_path, config.REPORTS_DIR))

        self.zip_file_path = Path(os.path.join(self.raw_path, config.ZIP_FILE_NAME))
        self.data_file_path = Path(os.path.join(self.data_path, CSV_NAME))

    def make_dirs(self):
        dirs = [self.model_path, self.reports_path]
        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created model directory {self.model_path}")
        logger.info(f"Created reports directory {self.reports_path}")
        logger.info(f"Once unzipped, raw data will be available here: {self.raw_path}")

    @timing
    def build(self):
        logger.info("Unpacking file {zip_file_path}")
        unzip(self.zip_file_path, self.raw_path)

        logger.info("Reading raw data into CSV for pre-processing")
        train_examples = pd.read_csv(
            os.path.join(self.raw_path, self.config.ZIP_RELEVANT_FILES["train_examples"])
        )
        train_labels = pd.read_csv(
            os.path.join(self.raw_path, self.config.ZIP_RELEVANT_FILES["train_labels"])
        )
        test_examples = pd.read_csv(
            os.path.join(self.raw_path, self.config.ZIP_RELEVANT_FILES["test_examples"])
        )
        train_examples[SPLIT] = TRAIN
        train_examples[TARGET] = train_labels[TARGET]
        test_examples[SPLIT] = TEST
        test_examples[TARGET] = np.nan

        logger.info("Splitting data available for model-building into train and validation sets")
        train_examples, val_examples = train_test_split(
            train_examples,
            test_size=self.config.SPLIT_VALIDATION_SIZE,
            shuffle=True,
            random_state=self.config.SPLIT_RANDOM_SEED
        )
        val_examples[SPLIT] = VAL

        df = pd.concat([train_examples, val_examples, test_examples], ignore_index=True).copy()
        df.drop(columns=[ID], inplace=True)
        df.to_csv(self.data_file_path, index=False, mode="w+")
        logger.info(f"Saved CSV to {self.data_file_path}")
        logger.info("Ready for pre-processing")

        del train_examples, val_examples, test_examples, train_labels
        gc.collect()
        return df


class BasicDataset(TensorDataset):
    def __init__(self, data, target):
        self.x = data
        self.y = target
        self.n_samples = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.n_samples

