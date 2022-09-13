import os
import gc
import logging
import pickle
from pathlib import Path
from typing import Optional
from urllib.request import ProxyDigestAuthHandler

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import TensorDataset

from dnn.config import DNNConfig
from dnn.features.helpers import processor_init
from dnn.utils import timing
from dnn.utils.constants import (
    F0, TARGET, SPLIT, TRAIN, HOUR, DAY, MONTH
)

logger = logging.getLogger(__name__)


class Features(BaseEstimator, TransformerMixin):
    def __init__(self, config: DNNConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.model_path = Path(os.path.join(self.current_path, config.MODEL_DIR))

        self.processor = processor_init(config)
        self.processor_path = os.path.join(self.model_path, config.PROCESSOR_FILE_NAME)

    @timing
    def build(self, df: pd.DataFrame):
        df = df.copy()
        df[HOUR] = df[F0].apply(lambda x: int(x.split(" ")[0].split('-')[0]))
        df[DAY] = df[F0].apply(lambda x: int(x.split(" ")[0].split('-')[1]))
        df[MONTH] = df[F0].apply(lambda x: int(x.split(" ")[0].split('-')[0]))

        logger.info("Fitting processor on train set")
        train = df[df[SPLIT].isin([TRAIN])]
        self.fit(train)

        logger.info("Transforming data")
        Y = df[[SPLIT, TARGET]]
        X = self.transform(df)
        df = pd.concat([X, Y], axis=1).copy()

        del X, Y, train
        gc.collect()
        return df

    @timing
    def fit(self, df: pd.DataFrame):
        return self._featurize(df, "build")

    @timing
    def transform(self, df: pd.DataFrame):
        return self._featurize(df)

    @timing
    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def save(self):
        logger.info(f"Saving processor to {self.processor_path}")
        pickle.dump(self.processor, open(self.processor_path, "wb"))
        return self

    @timing
    def load(self):
        logger.info(f"Loading processor from {self.processor_path}")
        self.vectorizer = pickle.load(open(self.processor_path, "rb"))
        return self

    def _featurize(self, df: pd.DataFrame, context: Optional[str] = None):
        if context == "build":
            self.processor.fit(df)
            return self

        X = self.processor.transform(df)
        X = pd.DataFrame(X.toarray(), columns=self.processor.get_feature_names_out())
        return X
