import logging
from typing import Callable

from dnn.config import DNNConfig
from dnn.data_access import DataClass
from dnn.features import Features
from dnn.model import Model

logger = logging.getLogger(__name__)


def train(config: DNNConfig = DNNConfig) -> None:
    # Parse raw data for pre-processing
    data = DataClass(config)
    data.make_dirs()
    df = data.build()

    # Preprocess and build features
    features = Features(config)
    df_features = features.build(df)
    # Train model and generate submission file
    model = Model(config)
    model.build(df_features)

    # Save features
    features.save()
    # Save model
    model.save()
