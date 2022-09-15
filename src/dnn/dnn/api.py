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
    df_features = features.build()
    # Train model and generate submission file
    model = Model(config)
    model.build()

    # Save features
    features.save()
    # Save model
    model.save()


# def server(config: DNNConfig = DNNConfig) -> Callable[[dict], dict]:
#     # Load features
#     features = Features(config)
#     features.load()

#     # Load model
#     model = XGBoostModel(config)
#     model.load()

#     def predict(body: dict) -> dict:
#         df = pd.DataFrame([body])
#         df_features = features.transform(df)
#         return model.return_response(df_features)

#     return predict
