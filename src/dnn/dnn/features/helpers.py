from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from dnn.config import DNNConfig


def processor_init(config: DNNConfig):
    numeric_features = config.NUMERIC_FEATURES
    numeric_transformer = Pipeline(steps=[(config.NUMERIC_SCALAR, StandardScaler())])

    nominal_features = config.NOMINAL_FEATURES
    nominal_transformer = OneHotEncoder(
        categories=config.NOMINAL_CATEGORIES, handle_unknown=config.NOMINAL_HANDLE_UNKNOWN
    )

    # Remaining features will be dropped by default
    preprocessor = ColumnTransformer(
        transformers=[
            (config.NUMERIC_TRANSFORMER, numeric_transformer, numeric_features),
            (config.NOMINAL_TRANSFORMER, nominal_transformer, nominal_features)])

    return preprocessor
