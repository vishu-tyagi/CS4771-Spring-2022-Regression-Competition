import os
import gc
import json
import logging
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split

from dnn.config import DNNConfig
from dnn.model.mlp import MLPModel
from sentiment_analysis.utils import timing
from sentiment_analysis.utils.constants import (
    TEXT,
    TARGET,
    SPLIT,
    TRAIN,
    VAL,
    HOLDOUT,
    TEST,
    CONFIDENCE
)

logger = logging.getLogger(__name__)


class Model():
    def __init__(self, config: DNNConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.model_path = Path(os.path.join(self.current_path, config.MODEL_DIR))
        self.reports_path = Path(os.path.join(self.current_path, config.REPORTS_DIR))

        self._model = None
        # self.model_name = config.MODEL_NAME
        # self.params = None
        # self.params_name = config.MODEL_PARAMETERS_NAME

        # self.custom_evaluation = CustomEvaluation(config)
        # self.obj = None
        # self.custom_metric = None

    def _get_params(self):
        self.params = dict()
        self.params["num_boost_round"] = self.config.MODEL_NUM_BOOST_ROUND
        self.params["early_stopping_rounds"] = self.config.MODEL_EARLY_STOPPING_ROUNDS
        self.params["params"] = self.config.MODEL_PARAMETERS
        self.params["features"] = []
        self.obj = self.custom_evaluation.binary_logistic
        self.custom_metric = self.custom_evaluation.fbeta_eval
        return self

    @timing
    def save(self):
        model_file_path = os.path.join(self.model_path, self.model_name)
        logger.info(f"Saving model to {model_file_path}")
        self._model.save_model(model_file_path)

        params_file_path = os.path.join(self.model_path, self.params_name)
        logger.info(f"Saving model parameters to {params_file_path}")
        with open(params_file_path, "w") as f:
            json.dump(self.params, f)

        return self

    @timing
    def load(self):
        model_file_path = os.path.join(self.model_path, self.model_name)
        logger.info(f"Loading model from {model_file_path}")
        self._model = xgb.Booster().load_model(model_file_path)

        params_file_path = os.path.join(self.model_path, self.params_name)
        logger.info(f"Loading model parameters from {params_file_path}")
        with open(params_file_path) as f:
            self.params = json.load(f)

        self.threshold = self.params["threshold"]
        logger.info(f"Loaded model with threshold {self.threshold:0.3f}")
        return self

    def build(self, df: pd.DataFrame):
        X_train = df[df[SPLIT].isin([TRAIN])]
        Y_train = X_train[TARGET].values
        X_train = X_train.drop(columns=[TARGET]).copy()
        

        X_val = df[df[SPLIT].isin([VAL])]
        Y_val = X_val[TARGET].values
        X_val = X_val.drop(columns=[TEXT, TARGET, SPLIT]).copy()

        logger.info("Training model")
        self.train(X_train, Y_train, X_val, Y_val)

        logger.info("Computing feature importances")
        self.compute_feature_importances()

        logger.info("Discovering model threshold on holdout set")
        X = df[df[SPLIT].isin([HOLDOUT])]
        Y = X[TARGET].values
        pred_score = self._predict_prob(X)

        conf_matrix = self.custom_evaluation\
            .compute_confusion_matrix(pred_score, Y, save_to=self.confusion_matrix_path)
        with pd.option_context("display.max_rows", 50, "display.max_columns", 12):
            logger.info("Confusion matrix on holdout set:\n\n")
            logger.info(f"{conf_matrix}")

        self.threshold = self.custom_evaluation\
            .discover_threshold(conf_matrix)
        self.params["threshold"] = self.threshold
        logger.info(f"Setting threshold to {self.threshold}")

        logger.info(f"Saving ROC AUC curve on holdout set {self.roc_auc_path}")
        self.custom_evaluation.compute_roc_auc(pred_score, Y, save_to=self.roc_auc_path)

        logger.info(f"Evaluating on test set at model threshold: {self.threshold}")
        X = df[df[SPLIT].isin([TEST])]
        Y = X[TARGET].values
        pred = self.predict(X)
        scores = self.custom_evaluation.compute_scores(pred, Y)
        logger.info(f"F1 Score on test set at: {scores['F1']:0.2f}")
        logger.info(f"Accuracy on test set: {scores['Accuracy']:0.2f}")

        return self

    @timing
    def train(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        Y_val: Optional[pd.DataFrame] = None
    ):
        self._get_params()
        self.params["features"] = X_train.columns.tolist()

        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info(f"Number of training samples: {X_train.shape[0]}")
        pos_prob = sum(Y_train) / len(Y_train)
        logger.info(f"Training set label distribution: \
            pos:{pos_prob:0.2f}, neg:{1-pos_prob:0.2f}")

        d_train = xgb.DMatrix(X_train, Y_train)

        if X_val is not None:
            logger.info("Running validation...")
            logger.info(f"Number of validation samples: {X_val.shape[0]}")
            pos_prob = sum(Y_val) / len(Y_val)
            logger.info(f"Validation set label distribution: \
                pos:{pos_prob:0.2f}, neg:{1-pos_prob:0.2f}")

            d_val = xgb.DMatrix(X_val, Y_val)
            evals = [(d_train, 'train'), (d_val, 'validation')]

            self._model = xgb.train(
                params=self.params["params"],
                dtrain=d_train,
                num_boost_round=self.params["num_boost_round"],
                evals=evals,
                obj=self.obj,
                custom_metric=self.custom_metric,
                maximize=True,
                early_stopping_rounds=self.params["early_stopping_rounds"],
                verbose_eval=10
            )
            num_boost_rounds = self._model.best_iteration
            self.params["num_boost_round"] = num_boost_rounds
            logger.info(f"Learnt num_boost_round after validation:\
                {self.params['num_boost_round']}")

            del d_train, d_val, evals, self._model
            gc.collect()

            X_train = pd.concat([X_train, X_val], ignore_index=True)
            Y_train = np.concatenate([Y_train, Y_val])

            d_train = xgb.DMatrix(X_train, Y_train)
            logger.info("Now training on development (train+val) set...")

        evals = [(d_train, 'train')]

        self._model = xgb.train(
            params=self.params["params"],
            dtrain=d_train,
            num_boost_round=self.params["num_boost_round"],
            evals=evals,
            obj=self.obj,
            custom_metric=self.custom_metric,
            maximize=True,
            verbose_eval=10
        )

        del X_train, Y_train, X_val, Y_val, d_train, evals
        gc.collect()

        return self

    def _predict_prob(self, df: pd.DataFrame):
        features = self.params["features"]
        missing_features = set(features) - set(df.columns.tolist())
        if len(missing_features):
            raise ValueError(f"Features missing from given dataframe: {missing_features}")

        df = df[features].copy()
        dmatrix = xgb.DMatrix(df)

        predt = self._model.predict(dmatrix)
        y_score = expit(predt)

        return y_score

    def predict(self, df: pd.DataFrame):
        pred_score = self._predict_prob(df)
        pred = (pred_score >= self.threshold).astype(int)
        return pred

    def compute_feature_importances(self):
        return self

    def return_response(self, df: pd.DataFrame):
        score = np.asscalar(self.model.predict(df))
        sentiment = score >= self.model.threshold
        response = {CONFIDENCE: score, TARGET: sentiment}
        return response