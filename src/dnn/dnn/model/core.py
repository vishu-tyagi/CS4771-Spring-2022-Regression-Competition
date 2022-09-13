import os
import gc
import json
import logging
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dnn.config import DNNConfig
from dnn.data_access import BasicDataset
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

        self.num_features = None
        self.num_train_samples = None
        self.num_val_samples = None

        self._model = None
        self.device = None
        self.batch_size = None
        self.learning_rate = None
        self.epochs = None
        self.num_wokers = None
        self.criterion = None
        self.optimizer = None
        # self.model_name = config.MODEL_NAME
        # self.params = None
        # self.params_name = config.MODEL_PARAMETERS_NAME

        # self.custom_evaluation = CustomEvaluation(config)
        # self.obj = None
        # self.custom_metric = None

    def _get_params(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config.TRAIN_BATCH_SIZE
        self.learning_rate = self.config.TRAIN_LEARNING_RATE
        self.epochs = self.config.TRAIN_EPOCHS
        self.num_wokers = self.config.TRAIN_NUM_WORKERS
        self.criterion = self.config.TRAIN_CRITERION
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )
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
        Y_train = X_train[TARGET].values.reshape(-1,)
        X_train = X_train.drop(columns=[SPLIT, TARGET])

        X_val = df[df[SPLIT].isin([VAL])]
        Y_val = X_val[TARGET].values.reshape(-1,)
        X_val = X_val.drop(columns=[SPLIT, TARGET])

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
        self._model = MLPModel(X_train.shape[1], 1, self.config)
        self._get_params()

        self.num_features = X_train.shape[1]
        self.num_train_samples = X_train.shape[0]
        self.num_val_samples = X_val.shape[0]
        logger.info(f"Number of features: {self.num_features}")
        logger.info(f"Number of training samples: {self.num_train_samples}")
        logger.info(f"Number of validation samples: {self.num_val_samples}")

        X_train = X_train.to_numpy().astype(np.float32)
        X_train = torch.from_numpy(X_train)
        Y_train = Y_train.reshape(-1,).astype(np.float32)
        Y_train = torch.from_numpy(Y_train)

        X_val = X_val.to_numpy().astype(np.float32)
        X_val = torch.from_numpy(X_val)
        Y_val = Y_val.reshape(-1,).astype(np.float32)
        Y_val = torch.from_numpy(Y_val)

        train = BasicDataset(X_train, Y_train)
        train_loader = DataLoader(
            dataset=train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        val = BasicDataset(X_val, Y_val)
        val_loader = DataLoader(
            dataset=val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        del X_train, Y_train, X_val, Y_val, train, val
        gc.collect()

        val_loss = self._train(train_loader, val_loader)
        
        return self

    def _train(self, train_loader, val_loader):
        min_val_loss = np.Inf
        train_loss, val_loss = [], []

        for epoch in range(self.epochs):
            self._model.train()
            train_batch_loss = []
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self._model(x).reshape(-1,)
                loss = self.criterion(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_batch_loss.append(loss.item())
                logger.info(
                    f"EPOCH:{epoch+1}/{self.epochs}, step:{i+1}/{self.num_train_samples//self.batch_size}, loss={loss.item():.4f}", end="\r"
                )
            train_loss.append(np.array(train_batch_loss).mean())

            self._model.eval()
            val_batch_loss = []

            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    outputs = self._model(x).reshape(-1,)
                    loss = self.criterion(outputs, y)

                    val_batch_loss.append(loss.item())
                    logger.info(
                        f"EPOCH:{epoch+1}/{self.epochs}, step:{i+1}/{self.num_train_samples//self.batch_size}, loss={loss.item():.4f}", end="\r"
                    )
            val_loss.append(np.array(val_batch_loss).mean())
            logger.info(
                f"EPOCH:{epoch+1}/{self.epochs} - Training Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}"
            )

            model_state = f"{self._model.name}_epoch_{epoch+1:03}.pth"
            model_state_path = os.path.join(self.path.join(self.model_path, model_state))
            torch.save(self._model.state_dict(), model_state_path)

        return val_loss

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