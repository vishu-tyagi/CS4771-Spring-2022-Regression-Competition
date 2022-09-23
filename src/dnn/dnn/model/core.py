import os
import gc
import sys
import json
import logging
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dnn.config import DNNConfig
from dnn.data_access import BasicDataset
from dnn.model.mlp import MLPModel
from dnn.utils import timing
from dnn.utils.constants import (
    ID, TARGET, SPLIT, TRAIN, VAL, TEST, TRAINED_DIR, SUBMISSION_CSV, LOSS_CURVE
)

sns.set_theme(style="darkgrid")
logger = logging.getLogger(__name__)


class Model():
    def __init__(self, config: DNNConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.model_path = Path(os.path.join(self.current_path, config.MODEL_DIR))
        self.trained_models_path = Path(os.path.join(self.model_path, TRAINED_DIR))
        self.reports_path = Path(os.path.join(self.current_path, config.REPORTS_DIR))

        self._model = None
        self.state_dict_name = config.MODEL_STATE_DICT_NAME
        self.params = None
        self.params_name = config.MODEL_PARAMETERS_NAME

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config.TRAIN_BATCH_SIZE
        self.learning_rate = self.config.TRAIN_LEARNING_RATE
        self.epochs = self.config.TRAIN_EPOCHS
        self.num_workers = self.config.TRAIN_NUM_WORKERS

        self.criterion = None
        self.optimizer = None

        self.num_features = None
        self.num_train_samples = None
        self.num_val_samples = None

        self.train_loss = None
        self.val_loss = None
        self.best_epoch = None

    def _make_data_loader(
        self,
        X: pd.DataFrame,
        Y: np.ndarray,
        context: Optional[str] = None,
        shuffle: Optional[bool] = True
    ):
        X = X.to_numpy().astype(np.float32)
        X = torch.from_numpy(X)
        Y = Y.reshape(-1,).astype(np.float32)
        Y = torch.from_numpy(Y)

        if context == "validation":
            shuffle = False

        data = BasicDataset(X, Y)
        data_loader = DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )
        return data_loader

    def _plot_loss_curve(self):
        fig = plt.figure(figsize=(8, 6))
        epochs = [(i + 1) for i in range(self.epochs)]
        plt.plot(epochs, self.train_loss, label="Train Loss", color="orange")
        plt.plot(epochs, self.val_loss, label="Validation Loss", color="blue")
        best_epoch = np.argmin(self.val_loss) + 1
        plt.axvline(x=best_epoch, label="Best Epoch", color="darkgreen", linestyle="dashdot")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (L1)")
        plt.title(f"Best Epoch: {best_epoch}")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.reports_path, LOSS_CURVE), bbox_inches="tight")
        plt.close()

    @timing
    def save(self):
        state_dict_path = os.path.join(self.model_path, self.state_dict_name)
        logger.info(f"Saving model state dictionary to {state_dict_path}")
        torch.save(self._model.state_dict(), state_dict_path)

        params_file_path = os.path.join(self.model_path, self.params_name)
        logger.info(f"Saving model parameters to {params_file_path}")
        with open(params_file_path, "w") as f:
            json.dump(self.params, f)

    @timing
    def load(self):
        params_file_path = os.path.join(self.model_path, self.params_name)
        logger.info(f"Loading model parameters from {params_file_path}")
        with open(params_file_path) as f:
            self.params = json.load(f)

        self._model = MLPModel(**self.params)
        state_dict_path = os.path.join(self.model_path, self.state_dict_name)
        logger.info(f"Loading model from {state_dict_path}")
        self._model.load_state_dict(torch.load(state_dict_path))

    @timing
    def build(self, df: pd.DataFrame):
        X_train = df[df[SPLIT].isin([TRAIN])]
        Y_train = X_train[TARGET].values
        X_train = X_train.drop(columns=[SPLIT, TARGET])

        X_val = df[df[SPLIT].isin([VAL])]
        Y_val = X_val[TARGET].values
        X_val = X_val.drop(columns=[SPLIT, TARGET])

        logger.info("Training model")
        self.train(X_train, Y_train, X_val, Y_val)

        logger.info(f"Predicting on test set")
        X = df[df[SPLIT].isin([TEST])]
        X = X.drop(columns=[SPLIT, TARGET])
        pred = self.predict(X)

        logger.info("Generating submission for kaggle competition")
        submission = pd.DataFrame({
            ID: [i for i in range(pred.shape[0])],
            TARGET: pred
        })
        submission_path = os.path.join(self.reports_path, SUBMISSION_CSV)
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved to {submission_path}")

    @timing
    def train(
        self,
        X_train: pd.DataFrame,
        Y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        Y_val: Optional[np.ndarray] = None
    ):
        self.params = dict()
        self.params["dimensions"] = [X_train.shape[1]]
        self.params["dimensions"].extend(self.config.MODEL_HIDDEN_DIMENSIONS)
        self.params["dimensions"].append(1)
        self.params["dropout"] = self.config.MODEL_DROPOUT_SIZE
        self.params["features"] = X_train.columns.tolist()

        self._model = MLPModel(**self.params)
        self.criterion = nn.L1Loss(reduction="mean")
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        self.num_features = X_train.shape[1]
        self.num_train_samples = X_train.shape[0]
        self.num_val_samples = X_val.shape[0]
        logger.info(f"Number of features: {self.num_features}")
        logger.info(f"Number of training samples: {self.num_train_samples}")
        logger.info(f"Number of validation samples: {self.num_val_samples}")

        train_loader = self._make_data_loader(X_train, Y_train)
        val_loader = self._make_data_loader(X_val, Y_val, context="validation")

        del X_train, Y_train, X_val, Y_val
        gc.collect()

        self._train(train_loader, val_loader)
        return self

    def _train(self, train_loader, val_loader):
        # Move model to device
        self._model.to(self.device)
        # Two lists to keep the losses at the end of each epoch
        train_loss, val_loss = list(), list()
        for epoch in range(self.epochs):
            # Dummy lists to keep the losses at the end of each iteration
            # (one batch forward and backward process)
            train_batch_loss, val_batch_loss = list(), list()
            # Training mode
            self._model.train()
            for i, (x, y) in enumerate(train_loader):
                # Move input to device
                x = x.to(self.device)
                y = y.to(self.device)
                # Forward pass
                outputs = self._model(x).reshape(-1,)
                # Calculate loss
                loss = self.criterion(outputs, y)
                # Reset gradients
                self.optimizer.zero_grad()
                # Backward pass (backpropogation)
                loss.backward()
                # Update weights
                self.optimizer.step()
                train_batch_loss.append(loss.item())
                message = \
                    f"EPOCH:{epoch+1}/{self.epochs}, " + \
                    f"step:{i+1}/{(self.num_train_samples//self.batch_size)+1}, " + \
                    f"loss={loss.item():.4f}"
                print(message, end="\r", file=sys.stderr)
            # Take the average of iteration losses and append it
            # to the epoch losses list
            train_loss.append(np.array(train_batch_loss).mean())
            # Evaluate model
            self._model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    # Move input to device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # Forward pass
                    outputs = self._model(x).reshape(-1,)
                    # Calculate loss
                    loss = self.criterion(outputs, y)
                    val_batch_loss.append(loss.item())
                    message = \
                        f"EPOCH:{epoch+1}/{self.epochs}, " + \
                        f"step:{i+1}/{(self.num_train_samples//self.batch_size)+1}, " + \
                        f"loss={loss.item():.4f}"
                    print(message, end="\r", file=sys.stderr)
            # Take the average of iteration losses and append it
            # to the epoch losses list
            val_loss.append(np.array(val_batch_loss).mean())
            message = \
                f"EPOCH:{epoch+1}/{self.epochs} - " + \
                f"Training Loss: {train_loss[-1]:.4f}, " + \
                f"Validation Loss: {val_loss[-1]:.4f}"
            print(message, file=sys.stderr)
            # Save model
            state = f"epoch_{epoch+1:03}.pth"
            state_dict_path = os.path.join(self.trained_models_path, state)
            torch.save(self._model.state_dict(), state_dict_path)
        # Save losses for plotting train-val curve
        self.train_loss = train_loss
        self.val_loss = val_loss
        self._plot_loss_curve()
        # Load the best model based on validation loss
        best_epoch = np.argmin(self.val_loss) + 1
        logger.info(f"Best epoch: {best_epoch}")
        state = f"epoch_{best_epoch:03}.pth"
        state_dict_path = os.path.join(self.trained_models_path, state)
        state_dict = torch.load(state_dict_path)
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()
        return self

    @timing
    def predict(self, X: pd.DataFrame):
        features = self.params["features"]
        missing_features = set(features) - set(X.columns.tolist())
        if len(missing_features):
            raise ValueError(f"Features missing from given dataframe: {missing_features}")

        X = X[features].copy()
        X = X.to_numpy().astype(np.float32)
        X = torch.from_numpy(X)

        self._model.to(self.device)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X.to(self.device))
        pred = pred.to("cpu").numpy().reshape(-1,)
        return pred
