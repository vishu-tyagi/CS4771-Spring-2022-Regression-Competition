import os
from pathlib import Path

os.chdir('/content/drive/MyDrive/COMS4771')
DATA_PATH = Path.cwd() / 'data'
RAW = DATA_PATH / 'raw'
PROCESSED = DATA_PATH / 'processed'
SUBMISSION = DATA_PATH / 'submission'

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from models import *



class BasicDataset(TensorDataset):
    def __init__(self, data, target):
        self.x = data
        self.y = target
        self.n_samples = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.n_samples


def preprocess(data=None):
    
    data[['day']] = data.apply({'feature_0': lambda x: int(x.split(' ')[0].split('-')[1])})
    data[['month']] = data.apply({'feature_0': lambda x: int(x.split(' ')[0].split('-')[0])})
    data[['hour']] = data.apply({'feature_0': lambda x: int(x.split(' ')[1].split(':')[0])})

    return data


def load_data(preprocess_flag=True):

    xdev = pd.read_csv(RAW / 'train_examples.csv')
    if preprocess_flag:
        xdev = preprocess(xdev)

    ydev = pd.read_csv(RAW / 'train_labels.csv')
    ydev = np.array(ydev.duration.values).astype(np.float32)

    xtest = pd.read_csv(RAW / 'test_examples.csv')
    if preprocess_flag:
        xtest = preprocess(xtest)

    return xdev, ydev, xtest


def process_and_split(xdev=None, ydev=None, xtest=None, random_seed=1024):

    numeric_features = ['feature_2', 'feature_8', 'feature_9']
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    nominal_features = ['feature_1', 'feature_3', 'feature_4', 'hour', 'month', 'day']
    nominal_transformer = OneHotEncoder(
        handle_unknown='ignore',
        categories=[[i for i in range(10)], [i for i in range(1, 266)],
                    [i for i in range(1, 266)], [i for i in range(24)],
                    [i for i in range(1, 13)], [i for i in range(1, 32)]]
                    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('nominal', nominal_transformer, nominal_features)]) # remaining columns will be dropped by default

    xtrain, xval, ytrain, yval = train_test_split(xdev, ydev, test_size=.2, random_state=random_seed)

    xtrain = torch.from_numpy(preprocessor.fit_transform(xtrain).toarray().astype(np.float32))
    ytrain = torch.from_numpy(ytrain.reshape(-1,).astype(np.float32))

    xval = torch.from_numpy(preprocessor.transform(xval).toarray().astype(np.float32))
    yval = torch.from_numpy(yval.reshape(-1,).astype(np.float32))

    xtest = torch.from_numpy(preprocessor.transform(xtest).toarray().astype(np.float32))

    return xtrain, ytrain, xval, yval, xtest


def make_data_loader(xtrain=None, ytrain=None, xval=None, yval=None, batch_size=764):

    train = BasicDataset(xtrain, ytrain)
    val = BasicDataset(xval, yval)

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader


def build_model(input_size=None, output_size=1):
    # model instance
    model = MLPModel(name='mlp', input_size=input_size, output_size=1)

    # If GPU available, move the model to GPU.
    if torch.cuda.is_available():
        model.to(device)

    return model


def train(model=None, train_loader=None, valid_loader=None, learning_rate=0.001, EPOCHS=100, batch_size=764, train_shape=None):

    criterion = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    min_valid_loss = np.Inf

    train_loss = []
    valid_loss = []

    for epoch in range(EPOCHS):
        model.train()
        train_batch_loss = []

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x).reshape(-1,)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_batch_loss.append(loss.item())

            print(f'EPOCH:{epoch+1}/{EPOCHS}, step:{i+1}/{train_shape//batch_size}, loss={loss.item():.4f}', end='\r')
        
        train_loss.append(np.array(train_batch_loss).mean())
        
        model.eval()
        valid_batch_loss = []

        with torch.no_grad():
            for i, (x, y) in enumerate(valid_loader):  
                x = x.to(device)
                y = y.to(device)

                outputs = model(x).reshape(-1,)
                loss = criterion(outputs, y)
                
                valid_batch_loss.append(loss.item())

                print(f'EPOCH:{epoch+1}/{EPOCHS}, step:{i+1}/{train_shape//batch_size}, loss={loss.item():.4f}', end='\r')

        valid_loss.append(np.array(valid_batch_loss).mean())
        
        print(f'EPOCH:{epoch+1}/{EPOCHS} - Training Loss: {train_loss[-1]}, Validation Loss: {valid_loss[-1]}')

        torch.save(model.state_dict(), f'models/{model.name}_epoch_{epoch+1:03}.pth')
        
    return valid_loss


def predict(model=None, xtest=None, valid_loss=None, file_name=None):
    best_epoch = np.argmin(valid_loss) + 1 
    print(f'Best epoch is epoch: {best_epoch}')

    state_dict = torch.load(f'models/{model.name}_epoch_{best_epoch:03}.pth')

    model.load_state_dict(state_dict)
    model.to(device);

    with torch.no_grad():
        ytest = model(xtest.to(device)).to('cpu').numpy()
    
    submission = pd.DataFrame({'id':[i for i in range(ytest.shape[0])], 'duration': ytest.reshape(-1,)})
    submission.to_csv(SUBMISSION / file_name, index=False)
    print('Predictions saved')
    
    return ytest


def ensemble(files=None, test_size=None, file_name=None):
    N = len(files)
    nu = np.array([0 for _ in range(test_size)]).astype(np.float32)

    for file_ in files:
        pred = pd.read_csv(SUBMISSION / file_)
        nu += (1/N) * pred.duration.values.reshape(-1,)

    submission = pd.DataFrame({'id':[i for i in range(nu.shape[0])], 'duration': nu.reshape(-1,)})
    submission.to_csv(SUBMISSION / file_name, index=False)

