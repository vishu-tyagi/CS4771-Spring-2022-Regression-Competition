import torch
import torch.nn as nn


class MLPModel(nn.Module):

    def __init__(self, name, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size
        self.name = name

        '''model layers'''
        super(MLPModel, self).__init__()
        self.l1 = nn.Linear(self.input_size, 784)
        self.l2 = nn.Linear(784, 624)
        self.l3 = nn.Linear(624, 312)
        self.l4 = nn.Linear(312, 156)
        self.l5 = nn.Linear(156, 78)
        self.l6 = nn.Linear(78, self.output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        
        '''model architecture'''
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l4(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l5(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l6(out)

        return out


