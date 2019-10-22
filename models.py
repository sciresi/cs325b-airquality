from abc import ABC, abstractmethod
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
import copy
from sklearn import base
import matplotlib.pyplot as plt


class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def test(self, X_test, y_test):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass

class ScikitLearnModel(Model):
    def __init__(self, model):
        super(ScikitLearnModel, self).__init__()
        self.model = base.clone(model)
        
    def train(self, X_train, y_train):
        self.model = self.model.fit(X_train, y_train)
        
    def test(self, X_test, y_test):
        return self.model.score(X_test, y_test)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
class PyTorchModel(Model):
    NUM_ITERATIONS = 1000
    LEARNING_RATE = 1
    DEFAULT_OPTIMIZER = optim.SGD
    DEFAULT_REGRESSION_LOSS = nn.MSELoss()
    DEFAULT_CLASSIFIER_LOSS = nn.CrossEntropyLoss()
    
    def __init__(self, model, optimizer = None, loss_fn = None, normalize = True, classifier = True):
        super(PyTorchModel, self).__init__()
        self.model = copy.deepcopy(model)
        self.optimizer = optimizer or self.DEFAULT_OPTIMIZER(self.model.parameters(), lr=self.LEARNING_RATE)
        self.loss_fn = loss_fn or self.DEFAULT_CLASSIFIER_LOSS if classifier else self.DEFAULT_REGRESSION_LOSS
        self.train_loss = []
        self.normalize = normalize
        
    def normalize_data(self, data, recompute):
        if recompute:
            self.means = data.mean(axis=0)
            self.stds = data.std(axis=0)
            self.stds[self.stds == 0] = 1
        return (data - self.means) / self.stds
        
    def train(self, X_train, y_train, num_iterations = None):
        num_iterations = num_iterations or self.NUM_ITERATIONS
        self.model = self.model.train()
        if self.normalize:
            X_train = self.normalize_data(X_train, recompute = True)
            
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        self.train_loss.clear()
        for epoch in range(num_iterations):
            self.optimizer.zero_grad()
            output = self.model(X_train)
            loss = self.loss_fn(output, y_train)
            self.train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
    
    def test(self, X_test, y_test):
        if self.normalize:
            X_test = self.normalize_data(X_test, recompute = False)
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
        return self.loss_fn(self.model(X_test), y_test).item()
    
    def predict(self, X_test):
        if self.normalize:
            X_test = self.normalize_data(X_test, recompute = False)
        X_test = torch.from_numpy(X_test).float()
        logits = self.model(X_test)
        axis = len(logits.shape) - 1
        return logits.argmax(axis=axis)
    
    def print_loss_curve(self):
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.plot(self.train_loss)
    
class EPASplittingModel(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(EPASplittingModel, self).__init__()
        self.input_to_hidden = nn.Linear(num_features, hidden_size)
        self.hidden_to_logits = nn.Linear(hidden_size, 2)
        
    def forward(self, data):
        relu_input = self.input_to_hidden(data)
        hidden = F.relu(relu_input)
        logits = self.hidden_to_logits(hidden)
        return logits
