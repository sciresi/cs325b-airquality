import os
from abc import ABC, abstractmethod
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
import copy
from sklearn import base
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class Model(ABC):
    @abstractmethod
    def train(self, train_data, **kwargs):
        pass
    
    @abstractmethod
    def test(self, test_data):
        pass
    
    @abstractmethod
    def predict(self, data):
        pass

class Model(ABC):
    def __init__(self, experiment_name = "Experiment", log_file = None):
        self.model = None
        self.experiment_name = experiment_name
        self.log_file = log_file
        self.train_losses = []
        self.val_losses = None
    
    def __getattr__(self, attr):
        return getattr(self.model, attr)
    
    def log(self, string):
        if self.log_file:
            self.log_file.write(string + "\n")
        print(string)
        
    def set_log_file(self, log_file):
        self.log_file = log_file
        
    def plot_loss_curve(self, plot_save_path = None):
        """
        Plots the training losses (and validation if present) for the experiment.
        If a save path is specified, the plot is saved to that location.

        Parameters
        ----------
        plot_save_path : str
            Path to save the plotted losses.
        """
        plt.plot(np.arange(len(self.train_losses)), self.train_losses, label="Train loss")
        if self.val_losses:
            plt.plot(np.arange(len(self.val_losses)), self.val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average loss")
        title = "{}: Training ".format(self.experiment_name)
        if self.val_losses:
            title += "and validation "
        title += "losses"
        plt.title(title)
        plt.legend(loc = "upper right")
        plt.show() # todo: remove
        if plot_save_path:
            self.log("Saving loss graph to {}".format(plot_save_path))
            plt.savefig(plot_save_path)
        else:
            plt.show()

    @abstractmethod
    def train(self, train_data, **kwargs):
        pass
    
    @abstractmethod
    def test(self, test_data):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass
    
    @abstractmethod
    def save(self, save_path, model_name = None):
        pass
    
    # TODO: make this abstract and implement it
    #@abstractmethod
    def load(self, load_path):
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
    NUM_EPOCHS = 1000
    LEARNING_RATE = 1e-1
    DEFAULT_OPTIMIZER = optim.SGD
    DEFAULT_REGRESSION_LOSS = nn.MSELoss()
    DEFAULT_CLASSIFIER_LOSS = nn.CrossEntropyLoss()
    
    # TODO: maybe omit all the optional parameters and just do **kwargs
    def __init__(self, model, optimizer = None, loss_fn = None, 
                 normalize = True, is_classifier = False, use_cuda = True, 
                 log_file = None):
        super(PyTorchModel, self).__init__(log_file)
        self.model = copy.deepcopy(model)
        self.set_cuda(use_cuda)
            
        self.optimizer = optimizer or self.DEFAULT_OPTIMIZER(self.model.parameters(), lr=self.LEARNING_RATE)
        self.is_classifier = is_classifier
        self.loss_fn = loss_fn or self.DEFAULT_CLASSIFIER_LOSS if is_classifier else self.DEFAULT_REGRESSION_LOSS
        self.normalize = normalize
        
    def __call__(self, *input_, **kwargs):
        return self.model(*input_, **kwargs)
    
    def set_loss_fn(self, loss_fn):
        """
        Sets the model's loss function.
        
        Parameters
        ----------
        loss_fn : torch.nn.modules.loss._Loss
            Loss function to use for training.
        """
        self.loss_fn = loss_fn
    
    def set_optimizer(self, optimizer):
        """
        Sets the model's optimizer.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to use for training.
        """
        self.optimizer = optimizer
        
    def set_cuda(self, use_cuda):
        """
        Attempts to set the model to use GPU.
        
        Parameters
        ----------
        use_cuda : bool
            Whether to use GPU. Note that CUDA must actually be available on
            the machine for GPU use.
        """
        if use_cuda and torch.cuda.is_available():
            self.log("CUDA is available, model will use GPU...")
            self.model = self.model.cuda()
            self.use_cuda = True
        else:
            self.log("Model will use CPU...")
            self.use_cuda = False
        
    # TODO: figure out normalization for torch Dataset
    def normalize_data(self, data, recompute):
        if recompute:
            self.means = data.mean(axis=0)
            self.stds = data.std(axis=0)
            self.stds[self.stds == 0] = 1
        return (data - self.means) / self.stds
    
    def run_dataset(self, dataloader, is_train = False):
        """
        Runs the model over the entire dataset and returns the average loss
        (i.e., the sum of losses for each data point divided by the length of
        the dataset).
        
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader to iterate over.
        is_train : bool, optional
            Whether we are training on this data (i.e., computing gradients).
            This defaults to False to avoid inadvertently training on
            non-training data.
            
        Returns
        -------
        average_loss : float
            The sum of losses over the entire dataset divided by the length of
            the dataset.
        """
        total_loss = 0
        num_batches = len(dataloader)
        for i, batch in enumerate(dataloader):
            
            non_image_batch, labels_batch = batch["non_image"], batch["label"]
            image_batch = batch.get("image")
            if self.use_cuda:
                non_image_batch, labels_batch = non_image_batch.cuda(), labels_batch.cuda()
                if image_batch is not None:
                    image_batch = image_batch.cuda()
            output_batch = self.model((non_image_batch, image_batch))
            batch_loss = self.loss_fn(output_batch, labels_batch)
            if is_train: 
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            total_loss += batch_loss.item()
            if (i + 1) % 10 == 0 and is_train: # TODO: custom print-every
                self.log("Finished batch {} out of {}".format(i + 1, num_batches))
            
        return total_loss / len(dataloader)
        
        
    def train(self, train_dataloader, num_epochs = None, val_dataloader = None):
        """
        Trains the model.
        
        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader storing the training data to iterate over.
        num_epochs : int, optional
            Number of epochs to train for. Defaults the constant NUM_EPOCHS
            if not provided.
        val_dataloader : torch.utils.data.DataLoader
            DataLoader storing the validation data to evaluate on after
            every epoch. If not provided, no validation is performed.
        """
        assert isinstance(train_dataloader, torch.utils.data.DataLoader), "Must use DataLoader"
        num_epochs = num_epochs or self.NUM_EPOCHS
        self.model = self.model.train()
        if self.normalize:
            # TODO: figure out normalization for torch Dataset
            pass
            
        self.train_losses = []
        if val_dataloader:
            assert isinstance(val_dataloader, torch.utils.data.DataLoader), "Must use DataLoader"
            self.val_losses = []
        print("-" * 80)
        print("Training for {} epochs".format(num_epochs))
        for epoch in range(num_epochs):
            print("Running epoch {}...".format(epoch))
            epoch_loss = self.run_dataset(train_dataloader, is_train = True)
            self.train_losses.append(epoch_loss)
            if val_dataloader:
                self.model = self.model.eval()
                val_loss = self.run_dataset(val_dataloader, is_train = False)
                self.log("Validation loss: {}".format(val_loss))
                self.val_losses.append(val_loss)
                self.model = self.model.train()
        print("Finished training!")
        print("-" * 80)
    
    def test(self, test_dataloader):
        """
        Runs the model on a test set.
        
        Parameters
        ----------
        test_dataloader : torch.utils.data.DataLoader
            DataLoader storing the test data to iterate over.
            
        Returns
        -------
        test_loss : float
            The sum of losses over the entire dataset divided by the number
            of points in the dataset.
        """
        assert isinstance(test_dataloader, torch.utils.data.DataLoader), "Must use DataLoader"
        
        if self.normalize:
            # TODO: figure out normalization for torch Dataset
            pass
        self.model.eval()
        return self.run_dataset(test_dataloader, is_train = False).item()
    
    # TODO: figure out how to take in the input. Then rework for both
    # classification and regression.
    def predict(self, X_test):
        self.model = self.model.eval()
        if self.normalize:
            X_test = self.normalize_data(X_test, recompute = False)
        X_test = torch.from_numpy(X_test).float()
        logits = self.model(X_test)
        axis = len(logits.shape) - 1
        return logits.argmax(axis=axis)
    
    # TODO: add checkpoint functionality. Maybe new method altogether?
    # Since may not be applicable to ScikitLearn.
    # Maybe have this save a bunch of metadata about the Model too?
    def save(self, model_save_path):
        """
        Saves the PyTorch model to the specified path.
        
        Parameters
        ----------
        model_save_path : str
            Path to save the model.
        """
        if not model_save_path.endswith(".pt"):
            model_save_path += ".pt"
        self.log("Saving model to {}".format(model_save_path))
        torch.save(self.model.state_dict(), model_save_path)
        