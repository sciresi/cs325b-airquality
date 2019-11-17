from torch import nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, device = "cpu"):
        super(SmallCNN, self).__init__()

        in_channels = 13
        out_channels1 = 32
        out_channels2 = 64
        out_channels3 = 128
        
        self.device = device
        
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels2, out_channels3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) 

    def forward(self, x):
        _, x = x
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = x.reshape(x.size(0), 128 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(-1)
        
        return x

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

class SmallFF(nn.Module):
    def __init__(self):
        super(SmallFF, self).__init__()
        #12,50,1 works for MSE ~ 1 on 10 datapoints
        #12,50,20,1 with lr .02 gets MSE ~21 on all of 2016
        
        self.fc1 = nn.Linear(16,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,1)
    def forward(self,x):
        x, _ = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
    
class CombinedModel(nn.Module):
    def __init__(self, non_image_model, image_model):
        