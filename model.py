import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8*8*73
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=64, planes=64, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(64, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s)))
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))

        return v    # value output for the state s
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(10):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(10):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

# train
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import os
import time

class ChessNetTrainer:
    def __init__(self,net):
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def train(self,examples):
        states, values = list(zip(*examples))
        batch_size = 32
        dataset = list(zip(states,values))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.net.train()
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                # data was saved as uint8, and we need to convert to float for matmul
                inputs, labels = Variable(inputs.float()), Variable(labels.float())
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
            print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(dataloader)))
    
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({'state_dict': self.net.state_dict(),
                    'optimizer' : self.optimizer.state_dict()}, filepath)
        print("Model saved! ")
    
    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Model loaded! ")
    
    def predict(self, state):
        self.net.eval()
        state = Variable(torch.FloatTensor(state).unsqueeze(0))
        return self.net(state)