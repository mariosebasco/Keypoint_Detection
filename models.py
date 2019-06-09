#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

#modeled after research paper
class Net_orig(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #input image 1x224x224
        
        self.pool = nn.MaxPool2d(2, 2) #all pooling layers will be the same

        self.conv1 = nn.Conv2d(1, 32, 4)    #32x221x221 -> #32x110x110 after pooling
       
        self.conv2 = nn.Conv2d(32, 64, 3)   #64x108x108 -> #64x54x54 after pooling

        self.conv3 = nn.Conv2d(64, 128, 2)  #128x53x53 -> #128x26x26 after pooling

        self.conv4 = nn.Conv2d(128, 256, 1) #256x26x26 -> #256x13x13 after pooling

        self.fc1 = nn.Linear(256*13*13,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,68*2)

        #initializations found in research paper -- not working properly, need to fix
        # I.uniform_(self.conv1.weight)
        # I.uniform_(self.conv2.weight)
        # I.uniform_(self.conv3.weight)
        # I.xavier_uniform_(self.fc1.weight)
        # I.xavier_uniform_(self.fc2.weight)
        # I.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):

        dropout_1 = nn.Dropout(0.1)
        dropout_2 = nn.Dropout(0.2)
        dropout_3 = nn.Dropout(0.3)
        dropout_4 = nn.Dropout(0.4)
        dropout_5 = nn.Dropout(0.5)
        dropout_6 = nn.Dropout(0.6)
                
        x = dropout_1(self.pool(F.elu(self.conv1(x))))
        x = dropout_2(self.pool(F.elu(self.conv2(x))))
        x = dropout_3(self.pool(F.elu(self.conv3(x))))
        x = dropout_4(self.pool(F.elu(self.conv4(x))))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = dropout_5(F.elu(self.fc1(x)))
        x = dropout_6(self.fc2(x))
        x = self.fc3(x)
        
        return x


#some changes to research paper model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #input image 1x224x224

        self.pool = nnMaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(1, 32, 4)    #32x221x221 -> 32x110x110 after pooling
       
        self.conv2 = nn.Conv2d(32, 64, 3)   #64x108x108 -> 64x54x54 after pooling

        self.conv3 = nn.Conv2d(64, 128, 2)  #128x53x53 -> 128x26x26 after pooling

        self.conv4 = nn.Conv2d(128, 256, 1) #256x26x26 -> 256x13x13 after pooling

        self.lin1 = nn.Linear(256*13*13,1000)
        self.lin2 = nn.Linear(1000,1000)
        self.lin3 = nn.Linear(1000,68*2)
        
    def forward(self, x):
        #ascending dropout probabilities similar to research paper
        dropout_1 = nn.Dropout(0.1)
        dropout_2 = nn.Dropout(0.2)
        dropout_3 = nn.Dropout(0.3)
        dropout_4 = nn.Dropout(0.4)
        dropout_5 = nn.Dropout(0.5)
        dropout_6 = nn.Dropout(0.6)
        
        x = dropout_1(self.pool(F.elu(self.conv1(x))))
        x = dropout_2(self.pool(F.elu(self.conv2(x))))
        x = dropout_3(self.pool(F.elu(self.conv3(x))))
        x = dropout_4(self.pool(F.elu(self.conv4(x))))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = dropout_5(F.elu(self.lin1(x)))
        x = dropout_6(self.lin2(x))
        x = self.lin3(x)
        
        return x
